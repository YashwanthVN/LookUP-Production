"""Precision audit of what the product shows.

Step 1 (export):  uv run python scripts/label_sample.py export
    Runs the full pipeline for a FIXED random sample of tickers (seeded, so the
    sample is decided before any result is seen). The price check is bypassed
    so we measure the news itself on every ticker.
    Writes into data/label/<date>/:
      labels_blind.csv  <- fill the two label columns; no scores or providers shown
      key.csv           <- scores / scope / provider (join key; do not open while labelling)
      tickers.csv       <- per-ticker coverage (pool size, GDELT errors, abstentions)

Step 2 (label):   open labels_blind.csv, fill, for each headline:
      about_company : 1 if the article's main subject is this company, its
                      business or its stock; 0 for market-wide pieces, peers,
                      passing mentions, or a different entity
      explains_move : 1 if it reports a company-specific event that could
                      plausibly move the stock (results, guidance, deal,
                      regulatory/legal, management change, rating action); else 0
    Label from the headline and company name only. Do not look at key.csv.

Step 3 (score):   uv run python scripts/label_sample.py score data/label/<date>
"""
import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import labels as L  # noqa: E402

QUERY = "why did the stock move"


def export(args):
    os.environ["LOOKUP_SKIP_MOVE_CHECK"] = "1"
    from src.agents.coordinator import create_graph
    from src.agents.retriever_agent import get_model

    universes = {"IN": L.load_universe(ROOT / "src/utils/in_tickers.json"),
                 "US": L.load_universe(ROOT / "src/utils/us_tickers.json")}
    sample = [(m, t) for m, n in (("IN", args.n_in), ("US", args.n_us))
              for t in L.sample_tickers(universes[m], n, args.seed)]
    out = ROOT / "data" / "label" / (args.tag or dt.date.today().isoformat())
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(sample)} tickers -> {out}  (Indian ones wait on GDELT: expect minutes)")

    get_model()
    app = create_graph()
    blind, key, tick = [], [], []
    for i, (market, ticker) in enumerate(sample, 1):
        t0 = time.time()
        try:
            r = app.invoke({"query": QUERY, "tickers": [ticker]})
        except Exception as e:
            print(f"[{i}/{len(sample)}] {ticker}: FAILED {type(e).__name__}: {e}")
            tick.append({"ticker": ticker, "market": market, "error": type(e).__name__})
            continue
        primary = r["tickers"][0]
        diag = r.get("diagnostics", {})
        g = diag.get("gdelt") or {}
        tick.append({"ticker": primary, "market": market, "pool_size": diag.get("pool_size", 0),
                     "own_count": diag.get("own_count", 0), "market_count": diag.get("market_count", 0),
                     "gdelt_error": g.get("error", ""), "abstained": r.get("abstained"),
                     "abstain_reason": r.get("abstain_reason", "")})
        for rank, d in enumerate(r.get("ranked", []), 1):
            rid = f"{primary}|{rank}"
            blind.append({"id": rid, "ticker": primary, "company": r["profile"]["name"],
                          "headline": d["headline"], "source": d.get("source"),
                          "published_at": d["published_at"], "about_company": "", "explains_move": ""})
            key.append({"id": rid, "ticker": primary, "market": market, "rank": rank,
                        "rerank_score": d["rerank_score"], "scope": d["scope"],
                        "provider": d.get("provider")})
        print(f"[{i}/{len(sample)}] {primary}: pool={diag.get('pool_size', 0)} "
              f"own={diag.get('own_count', 0)} ({time.time() - t0:.0f}s)")
    pd.DataFrame(blind).sample(frac=1, random_state=args.seed).to_csv(out / "labels_blind.csv", index=False)
    pd.DataFrame(key).to_csv(out / "key.csv", index=False)
    pd.DataFrame(tick).to_csv(out / "tickers.csv", index=False)
    print(f"\nLabel: {out / 'labels_blind.csv'}  (rows are shuffled on purpose)")


def score(args):
    d = Path(args.dir)
    blind, key, tick = (pd.read_csv(d / f) for f in ("labels_blind.csv", "key.csv", "tickers.csv"))
    for c in ("about_company", "explains_move"):
        blind[c] = pd.to_numeric(blind[c], errors="coerce")
    missing = int(blind[["about_company", "explains_move"]].isna().any(axis=1).sum())
    if missing:
        print(f"WARNING: {missing}/{len(blind)} rows are not fully labelled; they are skipped.")
    m = blind.merge(key.drop(columns=["ticker"]), on="id")
    pd.set_option("display.width", 140)
    for col in ("about_company", "explains_move"):
        print(f"\n=== precision@5 of '{col}' (mean over tickers, 95% bootstrap CI over tickers)")
        print(L.summarize(m, col, 5).to_string(index=False))
        print(L.summarize(m, col, 5, group_col="market").to_string(index=False))
    print("\n=== item-level share about_company, by provider")
    print(m.dropna(subset=["about_company"]).groupby("provider")["about_company"]
          .agg(["mean", "count"]).to_string())
    print("\n=== coverage (all sampled tickers)")
    tick["has_company_news"] = tick["own_count"].fillna(0) > 0
    print(tick.groupby("market")["has_company_news"].agg(["mean", "count"]).to_string())
    if "gdelt_error" in tick:
        print("GDELT errors among Indian tickers:",
              int((tick.query("market == 'IN'")["gdelt_error"].fillna("") != "").sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--n-in", type=int, default=40)
    e.add_argument("--n-us", type=int, default=20)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--tag", default="")
    e.set_defaults(fn=export)
    s = sub.add_parser("score")
    s.add_argument("dir")
    s.set_defaults(fn=score)
    a = ap.parse_args()
    a.fn(a)