"""Gather news for the primary ticker and its peers, escalating until there is enough.

Tiers (each runs only if the earlier ones left fewer than ENOUGH headlines that
name the company; every tier is recorded in diagnostics['tiers']):
  1. providers             window (as_of-72h, as_of]
  2. Google News, then GDELT only if still short   same window
  3. Google News, wider     (as_of-7d, as_of], plus "<name> shares" / "<name> stock"
  4. Google News, post-open (events only) (open, min(now, open+48h)]

Phases: 'pre' items were published before as_of, which for an event is the open
of the explained session (the validated protocol's evidence window, widened to 7
days in tier 3). 'post' items were published after the open. They are where
explanations usually appear, but they may report the move rather than cause it,
so they are kept apart and labelled. Tier 4 is NOT part of the validated protocol.

Items without a parseable timestamp are dropped, never defaulted. Scope:
  company - primary ticker, headline names the company
  market  - from the primary's feed but does not name the company
  peer    - from a sector peer's feed
GDELT_DOC_ENABLED=0 / GNEWS_ENABLED=0 switch the sources off.
"""
import logging
import os

import pandas as pd

from src.streaming import gdelt_doc, gnews_rss

from . import profile as profile_mod
from .state import LOOKUPState

log = logging.getLogger("lookup.news")
LOOKBACK_H = 72
WIDE_DAYS = 7
POST_HOURS = 48
ENOUGH = 5
PER_TICKER_LIMIT = 25


def _make_fetcher():
    from src.streaming.news_fetcher import UnifiedNewsFetcher
    return UnifiedNewsFetcher()


def _gdelt_enabled() -> bool:
    return os.getenv("GDELT_DOC_ENABLED", "1") != "0"


def _gnews_enabled() -> bool:
    return os.getenv("GNEWS_ENABLED", "1") != "0"


def _clean(rep: dict) -> dict:
    return {k: v for k, v in rep.items() if k not in ("query", "body")}


def _gdelt_items(ticker: str, since, until):
    symbol = ticker.split(".")[0]
    name = profile_mod.get_profile(ticker)["name"]
    items, rep = gdelt_doc.fetch_company_news(name, since, until, symbol=symbol,
                                              limit=PER_TICKER_LIMIT)
    return items, _clean(rep)


def news_agent(state: LOOKUPState) -> dict:
    mv = state.get("move") or {}
    if mv.get("requested_date") and mv.get("status") == "unavailable":
        return {"news_pool": [], "diagnostics": {"skipped": "requested_date_unavailable"}}
    until = pd.Timestamp(state["as_of"])
    since = until - pd.Timedelta(hours=LOOKBACK_H)
    primary = state["tickers"][0]
    prof = state.get("profile") or profile_mod.get_profile(primary)
    core = gdelt_doc.core_name(prof.get("name") or primary)
    symbol = primary.split(".")[0]
    indian = primary.endswith(".NS") or primary.endswith(".BO")
    region = "IN" if indian else "US"

    pool, seen, tiers = [], set(), []

    def add(items, t, phase, tier):
        n = 0
        for it in items:
            headline = it["headline"].strip()
            key = headline.lower()
            if key in seen:
                continue
            seen.add(key)
            if t != primary:
                scope = "peer"
            else:
                scope = "company" if gdelt_doc.names_company(headline, core, symbol) else "market"
            pool.append({
                "headline": headline, "source": it.get("source"), "url": it.get("url"),
                "published_at": it["published_at"].isoformat(),
                "provider": it.get("provider"), "ticker": t,
                "is_primary": t == primary, "scope": scope, "phase": phase, "tier": tier,
            })
            n += scope == "company"
        return n

    def own_pre() -> int:
        return sum(1 for d in pool if d["scope"] == "company" and d["phase"] == "pre")

    def record(tier, added, rep=None):
        row = {"tier": tier, "company_added": added}
        if rep and rep.get("error"):
            row["error"] = rep["error"]
            log.warning("%s %s failed: %s", tier, primary, rep["error"])
        tiers.append(row)

    # ── tier 1: providers (+ GDELT) ──────────────────────────────────────────
    primary_n = 0
    for t in state["tickers"]:
        try:
            items = _make_fetcher().fetch(t, limit=PER_TICKER_LIMIT, since=since, until=until)
        except Exception as e:  # one bad provider must not sink the request
            log.warning("news fetch failed for %s: %s", t, e)
            items = []
        added = add(items, t, "pre", "providers")
        if t == primary:
            primary_n = len(items)
            record("providers", added)

    diag = {"primary_from_providers": primary_n, "gdelt": None}

    def run_gdelt():
        # GDELT is slow (8 s spacing, 10 s retry on 429), so it runs only if needed
        if not (_gdelt_enabled() and (indian or primary_n == 0)):
            return
        rep = {}
        try:
            items, rep = _gdelt_items(primary, since, until)
        except Exception as e:
            rep = {"error": f"exception:{type(e).__name__}"}
            items = []
        record("gdelt", add(items, primary, "pre", "gdelt"), rep)
        diag["gdelt"] = rep

    # ── tiers 2-4: Google News ───────────────────────────────────────────────
    def gn(term, s, u, phase, tier):
        try:
            items, rep = gnews_rss.fetch_news(term, s, u, region=region, name=core,
                                              symbol=symbol, limit=PER_TICKER_LIMIT)
            rep = _clean(rep)
        except Exception as e:
            items, rep = [], {"error": f"exception:{type(e).__name__}"}
        record(f"{tier}:{term}" if term != core else tier, add(items, primary, phase, tier), rep)

    if not (_gnews_enabled() and core):
        run_gdelt()
    else:
        gn(core, since, until, "pre", "google_news")
        if own_pre() < ENOUGH:
            run_gdelt()
        if own_pre() < ENOUGH:
            wide = until - pd.Timedelta(days=WIDE_DAYS)
            for term in (core, f"{core} shares", f"{core} stock"):
                gn(term, wide, until, "pre", "google_news_7d")
                if own_pre() >= ENOUGH:
                    break
        if state.get("mode") == "explained" or (mv.get("facts") or {}).get("intraday_flag"):
            now = pd.Timestamp(state.get("generated_at") or pd.Timestamp.now(tz="UTC"))
            post_until = min(now, until + pd.Timedelta(hours=POST_HOURS))
            if post_until > until:
                for term in (core, f"{core} shares"):
                    gn(term, until, post_until, "post", "google_news_after_open")

    diag["tiers"] = tiers
    diag["pool_size"] = len(pool)
    diag["own_count"] = own_pre()
    diag["post_count"] = sum(1 for d in pool if d["scope"] == "company" and d["phase"] == "post")
    diag["market_count"] = sum(1 for d in pool if d["scope"] == "market")
    return {"news_pool": pool, "diagnostics": diag}