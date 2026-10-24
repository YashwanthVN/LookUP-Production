"""Live smoke check: does every configured news provider actually return
usable publish timestamps?

The unit tests prove the parsing logic is correct. This proves the live APIs
hand us fields we can parse -- which is the part that silently breaks when a
provider changes schema or a key is missing.

Run from the backend directory:
    .\source\Scripts\python.exe scripts\check_news_timestamps.py

Exit code 0 if every reachable provider yielded at least one timestamped item,
1 otherwise. Providers with no API key are reported as SKIP, not failure.
"""

import os
import sys

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.streaming.news_fetcher import UnifiedNewsFetcher  # noqa: E402

load_dotenv()

CASES = [
    ("RELIANCE.NS", "Indian equity (GNews -> yfinance)"),
    ("TCS.NS", "Indian equity (GNews -> yfinance)"),
    ("AAPL", "US equity (Finnhub -> yfinance)"),
    ("^NSEI", "Indian index (NewsAPI -> yfinance)"),
    ("XAUUSD", "Commodity (yfinance search)"),
]

KEYS = {
    "GNEWS_API_KEY": "GNews (Indian equities)",
    "FINNHUB_API_KEY": "Finnhub (US equities)",
    "NEWSAPI_KEY": "NewsAPI (Indian indices)",
}


def main() -> int:
    print("=" * 72)
    print("API keys present")
    print("=" * 72)
    for env, label in KEYS.items():
        print(f"  {'yes' if os.getenv(env) else 'NO '}  {label:38s} ({env})")

    fetcher = UnifiedNewsFetcher()
    now = pd.Timestamp.now(tz="UTC")
    since = now - pd.Timedelta(hours=72)

    failures = []
    print()
    print("=" * 72)
    print(f"Fetching window ({since:%Y-%m-%d %H:%M} , {now:%Y-%m-%d %H:%M}] UTC")
    print("=" * 72)

    for ticker, desc in CASES:
        print(f"\n--- {ticker}  [{desc}]")
        try:
            items = fetcher.fetch(ticker, limit=5, since=since, until=now)
        except Exception as e:
            print(f"    EXCEPTION: {type(e).__name__}: {e}")
            failures.append(f"{ticker}: exception")
            continue

        rep = fetcher.last_report or {}
        print(
            f"    provider={rep.get('provider', '?')}  raw={rep.get('raw', 0)}  "
            f"kept={rep.get('kept', 0)}  no_ts={rep.get('dropped_no_timestamp', 0)}  "
            f"out_of_window={rep.get('dropped_out_of_window', 0)}"
        )

        if not items:
            if rep.get("raw", 0) == 0:
                print("    SKIP: provider returned nothing (no key, rate limit, or no news)")
            else:
                print("    FAIL: provider returned items but none survived the timestamp contract")
                failures.append(f"{ticker}: {rep.get('raw')} raw, 0 usable")
            continue

        for it in items[:3]:
            age_h = (now - it["published_at"]).total_seconds() / 3600
            print(f"      {it['published_at']:%Y-%m-%d %H:%M}Z  ({age_h:5.1f}h ago)  {it['headline'][:60]}")

        assert all(it["published_at"].tzinfo is not None for it in items)
        print("    OK: all items carry tz-aware UTC timestamps")

    print()
    print("=" * 72)
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        print("=" * 72)
        return 1
    print("PASS: every reachable provider yielded timestamped items.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())