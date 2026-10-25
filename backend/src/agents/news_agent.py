"""Fetch point-in-time news for the primary ticker and its peers.

Evidence window is (as_of - LOOKBACK_H, as_of], matching the research
protocol's half-open window. Items without a parseable timestamp are dropped
by UnifiedNewsFetcher, never defaulted.
"""
import logging
import pandas as pd

from .state import LOOKUPState

log = logging.getLogger("lookup.news")
LOOKBACK_H = 72
PER_TICKER_LIMIT = 25

def news_agent(state: LOOKUPState) -> dict:
    until = pd.Timestamp(state["as_of"])
    since = until - pd.Timedelta(hours=LOOKBACK_H)
    primary = state["tickers"][0]

    pool, seen = [], set()
    for t in state["tickers"]:
        try:
            # A fresh fetcher per call: last_report/state is not shared.
            items = _make_fetcher().fetch(t, limit=PER_TICKER_LIMIT, since=since, until=until)
        except Exception as e:  # one bad provider must not sink the request
            log.warning("news fetch failed for %s: %s", t, e)
            continue
        for it in items:
            key = it["headline"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            pool.append({
                "headline": it["headline"].strip(),
                "source": it.get("source"),
                "url": it.get("url"),
                "published_at": it["published_at"].isoformat(),
                "ticker": t,
                "is_primary": t == primary,
            })
    return {"news_pool": pool}


def _make_fetcher():
    from src.streaming.news_fetcher import UnifiedNewsFetcher
    return UnifiedNewsFetcher()
