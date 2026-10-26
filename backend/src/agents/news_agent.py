"""Fetch point-in-time news for the primary ticker and its peers.

Evidence window is (as_of - LOOKBACK_H, as_of], matching the research
protocol's half-open window. Items without a parseable timestamp are dropped,
never defaulted.

For the PRIMARY ticker only, GDELT DOC search is added when the ticker is
Indian (.NS/.BO) or the normal providers returned nothing: the free news APIs
barely cover Indian companies. One GDELT call per request (it is rate-limited).
Disable with GDELT_DOC_ENABLED=0.
"""
import logging
import os
from functools import lru_cache

import pandas as pd

from .state import LOOKUPState

log = logging.getLogger("lookup.news")
LOOKBACK_H = 72
PER_TICKER_LIMIT = 25


def _make_fetcher():
    from src.streaming.news_fetcher import UnifiedNewsFetcher
    return UnifiedNewsFetcher()


@lru_cache(maxsize=512)
def _company_name(ticker: str) -> str:
    """Long company name, falling back to the bare symbol."""
    try:
        import yfinance as yf
        name = (yf.Ticker(ticker).info or {}).get("longName")
        if name:
            return name
    except Exception as e:
        log.warning("name lookup failed for %s: %s", ticker, e)
    return ticker.split(".")[0]


def _gdelt_items(ticker: str, since, until):
    from src.streaming import gdelt_doc
    symbol = ticker.split(".")[0]
    items, rep = gdelt_doc.fetch_company_news(
        _company_name(ticker), since, until, symbol=symbol, limit=PER_TICKER_LIMIT)
    rep = {k: v for k, v in rep.items() if k not in ("query", "body")}
    if rep.get("error"):
        log.warning("gdelt_doc %s failed: %s", ticker, rep["error"])
    else:
        log.info("gdelt_doc %s: %s", ticker, rep)
    return items, rep


def _gdelt_enabled() -> bool:
    return os.getenv("GDELT_DOC_ENABLED", "1") != "0"


def news_agent(state: LOOKUPState) -> dict:
    until = pd.Timestamp(state["as_of"])
    since = until - pd.Timedelta(hours=LOOKBACK_H)
    primary = state["tickers"][0]

    pool, seen = [], set()

    def add(items, t):
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
                "provider": it.get("provider"),
                "ticker": t,
                "is_primary": t == primary,
            })

    primary_n = 0
    for t in state["tickers"]:
        try:
            items = _make_fetcher().fetch(t, limit=PER_TICKER_LIMIT, since=since, until=until)
        except Exception as e:  # one bad provider must not sink the request
            log.warning("news fetch failed for %s: %s", t, e)
            items = []
        add(items, t)
        if t == primary:
            primary_n = len(items)

    diag = {"primary_from_providers": primary_n, "gdelt": None}
    indian = primary.endswith(".NS") or primary.endswith(".BO")
    if _gdelt_enabled() and (indian or primary_n == 0):
        try:
            items, rep = _gdelt_items(primary, since, until)
            add(items, primary)
            diag["gdelt"] = rep
        except Exception as e:
            log.warning("gdelt failed for %s: %s", primary, e)
            diag["gdelt"] = {"error": f"exception:{type(e).__name__}"}

    diag["pool_size"] = len(pool)
    diag["own_count"] = sum(1 for d in pool if d["is_primary"])
    return {"news_pool": pool, "diagnostics": diag}