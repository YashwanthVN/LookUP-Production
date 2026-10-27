"""Company profile (name + business summary) from yfinance, cached.

Only successful lookups are cached, so a transient failure is retried.
"""
import logging
import time
from typing import Dict

log = logging.getLogger("lookup.profile")
TTL_S = 6 * 3600
_cache: Dict[str, tuple] = {}


def get_profile(ticker: str) -> Dict[str, str]:
    now = time.monotonic()
    hit = _cache.get(ticker)
    if hit and now - hit[0] < TTL_S:
        return hit[1]
    info = {}
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
    except Exception as e:
        log.warning("profile lookup failed for %s: %s", ticker, e)
    name = info.get("longName") or info.get("shortName")
    prof = {"name": name or ticker.split(".")[0],
            "summary": info.get("longBusinessSummary") or ""}
    if name:
        _cache[ticker] = (now, prof)
    return prof