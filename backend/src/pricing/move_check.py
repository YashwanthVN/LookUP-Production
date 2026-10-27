"""Did the stock make an unusual move in its latest COMPLETE session?

Follows the research event definition in spirit: close-to-close return,
market-adjusted with a lagged rolling beta, flagged when |z| >= 2.5.

PARITY NOT YET VERIFIED against lookup-attribution (data/prices.py,
eval/events.py). The window, lag and threshold below are my reading of the
protocol. Do not quote numbers from this module as research results until a
parity test against the research code passes.
"""
import datetime as dt
import re
import time
from typing import Any, Callable, Dict, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BETA_WINDOW = 250   # trading days in the beta estimation window
BETA_LAG = 30       # window ends this many trading days before the event day
MIN_EST = 120
Z_THRESHOLD = 2.5
MAX_STALE_DAYS = 6
LOOKBACK_SESSIONS = 5   # how many recent complete sessions are scanned for an event

EXCHANGES = {
    "IN": {"tz": "Asia/Kolkata", "open": dt.time(9, 15), "close": dt.time(15, 30),
           "bench": "^NSEI", "bench_name": "NIFTY 50"},
    "US": {"tz": "America/New_York", "open": dt.time(9, 30), "close": dt.time(16, 0),
           "bench": "^GSPC", "bench_name": "S&P 500"},
}
_US = re.compile(r"^[A-Z]{1,5}([.-][A-Z])?$")

_DOWN = {"fall", "falls", "falling", "fallen", "fell", "drop", "drops", "dropping",
         "dropped", "dip", "dips", "dipped", "dipping", "decline", "declined",
         "declining", "tumble", "tumbled", "tumbling", "slump", "slumped", "crash",
         "crashed", "plunge", "plunged", "sink", "sank", "down", "lower", "slide", "slid"}
_UP = {"rise", "rises", "rising", "rose", "risen", "gain", "gains", "gained",
       "gaining", "jump", "jumped", "surge", "surged", "soar", "soared", "rally",
       "rallied", "climb", "climbed", "up", "higher", "spike", "spiked"}


def exchange_for(ticker: str) -> Optional[str]:
    t = (ticker or "").upper()
    if t.endswith(".NS") or t.endswith(".BO"):
        return "IN"
    if _US.match(t):
        return "US"
    return None


def query_direction(query: str) -> Optional[str]:
    words = set(re.findall(r"[a-z]+", (query or "").lower()))
    d, u = bool(words & _DOWN), bool(words & _UP)
    if d and not u:
        return "down"
    if u and not d:
        return "up"
    return None


def session_open_utc(d: dt.date, ex: str) -> pd.Timestamp:
    cfg = EXCHANGES[ex]
    local = dt.datetime.combine(d, cfg["open"], tzinfo=ZoneInfo(cfg["tz"]))
    return pd.Timestamp(local.astimezone(dt.timezone.utc)).tz_convert("UTC")


def drop_partial(close: pd.Series, now_utc: pd.Timestamp, ex: str) -> pd.Series:
    """Remove today's bar if that exchange's session has not closed yet."""
    cfg = EXCHANGES[ex]
    local_now = now_utc.tz_convert(cfg["tz"])
    if close.empty:
        return close
    if close.index[-1].date() == local_now.date() and local_now.time() < cfg["close"]:
        return close.iloc[:-1]
    return close


def market_adjusted_move(stock: pd.Series, market: pd.Series, window: int = BETA_WINDOW,
                         lag: int = BETA_LAG, min_est: int = MIN_EST,
                         offset: int = 0) -> Dict[str, Any]:
    """Market-adjusted move for the session `offset` sessions before the latest."""
    closes = pd.concat({"s": stock, "m": market}, axis=1, join="inner").dropna()
    rets = closes.pct_change().dropna()
    pos = len(rets) - 1 - offset
    if pos < lag + min_est:
        return {"status": "unavailable", "reason": "insufficient_history"}
    est = rets.iloc[max(0, pos - lag - window): pos - lag]
    var_m = est["m"].var(ddof=1)
    if not var_m > 0:
        return {"status": "unavailable", "reason": "flat_market"}
    beta = est["s"].cov(est["m"]) / var_m
    sigma = (est["s"] - beta * est["m"]).std(ddof=1)
    if not sigma > 0:
        return {"status": "unavailable", "reason": "flat_stock"}
    t = rets.iloc[pos]
    ar = t["s"] - beta * t["m"]
    return {"status": "ok", "date": rets.index[pos].date().isoformat(),
            "ret": float(t["s"]), "bench_ret": float(t["m"]), "beta": float(beta),
            "abnormal": float(ar), "z": float(ar / sigma), "n_est": int(len(est))}


_cache: Dict[str, Any] = {}


def fetch_closes(symbol: str, period: str = "3y", ttl: float = 600.0) -> Optional[pd.Series]:
    now = time.monotonic()
    hit = _cache.get(symbol)
    if hit and now - hit[0] < ttl:
        return hit[1]
    import yfinance as yf
    h = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
    if h is None or h.empty or "Close" not in h:
        return None
    s = h["Close"].astype(float)
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")].dropna()
    _cache[symbol] = (now, s)
    return s


def check_move(ticker: str, now: Optional[pd.Timestamp] = None,
               fetch: Optional[Callable[[str], Optional[pd.Series]]] = None,
               direction: Optional[str] = None,
               lookback: int = LOOKBACK_SESSIONS) -> Dict[str, Any]:
    """Scan the last `lookback` complete sessions. Never raises.

    status: 'event' | 'no_event' | 'unavailable'.
    If several sessions qualify, the MOST RECENT one is explained, preferring
    one whose direction matches `direction` ('down'/'up') when given.
    Note: scanning k sessions raises the chance that a quiet stock shows an
    event by luck, roughly k x the per-session false-positive rate.
    """
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    ex = exchange_for(ticker)
    if ex is None:
        return {"status": "unavailable", "reason": "unsupported_instrument"}
    cfg = EXCHANGES[ex]
    fetch = fetch or fetch_closes
    try:
        stock, market = fetch(ticker), fetch(cfg["bench"])
    except Exception as e:
        return {"status": "unavailable", "reason": f"fetch_failed:{type(e).__name__}"}
    if stock is None or market is None or len(stock) == 0 or len(market) == 0:
        return {"status": "unavailable", "reason": "no_prices"}
    stock, market = drop_partial(stock, now, ex), drop_partial(market, now, ex)

    sessions, first_fail = [], None
    for off in range(lookback):
        r = market_adjusted_move(stock, market, offset=off)
        if r["status"] == "ok":
            sessions.append(r)           # newest first
        elif first_fail is None:
            first_fail = r
    if not sessions:
        return first_fail or {"status": "unavailable", "reason": "insufficient_history"}

    latest = sessions[0]
    if (now.date() - dt.date.fromisoformat(latest["date"])).days > MAX_STALE_DAYS:
        return {"status": "unavailable", "reason": "stale_prices"}

    dir_of = lambda r: "down" if r["abnormal"] < 0 else "up"
    events = [r for r in sessions if abs(r["z"]) >= Z_THRESHOLD]
    if events:
        preferred = [r for r in events if direction and dir_of(r) == direction]
        chosen = (preferred or events)[0]
    else:
        chosen = latest

    d = dt.date.fromisoformat(chosen["date"])
    res = dict(chosen)
    res.update(
        status="event" if events else "no_event",
        exchange=ex, benchmark=cfg["bench_name"], direction=dir_of(chosen),
        session_open=session_open_utc(d, ex).isoformat(),
        z_threshold=Z_THRESHOLD, lookback=len(sessions), latest_date=latest["date"],
        recent=[{"date": r["date"], "ret": r["ret"], "abnormal": r["abnormal"], "z": r["z"]}
                for r in sessions],
    )
    return res


def peer_context(peers, date: str, fetch: Optional[Callable[[str], Optional[pd.Series]]] = None,
                 now: Optional[pd.Timestamp] = None, scan: int = 10) -> list:
    """Each peer's market-adjusted move on the SAME session `date` (YYYY-MM-DD).

    Price facts only: they say whether a move looks sector-wide or specific to the
    stock, not why it happened. Never raises; peers with no data are skipped.
    """
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    fetch = fetch or fetch_closes
    out = []
    for p in peers:
        try:
            ex = exchange_for(p)
            if ex is None:
                continue
            stock, market = fetch(p), fetch(EXCHANGES[ex]["bench"])
            if stock is None or market is None:
                continue
            stock, market = drop_partial(stock, now, ex), drop_partial(market, now, ex)
            for off in range(scan):
                r = market_adjusted_move(stock, market, offset=off)
                if r["status"] != "ok":
                    break
                if r["date"] == date:
                    out.append({"ticker": p, "ret": r["ret"], "abnormal": r["abnormal"],
                                "z": r["z"]})
                    break
                if r["date"] < date:
                    break
        except Exception:
            continue
    return out