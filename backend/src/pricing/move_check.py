"""Did the stock make an unusual move in its latest COMPLETE session?

Follows the research event definition in spirit: close-to-close return,
market-adjusted with a lagged rolling beta, flagged when |z| >= 2.5 ("event").
Below that, a session is "notable" if its raw move is >= 3% or it had a large
intraday swing. NOTABLE is a product judgment, not part of the research protocol,
and is always reported as a weaker signal.

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
NOTABLE_RET = 0.03      # below the event threshold, a raw move this large (or a big intraday swing) is 'notable'

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


def fetch_ohlc(symbol: str, period: str = "3y", ttl: float = 600.0) -> Optional[pd.DataFrame]:
    """Daily Open/High/Low/Close, indexed by session date (one yfinance call, cached)."""
    now = time.monotonic()
    hit = _cache.get(symbol)
    if hit and now - hit[0] < ttl:
        return hit[1]
    import yfinance as yf
    h = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
    if h is None or h.empty or "Close" not in h:
        return None
    cols = [c for c in ("Open", "High", "Low", "Close") if c in h]
    df = h[cols].astype(float)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["Close"])
    _cache[symbol] = (now, df)
    return df


def fetch_closes(symbol: str, period: str = "3y", ttl: float = 600.0) -> Optional[pd.Series]:
    df = fetch_ohlc(symbol, period, ttl)
    return None if df is None else df["Close"]


EXCURSION_MIN = 0.03     # an intraday swing must be at least 3% of the previous close...
EXCURSION_MULT = 2.0     # ...and at least 2x the stock's median daily swing. Descriptive, not validated.


def session_facts(ohlc: Optional[pd.DataFrame], date: str, now: pd.Timestamp, ex: str) -> Optional[Dict[str, Any]]:
    """Open/low/high/close of one session versus the previous close. Never raises."""
    try:
        if ohlc is None or not {"Open", "High", "Low", "Close"} <= set(ohlc.columns):
            return None
        keep = drop_partial(ohlc["Close"], now, ex).index
        df = ohlc.loc[ohlc.index.isin(keep)]
        d = pd.Timestamp(date)
        if d not in df.index:
            return None
        pos = df.index.get_loc(d)
        if pos < 1:
            return None
        prev = df["Close"].iloc[pos - 1]
        row = df.iloc[pos]
        win = df.iloc[max(0, pos - 250): pos + 1]
        prevs = win["Close"].shift(1)
        exc = pd.concat([(win["Low"] / prevs - 1).abs(), (win["High"] / prevs - 1).abs()], axis=1).max(axis=1)
        typical = float(exc.iloc[:-1].median()) if len(exc) > 20 else float("nan")
        low_ret, high_ret = float(row["Low"] / prev - 1), float(row["High"] / prev - 1)
        worst = max(abs(low_ret), abs(high_ret))
        flag = bool(worst >= EXCURSION_MIN and typical == typical and worst >= EXCURSION_MULT * typical)
        return {"date": date, "prev_close": float(prev), "open": float(row["Open"]),
                "gap": float(row["Open"] / prev - 1), "low_ret": low_ret, "high_ret": high_ret,
                "close_ret": float(row["Close"] / prev - 1), "typical_excursion": typical,
                "intraday_flag": flag}
    except Exception:
        return None


def offset_for_date(stock: pd.Series, market: pd.Series, date: str) -> Optional[int]:
    """Sessions between `date` and the latest return, or None if `date` is not a session."""
    closes = pd.concat({"s": stock, "m": market}, axis=1, join="inner").dropna()
    idx = closes.pct_change().dropna().index
    d = pd.Timestamp(date)
    if d not in idx:
        return None
    return len(idx) - 1 - idx.get_loc(d)


def _facts(ohlc_fetch, ticker, date, now, ex):
    if ohlc_fetch is None:
        return None
    try:
        return session_facts(ohlc_fetch(ticker), date, now, ex)
    except Exception:
        return None


def _check_one_session(ticker, ex, cfg, stock, market, target: dt.date, now, ohlc_fetch):
    """The user named a day: test exactly that session (one test, no multiple-testing inflation)."""
    date = target.isoformat()
    base = {"requested_date": date}
    off = offset_for_date(stock, market, date)
    if off is None:
        return {"status": "unavailable", "reason": "no_complete_session_on_that_date", **base}
    r = market_adjusted_move(stock, market, offset=off)
    if r["status"] != "ok":
        return {**r, "reason": r.get("reason", "insufficient_history"), "status": "unavailable", **base}
    d = dt.date.fromisoformat(r["date"])
    event = abs(r["z"]) >= Z_THRESHOLD
    facts = _facts(ohlc_fetch, ticker, r["date"], now, ex)
    big = abs(r["ret"]) >= NOTABLE_RET or bool(facts and facts["intraday_flag"])
    res = dict(r)
    res.update(
        status="event" if event else ("notable" if big else "no_event"), exchange=ex, benchmark=cfg["bench_name"],
        direction="down" if r["abnormal"] < 0 else "up",
        session_open=session_open_utc(d, ex).isoformat(), z_threshold=Z_THRESHOLD,
        lookback=1, latest_date=r["date"], recent=[{"date": r["date"], "ret": r["ret"],
                                                  "abnormal": r["abnormal"], "z": r["z"]}],
        facts=facts, **base)
    return res


def check_move(ticker: str, now: Optional[pd.Timestamp] = None,
               fetch: Optional[Callable[[str], Optional[pd.Series]]] = None,
               direction: Optional[str] = None,
               lookback: int = LOOKBACK_SESSIONS,
               target_date: Optional[dt.date] = None,
               ohlc_fetch: Optional[Callable[[str], Optional[pd.DataFrame]]] = None) -> Dict[str, Any]:
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
    if ohlc_fetch is None and fetch is None:
        ohlc_fetch = fetch_ohlc          # facts only when real prices are being fetched
    fetch = fetch or fetch_closes
    try:
        stock, market = fetch(ticker), fetch(cfg["bench"])
    except Exception as e:
        return {"status": "unavailable", "reason": f"fetch_failed:{type(e).__name__}"}
    if stock is None or market is None or len(stock) == 0 or len(market) == 0:
        return {"status": "unavailable", "reason": "no_prices"}
    stock, market = drop_partial(stock, now, ex), drop_partial(market, now, ex)

    if target_date is not None:
        return _check_one_session(ticker, ex, cfg, stock, market, target_date, now, ohlc_fetch)

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
    notable = []
    if not events:
        for r in sessions:
            f = _facts(ohlc_fetch, ticker, r["date"], now, ex)
            r["_facts"] = f
            if abs(r["ret"]) >= NOTABLE_RET or (f and f["intraday_flag"]):
                notable.append(r)
    pool = events or notable
    if pool:
        preferred = [r for r in pool if direction and dir_of(r) == direction]
        chosen = (preferred or pool)[0]
    else:
        chosen = latest

    d = dt.date.fromisoformat(chosen["date"])
    res = dict(chosen)
    res.update(
        status="event" if events else ("notable" if notable else "no_event"),
        exchange=ex, benchmark=cfg["bench_name"], direction=dir_of(chosen),
        session_open=session_open_utc(d, ex).isoformat(),
        z_threshold=Z_THRESHOLD, lookback=len(sessions), latest_date=latest["date"],
        recent=[{"date": r["date"], "ret": r["ret"], "abnormal": r["abnormal"], "z": r["z"]}
                for r in sessions],
    )
    res.pop("_facts", None)
    res["facts"] = chosen.get("_facts") or _facts(ohlc_fetch, ticker, chosen["date"], now, ex)
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