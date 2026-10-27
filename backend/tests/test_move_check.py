"""Offline tests for the price-move check (synthetic series; PROTOCOL rule 12:
used only to test code correctness, never to produce reported numbers)."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.pricing import move_check as mc

# conftest replaces mc.check_move with a stub for every test; keep the real one.
REAL_CHECK_MOVE = mc.check_move


def series(n=400, last_stock=0.002, last_mkt=0.0, seed=0, shocks=None):
    """shocks: {sessions_before_last: stock_return}; market return 0 on those days."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-23", periods=n)
    m = rng.normal(0.0004, 0.01, n)
    s = 1.2 * m + rng.normal(0, 0.01, n)
    # quiet days in the scan window so only explicit shocks can be events
    for k in range(1, 8):
        m[-1 - k], s[-1 - k] = 0.0, 0.001
    for k, r in (shocks or {}).items():
        m[-1 - k], s[-1 - k] = 0.0, r
    m[-1], s[-1] = last_mkt, last_stock
    close = lambda r: pd.Series(100 * np.cumprod(1 + r), index=idx)
    return close(s), close(m)


def test_exchange_for():
    assert mc.exchange_for("RELIANCE.NS") == "IN"
    assert mc.exchange_for("tcs.bo") == "IN"
    assert mc.exchange_for("AAPL") == "US"
    assert mc.exchange_for("BRK-B") == "US"
    assert mc.exchange_for("XAUUSD") is None
    assert mc.exchange_for("^NSEI") is None
    assert mc.exchange_for("GC=F") is None


def test_query_direction():
    assert mc.query_direction("Why is AAPL dropping?") == "down"
    assert mc.query_direction("why did gold rally") == "up"
    assert mc.query_direction("why did it rise and then fall") is None
    assert mc.query_direction("Infosys outlook") is None


def test_session_open_utc_handles_dst():
    assert mc.session_open_utc(dt.date(2026, 10, 23), "IN") == pd.Timestamp("2026-10-23T03:45:00Z")
    assert mc.session_open_utc(dt.date(2026, 10, 23), "US") == pd.Timestamp("2026-10-23T13:30:00Z")  # EDT
    assert mc.session_open_utc(dt.date(2026, 11, 2), "US") == pd.Timestamp("2026-11-02T14:30:00Z")   # EST


def test_drop_partial_indian_session():
    s, _ = series()
    s.index = pd.bdate_range(end="2026-10-26", periods=len(s))
    midday = pd.Timestamp("2026-10-26T06:00:00Z")          # 11:30 IST, session open
    assert mc.drop_partial(s, midday, "IN").index[-1] == pd.Timestamp("2026-10-23")
    after = pd.Timestamp("2026-10-26T11:00:00Z")           # 16:30 IST, closed
    assert mc.drop_partial(s, after, "IN").index[-1] == pd.Timestamp("2026-10-26")


def test_drop_partial_us_session():
    s, _ = series()
    s.index = pd.bdate_range(end="2026-10-26", periods=len(s))
    assert mc.drop_partial(s, pd.Timestamp("2026-10-26T15:00:00Z"), "US").index[-1] == pd.Timestamp("2026-10-23")
    assert mc.drop_partial(s, pd.Timestamp("2026-10-26T21:00:00Z"), "US").index[-1] == pd.Timestamp("2026-10-26")


def test_normal_day_is_not_an_event():
    s, m = series(last_stock=0.002)
    r = mc.market_adjusted_move(s, m)
    assert r["status"] == "ok" and abs(r["z"]) < mc.Z_THRESHOLD
    assert 0.9 < r["beta"] < 1.5          # data was generated with beta 1.2


def test_shock_day_is_an_event_and_signed():
    s, m = series(last_stock=-0.08)
    r = mc.market_adjusted_move(s, m)
    assert r["z"] < -mc.Z_THRESHOLD and r["abnormal"] < 0
    s, m = series(last_stock=0.08)
    assert mc.market_adjusted_move(s, m)["z"] > mc.Z_THRESHOLD


def test_market_move_is_removed():
    # stock falls 2.4% but the market fell 2% with beta ~1.2 -> little abnormal move
    s, m = series(last_stock=-0.024, last_mkt=-0.02)
    r = mc.market_adjusted_move(s, m)
    assert abs(r["abnormal"]) < 0.015


def test_insufficient_history():
    s, m = series(n=100)
    assert mc.market_adjusted_move(s, m) == {"status": "unavailable", "reason": "insufficient_history"}


def _fetcher(stock, market):
    return lambda sym: market if sym.startswith("^") else stock


NOW = pd.Timestamp("2026-10-24T10:00:00Z")   # Saturday; last bar is Friday 23rd


def test_check_move_event_carries_session_open():
    s, m = series(last_stock=-0.08)
    r = REAL_CHECK_MOVE("RELIANCE.NS", now=NOW, fetch=_fetcher(s, m))
    assert r["status"] == "event" and r["direction"] == "down"
    assert r["date"] == "2026-10-23" and r["benchmark"] == "NIFTY 50"
    assert pd.Timestamp(r["session_open"]) == pd.Timestamp("2026-10-23T03:45:00Z")


def test_check_move_no_event():
    s, m = series(last_stock=0.001)
    assert REAL_CHECK_MOVE("AAPL", now=NOW, fetch=_fetcher(s, m))["status"] == "no_event"


def test_check_move_unavailable_paths():
    s, m = series()
    assert REAL_CHECK_MOVE("XAUUSD", now=NOW)["reason"] == "unsupported_instrument"
    assert REAL_CHECK_MOVE("AAPL", now=NOW, fetch=lambda sym: None)["reason"] == "no_prices"
    def boom(sym): raise ConnectionError("x")
    assert REAL_CHECK_MOVE("AAPL", now=NOW, fetch=boom)["reason"].startswith("fetch_failed")
    late = pd.Timestamp("2026-11-30T10:00:00Z")
    assert REAL_CHECK_MOVE("AAPL", now=late, fetch=_fetcher(s, m))["reason"] == "stale_prices"


def test_event_a_few_sessions_back_is_found():
    s, m = series(shocks={3: -0.08})
    r = REAL_CHECK_MOVE("RELIANCE.NS", now=NOW, fetch=_fetcher(s, m))
    assert r["status"] == "event" and r["direction"] == "down"
    assert r["date"] == s.index[-4].date().isoformat()       # 3 sessions before the last bar
    assert r["latest_date"] == "2026-10-23" and r["lookback"] == 5 and len(r["recent"]) == 5
    assert pd.Timestamp(r["session_open"]) == mc.session_open_utc(s.index[-4].date(), "IN")


def test_event_outside_the_window_is_ignored():
    s, m = series(shocks={7: -0.08})
    r = REAL_CHECK_MOVE("AAPL", now=NOW, fetch=_fetcher(s, m))
    assert r["status"] == "no_event" and len(r["recent"]) == 5


def test_most_recent_event_wins_unless_direction_requested():
    s, m = series(shocks={3: -0.08, 1: 0.08})
    f = _fetcher(s, m)
    assert REAL_CHECK_MOVE("AAPL", now=NOW, fetch=f)["direction"] == "up"                       # most recent
    assert REAL_CHECK_MOVE("AAPL", now=NOW, fetch=f, direction="down")["direction"] == "down"   # asked about a fall
    assert REAL_CHECK_MOVE("AAPL", now=NOW, fetch=f, direction="up")["direction"] == "up"


def test_requested_direction_absent_falls_back_to_the_event_that_exists():
    s, m = series(shocks={2: 0.08})
    r = REAL_CHECK_MOVE("AAPL", now=NOW, fetch=_fetcher(s, m), direction="down")
    assert r["status"] == "event" and r["direction"] == "up"