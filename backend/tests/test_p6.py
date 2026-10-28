"""P6: named dates, session facts (open/low/high/close), headline price-claim checks."""
import datetime as dt

import numpy as np
import pandas as pd

from src.agents import coordinator, news_agent, retriever_agent
from src.pricing import claims as cl
from src.pricing import move_check as mc
from src.pricing import query_dates as qd
from tests.test_move_check import NOW, _fetcher, series

REAL_CHECK_MOVE = mc.check_move
TODAY = dt.date(2026, 10, 27)


# ── dates in questions ───────────────────────────────────────────────────────
def test_find_date_forms():
    f = lambda s: qd.find_date(s, TODAY)[0]
    assert f("why did Infosys fall on 16 Oct") == dt.date(2026, 10, 16)
    assert f("Infosys 16th October 2026") == dt.date(2026, 10, 16)
    assert f("what moved TSLA on Oct 16th?") == dt.date(2026, 10, 16)
    assert f("on 2026-10-16") == dt.date(2026, 10, 16)
    assert f("why did it drop yesterday") == dt.date(2026, 10, 26)
    assert f("Reliance 5 Nov") == dt.date(2025, 11, 5)          # future date -> last year
    assert f("it may rise soon") is None and f("30 feb") is None and f("3/4") is None


def test_strip_date_leaves_the_subject():
    assert qd.strip_date("why did Infosys fall on 16 Oct", TODAY).split() == ["why", "did", "Infosys", "fall", "on"]
    assert qd.strip_date("Infosys outlook", TODAY) == "Infosys outlook"


# ── headline claims ──────────────────────────────────────────────────────────
FACTS = {"close_ret": 0.0106, "gap": -0.0307, "low_ret": -0.039, "high_ret": 0.0116}


def test_claim_matches_intraday_low():
    r = cl.check("Infosys shares crash 4% - Should investors buy, sell, hold?", "Infosys", FACTS)
    assert r["verdict"] == "matches" and "intraday low" in r["detail"]


def test_claim_inside_range_is_an_intraday_snapshot():
    r = cl.check("Infosys shares fall 2.7% as company cuts forecast", "Infosys", FACTS)
    assert r["verdict"] == "within_range"


def test_claim_beyond_the_range_is_contradicted():
    r = cl.check("Infosys shares drop 6.5% after Q2 EPS miss", "Infosys", FACTS)
    assert r["verdict"] == "contradicted" and "NOT supported" in r["detail"]


def test_claim_direction_opposite_to_the_session_is_contradicted():
    up = {**FACTS, "low_ret": 0.001, "gap": 0.002}
    assert cl.check("Infosys shares crash 4%", "Infosys", up)["verdict"] == "contradicted"


def test_claims_about_other_things_are_ignored():
    assert cl.check("JSW Steel net leaps 225% to Rs 5,359 crore", "JSW Steel", FACTS) is None
    assert cl.check("Dr. Reddy's rallies 7%, Shriram falls 5%", "JSW Steel", FACTS) is None
    assert cl.check("Dr. Reddy's rallies 7%, JSW Steel gains 6%", "JSW Steel",
                    {**FACTS, "close_ret": 0.06, "high_ret": 0.07})["verdict"] == "matches"
    assert cl.check("Infosys shares dip as outlook weakens", "Infosys", FACTS) is None
    assert cl.check("anything", "Infosys", None) is None


def test_tolerance_follows_headline_rounding():
    assert cl.extract_claim("Infosys shares crash 4%", "Infosys")["tol"] > cl.extract_claim("Infosys shares crash 3.9%", "Infosys")["tol"]


# ── session facts ────────────────────────────────────────────────────────────
def ohlc_from(close: pd.Series, last_open=None, last_low=None, last_high=None):
    df = pd.DataFrame({"Open": close.shift(1).fillna(close.iloc[0]), "Close": close})
    df["High"] = df[["Open", "Close"]].max(axis=1) * 1.004
    df["Low"] = df[["Open", "Close"]].min(axis=1) * 0.996
    if last_open is not None:
        df.iloc[-1, df.columns.get_loc("Open")] = last_open * df["Close"].iloc[-2]
    if last_low is not None:
        df.iloc[-1, df.columns.get_loc("Low")] = last_low * df["Close"].iloc[-2]
    if last_high is not None:
        df.iloc[-1, df.columns.get_loc("High")] = last_high * df["Close"].iloc[-2]
    return df


def test_session_facts_flags_a_swing_that_recovered():
    s, m = series(last_stock=0.011)
    df = ohlc_from(s, last_open=0.9693, last_low=0.95)
    f = mc.session_facts(df, "2026-10-23", NOW, "IN")
    assert abs(f["gap"] + 0.0307) < 1e-6 and abs(f["low_ret"] + 0.05) < 1e-6
    assert abs(f["close_ret"] - 0.011) < 1e-3 and f["intraday_flag"] is True


def test_session_facts_quiet_day_is_not_flagged_and_bad_input_is_safe():
    s, m = series(last_stock=0.002)
    f = mc.session_facts(ohlc_from(s), "2026-10-23", NOW, "IN")
    assert f["intraday_flag"] is False
    assert mc.session_facts(None, "2026-10-23", NOW, "IN") is None
    assert mc.session_facts(ohlc_from(s), "2026-10-24", NOW, "IN") is None      # Saturday


def test_check_move_attaches_facts_when_ohlc_is_given():
    s, m = series(last_stock=0.011)
    df = ohlc_from(s, last_open=0.9693, last_low=0.95)
    r = REAL_CHECK_MOVE("INFY.NS", now=NOW, fetch=_fetcher(s, m), ohlc_fetch=lambda t: df)
    assert r["facts"]["intraday_flag"] is True
    assert REAL_CHECK_MOVE("INFY.NS", now=NOW, fetch=_fetcher(s, m))["facts"] is None


# ── a named day ──────────────────────────────────────────────────────────────
def test_named_day_is_tested_alone():
    s, m = series(shocks={3: -0.08})
    day = s.index[-4].date()
    r = REAL_CHECK_MOVE("RELIANCE.NS", now=NOW, fetch=_fetcher(s, m), target_date=day)
    assert r["status"] == "event" and r["date"] == day.isoformat() and r["requested_date"] == day.isoformat()
    assert r["lookback"] == 1 and len(r["recent"]) == 1
    quiet = REAL_CHECK_MOVE("RELIANCE.NS", now=NOW, fetch=_fetcher(s, m), target_date=s.index[-6].date())
    assert quiet["status"] == "no_event" and quiet["requested_date"]


def test_named_day_that_is_not_a_session():
    s, m = series()
    r = REAL_CHECK_MOVE("RELIANCE.NS", now=NOW, fetch=_fetcher(s, m), target_date=dt.date(2026, 10, 24))  # Saturday
    assert r["status"] == "unavailable" and r["reason"] == "no_complete_session_on_that_date"
    assert r["requested_date"] == "2026-10-24"


# ── graph ────────────────────────────────────────────────────────────────────
class _FakeModel:
    def predict(self, pairs): return [float(len(h)) for _, h in pairs]


class F:
    """Returns one headline just before the window end (pre) for the primary ticker."""
    def __init__(self): self.untils = []
    def fetch(self, ticker, limit, since, until):
        self.untils.append(until)
        if ticker != "XYZ":
            return []
        return [{"headline": "XYZ shares crash 4% - should you buy?", "source": "Mint", "url": "u",
                 "published_at": until - pd.Timedelta(hours=1), "provider": "f"}]


def _graph(monkeypatch, move, fetcher=None):
    from src.pricing import move_check
    monkeypatch.setattr(news_agent, "_make_fetcher", lambda: fetcher or F())
    monkeypatch.setattr(retriever_agent, "get_model", lambda: _FakeModel())
    monkeypatch.setattr(coordinator, "resolve_peers", lambda t: (t, [], "S"))
    monkeypatch.setattr(move_check, "check_move", lambda t, *a, **k: move)
    return coordinator.create_graph()


NOEV = {"status": "no_event", "date": "2026-10-16", "ret": 0.011, "bench_ret": 0.007, "abnormal": 0.004,
        "z": 0.3, "direction": "up", "benchmark": "NIFTY 50", "z_threshold": 2.5, "lookback": 1,
        "session_open": "2026-10-16T03:45:00+00:00", "requested_date": "2026-10-16",
        "recent": [{"date": "2026-10-16", "ret": 0.011, "abnormal": 0.004, "z": 0.3}],
        "facts": {"date": "2026-10-16", "prev_close": 1154.5, "open": 1119.05, "gap": -0.0307,
                  "low_ret": -0.039, "high_ret": 0.0116, "close_ret": 0.0106, "typical_excursion": 0.01,
                  "intraday_flag": True}}


def _gn_post(monkeypatch, item_hours_after_open=2):
    from src.streaming import gnews_rss
    monkeypatch.setenv("GNEWS_ENABLED", "1")
    def gn(term, since, until, region="IN", name=None, symbol=None, limit=40, session=None):
        open_ = pd.Timestamp(NOEV["session_open"])
        if since == open_:
            return [{"headline": "XYZ shares crash 4% on guidance cut", "source": "Mint", "url": "u",
                     "published_at": open_ + pd.Timedelta(hours=item_hours_after_open),
                     "provider": "google_news"}], {}
        return [], {}
    monkeypatch.setattr(gnews_rss, "fetch_news", gn)


def test_named_day_with_an_intraday_swing_gets_post_open_coverage_and_claim_checks(monkeypatch):
    _gn_post(monkeypatch)
    f = F()
    out = _graph(monkeypatch, NOEV, f).invoke({"query": "why did XYZ fall on 16 Oct", "tickers": ["XYZ"],
                                              "target_date": "2026-10-16"})
    assert out["as_of"].startswith("2026-10-16T03:45")
    assert max(f.untils) == pd.Timestamp("2026-10-16T03:45:00Z")        # providers stop at the open
    assert out["abstain_reason"] == "intraday_only" and out["abstained"]
    r = out["final_report"]
    assert "on 2026-10-16" in r and "open -3.1%" in r and "large intraday swing" in r
    assert "Published after the session opened" in r
    assert "price check: headline figure -4.0% matches the session's intraday low" in r
    post = [d for d in out["ranked"] if d["phase"] == "post"]
    assert post and post[0]["claim_check"]["verdict"] == "matches"


def test_quiet_close_without_a_swing_has_no_post_open_search(monkeypatch):
    _gn_post(monkeypatch)
    calm = {**NOEV, "facts": {**NOEV["facts"], "intraday_flag": False}}
    out = _graph(monkeypatch, calm).invoke({"query": "why", "tickers": ["XYZ"], "target_date": "2026-10-16"})
    assert out["abstain_reason"] == "no_unusual_move"
    assert not any(d["phase"] == "post" for d in out["ranked"])


def test_unavailable_named_day_abstains_without_fetching_news(monkeypatch):
    class Boom:
        def fetch(self, *a, **k): raise AssertionError("no news fetch expected")
    mv = {"status": "unavailable", "reason": "no_complete_session_on_that_date", "requested_date": "2026-10-24"}
    out = _graph(monkeypatch, mv, Boom()).invoke({"query": "why on 24 Oct", "tickers": ["XYZ"],
                                                  "target_date": "2026-10-24"})
    assert out["abstain_reason"] == "date_not_available" and "2026-10-24" in out["final_report"]


def test_date_in_the_question_is_read_when_none_is_passed(monkeypatch):
    seen = {}
    from src.pricing import move_check
    def spy(t, *a, **k):
        seen.update(k); return {"status": "unavailable", "reason": "x"}
    monkeypatch.setattr(news_agent, "_make_fetcher", lambda: F())
    monkeypatch.setattr(retriever_agent, "get_model", lambda: _FakeModel())
    monkeypatch.setattr(coordinator, "resolve_peers", lambda t: (t, [], "S"))
    monkeypatch.setattr(move_check, "check_move", spy)
    coordinator.create_graph().invoke({"query": "why did XYZ fall on 2026-10-16", "tickers": ["XYZ"]})
    assert seen["target_date"] == dt.date(2026, 10, 16)


def test_headlines_outside_the_sessions_day_are_not_claim_checked(monkeypatch):
    out = _graph(monkeypatch, NOEV).invoke({"query": "why", "tickers": ["XYZ"], "target_date": "2026-10-16"})
    pre = [d for d in out["ranked"] if d["phase"] == "pre"]
    assert pre and "claim_check" not in pre[0]       # published before the open: describes an earlier session