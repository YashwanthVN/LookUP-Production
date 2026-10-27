"""Tiered news search, post-open phase, peer context, writer wording."""
import pandas as pd

from src.agents import coordinator, news_agent, retriever_agent
from src.agents.writer import peer_line
from src.pricing import move_check
from src.streaming import gnews_rss

EVENT = {"status": "event", "date": "2026-10-26", "ret": 0.063, "bench_ret": 0.007,
         "abnormal": 0.055, "z": 4.9, "direction": "up", "benchmark": "NIFTY 50",
         "z_threshold": 2.5, "lookback": 5, "session_open": "2026-10-26T03:45:00+00:00"}
OPEN = pd.Timestamp("2026-10-26T03:45:00Z")
REAL_PEER_CONTEXT = move_check.peer_context   # conftest stubs the module attribute per test


class _FakeModel:
    def predict(self, pairs):
        return [float(len(h)) for _, h in pairs]


class NoProviders:
    def fetch(self, *a, **k): return []


def _graph(monkeypatch, gnews, move=EVENT, fetcher=None, peers=("PEER1",)):
    monkeypatch.setenv("GNEWS_ENABLED", "1")
    monkeypatch.setattr(news_agent, "_make_fetcher", lambda: fetcher or NoProviders())
    monkeypatch.setattr(retriever_agent, "get_model", lambda: _FakeModel())
    monkeypatch.setattr(coordinator, "resolve_peers", lambda t: (t, list(peers), "S"))
    monkeypatch.setattr(move_check, "check_move", lambda t, *a, **k: move)
    monkeypatch.setattr(gnews_rss, "fetch_news", gnews)
    return coordinator.create_graph()


def _item(h, ts, src="Mint"):
    return {"headline": h, "source": src, "url": "u", "published_at": pd.Timestamp(ts), "provider": "google_news"}


def test_zero_provider_news_is_rescued_by_google_news(monkeypatch):
    def gn(term, since, until, region="IN", name=None, symbol=None, limit=40, session=None):
        if until == OPEN:
            return [_item("XYZ Corp wins order", OPEN - pd.Timedelta(hours=5))], {"kept": 1}
        return [], {"kept": 0}
    out = _graph(monkeypatch, gn).invoke({"query": "why", "tickers": ["XYZ"]})
    assert not out["abstained"]
    assert "XYZ Corp wins order" in out["final_report"]
    assert out["diagnostics"]["own_count"] == 1
    tiers = [t["tier"] for t in out["diagnostics"]["tiers"]]
    assert "google_news" in tiers and "google_news_7d" in tiers        # kept escalating: only 1 < 5


def test_escalation_stops_once_enough_company_items(monkeypatch):
    calls = []
    def gn(term, since, until, region="IN", name=None, symbol=None, limit=40, session=None):
        calls.append((term, since, until))
        if until == OPEN and since == OPEN - pd.Timedelta(hours=72):
            return [_item(f"XYZ Corp story {i}", OPEN - pd.Timedelta(hours=i + 1)) for i in range(6)], {}
        return [], {}
    out = _graph(monkeypatch, gn).invoke({"query": "why", "tickers": ["XYZ"]})
    assert out["diagnostics"]["own_count"] == 6
    assert not any(t["tier"] == "google_news_7d" for t in out["diagnostics"]["tiers"])
    assert len([c for c in calls if c[2] == OPEN and c[1] < OPEN - pd.Timedelta(hours=72)]) == 0


def test_wide_window_never_ends_after_the_open(monkeypatch):
    seen = []
    def gn(term, since, until, region="IN", name=None, symbol=None, limit=40, session=None):
        seen.append((since, until)); return [], {}
    _graph(monkeypatch, gn).invoke({"query": "why", "tickers": ["XYZ"]})
    pre_calls = [c for c in seen if c[1] <= OPEN]
    assert pre_calls and all(c[1] == OPEN for c in pre_calls)
    assert min(c[0] for c in pre_calls) == OPEN - pd.Timedelta(days=7)


def test_post_open_items_are_labelled_and_kept_separate(monkeypatch):
    def gn(term, since, until, region="IN", name=None, symbol=None, limit=40, session=None):
        if since == OPEN:
            return [_item("XYZ Corp shares surge after results", OPEN + pd.Timedelta(hours=3))], {}
        return [], {}
    out = _graph(monkeypatch, gn).invoke({"query": "why", "tickers": ["XYZ"]})
    assert not out["abstained"]
    post = [d for d in out["ranked"] if d["phase"] == "post"]
    assert len(post) == 1 and post[0]["tier"] == "google_news_after_open"
    r = out["final_report"]
    assert "Published after the session opened" in r and "may report the move" in r
    assert "No headlines naming XYZ were published before the session opened" in r
    assert out["diagnostics"]["post_count"] == 1 and out["diagnostics"]["own_count"] == 0


def test_post_open_never_searched_without_an_event(monkeypatch):
    seen = []
    def gn(term, since, until, region="IN", name=None, symbol=None, limit=40, session=None):
        seen.append(term); return [], {}
    out = _graph(monkeypatch, gn, move={"status": "unavailable", "reason": "t"}).invoke(
        {"query": "why", "tickers": ["XYZ"]})
    assert not any(t["tier"] == "google_news_after_open" for t in out["diagnostics"]["tiers"])


def test_total_failure_reports_what_was_tried(monkeypatch):
    def gn(*a, **k): return [], {"error": "http_503"}
    out = _graph(monkeypatch, gn).invoke({"query": "why", "tickers": ["XYZ"]})
    assert out["abstained"] and out["abstain_reason"] == "no_timestamped_news_in_window"
    assert "Searched:" in out["final_report"] and "error http_503" in out["final_report"]


def test_google_news_exception_does_not_sink_request(monkeypatch):
    def gn(*a, **k): raise RuntimeError("boom")
    out = _graph(monkeypatch, gn).invoke({"query": "why", "tickers": ["XYZ"]})
    assert out["abstained"] and "exception:RuntimeError" in str(out["diagnostics"]["tiers"])


def test_peer_context_flows_to_report(monkeypatch):
    monkeypatch.setattr(move_check, "peer_context",
                        lambda peers, date, **k: [{"ticker": "PEER1.NS", "ret": 0.04, "abnormal": 0.03, "z": 2.2}])
    def gn(*a, **k): return [], {}
    out = _graph(monkeypatch, gn).invoke({"query": "why", "tickers": ["XYZ"]})
    assert out["peer_context"][0]["ticker"] == "PEER1.NS"
    assert "sector-wide" in out["final_report"]


def test_peer_line_reads():
    up = {"abnormal": 0.05}
    P = lambda a, z: {"ticker": "A.NS", "abnormal": a, "z": z}
    assert "sector-wide" in peer_line(up, [P(.03, 2.0), P(.02, 1.8), P(-.01, -.3)])
    assert "specific to this stock" in peer_line(up, [P(.001, .1), P(-.002, -.2)])
    assert "mixed" in peer_line(up, [P(.03, 2.0), P(.0, .1), P(.0, .1)])
    assert peer_line(up, []) == ""


def test_peer_context_function_matches_the_date():
    idx = pd.bdate_range("2025-01-01", periods=400)
    import numpy as np
    rng = np.random.default_rng(0)
    mkt = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, .01, 400))), index=idx)
    stk = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, .01, 400))), index=idx)
    d = idx[-3].date().isoformat()
    out = REAL_PEER_CONTEXT(["AAA.NS"], d, fetch=lambda s: mkt if s.startswith("^") else stk,
                                  now=pd.Timestamp("2027-06-01T00:00:00Z"))
    assert len(out) == 1 and set(out[0]) == {"ticker", "ret", "abnormal", "z"}


def test_post_open_items_use_a_move_aware_query(monkeypatch):
    seen = {}
    class Spy:
        def predict(self, pairs):
            for q, h in pairs: seen[h] = q
            return [1.0] * len(pairs)
    def gn(term, since, until, region="IN", name=None, symbol=None, limit=40, session=None):
        if since == OPEN:
            return [_item("XYZ Corp shares jump on results", OPEN + pd.Timedelta(hours=2))], {}
        return [_item("XYZ Corp board meeting", OPEN - pd.Timedelta(hours=2))], {}
    g = _graph(monkeypatch, gn)
    monkeypatch.setattr(retriever_agent, "get_model", lambda: Spy())
    g.invoke({"query": "why", "tickers": ["XYZ"]})
    assert seen["XYZ Corp shares jump on results"] == "Why did XYZ shares rise 6%?"
    assert seen["XYZ Corp board meeting"].startswith("XYZ. ")


def test_gdelt_is_skipped_when_google_news_already_has_enough(monkeypatch):
    monkeypatch.setenv("GDELT_DOC_ENABLED", "1")
    called = []
    monkeypatch.setattr(news_agent, "_gdelt_items", lambda *a: called.append(1) or ([], {}))
    def gn(term, since, until, region="IN", name=None, symbol=None, limit=40, session=None):
        if until == OPEN and since == OPEN - pd.Timedelta(hours=72):
            return [_item(f"XYZ.NS Corp story {i}", OPEN - pd.Timedelta(hours=i + 1)) for i in range(6)], {}
        return [], {}
    g = _graph(monkeypatch, gn)
    monkeypatch.setenv("GDELT_DOC_ENABLED", "1")
    g.invoke({"query": "why", "tickers": ["XYZ.NS"]})
    assert called == []