"""Guards for P1: GNN path removed, request path stateless, trigger bug gone."""
import ast
import pathlib
import types

import pandas as pd
import pytest

AGENTS = pathlib.Path(__file__).resolve().parents[1] / "src" / "agents"
FORBIDDEN = ("torch_geometric", "kg_holder", "financial_kg", "gnn_rag",
             "temporal_gnn", "reasoning_engine", "chromadb", "vector_store",
             "MemorySaver", "operator")


def test_agents_do_not_import_removed_modules():
    for f in AGENTS.glob("*.py"):
        src = f.read_text(encoding="utf-8")
        for bad in FORBIDDEN:
            assert bad not in src, f"{f.name} still references {bad}"


def test_state_has_no_reducers():
    tree = ast.parse((AGENTS / "state.py").read_text(encoding="utf-8"))
    assert not any(isinstance(n, ast.Name) and n.id == "Annotated" for n in ast.walk(tree))


class _FakeFetcher:
    def fetch(self, ticker, limit, since, until):
        t = until - pd.Timedelta(hours=2)
        return [{"headline": f"{ticker} news A", "source": "S", "url": "u",
                 "published_at": t, "provider": "fake"},
                {"headline": "Shared headline", "source": "S", "url": "u",
                 "published_at": t, "provider": "fake"}]


class _FakeModel:
    def predict(self, pairs):
        return [float(len(h)) for _, h in pairs]


@pytest.fixture
def app(monkeypatch):
    from src.agents import coordinator, news_agent, retriever_agent
    monkeypatch.setattr(news_agent, "_make_fetcher", lambda: _FakeFetcher())
    monkeypatch.setattr(retriever_agent, "get_model", lambda: _FakeModel())
    monkeypatch.setattr(coordinator, "resolve_peers",
                        lambda t: (t.upper(), ["PEER1"], "Tech"))
    return coordinator.create_graph()


def test_repeated_requests_do_not_accumulate(app):
    outs = [app.invoke({"query": "why", "tickers": ["XYZ"]}) for _ in range(3)]
    assert outs[0]["reranked_documents"] == outs[2]["reranked_documents"]
    assert len(outs[2]["news_pool"]) == len(outs[0]["news_pool"]) == 3  # deduped


def test_pool_is_deduplicated_and_window_stamped(app):
    out = app.invoke({"query": "why", "tickers": ["XYZ"]})
    heads = [d["headline"] for d in out["news_pool"]]
    assert heads.count("Shared headline") == 1
    assert out["ranked"][0]["rerank_score"] >= out["ranked"][-1]["rerank_score"]
    assert out["abstained"] is False


def test_empty_pool_abstains(monkeypatch):
    from src.agents import coordinator, news_agent
    class Empty:
        def fetch(self, *a, **k): return []
    monkeypatch.setattr(news_agent, "_make_fetcher", lambda: Empty())
    monkeypatch.setattr(coordinator, "resolve_peers", lambda t: (t, [], None))
    out = coordinator.create_graph().invoke({"query": "why", "tickers": ["XYZ"]})
    assert out["abstained"] is True and out["abstain_reason"]


def test_provider_exception_does_not_sink_request(monkeypatch):
    from src.agents import coordinator, news_agent
    class Boom:
        def fetch(self, *a, **k): raise RuntimeError("down")
    monkeypatch.setattr(news_agent, "_make_fetcher", lambda: Boom())
    monkeypatch.setattr(coordinator, "resolve_peers", lambda t: (t, [], None))
    out = coordinator.create_graph().invoke({"query": "q", "tickers": ["XYZ"]})
    assert out["abstained"] is True


def test_query_uses_profile_not_user_words():
    from src.agents.retriever_agent import build_query
    prof = {"name": "Reliance Industries Limited", "summary": "Reliance makes oil. " * 50}
    q = build_query(prof, "why did it fall", "RELIANCE.NS")
    assert q.startswith("Reliance Industries. Reliance makes oil.")
    assert "fall" not in q and len(q) <= len("Reliance Industries. ") + 300


def test_query_falls_back_to_user_words_without_profile_summary():
    from src.agents.retriever_agent import build_query
    assert build_query({"name": "Gold", "summary": ""}, "why did gold fall", "XAUUSD") == "Gold. why did gold fall"
    assert "move" in build_query(None, "", "TCS.NS")


def test_peer_only_pool_abstains_but_shows_context(monkeypatch):
    from src.agents import coordinator, news_agent, retriever_agent

    class PeerOnly:
        def fetch(self, ticker, limit, since, until):
            if ticker == "XYZ":
                return []
            return [{"headline": f"{ticker} result", "source": "S", "url": "u",
                     "published_at": until - pd.Timedelta(hours=1), "provider": "f"}]

    monkeypatch.setattr(news_agent, "_make_fetcher", lambda: PeerOnly())
    monkeypatch.setattr(retriever_agent, "get_model", lambda: _FakeModel())
    monkeypatch.setattr(coordinator, "resolve_peers", lambda t: (t, ["PEER1"], "S"))
    out = coordinator.create_graph().invoke({"query": "why", "tickers": ["XYZ"]})
    assert out["abstained"] is True
    assert out["abstain_reason"] == "no_news_for_primary_ticker"
    assert "PEER1 result" in out["final_report"]


def _graph(monkeypatch, fetcher, move):
    from src.agents import coordinator, news_agent, retriever_agent
    from src.pricing import move_check
    monkeypatch.setattr(news_agent, "_make_fetcher", lambda: fetcher)
    monkeypatch.setattr(retriever_agent, "get_model", lambda: _FakeModel())
    monkeypatch.setattr(coordinator, "resolve_peers", lambda t: (t, ["PEER1"], "S"))
    monkeypatch.setattr(move_check, "check_move", lambda t, *a, **k: move)
    return coordinator.create_graph()


EVENT = {"status": "event", "date": "2026-10-20", "ret": -0.05, "bench_ret": -0.01,
         "abnormal": -0.04, "z": -4.1, "direction": "down", "benchmark": "NIFTY 50",
         "z_threshold": 2.5, "session_open": "2026-10-20T03:45:00+00:00"}
NO_EVENT = {**EVENT, "status": "no_event", "ret": 0.002, "abnormal": 0.001, "z": 0.3,
            "direction": "up", "lookback": 5,
            "recent": [{"date": "2026-10-20", "ret": 0.002, "abnormal": 0.001, "z": 0.3},
                       {"date": "2026-10-19", "ret": -0.01, "abnormal": -0.008, "z": -1.9}]}


def test_event_moves_the_evidence_window_to_the_session_open(monkeypatch):
    seen = {}
    class F:
        def fetch(self, ticker, limit, since, until):
            seen["until"] = until
            return [{"headline": f"{ticker} result", "source": "S", "url": "u",
                     "published_at": until - pd.Timedelta(hours=1), "provider": "f"}]
    out = _graph(monkeypatch, F(), EVENT).invoke({"query": "why is it falling", "tickers": ["XYZ"]})
    assert seen["until"] == pd.Timestamp("2026-10-20T03:45:00Z")
    assert out["mode"] == "explained" and out["as_of"].startswith("2026-10-20T03:45")
    assert "fell 5.0% on 2026-10-20" in out["final_report"]
    assert out["abstained"] is False


def test_no_event_abstains_but_shows_context_only(monkeypatch):
    class F:
        def fetch(self, ticker, limit, since, until):
            return [{"headline": f"{ticker} Corp opens new plant", "source": "S", "url": "u",
                     "published_at": until - pd.Timedelta(hours=1), "provider": "f"}]
    out = _graph(monkeypatch, F(), NO_EVENT).invoke({"query": "why is it falling", "tickers": ["XYZ"]})
    assert out["abstained"] and out["abstain_reason"] == "no_unusual_move"
    r = out["final_report"]
    assert "No unusual move" in r and "last 5 complete sessions" in r
    assert "Largest in the window: 2026-10-19" in r and "z = -1.9" in r
    assert "assumes the stock fell, but it rose" in r
    assert "context only" in r and "XYZ Corp opens new plant" in r
    assert "PEER1" not in r.split("context only", 1)[1]   # peers never listed as company coverage


def test_no_event_with_no_news_still_abstains_cleanly(monkeypatch):
    class F:
        def fetch(self, *a, **k): return []
    out = _graph(monkeypatch, F(), NO_EVENT).invoke({"query": "why", "tickers": ["XYZ"]})
    assert out["abstain_reason"] == "no_unusual_move" and "context only" not in out["final_report"]


def test_unavailable_check_proceeds_and_says_so(monkeypatch):
    class F:
        def fetch(self, ticker, limit, since, until):
            return [{"headline": f"{ticker} result", "source": "S", "url": "u",
                     "published_at": until - pd.Timedelta(hours=1), "provider": "f"}]
    out = _graph(monkeypatch, F(), {"status": "unavailable", "reason": "unsupported_instrument"}) \
        .invoke({"query": "why", "tickers": ["XYZ"]})
    assert out["mode"] == "unchecked" and "Price move not verified" in out["final_report"]


def test_scope_labels_company_market_and_peer(monkeypatch):
    class F:
        def fetch(self, ticker, limit, since, until):
            t = until - pd.Timedelta(hours=1)
            if ticker == "XYZ":
                return [{"headline": "XYZ posts record profit", "source": "S", "url": "u", "published_at": t, "provider": "f"},
                        {"headline": "Two-thirds of S&P 500 stocks rose", "source": "S", "url": "u", "published_at": t, "provider": "f"}]
            return [{"headline": "PEER1 hikes prices", "source": "S", "url": "u", "published_at": t, "provider": "f"}]
    out = _graph(monkeypatch, F(), {"status": "unavailable", "reason": "t"}).invoke(
        {"query": "why", "tickers": ["XYZ"]})
    scopes = {d["headline"]: d["scope"] for d in out["news_pool"]}
    assert scopes == {"XYZ posts record profit": "company",
                      "Two-thirds of S&P 500 stocks rose": "market",
                      "PEER1 hikes prices": "peer"}
    assert out["diagnostics"]["own_count"] == 1 and out["diagnostics"]["market_count"] == 1
    assert "[market news]" in out["final_report"]


def test_market_only_primary_pool_abstains(monkeypatch):
    class F:
        def fetch(self, ticker, limit, since, until):
            if ticker != "XYZ":
                return []
            return [{"headline": "Stocks rose on Thursday", "source": "S", "url": "u",
                     "published_at": until - pd.Timedelta(hours=1), "provider": "f"}]
    out = _graph(monkeypatch, F(), {"status": "unavailable", "reason": "t"}).invoke(
        {"query": "why", "tickers": ["XYZ"]})
    assert out["abstained"] and out["abstain_reason"] == "no_news_for_primary_ticker"