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


def test_query_names_the_company():
    from src.agents.retriever_agent import build_query
    assert build_query("why did it fall", "RELIANCE.NS").startswith("RELIANCE.")
    assert "move" in build_query("", "TCS.NS")


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