"""Rank the request's own news pool with the cross-encoder.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 over the FULL pool (no prefilter),
the ranker validated in the research repo. The pool here (primary + sector
peers by static list) is NOT the validated kNN-neighbour pool; see P4.
"""
import threading
from .state import LOOKUPState

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 5

_model = None
_lock = threading.Lock()


def get_model():
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import CrossEncoder
            _model = CrossEncoder(MODEL_NAME)
    return _model


def build_query(query: str, primary: str) -> str:
    """Name the company: the validated ranker's query always starts with it.

    A bare 'why did it fall' matches on 'fall', not on the company.
    """
    base = primary.split(".")[0].upper()
    q = (query or "").strip() or "why did the stock move"
    return f"{base}. {q}"


def retriever_agent(state: LOOKUPState) -> dict:
    pool = state.get("news_pool") or []
    if not pool:
        return {"ranked": [], "reranked_documents": []}

    q = build_query(state.get("query", ""), state["tickers"][0])
    model = get_model()
    with _lock:  # one predict at a time; the model object is shared
        scores = model.predict([(q, d["headline"]) for d in pool])

    ranked = []
    for d, s in zip(pool, scores):
        ranked.append({**d, "rerank_score": float(s)})
    ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    top = ranked[:TOP_K]
    docs = [f"{d['headline']} (relevance: {d['rerank_score']:.3f})" for d in top]
    return {"ranked": top, "reranked_documents": docs}