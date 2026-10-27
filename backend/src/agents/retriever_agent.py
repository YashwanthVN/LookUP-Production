"""Rank the request's own news pool with the cross-encoder.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 over the FULL pool (no prefilter).
Query follows the validated form: "{company name}. {business summary[:300]}".
The user's own words (fell, dropping...) are NOT used when a profile exists:
they pull in any headline containing those verbs, whatever company it is about.

The pool (primary + static sector peers; pre-open and post-open items ranked
together, top-k kept per phase) is NOT the validated kNN-neighbour
pool; see the roadmap.
"""
import threading
from typing import Dict, Optional

from src.streaming.gdelt_doc import core_name

from .state import LOOKUPState

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 5
SUMMARY_CHARS = 300

_model = None
_lock = threading.Lock()


def get_model():
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import CrossEncoder
            _model = CrossEncoder(MODEL_NAME)
    return _model


def build_query(profile: Optional[Dict[str, str]], user_query: str, primary: str) -> str:
    name = core_name((profile or {}).get("name") or primary.split(".")[0])
    summary = ((profile or {}).get("summary") or "").strip()
    if summary:
        return f"{name}. {summary[:SUMMARY_CHARS]}"
    q = (user_query or "").strip() or "why did the stock move"
    return f"{name}. {q}"


def retriever_agent(state: LOOKUPState) -> dict:
    pool = state.get("news_pool") or []
    if not pool:
        return {"ranked": [], "reranked_documents": []}

    q = build_query(state.get("profile"), state.get("query", ""), state["tickers"][0])
    model = get_model()
    with _lock:  # one predict at a time; the model object is shared
        scores = model.predict([(q, d["headline"]) for d in pool])

    ranked = sorted(({**d, "rerank_score": float(s)} for d, s in zip(pool, scores)),
                    key=lambda x: x["rerank_score"], reverse=True)
    # top-k per phase, so post-open coverage cannot crowd out pre-open evidence
    top = ([d for d in ranked if d.get("phase", "pre") == "pre"][:TOP_K]
           + [d for d in ranked if d.get("phase") == "post"][:TOP_K])
    docs = [f"{d['headline']} (relevance: {d['rerank_score']:.3f})" for d in top]
    return {"ranked": top, "reranked_documents": docs}