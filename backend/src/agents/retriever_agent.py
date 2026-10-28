"""Rank the request's own news pool with the cross-encoder.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 over the FULL pool (no prefilter).
Query follows the validated form: "{company name}. {business summary[:300]}".
Headlines published after the open (events only) are scored with a move-aware query
instead: "Why did {name} shares rise 6%?". That choice is unvalidated.
The user's own words (fell, dropping...) are NOT used when a profile exists:
they pull in any headline containing those verbs, whatever company it is about.

The pool (primary + static sector peers; pre-open and post-open items ranked
together, top-k kept per phase) is NOT the validated kNN-neighbour
pool; see the roadmap.
"""
import threading
from typing import Dict, Optional

import pandas as pd

from src.pricing import claims
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


def build_post_query(profile, move, primary: str) -> str:
    """Query for headlines published AFTER the open. Unvalidated: they are scored on
    whether they answer 'why did it move', not on being about the company's business."""
    name = core_name((profile or {}).get("name") or primary.split(".")[0])
    ret = (move or {}).get("ret")
    if ret is None:
        return f"Why did {name} shares move?"
    return f"Why did {name} shares {'fall' if ret < 0 else 'rise'} {abs(ret) * 100:.0f}%?"


def retriever_agent(state: LOOKUPState) -> dict:
    pool = state.get("news_pool") or []
    if not pool:
        return {"ranked": [], "reranked_documents": []}

    primary = state["tickers"][0]
    q = build_query(state.get("profile"), state.get("query", ""), primary)
    q_post = build_post_query(state.get("profile"), state.get("move"), primary)
    model = get_model()
    with _lock:  # one predict at a time; the model object is shared
        scores = model.predict([(q_post if d.get("phase") == "post" else q, d["headline"])
                                for d in pool])

    ranked = sorted(({**d, "rerank_score": float(s)} for d, s in zip(pool, scores)),
                    key=lambda x: x["rerank_score"], reverse=True)
    # top-k per phase, so post-open coverage cannot crowd out pre-open evidence
    top = ([d for d in ranked if d.get("phase", "pre") == "pre"][:TOP_K]
           + [d for d in ranked if d.get("phase") == "post"][:TOP_K])
    docs = [f"{d['headline']} (relevance: {d['rerank_score']:.3f})" for d in top]
    top = _attach_claim_checks(top, state)
    return {"ranked": top, "reranked_documents": docs}


CLAIM_WINDOW_H = 30


def _attach_claim_checks(items, state):
    """Compare each headline's own price figure with the explained session's prices."""
    move = state.get("move") or {}
    facts, opened = move.get("facts"), move.get("session_open")
    if not facts or not opened or move.get("status") not in ("event", "no_event"):
        return items
    start = pd.Timestamp(opened)
    end = start + pd.Timedelta(hours=CLAIM_WINDOW_H)
    name = core_name((state.get("profile") or {}).get("name") or state["tickers"][0].split(".")[0])
    out = []
    for d in items:
        d = dict(d)
        if d.get("scope") == "company" and start <= pd.Timestamp(d["published_at"]) < end:
            chk = claims.check(d["headline"], name, facts)
            if chk:
                d["claim_check"] = chk
        out.append(d)
    return out