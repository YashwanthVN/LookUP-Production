"""Request state for the LookUP agent graph.

Deliberately has NO reducers. Every field is overwritten by the node that
owns it, so a request can never inherit another request's data.
"""
from typing import Any, Dict, List, TypedDict


class LOOKUPState(TypedDict, total=False):
    query: str
    tickers: List[str]            # [primary, *peers]
    profile: Dict[str, str]       # name, summary (primary ticker)
    generated_at: str             # ISO UTC, when the request ran
    target_date: str              # YYYY-MM-DD the user asked about (optional)
    as_of: str                    # ISO UTC; end of the evidence window
    move: Dict[str, Any]          # price-move check result
    mode: str                     # explained | none | unchecked
    peer_context: List[Dict[str, Any]]  # peers' moves on the same session (price facts)
    news_pool: List[Dict[str, Any]]
    ranked: List[Dict[str, Any]]  # top-k with rerank_score
    reranked_documents: List[str]
    final_report: str
    abstained: bool
    abstain_reason: str
    diagnostics: Dict[str, Any]