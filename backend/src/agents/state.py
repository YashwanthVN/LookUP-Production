"""Request state for the LookUP agent graph.

Deliberately has NO reducers (no reducers at all). Every field is overwritten by
the node that owns it, so a request can never inherit another request's data.
"""
from typing import Any, Dict, List, TypedDict


class LOOKUPState(TypedDict, total=False):
    query: str
    tickers: List[str]            # [primary, *peers]
    as_of: str                    # ISO UTC; end of the evidence window
    news_pool: List[Dict[str, Any]]
    ranked: List[Dict[str, Any]]  # top-k with rerank_score
    reranked_documents: List[str]
    final_report: str
    abstained: bool
    abstain_reason: str
