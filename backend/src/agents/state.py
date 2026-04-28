from typing import TypedDict, Annotated, List, Any
import operator

class LOOKUPState(TypedDict):
    query: str
    tickers: List[str]
    next_step: str
    gnn_evidence: Annotated[List[str], operator.add] 
    competitor_candidates: List[str]
    temporal_insights: str
    reranked_documents: List[str]
    final_report: str
    iteration_count: int