"""
LookUP FastAPI backend.
Run from backend/:  uv run uvicorn api:app --host 0.0.0.0 --port 8000
"""
import os
import sys
import threading
from contextlib import asynccontextmanager

import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv()

langgraph_app = None
is_ready = False


def _warm():
    global langgraph_app, is_ready
    try:
        from src.agents.coordinator import create_graph
        from src.agents.retriever_agent import get_model
        langgraph_app = create_graph()
        get_model()  # download/load the cross-encoder once
        is_ready = True
        print("API ready.", flush=True)
    except Exception as e:
        print(f"Background loading failed: {e}", flush=True)


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="LookUP API", version="1.1.0", lifespan=lifespan)

_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,  # "*" with credentials is invalid and unsafe
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    query: str
    ticker: str


class GraphHealth(BaseModel):
    nodes: int = 0
    financial_edges: int = 0
    news_edges: int = 0


class RankedItem(BaseModel):
    headline: str
    source: str | None = None
    url: str | None = None
    published_at: str
    ticker: str
    is_primary: bool
    rerank_score: float


class AnalyzeResponse(BaseModel):
    ticker: str
    display_name: str
    final_report: str
    abstained: bool = False
    abstain_reason: str = ""
    as_of: str = ""
    ranked: list[RankedItem] = []
    reranked_documents: list[str] = []
    # Deprecated, always empty. Kept so the current frontend keeps parsing;
    # removed when the UI is updated.
    gnn_evidence: list[str] = []
    competitor_candidates: list[str] = []
    graph_health: GraphHealth = GraphHealth()


class ResolveResponse(BaseModel):
    ticker: str | None
    display_name: str | None


@app.get("/health")
def health():
    return {"status": "healthy" if is_ready else "loading"}


@app.get("/resolve", response_model=ResolveResponse)
def resolve_ticker(query: str):
    try:
        search = yf.Search(query, max_results=3)
        if search.quotes:
            best = search.quotes[0]
            return ResolveResponse(ticker=best["symbol"],
                                   display_name=best.get("shortname", best["symbol"]))
    except Exception:
        pass
    return ResolveResponse(ticker=None, display_name=None)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if not is_ready:
        raise HTTPException(status_code=503, detail="System initializing. Try again shortly.")
    state = {"query": req.query, "tickers": [req.ticker]}
    result = langgraph_app.invoke(state)  # stateless: no thread_id, no checkpointer
    return AnalyzeResponse(
        ticker=req.ticker,
        display_name=req.ticker,
        final_report=result.get("final_report", ""),
        abstained=result.get("abstained", False),
        abstain_reason=result.get("abstain_reason", ""),
        as_of=result.get("as_of", ""),
        ranked=result.get("ranked", []),
        reranked_documents=result.get("reranked_documents", []),
    )
