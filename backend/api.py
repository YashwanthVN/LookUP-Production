"""
LookUP FastAPI backend.
Run from backend/:  uv run uvicorn api:app --host 0.0.0.0 --port 8000
"""
import logging
import os
import sys
import threading
import warnings
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv()
# yfinance triggers a harmless pandas deprecation warning on every call
warnings.filterwarnings("ignore", message=".*inplace keyword.*")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("lookup").setLevel(logging.INFO)  # our logs only; third-party stay quiet

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


app = FastAPI(title="LookUP API", version="1.3.0", lifespan=lifespan)

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
    date: str | None = None   # YYYY-MM-DD; overrides a date written in the query


class RankedItem(BaseModel):
    headline: str
    source: str | None = None
    url: str | None = None
    published_at: str
    provider: str | None = None
    ticker: str
    is_primary: bool
    scope: str = ""
    phase: str = "pre"
    tier: str = ""
    claim_check: dict | None = None
    rerank_score: float


class AnalyzeResponse(BaseModel):
    ticker: str
    display_name: str
    final_report: str
    abstained: bool = False
    abstain_reason: str = ""
    as_of: str = ""
    mode: str = ""
    move: dict | None = None
    peer_context: list[dict] = []
    ranked: list[RankedItem] = []
    reranked_documents: list[str] = []
    diagnostics: dict = {}


class ResolveResponse(BaseModel):
    ticker: str | None
    display_name: str | None


@app.get("/health")
def health():
    return {"status": "healthy" if is_ready else "loading"}


@app.get("/resolve", response_model=ResolveResponse)
def resolve_ticker(query: str):
    from src.streaming.ticker_resolver import resolve
    import datetime as dt
    from src.pricing.query_dates import strip_date
    ticker, name = resolve(strip_date(query, dt.datetime.now(dt.timezone.utc).date()))
    return ResolveResponse(ticker=ticker, display_name=name)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if not is_ready:
        raise HTTPException(status_code=503, detail="System initializing. Try again shortly.")
    state = {"query": req.query, "tickers": [req.ticker]}
    if req.date:
        state["target_date"] = req.date
    result = langgraph_app.invoke(state)  # stateless: no thread_id, no checkpointer
    primary = (result.get("tickers") or [req.ticker])[0]
    name = (result.get("profile") or {}).get("name") or primary
    return AnalyzeResponse(
        ticker=primary,
        display_name=name,
        final_report=result.get("final_report", ""),
        abstained=result.get("abstained", False),
        abstain_reason=result.get("abstain_reason", ""),
        as_of=result.get("as_of", ""),
        mode=result.get("mode", ""),
        move=result.get("move"),
        peer_context=result.get("peer_context", []),
        ranked=result.get("ranked", []),
        reranked_documents=result.get("reranked_documents", []),
        diagnostics=result.get("diagnostics", {}),
    )