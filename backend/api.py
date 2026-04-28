"""
LookUP FastAPI Backend
Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
Place this file in your LookUP root directory (same level as streamlit_app.py)
"""

import os
import sys
import torch
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Ensure project root in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.graph.financial_kg import DynamicFinancialKG
from src.inference.reasoning_engine import LookUPReporter
from src.kg_holder import set_kg
from src.agents.coordinator import create_graph

load_dotenv()

app = FastAPI(title="LookUP API", version="1.0.0")

# Allow requests from the React frontend (localhost:5173 for dev, your domain for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup: load heavy models once
# ---------------------------------------------------------------------------
kg = None
langgraph_app = None

@app.on_event("startup")
async def startup_event():
    global kg, langgraph_app
    kg = DynamicFinancialKG()
    set_kg(kg)
    model_path = os.path.join(project_root, "calibrated_gnn_reasoning.pt")
    if os.path.exists(model_path):
        kg.gnn_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        kg.gnn_model.eval()
    langgraph_app = create_graph()
    print("✅ LookUP API ready.")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    query: str
    ticker: str  # already resolved on the client side


class GraphHealth(BaseModel):
    nodes: int
    financial_edges: int
    news_edges: int


class AnalyzeResponse(BaseModel):
    ticker: str
    display_name: str
    final_report: str
    gnn_evidence: list[str]
    reranked_documents: list[str]
    competitor_candidates: list[str]
    graph_health: GraphHealth


class ResolveResponse(BaseModel):
    ticker: str | None
    display_name: str | None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/resolve", response_model=ResolveResponse)
def resolve_ticker(query: str):
    """
    Resolve a natural-language query to a ticker symbol using yfinance Search.
    """
    stop_words = {
        "why", "is", "has", "the", "price", "of", "fallen", "risen",
        "today", "on", "what", "how", "situation", "in", "about", "drop", "dropped",
    }
    words = [w for w in query.lower().split() if w not in stop_words]
    clean_subject = " ".join(words).strip()

    if not clean_subject:
        return ResolveResponse(ticker=None, display_name=None)

    try:
        search = yf.Search(clean_subject, max_results=5)
        if search.quotes:
            for match in search.quotes:
                symbol = match.get("symbol")
                quote_type = match.get("quoteType")
                if quote_type in ["EQUITY", "INDEX", "CURRENCY"]:
                    name = match.get("shortname", match.get("longname", symbol))
                    return ResolveResponse(ticker=symbol, display_name=name)
            best = search.quotes[0]
            return ResolveResponse(
                ticker=best["symbol"],
                display_name=best.get("shortname", best["symbol"]),
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ticker resolution error: {e}")

    return ResolveResponse(ticker=None, display_name=None)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Run the full multi-agent LangGraph pipeline and return structured results.
    """
    global kg, langgraph_app

    initial_state = {
        "query": req.query,
        "tickers": [req.ticker],
        "competitor_candidates": [],
        "gnn_evidence": [],
        "reranked_documents": [],
        "final_report": "",
        "iteration_count": 0,
    }

    try:
        result = langgraph_app.invoke(
            initial_state,
            config={"configurable": {"thread_id": req.ticker}},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    # Graph metrics
    reported = len(
        [d for u, v, d in kg.graph.edges(data=True) if d.get("relation") == "REPORTED"]
    )
    impacts = len(
        [d for u, v, d in kg.graph.edges(data=True) if d.get("relation") == "IMPACTS"]
    )

    # Resolve display name
    try:
        info = yf.Ticker(req.ticker).info
        display_name = info.get("shortName", req.ticker)
    except Exception:
        display_name = req.ticker

    return AnalyzeResponse(
        ticker=req.ticker,
        display_name=display_name,
        final_report=result.get("final_report", "No report generated."),
        gnn_evidence=result.get("gnn_evidence", []),
        reranked_documents=result.get("reranked_documents", []),
        competitor_candidates=result.get("competitor_candidates", []),
        graph_health=GraphHealth(
            nodes=len(kg.graph.nodes),
            financial_edges=reported,
            news_edges=impacts,
        ),
    )
