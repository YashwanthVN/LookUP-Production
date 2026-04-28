"""
LookUP FastAPI Backend
Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
Place this file in your LookUP root directory (same level as streamlit_app.py)
"""

import os
import sys
import torch
import yfinance as yf
import threading
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Ensure backend folder is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.graph.financial_kg import DynamicFinancialKG
from src.kg_holder import set_kg
from src.agents.coordinator import create_graph

load_dotenv()

app = FastAPI(title="LookUP API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://lookup-flax.vercel.app",
        "https://lookup-git-main-yashwanthvns-projects.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
kg = None
langgraph_app = None
is_ready = False


# ---------------------------------------------------------------------------
# Background Loader
# ---------------------------------------------------------------------------
def load_models_background():
    global kg, langgraph_app, is_ready
    try:
        print("⏳ Starting background model loading...", flush=True)

        # 1. KG Init
        kg = DynamicFinancialKG()
        set_kg(kg)

        # 2. Load model
        model_path = os.path.join(project_root, "calibrated_gnn_reasoning.pt")
        if os.path.exists(model_path):
            kg.gnn_model.load_state_dict(
                torch.load(model_path, map_location="cpu")
            )
            kg.gnn_model.eval()
            print("➡️ GNN Weights loaded.", flush=True)
        else:
            print("⚠️ Model file not found, skipping load.", flush=True)

        # 3. LangGraph
        langgraph_app = create_graph()

        is_ready = True
        print("✅ API ready.", flush=True)

    except Exception as e:
        print(f"❌ Background loading failed: {e}", flush=True)


@app.on_event("startup")
async def startup_event():
    thread = threading.Thread(target=load_models_background)
    thread.daemon = True  # important for clean shutdown
    thread.start()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    query: str
    ticker: str


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
    # Always return 200 (Render requirement)
    if not is_ready:
        return JSONResponse(
            status_code=200,
            content={"status": "loading"}
        )
    return {"status": "healthy"}


@app.get("/resolve", response_model=ResolveResponse)
def resolve_ticker(query: str):
    stop_words = {
        "why", "is", "has", "the", "price", "of", "fallen", "risen",
        "today", "on", "what", "how", "situation", "in", "about",
        "drop", "dropped",
    }

    words = [w for w in query.lower().split() if w not in stop_words]
    clean_subject = " ".join(words).strip()

    if not clean_subject:
        return ResolveResponse(ticker=None, display_name=None)

    try:
        search = yf.Search(clean_subject, max_results=5)
        if search.quotes:
            for match in search.quotes:
                if match.get("quoteType") in ["EQUITY", "INDEX", "CURRENCY"]:
                    return ResolveResponse(
                        ticker=match.get("symbol"),
                        display_name=match.get("shortname", match.get("symbol")),
                    )

            best = search.quotes[0]
            return ResolveResponse(
                ticker=best["symbol"],
                display_name=best.get("shortname", best["symbol"]),
            )

    except Exception as e:
        raise HTTPException(500, f"Ticker resolution error: {e}")

    return ResolveResponse(ticker=None, display_name=None)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    global kg, langgraph_app, is_ready

    if not is_ready:
        raise HTTPException(
            status_code=503,
            detail="System still initializing. Try again shortly.",
        )

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
        raise HTTPException(500, f"Analysis failed: {e}")

    reported = len([
        d for _, _, d in kg.graph.edges(data=True)
        if d.get("relation") == "REPORTED"
    ])
    impacts = len([
        d for _, _, d in kg.graph.edges(data=True)
        if d.get("relation") == "IMPACTS"
    ])

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
