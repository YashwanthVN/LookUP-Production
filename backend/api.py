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
    allow_origins=["*"],
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
        print("⏳ Starting background heavy loading...", flush=True)

        # CRITICAL: Heavy imports moved INSIDE the thread
        import torch
        from src.graph.financial_kg import DynamicFinancialKG
        from src.kg_holder import set_kg
        from src.agents.coordinator import create_graph

        # 1. KG Init
        kg = DynamicFinancialKG()
        set_kg(kg)

        # 2. Load model
        model_path = os.path.join(project_root, "calibrated_gnn_reasoning.pt")
        if os.path.exists(model_path):
            kg.gnn_model.load_state_dict(torch.load(model_path, map_location="cpu"))
            kg.gnn_model.eval()
            print("➡️ GNN Weights loaded.", flush=True)

        # 3. LangGraph
        langgraph_app = create_graph()

        is_ready = True
        print("✅ API FULLY INITIALIZED.", flush=True)

    except Exception as e:
        print(f"❌ Background loading failed: {e}", flush=True)


@app.on_event("startup")
async def startup_event():
    # This thread starts immediately; uvicorn can now bind to the port in <1s
    thread = threading.Thread(target=load_models_background, daemon=True)
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
    return {"status": "loading" if not is_ready else "healthy"}

@app.get("/resolve", response_model=ResolveResponse)
def resolve_ticker(query: str):
    # This works immediately even while models are loading!
    try:
        search = yf.Search(query, max_results=3)
        if search.quotes:
            best = search.quotes[0]
            return ResolveResponse(ticker=best["symbol"], display_name=best.get("shortname", best["symbol"]))
    except:
        pass
    return ResolveResponse(ticker=None, display_name=None)

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    global kg, langgraph_app, is_ready
    if not is_ready:
        raise HTTPException(status_code=503, detail="System initializing. Try in 2 mins.")

    # Your existing analysis logic...
    initial_state = {"query": req.query, "tickers": [req.ticker], "competitor_candidates": [], 
                     "gnn_evidence": [], "reranked_documents": [], "final_report": "", "iteration_count": 0}
    
    result = langgraph_app.invoke(initial_state, config={"configurable": {"thread_id": req.ticker}})
    
    reported = len([d for _, _, d in kg.graph.edges(data=True) if d.get("relation") == "REPORTED"])
    impacts = len([d for _, _, d in kg.graph.edges(data=True) if d.get("relation") == "IMPACTS"])

    return AnalyzeResponse(
        ticker=req.ticker,
        display_name=req.ticker,
        final_report=result.get("final_report", ""),
        gnn_evidence=result.get("gnn_evidence", []),
        reranked_documents=result.get("reranked_documents", []),
        competitor_candidates=result.get("competitor_candidates", []),
        graph_health=GraphHealth(nodes=len(kg.graph.nodes), financial_edges=reported, news_edges=impacts)
    )
