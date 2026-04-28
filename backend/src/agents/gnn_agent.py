import torch
import torch.nn.functional as F
import numpy as np
import os
from .state import LOOKUPState
from src.kg_holder import get_kg
from src.inference.reasoning_engine import LookUPReporter

def gnn_rag_agent(state: LOOKUPState) -> dict:
    kg = get_kg()
    ticker = state["tickers"][0]
    
    # 1. Pathing for the weights
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    model_path = os.path.join(project_root, "calibrated_gnn_reasoning.pt")
    
    news_nodes = [n for n, d in kg.graph.nodes(data=True) if d.get('type') == 'news']
    if not news_nodes:
        # If no news found, the coordinator might have failed to inject
        return {"gnn_evidence": ["⚠️ Warning: No news nodes found in Graph. Check API keys."]}
    
    # 2. Use the PROVEN reporter from Phase 1
    # This automatically handles 'company_AAPL' and Index lookup
    from src.inference.reasoning_engine import LookUPReporter
    reporter = LookUPReporter(kg, model_path)
    
    # 3. REAL DRIVERS (The 'escalating Middle East crisis' logic)
    drivers = reporter.engine.get_causal_drivers(ticker)
    evidence = [f"GNN prioritized: {d['headline']} (Weight: {d['impact_score']:.4f})" for d in drivers]

    # 4. REAL COMPETITORS (Discovery logic)
    all_nodes = list(kg.graph.nodes)
    # This finds anyone in the graph who isn't the primary ticker
    comps = [n.replace("company_", "") for n in all_nodes if n.startswith("company_") and ticker not in n]

    return {
        "gnn_evidence": evidence
    }