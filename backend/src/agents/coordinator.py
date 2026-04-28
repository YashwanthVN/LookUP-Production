from typing import TypedDict, Annotated, List, Literal
import operator
import torch
import json
import os
import yfinance as yf
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import agent functions
from .gnn_agent import gnn_rag_agent
from .competitor_agent import competitor_agent
from .temporal_analyst import temporal_analyst
from .retriever_agent import retriever_agent
from .writer import writer_agent
from .state import LOOKUPState

# Import kg holder to access global kg
from src.kg_holder import get_kg

def normalize(ticker: str) -> str:
    """Standardizes tickers by removing exchange suffixes for matching."""
    return ticker.upper().replace(".NS", "").replace(".BO", "")

def coordinator_agent(state: LOOKUPState) -> dict:
    kg = get_kg()
    kg.graph.clear()
    
    input_ticker = state["tickers"][0].upper()
    
    # 1. Identify Market and Set Path
    # Using your test logic to detect India vs US
    is_indian = ".NS" in input_ticker or ".BO" in input_ticker or input_ticker in ["OLAELEC", "TVSMOTOR"]
    
    # Updated paths based on your directory structure
    json_path = "src/utils/in_tickers.json" if is_indian else "src/utils/us_tickers.json"
    
    discovered_peers = []
    final_primary = input_ticker # Default to input
    
    try:
        if not os.path.exists(json_path):
            print(f"❌ Error: {json_path} missing.")
        else:
            with open(json_path, 'r') as f:
                market_data = json.load(f)
            
            target_sector = None
            normalized_input = normalize(input_ticker)

            # 2. Search for Sector and Peers using Normalized Matching
            for sector, tickers in market_data.items():
                # We need to find the specific ticker in the list that matches our input
                for t in tickers:
                    if normalize(t) == normalized_input:
                        target_sector = sector
                        final_primary = t # Upgrade to the full ticker (e.g. OLAELEC.NS)
                        
                        # Get peers, excluding the current ticker
                        discovered_peers = [
                            peer for peer in tickers if normalize(peer) != normalized_input
                        ][:5]
                        break
                if target_sector:
                    break
            
            if target_sector:
                print(f"✅ Market: {'India' if is_indian else 'US'} | Sector: {target_sector}")
                print(f"🔭 Neighbors found: {discovered_peers}")
            else:
                print(f"⚠️ Ticker {input_ticker} not found in {json_path}. Falling back to single-node.")

    except Exception as e:
        print(f"⚠️ Coordinator Error: {e}")

    # 4. Update state with the neighborhood
    # Ensure final_primary (the qualified ticker) is first
    final_tickers = [final_primary] + discovered_peers
    state["tickers"] = final_tickers

    # 5. Build Graph and Inject News
    print(f"🕸️ Building Graph for: {final_tickers}")
    kg.build_for_tickers(final_tickers)
    for t in final_tickers:
        kg.inject_real_time_news(t)

    return {
        "tickers": final_tickers,
        "next_step": "retriever"
    }

def route_next_agent(state: LOOKUPState) -> Literal["gnn", "temporal", "competitor", "write", "retriever"]:
    return state.get("next_step", "write")

def create_graph():
    workflow = StateGraph(LOOKUPState)

    workflow.add_node("coordinator", coordinator_agent)
    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("competitor", competitor_agent)
    workflow.add_node("gnn", gnn_rag_agent)
    workflow.add_node("write", writer_agent)

    workflow.set_entry_point("coordinator")

    # Linear Chain for accumulation
    workflow.add_edge("coordinator", "retriever")
    workflow.add_edge("retriever", "gnn")
    workflow.add_edge("gnn", "competitor")
    workflow.add_edge("competitor", "write")
    workflow.add_edge("write", END)

    return workflow.compile(checkpointer=MemorySaver())