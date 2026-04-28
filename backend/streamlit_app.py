import re
import os
import torch
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
import requests

# Internal module imports
from src.graph.financial_kg import DynamicFinancialKG
from src.inference.reasoning_engine import LookUPReporter
from src.kg_holder import set_kg
from src.agents.coordinator import create_graph

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="LookUP | AI Summary",
    page_icon="📈",
    layout="wide"
)
load_dotenv()

# --- CACHE THE MODEL ---
@st.cache_resource
def load_engine():
    """
    Initializes the Knowledge Graph and defines the model path.
    Mapping query terms to tickers allows for commodity support.
    """
    kg = DynamicFinancialKG()
    set_kg(kg)
    model_path = "calibrated_gnn_reasoning.pt"
    if os.path.exists(model_path):
        kg.gnn_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        kg.gnn_model.eval()
        
    # Compile the LangGraph app
    app = create_graph()
    return kg, app

kg, app = load_engine()

# --- CACHE THE ANALYSIS RESULTS ---
@st.cache_data(ttl=3600)  # Cache results for 1 hour to save API tokens
def run_agentic_analysis(ticker, query):
    """
    Invokes the multi-agent graph with a consistent thread ID per ticker.
    """
    initial_state = {
        "query": query,
        "tickers": [ticker],
        "competitor_candidates": [],
        "gnn_evidence": [],
        "reranked_documents": [],
        "final_report": "",
        "iteration_count": 0
    }
    # Using the ticker as the thread_id ensures continuity for that specific stock
    return app.invoke(initial_state, config={"configurable": {"thread_id": ticker}})

# --- SIDEBAR: GRAPH METRICS ---
with st.sidebar:
    st.header("🧠 Graph Intelligence")
    health_placeholder = st.empty()
    with health_placeholder.container():
        st.subheader("🛠️ Graph Health")
        st.info("Run an analysis to see metrics.")

# --- UI HEADER ---
st.title("📈 LookUP: Financial Reasoning")
st.markdown("### Explain market movements using GNN-Attention, Vector RAG & Gemini for LLM")

# --- SEARCH BAR ---
query = st.text_input(
    "Ask about a stock or commodity:",
    placeholder="Why has the price of gold fallen?"
)

def resolve_ticker(query):
    """
    Robust ticker resolution using Yahoo Finance's internal search API.
    Handles company names, common abbreviations, and commodities.
    """
    # 1. Clean the query
    stop_words = {"why", "is", "has", "the", "price", "of", "fallen", "risen", 
                  "today", "on", "what", "how", "situation", "in", "about", "drop", "dropped"}
    words = [w for w in query.lower().split() if w not in stop_words]
    clean_subject = " ".join(words).strip()

    if not clean_subject:
        return None, None

    try:
        search = yf.Search(clean_subject, max_results=5)
        
        if search.quotes and len(search.quotes) > 0:
            # Prefer EQUITY or INDEX matches
            for match in search.quotes:
                symbol = match.get('symbol')
                quote_type = match.get('quoteType')
                
                if quote_type in ['EQUITY', 'INDEX', 'CURRENCY']:
                    name = match.get('shortname', match.get('longname', symbol))
                    return symbol, name
            
            # Fallback to the first available result
            best_match = search.quotes[0]
            return best_match['symbol'], best_match.get('shortname', best_match['symbol'])
            
    except Exception as e:
        st.error(f"🔍 Ticker resolution error: {e}")
    
    return None, None

# --- EXECUTION LOGIC ---
if st.button("Analyze Causal Drivers"):
    if query:
        target_ticker, display_name = resolve_ticker(query)
        
        if target_ticker is None:
            st.warning("Could not identify the stock. Please try a specific ticker like 'AAPL'.")
        else:
            st.success(f"Resolved to **{display_name}** ({target_ticker})")
            
            with st.spinner(f"Running Multi-Agent Analysis for {target_ticker}..."):
                try:
                    # 1. Invoke the Cached Multi-Agent Workflow
                    result = run_agentic_analysis(target_ticker, query)

                    # 2. Update Sidebar Stats (Reflects state of global KG)
                    reported = len([d for u,v,d in kg.graph.edges(data=True) if d.get('relation') == 'REPORTED'])
                    impacts = len([d for u,v,d in kg.graph.edges(data=True) if d.get('relation') == 'IMPACTS'])
                    with health_placeholder.container():
                        st.subheader("🛠️ Graph Health")
                        st.metric("Financial Edges", reported)
                        st.metric("News Edges", impacts)
                        st.success(f"Nodes: {len(kg.graph.nodes)}")

                    # --- PROFESSIONAL UI LAYOUT ---
                    st.markdown(f"## 🔎 LookUP Analysis: {display_name} ({target_ticker})")

                    # A. Top Level Report (Gemini's Synthesis)
                    st.info(result.get("final_report", "No report generated."))

                    # B. Specialist Data Columns
                    col1, col2 = st.columns(2)

                    with col1:
                        with st.expander("🧠 GNN Causal Drivers (Attention Mapping)", expanded=True):
                            evidence = result.get("gnn_evidence", [])
                            if evidence:
                                for insight in evidence:
                                    st.write(f"• {insight}")
                            else:
                                st.write("No high-influence GNN drivers identified.")

                    with col2:
                        with st.expander("📑 Top Relevant Documents (Vector RAG)", expanded=True):
                            docs = result.get("reranked_documents", [])
                            if docs:
                                for doc in docs:
                                    st.write(f"• {doc}")
                            else:
                                st.write("No deep-context documents retrieved.")

                    # --- C. PROFESSIONAL COMPETITOR DASHBOARD ---
                    st.divider()
                    st.subheader("🏢 Market Context: Peer Proximity (GNN Latent Space)")

                    comps = result.get("competitor_candidates", [])
                    if comps:
                        # Create a metric-style row for competitors
                        # This proves System 2 reasoning for your IEEE paper
                        cols = st.columns(len(comps) if len(comps) > 0 else 1)
                        for i, comp_data in enumerate(comps):
                            # Splitting "TICKER (Sim: 0.99)"
                            name_part = comp_data.split(" (")[0]
                            sim_part = comp_data.split("Sim: ")[1].replace(")", "")
                            
                            with cols[i]:
                                st.metric(label=f"Peer: {name_part}", value=f"{float(sim_part)*100:.1f}%", help="Cosine similarity in GNN feature space")
                    else:
                        st.warning("No mathematically similar peers identified in the current graph state.")

                except Exception as e:
                    st.error(f"Analysis failed: {e}")
    else:
        st.warning("Please enter a query or ticker first.")

# --- FOOTER ---
st.divider()
st.caption("LookUP v1.0 | Multi-Agent Graph RAG (GNN + Vector Store)")