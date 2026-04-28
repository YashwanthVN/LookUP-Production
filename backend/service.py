import streamlit as st
import os
from src.graph.financial_kg import DynamicFinancialKG
from src.inference.reasoning_engine import LookUPReporter
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def get_lookup_components():
    """Initializes and caches the KG and Reporter for high-speed reuse."""
    model_path = os.path.join(PROJECT_ROOT, "calibrated_gnn_reasoning.pt")
    
    # Initialize KG once
    kg = DynamicFinancialKG()
    
    # Initialize Reporter (which loads weights)
    reporter = LookUPReporter(kg, model_path)
    return kg, reporter

def run_analysis_pipeline(ticker):
    """Executes the System 1 and System 2 reasoning pass."""
    kg, reporter = get_lookup_components()
    
    # System 1: Structural Fundamentals
    kg.build_for_tickers([ticker])
    
    # System 2: Real-time News Injection
    kg.inject_real_time_news(ticker)
    
    # Extract GNN Drivers & Generate Narrative
    top_drivers = reporter.engine.get_causal_drivers(ticker)
    report = reporter.generate_report(ticker)
    
    return top_drivers, report