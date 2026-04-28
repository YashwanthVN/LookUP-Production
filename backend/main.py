import os
import sys
import torch
import argparse
from dotenv import load_dotenv

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.graph.financial_kg import DynamicFinancialKG
from src.inference.reasoning_engine import LookUPReporter

def run_lookup_analysis(ticker):
    load_dotenv()
    
    # 1. Path Setup
    model_path = os.path.join(project_root, "calibrated_gnn_reasoning.pt")
    if not os.path.exists(model_path):
        print(f"❌ Error: Model weights not found at {model_path}")
        return

    print(f"\n🚀 STARTING LOOKUP ANALYSIS FOR: {ticker}")
    print("-" * 50)

    # 2. Initialize Knowledge Graph
    # This loads FinBERT and the GNN architecture
    kg = DynamicFinancialKG()

    # 3. Build Graph Structure (System 1)
    print(f"🏗️  Building Structural Graph (Fundamentals)...")
    kg.build_for_tickers([ticker])

    # 4. Inject Live News & Sentiment (System 2)
    print(f"📡 Injecting Real-Time Narrative Data...")
    kg.inject_real_time_news(ticker)

    # 5. Initialize Reporter
    reporter = LookUPReporter(kg, model_path)

    # 6. Fetch Raw Drivers for the Table
    top_drivers = reporter.engine.get_causal_drivers(ticker)

    # 7. Display Attention Mapping Table
    print(f"\n🔍 GNN ATTENTION MAPPING")
    print("=" * 85)
    print(f"{'HEADLINE':<60} | {'WEIGHT':<10} | {'SENT'}")
    print("-" * 85)
    
    if not top_drivers:
        print("⚠️ No causal news drivers identified for this ticker.")
    else:
        for d in top_drivers:
            headline = (d['headline'][:57] + '..') if len(d['headline']) > 57 else d['headline']
            print(f"{headline:<60} | {d['impact_score']:.4f}   | {d['sentiment']:.2f}")
    print("=" * 85)

    # 8. Generate Gemini Causal Report
    print(f"\n📊 Generating AI Causal Report...")
    report = reporter.generate_report(ticker)
    
    print("\n" + "📜" + " " + "FINAL ANALYSIS REPORT")
    print("-" * 50)
    print(report)
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LookUP: Causal Financial Reasoning Engine")
    parser.add_argument("ticker", type=str, help="Stock ticker to analyze (e.g., TSLA, AAPL, NVDA)")
    
    args = parser.parse_args()
    
    try:
        run_lookup_analysis(args.ticker.upper())
    except Exception as e:
        print(f" An error occurred: {e}")