"""
LOOKUP: Main orchestration script.
Run with: python -m src.main --query "Compare AAPL and MSFT"
"""
import argparse
from src.agents.coordinator import create_graph
from src.streaming.mcp_client import YFinanceMCPClient
from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = setup_logger("lookup")

def main(query: str, tickers: list):
    logger.info(f"Processing query: {query}")
    
    # 1. Start MCP server
    config = load_config()
    mcp = YFinanceMCPClient(server_cmd=config["mcp_command"])
    mcp.start()
    
    # 2. Build agent graph
    graph = create_graph()
    
    # 3. Run
    initial_state = {
        "query": query,
        "tickers": tickers,
        "competitor_candidates": [],
        "gnn_evidence": [],
        "temporal_insights": "",
        "reranked_documents": [],
        "final_report": "",
        "iteration_count": 0,
    }
    
    result = graph.invoke(initial_state)
    print("\n=== FINAL REPORT ===\n")
    print(result.get("final_report", "No report generated."))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="Compare Apple and Microsoft financial health")
    parser.add_argument("--tickers", type=str, nargs="+", default=["AAPL", "MSFT"])
    args = parser.parse_args()
    main(args.query, args.tickers)