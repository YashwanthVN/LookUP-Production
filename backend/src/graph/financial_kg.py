import os
import torch
import networkx as nx
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

from torch_geometric.data import Data
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel

# Custom internal imports
from src.graph.gnn_rag import FinancialReasoningGNN
from src.streaming.news_fetcher import UnifiedNewsFetcher

class DynamicFinancialKG:
    """
    Builds and maintains a Dynamic Financial Knowledge Graph integrating 
    fundamental metrics, competitor influence, and real-time news sentiment.
    """
    METRIC_TYPES = ["revenue", "ebitda", "netProfitMargin", "eps", "pe_ratio"]

    def __init__(self, quarters: int = 4, api_key: Optional[str] = None):
        from src.streaming.financial_client import FMPClient
        from src.retrieval.vector_store import VectorStore
        
        self.vector_store = VectorStore()
        self.client = FMPClient(api_key)
        self.graph = nx.MultiDiGraph()
        self.quarters = quarters

        # Environment-Agnostic Pathing
        project_root = Path(__file__).resolve().parent.parent.parent
        adapter_path = project_root / "notebooks" / "finbert_sentiment_lora_final"

        base_model_name = "yiyanghkust/finbert-tone"
        self.tokenizer = BertTokenizer.from_pretrained(base_model_name)
        base_model = BertForSequenceClassification.from_pretrained(base_model_name)

        try:
            self.sentiment_model = PeftModel.from_pretrained(base_model, adapter_path)
            print("✅ Success: Loaded LoRA Sentiment Model.")
        except Exception:
            print("⚠️ Warning: LoRA not found. Using vanilla FinBERT.")
            self.sentiment_model = base_model

        self.gnn_model = FinancialReasoningGNN(
            in_channels=3, 
            edge_channels=1, 
            hidden=64, 
            out=32
        )
        print("🧠 Reasoning GNN initialized.")

    def build_for_tickers(self, tickers: List[str]):
        self.graph.clear()
        
        # Graceful fail for Commodities (Gold/Silver)
        try:
            bulk_data = self.client.get_bulk_data(tickers, self.quarters)
        except Exception:
            bulk_data = {"profiles": {}, "financials": {}}

        for ticker in tickers:
            prof = bulk_data["profiles"].get(ticker, {})
            sector = prof.get("sector", "Commodity/Other")
            self.graph.add_node(
                f"company_{ticker}", 
                type="company", 
                sector=sector, 
                feat=[1.0, 0.0, 0.0]
            )

            # Add Time Nodes
            for q in range(self.quarters):
                date_node = (datetime.now() - timedelta(days=90 * q)).strftime('%Y-%m')
                self.graph.add_node(f"time_{date_node}", type="time", feat=[0.0, 0.0, 1.0])

            # Build System 1 (Financials)
            fin_df = bulk_data["financials"].get(ticker, pd.DataFrame())
            for m in self.METRIC_TYPES:
                self.graph.add_node(f"metric_{m}", type="metric", feat=[0.0, 1.0, 0.0])
                if not fin_df.empty and m in fin_df.columns:
                    for _, row in fin_df.iterrows():
                        self._add_metric_edges(ticker, m, str(row['date'])[:7], row[m])

        self._add_competitor_edges(tickers)
        self._normalize_metrics()

    def _add_metric_edges(self, ticker, metric, date_prefix, val):
        comp_node = f"company_{ticker}"
        met_node = f"metric_{metric}"
        time_node = f"time_{date_prefix}"
        
        clean_val = float(val) if val and np.isfinite(val) else 0.0

        # Restore Temporal Edges
        self.graph.add_edge(
            comp_node, met_node, relation="REPORTED", value=clean_val, is_metric=True
        )
        if time_node in self.graph:
            self.graph.add_edge(met_node, time_node, relation="VALID_AT")

    def _add_competitor_edges(self, tickers):
        for ticker in tickers:
            u_node = f"company_{ticker}"
            u_sector = self.graph.nodes[u_node].get('sector')
            peers = [
                n for n, d in self.graph.nodes(data=True) 
                if d.get('sector') == u_sector and n != u_node
            ]

            for peer in peers:
                # Uses FMPClient's historical price method
                corr = self._get_correlation_weight(ticker, peer.replace("company_", ""))
                self.graph.add_edge(
                    peer, u_node, relation="COMPETITOR_INFLUENCE", value=float(corr)
                )

    def _get_correlation_weight(self, t1, t2):
        """
        Calculates Pearson correlation between two tickers with length safety checks.
        """
        try:
            p1 = self.client.get_historical_prices(t1)
            p2 = self.client.get_historical_prices(t2)
            
            # --- FIX: Minimum length requirement for valid correlation ---
            if len(p1) < 5 or len(p2) < 5:
                return 0.5  # Neutral fallback
            
            # Ensure we are comparing arrays of the same length
            min_len = min(len(p1), len(p2))
            
            # Calculate correlation on the most recent overlapping window
            correlation = np.corrcoef(p1[-min_len:], p2[-min_len:])[0, 1]
            
            # Handle potential NaNs from np.corrcoef (e.g., if one series is constant)
            if np.isnan(correlation):
                return 0.5
                
            return max(0.1, correlation)
            
        except Exception as e:
            print(f"⚠️ Correlation calculation failed for {t1}-{t2}: {e}")
            return 0.5

    def _normalize_metrics(self):
        for m_type in self.METRIC_TYPES:
            edges = [
                d for u, v, d in self.graph.edges(data=True) 
                if f"metric_{m_type}" in v and d.get('is_metric')
            ]
            if not edges:
                continue
            
            vals = [e['value'] for e in edges]
            mean, std = np.mean(vals), np.std(vals)
            for e in edges:
                e['value'] = (e['value'] - mean) / (std if std > 0 else 1.0)

    def add_news_event(self, ticker, sentiment, magnitude, headline, sector=None):
        news_id = f"news_{ticker}_{hash(headline) % 10000}"
        self.graph.add_node(
            news_id, 
            type="news", 
            sector=sector,
            feat=[0.0, 0.0, 1.0],
            sentiment=float(sentiment), 
            label=str(headline)
        )
        self.graph.add_edge(news_id, f"company_{ticker}", relation="IMPACTS", value=magnitude)

    def inject_real_time_news(self, ticker: str):
        """
        Tries multiple strategies to fetch news, only accepting a strategy if it yields valid headlines.
        """
        fetcher = UnifiedNewsFetcher()
        base_symbol = ticker.split('.')[0].upper()
        
        search_strategies = [
            ticker,                
            f"{base_symbol}.BO",   
            base_symbol            
        ]

        headlines = []
        actual_query_used = None
        raw_items = []  # we may need original items later for source info

        # --- Try ticker-based strategies ---
        for query in search_strategies:
            try:
                print(f"📡 System 2 Discovery: Attempting news for '{query}'...")
                items = fetcher.fetch(query, limit=12)
                # Extract headlines that are non-empty
                candidate_headlines = [item['headline'] for item in items if item.get('headline')]
                if candidate_headlines:
                    headlines = candidate_headlines
                    raw_items = items  # keep the full items for later (maybe source, etc.)
                    actual_query_used = query
                    print(f"✅ News headlines acquired via: {query}")
                    break
                else:
                    print(f"⚠️ No valid headlines for '{query}'. Trying next...")
            except Exception as e:
                print(f"❌ Error for '{query}': {e}. Moving to next strategy...")
                continue

        # --- Final fallback: company name search via yfinance ---
        if not headlines:
            try:
                print(f"📡 Final attempt: Searching by company name for {ticker}...")
                # Get company name from yfinance
                yf_ticker = yf.Ticker(ticker)
                company_name = yf_ticker.info.get('longName')
                if company_name:
                    # Use yfinance Search with the company name
                    search_results = yf.Search(company_name, max_results=12).news
                    candidate_headlines = [n.get('title') for n in search_results if n.get('title')]
                    if candidate_headlines:
                        headlines = candidate_headlines
                        # Create pseudo items for consistency (we don't have full metadata)
                        raw_items = [{'headline': h, 'source': 'yfinance_name_search'} for h in headlines]
                        actual_query_used = f"name_search:{company_name}"
                        print(f"✅ News found via company name: {company_name}")
            except Exception as e:
                print(f"❌ Company name search failed: {e}")

        if not headlines:
            print(f"❌ Critical: No news found for {ticker} after all strategies.")
            return

        # --- Sentiment processing (using headlines list) ---
        inputs = self.tokenizer(
            headlines, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        device = next(self.sentiment_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        self.sentiment_model.eval()
        with torch.no_grad():
            logits = self.sentiment_model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        hints = fetcher.TEMPLATES.get(base_symbol, fetcher.TEMPLATES["DEFAULT"])
        critical_triggers = ["strike", "war", "safety", "sales decline", "market share", "downgrade"]

        # --- Inject each headline as a news node ---
        for idx, headline in enumerate(headlines):
            # The corresponding raw item might be in raw_items; we'll use index to get source if needed
            source = raw_items[idx].get('source', 'unknown') if idx < len(raw_items) else 'unknown'
            score = probs[idx][1].item() - probs[idx][2].item()
            headline_lower = headline.lower()
            magnitude = abs(score) * 2.0 + 0.5

            is_critical = any(trigger in headline_lower for trigger in critical_triggers)
            if is_critical:
                magnitude *= 3.0

            self.add_news_event(ticker, score, magnitude, headline)
            self.vector_store.add_document(
                headline,
                metadata={
                    'ticker': ticker,
                    'sentiment': score,
                    'magnitude': magnitude,
                    'timestamp': datetime.now().isoformat(),
                    'query_source': actual_query_used,
                    'source': source
                }
            )

        print(f"✅ Successfully Injected {len(headlines)} news events for {ticker} (Source: {actual_query_used})")

    def get_node_index(self, label: str):
        """
        Robustly finds the integer index of a node based on its label or ticker.
        Handles special characters (like ^), prefixes, and potential ValueErrors.
        """
        node_list = list(self.graph.nodes)
        target = str(label).strip()
        
        try:
            # 1. Metric Match (Prioritize metric types for faster lookup)
            if target in self.METRIC_TYPES:
                metric_prefixed = f"metric_{target}"
                if metric_prefixed in node_list:
                    return node_list.index(metric_prefixed)

            # 2. Exact Match (e.g., "company_AAPL" or "news_XAUUSD_1234")
            if target in node_list:
                return node_list.index(target)

            # 3. Company Prefix Match (e.g., "^BSESN" -> "company_^BSESN")
            company_prefixed = f"company_{target}"
            if company_prefixed in node_list:
                return node_list.index(company_prefixed)

            # 4. Case-Insensitive Fallback (Last Resort)
            target_lower = target.lower()
            for i, node in enumerate(node_list):
                if node.lower() == target_lower or node.lower() == f"company_{target_lower}":
                    return i
            
            return None

        except ValueError:
            # Caught if .index() fails despite the 'in' checks
            return None

    def to_pyg_data(self) -> Data:
        node_list = list(self.graph.nodes)
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        
        # 1. Node Features
        x = torch.tensor(
            [self.graph.nodes[n].get('feat', [0, 0, 0]) for n in node_list], 
            dtype=torch.float
        )
        
        edge_index, edge_attr = [], []
        
        # 2. Synchronized Edge and Attribute Construction
        for u, v, d in self.graph.edges(data=True):
            if u in node_to_idx and v in node_to_idx:
                # Add exactly ONE edge
                edge_index.append([node_to_idx[u], node_to_idx[v]])
                
                # Add exactly ONE attribute for that edge
                val = d.get('value')
                if val is not None:
                    edge_attr.append([float(val)])
                else:
                    # Default for structural edges (like VALID_AT)
                    edge_attr.append([0.0])
        
        # 3. Final Tensor Conversion
        if not edge_index:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        else:
            # Ensure edge_index is (2, num_edges)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            # Ensure edge_attr is (num_edges, 1)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)