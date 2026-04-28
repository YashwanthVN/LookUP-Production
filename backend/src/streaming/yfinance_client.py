import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional

class YFinanceClient:
    """Fetches real and historical financial data for given tickers."""

    def __init__(self):
        self.cache = {}  # simple in‑memory cache to avoid repeated calls

    def get_company_info(self, ticker: str) -> Dict:
        """Get current company overview (market cap, sector, etc.)."""
        if ticker in self.cache:
            return self.cache[ticker]
        stock = yf.Ticker(ticker)
        info = stock.info
        # Extract relevant fields
        data = {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "revenue": info.get("totalRevenue"),
            "ebitda": info.get("ebitda"),
            "profit_margin": info.get("profitMargins"),
        }
        self.cache[ticker] = data
        return data

    def get_quarterly_financials(self, ticker: str, max_quarters: int = 4) -> pd.DataFrame:
        """Fetch quarterly income statement and return key metrics."""
        stock = yf.Ticker(ticker)
        # quarterly financials
        financials = stock.quarterly_financials.T  # transpose so rows are dates
        if financials.empty:
            return pd.DataFrame()
        # Keep only relevant columns if they exist
        keep_cols = ['Total Revenue', 'EBITDA', 'Net Income']
        available = [col for col in keep_cols if col in financials.columns]
        df = financials[available].copy()
        df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')  # format dates
        return df.head(max_quarters)

    def get_quarterly_ratios(self, ticker: str, max_quarters: int = 4) -> pd.DataFrame:
        """Fetch quarterly valuation ratios (P/E, etc.) – yfinance provides them in quarterly_earnings."""
        stock = yf.Ticker(ticker)
        earnings = stock.quarterly_earnings.T
        if earnings.empty:
            return pd.DataFrame()
        # earnings may contain 'Revenue' and 'Earnings' – we'll use them
        df = earnings.T if earnings.shape[0] == 1 else earnings  # ensure rows are dates
        df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
        return df.head(max_quarters)