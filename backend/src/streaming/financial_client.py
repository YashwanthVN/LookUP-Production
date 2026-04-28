import os
import requests
import pandas as pd
import yfinance as yf
import numpy as np
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

class FMPClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        self.base_url = "https://financialmodelingprep.com/api/v3/"

    def get_bulk_data(self, tickers: List[str], quarters: int = 4) -> Dict:
        ticker_str = ",".join(tickers)
        
        # 1. ATOMIC CALL: Batch Profile & Market Cap (1 call)
        # FMP allows multiple symbols in the quote and profile endpoints
        profiles_raw = self._get(f"profile/{ticker_str}")
        profiles = {p['symbol']: p for p in profiles_raw} if isinstance(profiles_raw, list) else {}

        # 2. OPTIMIZED CALL: Batch Financials
        # If your plan supports it, 'income-statement-bulk' is the fastest.
        # Otherwise, we use a Session to reuse the TCP connection for speed.
        all_financials = {}
        with requests.Session() as session:
            session.params = {"apikey": self.api_key, "period": "quarter", "limit": quarters}
            for ticker in tickers:
                url = f"{self.base_url}income-statement/{ticker}"
                resp = session.get(url, timeout=30)
                if resp.status_code == 200:
                    df = pd.DataFrame(resp.json())
                    # System 1 Calculation: Net Profit Margin
                    if 'netIncome' in df.columns and 'revenue' in df.columns:
                        df['netProfitMargin'] = df['netIncome'] / df['revenue'].replace(0, np.nan)
                    all_financials[ticker] = df
                else:
                    all_financials[ticker] = self._get_yfinance_fallback(ticker)

        return {"profiles": profiles, "financials": all_financials}

    def get_historical_prices(self, ticker, limit=30):
        url = f"{self.base_url}historical-price-full/{ticker}?timeseries={limit}&apikey={self.api_key}"
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json().get('historical', [])
            return [day['close'] for day in data][::-1] # Ascending order
        return []
    
    def _get(self, endpoint: str, params: dict = None) -> dict:
        if params is None: params = {}
        params["apikey"] = self.api_key
        url = f"{self.base_url}{endpoint.lstrip('/')}"
        try:
            resp = requests.get(url, params=params)
            if resp.status_code == 402: return "PAYMENT_REQUIRED"
            resp.raise_for_status()
            return resp.json()
        except Exception: return {}

    def _get_yfinance_fallback(self, ticker: str) -> pd.DataFrame:
        try:
            stock = yf.Ticker(ticker)
            df = stock.quarterly_financials.T.reset_index()
            # Standardizing columns to match FMP naming convention for System 1
            rename_map = {
                'index': 'date', 
                'Total Revenue': 'revenue', 
                'Basic EPS': 'eps',
                'Ebitda': 'ebitda',
                'Net Income': 'netIncome'
            }
            df = df.rename(columns=rename_map)
            if 'netIncome' in df.columns and 'revenue' in df.columns:
                df['netProfitMargin'] = df['netIncome'] / df['revenue']
            return df
        except: return pd.DataFrame()
        
    def get_stock_news(self, ticker: str, limit: int = 10):
        """Fetches real-time headlines for consensus sentiment analysis."""
        url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit={limit}&apikey={self.api_key}"
        try:
            response = requests.get(url, timeout=15)
            return response.json()
        except Exception as e:
            print(f"❌ News API Error: {e}")
            return []