import os
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class UnifiedNewsFetcher:
    """
    A multi-source news fetcher that retrieves financial news for stocks, 
    commodities, and Indian market indices using Finnhub, NewsAPI, and yfinance.
    """

    # 🏛️ Global Macro Templates for sentiment boosting and keyword matching
    TEMPLATES = {
        "^BSESN": [
            "Sensex crash", "Middle East war impact India", "crude oil price", 
            "FII selling", "RBI policy", "NIFTY", "BANKNIFTY"
        ],
        "^NSEI": [
            "Nifty fall", "conflict India", "Oil price surge", 
            "Indian stock market news", "Adani Reliance", "SENSEX", 
            "BANKNIFTY", "Dalal Street news"
        ],
        "XAUUSD": [
            "geopolitics", "war", "central bank buying", "inflation", 
            "Fed rate cuts", "unemployment", "safe-haven"
        ],
        "XAGUSD": [
            "industrial demand", "solar energy", "silver deficit", 
            "inflation hedge", "geopolitical risk"
        ],
        "USO": [
            "OPEC+", "shale production", "Strait of Hormuz", 
            "oil inventory", "global recession", "energy transition"
        ],
        "DEFAULT": [
            "market volatility", "macroeconomic data", "interest rates", "economic growth"
        ]
    }

    def __init__(self, finnhub_api_key: Optional[str] = None, newsapi_key: Optional[str] = None):
        self.finnhub_key = finnhub_api_key or os.getenv("FINNHUB_API_KEY")
        self.newsapi_key = newsapi_key or os.getenv("NEWSAPI_KEY")
        self.gnews_key = os.getenv("GNEWS_API_KEY")
        self.finnhub_url = "https://finnhub.io/api/v1/company-news"
        self.newsapi_url = "https://newsapi.org/v2/everything"

    def fetch(self, ticker: str, limit: int = 10) -> List[Dict]:
        ticker_upper = ticker.upper()

        if ".NS" in ticker_upper or ".BO" in ticker_upper:
            try:
                ticker_obj = yf.Ticker(ticker_upper)
                company_name = ticker_obj.info.get('longName', ticker_upper)
                news = self._fetch_gnews_india(company_name, limit)
                if news:
                    return news
                # Fallback to yfinance name search if GNews returns nothing
                return self._fetch_yfinance(ticker_upper, limit)
            except Exception as e:
                print(f"⚠️ Error resolving Indian company name: {e}")
                return self._fetch_yfinance(ticker_upper, limit)
        
        # Route request based on asset type
        elif ticker_upper in ["^BSESN", "SENSEX", "^NSEI", "NIFTY"]:
            return self._fetch_indian_index(ticker_upper, limit)

        elif ticker_upper in ["XAUUSD", "XAGUSD", "USO"]:
            return self._fetch_commodity(ticker_upper, limit)

        else:
            # Stocks – try Finnhub first, fallback to yfinance
            news = self._fetch_finnhub(ticker_upper, limit)
            if news:
                return news
            return self._fetch_yfinance(ticker_upper, limit)
        
    def _fetch_gnews_india(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Scrapes regional Indian news using GNews. 
        Ideal for OLAELEC.BO, COALINDIA.NS, etc.
        """
        if not self.gnews_key:
            return []
            
        url = f"https://gnews.io/api/v4/search?q={query}&lang=en&country=in&max={limit}&apikey={self.gnews_key}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                articles = response.json().get("articles", [])
                return [{"headline": a["title"], "source": a["source"]["name"]} for a in articles]
            else:
                print(f"❌ GNews API error {response.status_code}")
        except Exception as e:
            print(f"❌ GNews Error: {e}")
        return []

    def _fetch_indian_index(self, ticker: str, limit: int) -> List[Dict]:
        if not self.newsapi_key:
            print("⚠️ NewsAPI key not found. Falling back to yfinance search.")
            return self._fetch_yfinance_index_fallback(ticker, limit)

        # Build specific query based on ticker
        if ticker in ["^BSESN", "SENSEX"]:
            query = "Sensex OR BSE Sensex OR S&P BSE Sensex"
        elif ticker in ["^NSEI", "NIFTY"]:
            query = "Nifty OR NSE Nifty OR Nifty 50"
        else:
            query = "Indian stock market"

        params = {
            "q": query,
            "apiKey": self.newsapi_key,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit * 2,  # Fetch extra for relevance filtering
            "domains": (
                "economictimes.indiatimes.com,moneycontrol.com,"
                "business-standard.com,livemint.com,financialexpress.com"
            )
        }

        try:
            resp = requests.get(self.newsapi_url, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                articles = data.get("articles", [])
                headlines = []
                relevant_keywords = ["sensex", "bse", "nifty", "nse", "indian market", "dalal street"]
                
                for art in articles:
                    title = art.get("title", "")
                    if any(kw in title.lower() for kw in relevant_keywords):
                        headlines.append({
                            "headline": title,
                            "source": art.get("source", {}).get("name", "NewsAPI"),
                            "url": art.get("url")
                        })
                return headlines[:limit]
            else:
                print(f"❌ NewsAPI error {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"❌ NewsAPI exception: {e}")

        return self._fetch_yfinance_index_fallback(ticker, limit)

    def _fetch_yfinance_index_fallback(self, ticker: str, limit: int) -> List[Dict]:
        try:
            search_query = "Sensex OR Nifty OR Indian stock market"
            results = yf.Search(search_query, max_results=limit).news
            return [{"headline": n["title"], "source": "yfinance"} for n in results]
        except Exception:
            return []

    def _fetch_commodity(self, ticker: str, limit: int) -> List[Dict]:
        commodity_map = {
            "XAUUSD": "gold price",
            "XAGUSD": "silver price",
            "USO": "oil price"
        }
        query = commodity_map.get(ticker, "commodity market")
        try:
            results = yf.Search(query, max_results=limit).news
            return [{"headline": n["title"], "source": "yfinance"} for n in results]
        except Exception:
            return []

    def _fetch_finnhub(self, ticker: str, limit: int) -> List[Dict]:
        if not self.finnhub_key:
            return []
        try:
            end = datetime.now().strftime('%Y-%m-%d')
            start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            params = {
                'symbol': ticker, 
                'from': start, 
                'to': end, 
                'token': self.finnhub_key
            }
            resp = requests.get(self.finnhub_url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return [
                    {
                        "headline": n.get("headline"), 
                        "source": "Finnhub", 
                        "url": n.get("url")
                    } for n in data[:limit]
                ]
        except Exception:
            pass
        return []

    def _fetch_yfinance(self, ticker: str, limit: int) -> List[Dict]:
        try:
            ticker_obj = yf.Ticker(ticker)
            news = ticker_obj.news
            
            # If direct ticker news is empty (Common for India), try a Name Search
            if not news:
                # Get the long name (e.g., "Bharat Electronics Limited")
                company_name = ticker_obj.info.get('longName')
                if company_name:
                    print(f"🔍 Ticker news empty. Searching by name: {company_name}")
                    news = yf.Search(company_name, max_results=limit).news
            
            return [
                {"headline": n.get("title"), "source": "yfinance"} 
                for n in news[:limit]
            ]
        except Exception as e:
            print(f"❌ yfinance fetch failed: {e}")
            return []