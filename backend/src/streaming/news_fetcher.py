"""Unified multi-source news fetcher.

CONTRACT: every item returned carries a parsed, timezone-aware UTC publish
timestamp. Items whose timestamp cannot be parsed are DROPPED, never defaulted.

Rationale: the attribution pipeline windows evidence on `published_at`
(lookup.graph.timeutil.preopen_window, lookup.graph.pools.LinkIndex.links_in),
and `hours_before_open` is rendered into both the generator prompt and the
judge's reference block. An item with a guessed timestamp would silently cross
a leakage boundary and corrupt an explanation that looks well-formed.

Item schema (stable contract for downstream consumers):
    {
        "headline":     str,            # non-empty
        "source":       str,            # publisher name as reported
        "url":          str | None,
        "published_at": pd.Timestamp,   # tz-aware, UTC
        "provider":     str,            # which API produced it
    }

After every fetch, `self.last_report` holds counts for observability:
    {"provider": str, "raw": int, "dropped_no_timestamp": int,
     "dropped_no_headline": int, "dropped_out_of_window": int, "kept": int}
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf

# Timestamps outside this band are treated as unparseable rather than trusted.
_MIN_TS = pd.Timestamp("2000-01-01", tz="UTC")
_FUTURE_SLACK = pd.Timedelta(days=2)


def to_utc(value) -> Optional[pd.Timestamp]:
    """Parse epoch seconds, epoch milliseconds, or an ISO-8601 string into a
    tz-aware UTC Timestamp. Returns None if the value is missing, unparseable,
    or outside a sane band -- callers must drop those items.

    Naive strings are assumed UTC: every provider here documents UTC, and
    assuming local time would shift windows by the IST offset.
    """
    if value is None or value == "":
        return None
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            v = float(value)
            if v <= 0:
                return None
            if v > 1e11:  # milliseconds, not seconds
                v /= 1000.0
            ts = pd.Timestamp(v, unit="s", tz="UTC")
        else:
            ts = pd.Timestamp(value)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    except Exception:
        return None
    if ts is pd.NaT or pd.isna(ts):
        return None
    if not (_MIN_TS <= ts <= pd.Timestamp.now(tz="UTC") + _FUTURE_SLACK):
        return None
    return ts


def _first(d: dict, *keys):
    """First present, non-empty value among `keys`."""
    for k in keys:
        if isinstance(d, dict):
            v = d.get(k)
            if v not in (None, "", {}, []):
                return v
    return None


def _yf_news_item(n: dict) -> dict:
    """Normalise one yfinance news entry.

    yfinance changed schema: newer builds nest everything under `content` with
    `pubDate`; older ones are flat with `providerPublishTime`. Both are handled
    because the installed version is not guaranteed across machines.
    """
    if not isinstance(n, dict):
        return {}
    content = n.get("content") if isinstance(n.get("content"), dict) else n

    title = _first(content, "title", "headline") or _first(n, "title", "headline")

    raw_ts = _first(content, "pubDate", "displayTime", "providerPublishTime") or _first(
        n, "providerPublishTime", "provider_publish_time", "pubDate"
    )

    url = None
    for container in (content, n):
        if not isinstance(container, dict):
            continue
        for key in ("canonicalUrl", "clickThroughUrl"):
            sub = container.get(key)
            if isinstance(sub, dict) and sub.get("url"):
                url = sub["url"]
                break
            if isinstance(sub, str) and sub:
                url = sub
                break
        if url:
            break
        if container.get("link"):
            url = container["link"]
            break

    provider = _first(content, "provider") or _first(n, "provider")
    if isinstance(provider, dict):
        provider = provider.get("displayName") or provider.get("name")
    source = provider or n.get("publisher") or "yfinance"

    return {"headline": title, "source": str(source), "url": url, "raw_ts": raw_ts}


class UnifiedNewsFetcher:
    """Multi-source financial news fetcher for stocks, commodities and Indian indices."""

    TEMPLATES = {
        "^BSESN": [
            "Sensex crash", "Middle East war impact India", "crude oil price",
            "FII selling", "RBI policy", "NIFTY", "BANKNIFTY",
        ],
        "^NSEI": [
            "Nifty fall", "conflict India", "Oil price surge",
            "Indian stock market news", "Adani Reliance", "SENSEX",
            "BANKNIFTY", "Dalal Street news",
        ],
        "XAUUSD": [
            "geopolitics", "war", "central bank buying", "inflation",
            "Fed rate cuts", "unemployment", "safe-haven",
        ],
        "XAGUSD": [
            "industrial demand", "solar energy", "silver deficit",
            "inflation hedge", "geopolitical risk",
        ],
        "USO": [
            "OPEC+", "shale production", "Strait of Hormuz",
            "oil inventory", "global recession", "energy transition",
        ],
        "DEFAULT": [
            "market volatility", "macroeconomic data", "interest rates", "economic growth",
        ],
    }

    def __init__(self, finnhub_api_key: Optional[str] = None, newsapi_key: Optional[str] = None):
        self.finnhub_key = finnhub_api_key or os.getenv("FINNHUB_API_KEY")
        self.newsapi_key = newsapi_key or os.getenv("NEWSAPI_KEY")
        self.gnews_key = os.getenv("GNEWS_API_KEY")
        self.finnhub_url = "https://finnhub.io/api/v1/company-news"
        self.newsapi_url = "https://newsapi.org/v2/everything"
        self.last_report: Dict = {}

    # ------------------------------------------------------------------
    # normalisation
    # ------------------------------------------------------------------
    def _finalize(
        self,
        raw_items: List[dict],
        provider: str,
        limit: int,
        since: Optional[pd.Timestamp] = None,
        until: Optional[pd.Timestamp] = None,
    ) -> List[Dict]:
        """Drop untimestamped/untitled items, apply the (since, until] window,
        sort newest-first, trim to `limit`, and record a drop report.

        The window is half-open on the left and closed on the right, matching
        lookup.graph.pools.window_bounds.
        """
        report = {
            "provider": provider,
            "raw": len(raw_items),
            "dropped_no_timestamp": 0,
            "dropped_no_headline": 0,
            "dropped_out_of_window": 0,
            "kept": 0,
        }
        out: List[Dict] = []
        for it in raw_items:
            headline = it.get("headline")
            if not isinstance(headline, str) or not headline.strip():
                report["dropped_no_headline"] += 1
                continue
            ts = to_utc(it.get("raw_ts"))
            if ts is None:
                report["dropped_no_timestamp"] += 1
                continue
            if since is not None and not (ts > pd.Timestamp(since)):
                report["dropped_out_of_window"] += 1
                continue
            if until is not None and not (ts <= pd.Timestamp(until)):
                report["dropped_out_of_window"] += 1
                continue
            out.append(
                {
                    "headline": headline.strip(),
                    "source": str(it.get("source") or provider),
                    "url": it.get("url"),
                    "published_at": ts,
                    "provider": provider,
                }
            )

        out.sort(key=lambda d: d["published_at"], reverse=True)
        out = out[:limit]
        report["kept"] = len(out)
        self.last_report = report
        if report["dropped_no_timestamp"]:
            print(
                f"[news] {provider}: dropped {report['dropped_no_timestamp']}/{report['raw']} "
                f"items with no usable publish timestamp"
            )
        return out

    # ------------------------------------------------------------------
    # routing
    # ------------------------------------------------------------------
    def fetch(
        self,
        ticker: str,
        limit: int = 10,
        since: Optional[pd.Timestamp] = None,
        until: Optional[pd.Timestamp] = None,
    ) -> List[Dict]:
        """Fetch news for `ticker`, restricted to the half-open window (since, until].

        Both bounds are optional; pass tz-aware timestamps (or anything
        pd.Timestamp accepts). Callers that need a leakage-safe evidence window
        should always pass `until`.
        """
        ticker_upper = ticker.upper()

        if ".NS" in ticker_upper or ".BO" in ticker_upper:
            company_name = None
            try:
                company_name = yf.Ticker(ticker_upper).info.get("longName")
            except Exception as e:
                print(f"[news] could not resolve company name for {ticker_upper}: {e}")
            if company_name:
                news = self._fetch_gnews_india(company_name, limit, since, until)
                if news:
                    return news
            return self._fetch_yfinance(ticker_upper, limit, since, until)

        if ticker_upper in ["^BSESN", "SENSEX", "^NSEI", "NIFTY"]:
            return self._fetch_indian_index(ticker_upper, limit, since, until)

        if ticker_upper in ["XAUUSD", "XAGUSD", "USO"]:
            return self._fetch_commodity(ticker_upper, limit, since, until)

        news = self._fetch_finnhub(ticker_upper, limit, since, until)
        if news:
            return news
        return self._fetch_yfinance(ticker_upper, limit, since, until)

    # ------------------------------------------------------------------
    # providers
    # ------------------------------------------------------------------
    def _fetch_gnews_india(self, query, limit, since=None, until=None) -> List[Dict]:
        if not self.gnews_key:
            return []
        params = {
            "q": query, "lang": "en", "country": "in",
            "max": max(limit * 2, 10), "apikey": self.gnews_key,
        }
        if since is not None:
            params["from"] = pd.Timestamp(since).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        if until is not None:
            params["to"] = pd.Timestamp(until).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            resp = requests.get("https://gnews.io/api/v4/search", params=params, timeout=10)
            if resp.status_code != 200:
                print(f"[news] GNews error {resp.status_code}")
                return []
            raw = [
                {
                    "headline": a.get("title"),
                    "source": (a.get("source") or {}).get("name", "GNews"),
                    "url": a.get("url"),
                    "raw_ts": a.get("publishedAt"),
                }
                for a in resp.json().get("articles", [])
            ]
            return self._finalize(raw, "gnews", limit, since, until)
        except Exception as e:
            print(f"[news] GNews exception: {e}")
            return []

    def _fetch_indian_index(self, ticker, limit, since=None, until=None) -> List[Dict]:
        if not self.newsapi_key:
            print("[news] NewsAPI key not set; falling back to yfinance search")
            return self._fetch_yfinance_index_fallback(ticker, limit, since, until)

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
            "pageSize": min(max(limit * 3, 20), 100),
            "domains": (
                "economictimes.indiatimes.com,moneycontrol.com,"
                "business-standard.com,livemint.com,financialexpress.com"
            ),
        }
        if since is not None:
            params["from"] = pd.Timestamp(since).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        if until is not None:
            params["to"] = pd.Timestamp(until).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            resp = requests.get(self.newsapi_url, params=params, timeout=15)
            if resp.status_code != 200:
                print(f"[news] NewsAPI error {resp.status_code}")
                return self._fetch_yfinance_index_fallback(ticker, limit, since, until)
            kw = ["sensex", "bse", "nifty", "nse", "indian market", "dalal street"]
            raw = [
                {
                    "headline": a.get("title"),
                    "source": (a.get("source") or {}).get("name", "NewsAPI"),
                    "url": a.get("url"),
                    "raw_ts": a.get("publishedAt"),
                }
                for a in resp.json().get("articles", [])
                if any(k in (a.get("title") or "").lower() for k in kw)
            ]
            return self._finalize(raw, "newsapi", limit, since, until)
        except Exception as e:
            print(f"[news] NewsAPI exception: {e}")
            return self._fetch_yfinance_index_fallback(ticker, limit, since, until)

    def _fetch_yfinance_index_fallback(self, ticker, limit, since=None, until=None) -> List[Dict]:
        try:
            results = yf.Search(
                "Sensex OR Nifty OR Indian stock market", max_results=max(limit * 2, 10)
            ).news
            return self._finalize(
                [_yf_news_item(n) for n in (results or [])], "yfinance_search", limit, since, until
            )
        except Exception as e:
            print(f"[news] yfinance index fallback failed: {e}")
            return []

    def _fetch_commodity(self, ticker, limit, since=None, until=None) -> List[Dict]:
        query = {"XAUUSD": "gold price", "XAGUSD": "silver price", "USO": "oil price"}.get(
            ticker, "commodity market"
        )
        try:
            results = yf.Search(query, max_results=max(limit * 2, 10)).news
            return self._finalize(
                [_yf_news_item(n) for n in (results or [])], "yfinance_search", limit, since, until
            )
        except Exception as e:
            print(f"[news] commodity fetch failed: {e}")
            return []

    def _fetch_finnhub(self, ticker, limit, since=None, until=None) -> List[Dict]:
        if not self.finnhub_key:
            return []
        # Finnhub's from/to are date-granular; the precise window is enforced in
        # _finalize against each item's own timestamp.
        end_dt = pd.Timestamp(until).tz_convert("UTC") if until is not None else pd.Timestamp.now(tz="UTC")
        start_dt = pd.Timestamp(since).tz_convert("UTC") if since is not None else end_dt - pd.Timedelta(days=7)
        try:
            resp = requests.get(
                self.finnhub_url,
                params={
                    "symbol": ticker,
                    "from": start_dt.strftime("%Y-%m-%d"),
                    "to": end_dt.strftime("%Y-%m-%d"),
                    "token": self.finnhub_key,
                },
                timeout=10,
            )
            if resp.status_code != 200:
                print(f"[news] Finnhub error {resp.status_code}")
                return []
            raw = [
                {
                    "headline": n.get("headline"),
                    "source": n.get("source") or "Finnhub",
                    "url": n.get("url"),
                    "raw_ts": n.get("datetime"),  # epoch seconds
                }
                for n in resp.json()
            ]
            return self._finalize(raw, "finnhub", limit, since, until)
        except Exception as e:
            print(f"[news] Finnhub exception: {e}")
            return []

    def _fetch_yfinance(self, ticker, limit, since=None, until=None) -> List[Dict]:
        try:
            tk = yf.Ticker(ticker)
            news = tk.news or []
            if not news:
                company_name = None
                try:
                    company_name = tk.info.get("longName")
                except Exception:
                    pass
                if company_name:
                    print(f"[news] ticker news empty; searching by name: {company_name}")
                    news = yf.Search(company_name, max_results=max(limit * 2, 10)).news or []
            return self._finalize(
                [_yf_news_item(n) for n in news], "yfinance", limit, since, until
            )
        except Exception as e:
            print(f"[news] yfinance fetch failed: {e}")
            return []