"""Google News RSS search: keyless, date-filterable headlines.

Why it exists: yfinance returns little for Indian tickers and GDELT is
rate-limited (HTTP 429 is common), so a big move could end with zero articles.
This is a second, independent source.

NOT the validated corpus and NOT licensed for commercial redistribution.
Check Google's terms (or switch to a licensed news API) before charging for
the product. Titles are the publisher's headline; Google appends " - Source".

Timestamps come from <pubDate>, which is the publisher's time. Items without
a parseable pubDate are dropped. Day-level `after:`/`before:` operators are
padded by one day in the query and the exact window is applied here.
"""
import re
import threading
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import pandas as pd
import requests

from src.streaming.gdelt_doc import names_company

URL = "https://news.google.com/rss/search"
MIN_INTERVAL_S = 1.5
CACHE_TTL_S = 600
TIMEOUT_S = 10
REGIONS = {"IN": ("en-IN", "IN", "IN:en"), "US": ("en-US", "US", "US:en")}

_lock = threading.Lock()
_last_call = 0.0
_cache: Dict[Tuple, Tuple[float, List[Dict[str, Any]], Dict[str, Any]]] = {}


def build_query(term: str, since: pd.Timestamp, until: pd.Timestamp) -> str:
    """'"JSW Steel" after:2026-10-22 before:2026-10-27' (day-padded)."""
    a = (since.tz_convert("UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    b = (until.tz_convert("UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return f'"{term}" after:{a} before:{b}'


def parse_rss(xml_text: str) -> List[Dict[str, Any]]:
    """Pure: RSS text -> [{title, source, url, published_at}]. Bad items are skipped."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        src = (it.findtext("source") or "").strip()
        if not title:
            continue
        if src and title.endswith(f" - {src}"):
            title = title[: -(len(src) + 3)].strip()
        try:
            ts = pd.Timestamp(parsedate_to_datetime(it.findtext("pubDate") or ""))
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        except Exception:
            continue
        out.append({"title": title, "source": src or "google_news",
                    "url": (it.findtext("link") or "").strip() or None,
                    "published_at": ts})
    return out


def filter_items(raw, since, until, name: Optional[str], symbol: Optional[str], limit: int):
    """Pure: window (since, until], optional company-name filter, dedupe, newest first."""
    rep = {"raw": len(raw), "dropped_out_of_window": 0, "dropped_no_title_match": 0,
           "dropped_dup": 0}
    out, seen = [], set()
    for a in raw:
        ts = a["published_at"]
        if not (since < ts <= until):
            rep["dropped_out_of_window"] += 1
            continue
        if name and not names_company(a["title"], name, symbol):
            rep["dropped_no_title_match"] += 1
            continue
        key = re.sub(r"\W+", " ", a["title"].lower()).strip()
        if key in seen:
            rep["dropped_dup"] += 1
            continue
        seen.add(key)
        out.append({"headline": a["title"], "source": a["source"], "url": a["url"],
                    "published_at": ts, "provider": "google_news"})
    out.sort(key=lambda d: d["published_at"], reverse=True)
    rep["kept"] = min(len(out), limit)
    return out[:limit], rep


def fetch_news(term: str, since: pd.Timestamp, until: pd.Timestamp, region: str = "IN",
               name: Optional[str] = None, symbol: Optional[str] = None,
               limit: int = 40, session: Optional[Any] = None):
    """Return (items, report). Never raises. `name` turns on the title filter."""
    global _last_call
    if not term:
        return [], {"error": "empty_term"}
    q = build_query(term, since, until)
    key = (q, name, since.floor("1h"), until.floor("10min"), region)
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < CACHE_TTL_S:
        return hit[1], {**hit[2], "cached": True}

    hl, gl, ceid = REGIONS.get(region, REGIONS["US"])
    url = f"{URL}?q={quote_plus(q)}&hl={hl}&gl={gl}&ceid={quote_plus(ceid)}"
    http = session or requests
    with _lock:
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = http.get(url, timeout=TIMEOUT_S,
                            headers={"User-Agent": "Mozilla/5.0 (LookUP research)"})
        except Exception as e:
            _last_call = time.monotonic()
            return [], {"error": f"request_failed:{type(e).__name__}"}
        _last_call = time.monotonic()
    if getattr(resp, "status_code", 200) != 200:
        return [], {"error": f"http_{resp.status_code}"}

    items, rep = filter_items(parse_rss(resp.text), since, until, name, symbol, limit)
    rep["query"] = q
    _cache[key] = (time.monotonic(), items, rep)
    return items, rep