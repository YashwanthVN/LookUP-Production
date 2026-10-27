"""GDELT DOC 2.0 article search: live, keyless news for companies the free
news APIs do not cover (most Indian tickers).

NOT the validated corpus. The research corpus came from GDELT GKG joined to
entities; this is a keyword search over GDELT's article index. A title filter
keeps only articles that actually name the company. Precision is unmeasured.

`seendate` is when GDELT first saw the article (typically minutes to hours
after publication), not the publisher's timestamp. It is used as published_at.
Because it can only be later than the true time, windowing on it can drop a
borderline article but cannot admit one from after the cutoff.

Rate limit: GDELT asks for at most one request per ~5 s. Calls are serialised
and spaced; results are cached for 10 minutes.
"""
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MIN_INTERVAL_S = 8.0
RETRY_WAIT_S = 10.0
CACHE_TTL_S = 600
TIMEOUT_S = 12

_lock = threading.Lock()
_last_call = 0.0
_cache: Dict[Tuple, Tuple[float, List[Dict[str, Any]], Dict[str, Any]]] = {}

_SUFFIX = re.compile(
    r"\b(limited|ltd\.?|corporation|corp\.?|incorporated|inc\.?|company|co\.?|plc|"
    r"private|pvt\.?|the)\b", re.I)


def core_name(long_name: str) -> str:
    """'Reliance Industries Limited' -> 'Reliance Industries'."""
    s = _SUFFIX.sub(" ", long_name or "")
    return re.sub(r"\s+", " ", re.sub(r"[^\w&\- ]", " ", s)).strip()


def parse_seendate(s: Any) -> Optional[pd.Timestamp]:
    try:
        return pd.Timestamp(pd.to_datetime(str(s), format="%Y%m%dT%H%M%SZ", utc=True))
    except Exception:
        return None


def _fmt(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y%m%d%H%M%S")


def names_company(title: str, name: str, symbol: Optional[str]) -> bool:
    t = title.lower()
    if name and name.lower() in t:
        return True
    if symbol and len(symbol) >= 3:
        return re.search(rf"\b{re.escape(symbol.lower())}\b", t) is not None
    return False


def filter_articles(articles, name, symbol, since, until, limit):
    """Pure function: parse, drop unusable/off-topic/out-of-window, dedupe, sort."""
    rep = {"raw": len(articles), "dropped_no_title": 0, "dropped_no_timestamp": 0,
           "dropped_no_title_match": 0, "dropped_out_of_window": 0, "dropped_dup": 0}
    out, seen = [], set()
    for a in articles:
        title = (a.get("title") or "").strip()
        if not title:
            rep["dropped_no_title"] += 1
            continue
        ts = parse_seendate(a.get("seendate"))
        if ts is None:
            rep["dropped_no_timestamp"] += 1
            continue
        if not (since < ts <= until):
            rep["dropped_out_of_window"] += 1
            continue
        if not names_company(title, name, symbol):
            rep["dropped_no_title_match"] += 1
            continue
        key = re.sub(r"\W+", " ", title.lower()).strip()
        if key in seen:
            rep["dropped_dup"] += 1
            continue
        seen.add(key)
        out.append({"headline": title, "source": a.get("domain") or "gdelt",
                    "url": a.get("url"), "published_at": ts, "provider": "gdelt_doc"})
    out.sort(key=lambda d: d["published_at"], reverse=True)
    rep["kept"] = min(len(out), limit)
    return out[:limit], rep


def fetch_company_news(name: str, since: pd.Timestamp, until: pd.Timestamp,
                       symbol: Optional[str] = None, limit: int = 40,
                       session: Optional[Any] = None):
    """Return (items, report). Never raises; failures are reported."""
    global _last_call
    name = core_name(name)
    if not name:
        return [], {"error": "empty_name"}

    key = (name, since.floor("1h"), until.floor("10min"))
    now = time.monotonic()
    hit = _cache.get(key)
    if hit and now - hit[0] < CACHE_TTL_S:
        return hit[1], {**hit[2], "cached": True}

    params = {
        "query": f'"{name}" sourcelang:english',
        "mode": "ArtList", "format": "json", "sort": "DateDesc",
        "maxrecords": 250,
        "startdatetime": _fmt(since), "enddatetime": _fmt(until),
    }
    http = session or requests
    with _lock:
        resp = None
        for attempt in (0, 1):
            wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                resp = http.get(URL, params=params, timeout=TIMEOUT_S)
            except Exception as e:
                _last_call = time.monotonic()
                return [], {"error": f"request_failed:{type(e).__name__}"}
            _last_call = time.monotonic()
            if getattr(resp, "status_code", 200) == 429 and attempt == 0:
                time.sleep(RETRY_WAIT_S)  # rate limited: wait once, try again
                continue
            break

    if getattr(resp, "status_code", 200) != 200:
        return [], {"error": f"http_{resp.status_code}"}
    try:
        data = resp.json()
    except ValueError:
        # GDELT answers rate-limit / bad-query with plain text, not JSON.
        return [], {"error": "non_json_response", "body": (resp.text or "")[:120]}

    items, rep = filter_articles(data.get("articles") or [], name, symbol,
                                 since, until, limit)
    rep["query"] = params["query"]
    _cache[key] = (time.monotonic(), items, rep)
    return items, rep