"""Turn a free-text question into a ticker.

Ported from the old Streamlit app, which worked: strip question filler so the
search sees only the subject ("Why is Reliance falling?" -> "reliance"), then
prefer equities, indexes and currencies over other quote types.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

STOP_WORDS = {
    "why", "is", "are", "was", "were", "has", "have", "had", "did", "does", "do",
    "the", "a", "an", "of", "on", "in", "at", "to", "for", "and", "with", "about",
    "what", "how", "when", "which", "who", "will", "would", "can", "could",
    "today", "now", "currently", "recently", "lately", "news",
    "price", "prices", "stock", "stocks", "share", "shares", "market",
    "fall", "falls", "falling", "fallen", "fell",
    "drop", "drops", "dropping", "dropped", "dip", "dipped", "decline", "declined",
    "rise", "rises", "rising", "risen", "rose", "gain", "gained", "jump", "jumped",
    "up", "down", "going", "go", "went", "moving", "move", "moved", "movement",
    "situation", "outlook", "impact", "earnings", "result", "results",
}
PREFERRED_TYPES = {"EQUITY", "INDEX", "CURRENCY"}


def clean_subject(query: str) -> str:
    words = re.findall(r"[A-Za-z0-9&.\-^=]+", (query or "").lower())
    return " ".join(w for w in words if w not in STOP_WORDS).strip()


def pick_best(quotes: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    for q in quotes or []:
        if q.get("quoteType") in PREFERRED_TYPES and q.get("symbol"):
            return q["symbol"], q.get("shortname") or q.get("longname") or q["symbol"]
    for q in quotes or []:
        if q.get("symbol"):
            return q["symbol"], q.get("shortname") or q.get("longname") or q["symbol"]
    return None, None


def resolve(query: str) -> Tuple[Optional[str], Optional[str]]:
    subject = clean_subject(query)
    if not subject:
        return None, None
    import yfinance as yf
    try:
        quotes = yf.Search(subject, max_results=5, timeout=8).quotes
    except Exception:
        return None, None
    return pick_best(quotes)