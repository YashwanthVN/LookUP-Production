"""Find a calendar date written in a question ("why did Infosys fall on 16 Oct").

Pure and conservative: returns None unless a day number sits next to a month name,
or the text is an ISO date, a numeric date WITH a 4-digit year (26-10-2026,
26/10/2026, 26.10.2026; read day-first, the Indian convention, unless the second
number is above 12), or the word 'yesterday'. Short slash dates (3/4) are NOT
parsed: day/month versus month/day is ambiguous. A year is optional; a date that
would fall in the future is read as last year's.
"""
import datetime as dt
import re
from typing import Optional, Tuple

_MONTHS = {m: i for i, names in enumerate(
    [("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"),
     ("may",), ("jun", "june"), ("jul", "july"), ("aug", "august"),
     ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
     ("dec", "december")], 1) for m in names}
_MON = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ISO = re.compile(r"\b(20\d\d)-(\d{1,2})-(\d{1,2})\b")
_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MON})\b\.?(?:,?\s+(20\d\d))?", re.I)
_MDY = re.compile(rf"\b({_MON})\b\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(20\d\d))?", re.I)
_NUM = re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](20\d\d)\b")
_YESTERDAY = re.compile(r"\byesterday\b", re.I)


def _make(y: int, m: int, d: int, today: dt.date) -> Optional[dt.date]:
    try:
        out = dt.date(y, m, d)
    except ValueError:
        return None
    return out


def find_date(text: str, today: dt.date) -> Tuple[Optional[dt.date], Optional[Tuple[int, int]]]:
    """Return (date, (start, end)) of the first date in `text`, else (None, None)."""
    text = text or ""
    m = _ISO.search(text)
    if m:
        d = _make(int(m[1]), int(m[2]), int(m[3]), today)
        if d:
            return d, m.span()
    m = _NUM.search(text)
    if m:
        a, b = int(m[1]), int(m[2])
        day, mon = (b, a) if (b > 12 and a <= 12) else (a, b)   # day-first (Indian convention) unless clearly month-first
        d = _make(int(m[3]), mon, day, today)
        if d:
            return d, m.span()
    for rx, order in ((_DMY, "dm"), (_MDY, "md")):
        m = rx.search(text)
        if not m:
            continue
        day, mon = (int(m[1]), _MONTHS[m[2].lower()]) if order == "dm" else (int(m[2]), _MONTHS[m[1].lower()])
        if m[3]:
            d = _make(int(m[3]), mon, day, today)
        else:
            d = _make(today.year, mon, day, today)
            if d and d > today:
                d = _make(today.year - 1, mon, day, today)
        if d:
            return d, m.span()
    m = _YESTERDAY.search(text)
    if m:
        return today - dt.timedelta(days=1), m.span()
    return None, None


def strip_date(text: str, today: dt.date) -> str:
    """The question without its date words, for ticker search."""
    d, span = find_date(text, today)
    if d is None:
        return text
    return (text[:span[0]] + " " + text[span[1]:]).strip()