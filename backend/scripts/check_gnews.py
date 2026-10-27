"""Live check of the Google News source. Run from backend/:
    uv run python scripts/check_gnews.py "JSW Steel" IN 7
Prints how many headlines came back for the last N days and the newest few.
"""
import sys

import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.streaming import gnews_rss

name = sys.argv[1] if len(sys.argv) > 1 else "JSW Steel"
region = sys.argv[2] if len(sys.argv) > 2 else "IN"
days = int(sys.argv[3]) if len(sys.argv) > 3 else 7
until = pd.Timestamp.now(tz="UTC")
items, rep = gnews_rss.fetch_news(name, until - pd.Timedelta(days=days), until,
                                  region=region, name=name, limit=40)
print({k: v for k, v in rep.items() if k != "query"})
print("query:", rep.get("query"))
for it in items[:8]:
    print(it["published_at"].strftime("%Y-%m-%d %H:%MZ"), "|", it["source"], "|", it["headline"])
if not items:
    print("NO ITEMS - if 'error' is shown above, Google News is unreachable from this network.")