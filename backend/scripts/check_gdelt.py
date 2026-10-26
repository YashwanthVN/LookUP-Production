"""Live check of the GDELT DOC provider. Needs internet; not part of pytest.

    uv run python scripts/check_gdelt.py
    uv run python scripts/check_gdelt.py "Infosys" "Tata Motors"
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.streaming import gdelt_doc  # noqa: E402

names = sys.argv[1:] or ["Reliance Industries", "Tata Consultancy Services", "HDFC Bank"]
until = pd.Timestamp.now(tz="UTC")
since = until - pd.Timedelta(hours=72)
for n in names:
    items, rep = gdelt_doc.fetch_company_news(n, since, until, symbol=n.split()[0])
    print(f"\n=== {n}")
    print({k: v for k, v in rep.items() if k != "query"})
    for it in items[:8]:
        age = (until - it["published_at"]).total_seconds() / 3600
        print(f"  {age:5.1f}h  {it['source'][:22]:22s}  {it['headline'][:90]}")