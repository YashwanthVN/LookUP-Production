"""Sampling, blind labelling and scoring helpers for the precision audit.

Sampling is seeded and sorted so the sample is fixed before any result is seen.
Scoring is clustered by ticker: items within one ticker are not independent,
so the bootstrap resamples tickers, not rows.
"""
import json
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def sample_tickers(universe: Dict[str, List[str]], n: int, seed: int = 0) -> List[str]:
    flat = sorted({t for members in universe.values() for t in members})
    rng = random.Random(seed)
    return sorted(rng.sample(flat, min(n, len(flat))))


def load_universe(path) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def precision_at_k(df: pd.DataFrame, label_col: str, k: int = 5) -> pd.Series:
    """Mean label over each ticker's top-k labelled rows."""
    top = df[df["rank"] <= k].dropna(subset=[label_col])
    return top.groupby("ticker")[label_col].mean()


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    v = np.asarray(list(values), dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)])
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize(merged: pd.DataFrame, label_col: str, k: int = 5,
              group_col: Optional[str] = None) -> pd.DataFrame:
    p = precision_at_k(merged, label_col, k)
    meta = merged.drop_duplicates("ticker").set_index("ticker")
    rows = []
    groups = [("all", p)] if group_col is None else [
        (g, p[p.index.isin(meta.index[meta[group_col] == g])]) for g in sorted(meta[group_col].unique())]
    for name, series in groups:
        m, lo, hi = bootstrap_ci(series.values)
        rows.append({"group": name, "tickers": len(series), "precision@%d" % k: m,
                     "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)