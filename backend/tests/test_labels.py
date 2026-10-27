import numpy as np
import pandas as pd

from src.evaluation import labels as L


def test_sampling_is_deterministic_sorted_and_unique():
    uni = {"A": ["X1", "X2", "X3"], "B": ["X3", "X4", "X5", "X6"]}
    a = L.sample_tickers(uni, 4, seed=0)
    assert a == L.sample_tickers(uni, 4, seed=0)
    assert a == sorted(set(a)) and len(a) == 4
    assert len(L.sample_tickers(uni, 99, seed=0)) == 6


def test_precision_at_k_uses_top_k_and_skips_unlabelled():
    df = pd.DataFrame({"ticker": ["A"] * 4 + ["B"] * 2,
                       "rank": [1, 2, 3, 6, 1, 2],
                       "about_company": [1, 0, 1, 0, 0, np.nan]})
    p = L.precision_at_k(df, "about_company", 5)
    assert p["A"] == 2 / 3 and p["B"] == 0.0


def test_bootstrap_ci_contains_mean_and_is_seeded():
    m, lo, hi = L.bootstrap_ci([0.2, 0.4, 0.6, 0.8, 1.0])
    assert lo <= m <= hi and m == 0.6
    assert L.bootstrap_ci([0.2, 0.4, 0.6]) == L.bootstrap_ci([0.2, 0.4, 0.6])
    assert np.isnan(L.bootstrap_ci([])[0])


def test_summarize_groups_by_market():
    df = pd.DataFrame({"ticker": ["A", "A", "B", "B"], "rank": [1, 2, 1, 2],
                       "market": ["IN", "IN", "US", "US"], "about_company": [1, 1, 0, 0]})
    out = L.summarize(df, "about_company", 5, group_col="market").set_index("group")
    assert out.loc["IN", "precision@5"] == 1.0 and out.loc["US", "precision@5"] == 0.0