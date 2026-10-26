import pytest

from src.streaming.ticker_resolver import clean_subject, pick_best


@pytest.mark.parametrize("q,expected", [
    ("Why is Reliance falling?", "reliance"),
    ("Why has the price of gold fallen?", "gold"),
    ("Infosys outlook", "infosys"),
    ("TSLA earnings impact", "tsla"),
    ("Why is AAPL dropping?", "aapl"),
    ("RELIANCE.NS", "reliance.ns"),
    ("why is the", ""),
    ("", ""),
])
def test_clean_subject(q, expected):
    assert clean_subject(q) == expected


def test_pick_best_prefers_equity_over_first():
    quotes = [{"symbol": "GC=F", "quoteType": "FUTURE", "shortname": "Gold Jun"},
              {"symbol": "ABX", "quoteType": "EQUITY", "shortname": "Barrick"}]
    assert pick_best(quotes) == ("ABX", "Barrick")


def test_pick_best_falls_back_to_first_and_handles_empty():
    assert pick_best([{"symbol": "GC=F", "quoteType": "FUTURE", "longname": "Gold"}]) == ("GC=F", "Gold")
    assert pick_best([]) == (None, None)
    assert pick_best(None) == (None, None)