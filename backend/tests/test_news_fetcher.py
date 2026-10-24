"""Offline unit tests for the news-fetcher timestamp contract.

No network. Synthetic fixtures are used only to test code correctness
(PROTOCOL.md rule 12), never to produce reported numbers.

Run from the backend directory:
    uv run pytest tests/test_news_fetcher.py -v
"""

import pandas as pd
import pytest

from src.streaming.news_fetcher import UnifiedNewsFetcher, _yf_news_item, to_utc


# ----------------------------------------------------------------------
# to_utc
# ----------------------------------------------------------------------
def test_epoch_seconds():
    ts = to_utc(1760000000)
    assert ts is not None and str(ts.tz) == "UTC"


def test_epoch_milliseconds_detected():
    sec, ms = to_utc(1760000000), to_utc(1760000000000)
    assert sec is not None and ms is not None
    assert abs((sec - ms).total_seconds()) < 1


def test_iso_with_z_suffix():
    ts = to_utc("2026-10-24T06:30:00Z")
    assert ts == pd.Timestamp("2026-10-24T06:30:00", tz="UTC")


def test_iso_with_offset_converted_to_utc():
    ts = to_utc("2026-10-24T12:00:00+05:30")
    assert ts == pd.Timestamp("2026-10-24T06:30:00", tz="UTC")


def test_naive_string_assumed_utc():
    assert to_utc("2026-10-24 06:30:00") == pd.Timestamp("2026-10-24T06:30:00", tz="UTC")


@pytest.mark.parametrize("bad", [None, "", "not a date", 0, -5, True, False, float("nan")])
def test_unparseable_returns_none(bad):
    assert to_utc(bad) is None


def test_absurd_past_and_future_rejected():
    assert to_utc("1970-01-02") is None
    assert to_utc(pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=30)) is None


# ----------------------------------------------------------------------
# yfinance schema variants
# ----------------------------------------------------------------------
def test_yf_nested_content_schema():
    item = _yf_news_item(
        {
            "id": "abc",
            "content": {
                "title": "Infosys wins deal",
                "pubDate": "2026-10-24T04:00:00Z",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://example.com/a"},
            },
        }
    )
    assert item["headline"] == "Infosys wins deal"
    assert item["source"] == "Reuters"
    assert item["url"] == "https://example.com/a"
    assert to_utc(item["raw_ts"]) == pd.Timestamp("2026-10-24T04:00:00", tz="UTC")


def test_yf_flat_legacy_schema():
    item = _yf_news_item(
        {
            "title": "TCS Q2 results",
            "providerPublishTime": 1760000000,
            "publisher": "ET",
            "link": "https://example.com/b",
        }
    )
    assert item["headline"] == "TCS Q2 results"
    assert item["url"] == "https://example.com/b"
    assert to_utc(item["raw_ts"]) is not None


def test_yf_garbage_entry_does_not_raise():
    assert _yf_news_item({}) == {} or _yf_news_item({}).get("headline") is None
    assert _yf_news_item(None) == {}


# ----------------------------------------------------------------------
# _finalize: the contract that protects the leakage boundary
# ----------------------------------------------------------------------
def _fx():
    return UnifiedNewsFetcher.__new__(UnifiedNewsFetcher)


def test_untimestamped_items_are_dropped_not_defaulted():
    f = _fx()
    out = f._finalize(
        [
            {"headline": "has time", "raw_ts": "2026-10-24T04:00:00Z", "source": "A"},
            {"headline": "no time", "raw_ts": None, "source": "B"},
            {"headline": "bad time", "raw_ts": "whenever", "source": "C"},
        ],
        "test",
        limit=10,
    )
    assert [d["headline"] for d in out] == ["has time"]
    assert f.last_report["dropped_no_timestamp"] == 2
    assert all(d["published_at"] is not None for d in out)


def test_empty_headlines_dropped():
    f = _fx()
    out = f._finalize(
        [
            {"headline": "   ", "raw_ts": "2026-10-24T04:00:00Z"},
            {"headline": None, "raw_ts": "2026-10-24T04:00:00Z"},
        ],
        "test",
        limit=10,
    )
    assert out == []
    assert f.last_report["dropped_no_headline"] == 2


def test_window_is_half_open_left_closed_right():
    """Matches lookup.graph.pools.window_bounds: (since, until]."""
    f = _fx()
    since = pd.Timestamp("2026-10-24T00:00:00", tz="UTC")
    until = pd.Timestamp("2026-10-24T06:00:00", tz="UTC")
    out = f._finalize(
        [
            {"headline": "exactly at since", "raw_ts": since},      # excluded
            {"headline": "inside", "raw_ts": "2026-10-24T03:00:00Z"},
            {"headline": "exactly at until", "raw_ts": until},       # included
            {"headline": "after until", "raw_ts": "2026-10-24T09:00:00Z"},
        ],
        "test",
        limit=10,
        since=since,
        until=until,
    )
    assert sorted(d["headline"] for d in out) == ["exactly at until", "inside"]
    assert f.last_report["dropped_out_of_window"] == 2


def test_sorted_newest_first_and_trimmed():
    f = _fx()
    out = f._finalize(
        [
            {"headline": "old", "raw_ts": "2026-10-24T01:00:00Z"},
            {"headline": "new", "raw_ts": "2026-10-24T05:00:00Z"},
            {"headline": "mid", "raw_ts": "2026-10-24T03:00:00Z"},
        ],
        "test",
        limit=2,
    )
    assert [d["headline"] for d in out] == ["new", "mid"]


def test_every_item_matches_the_published_schema():
    f = _fx()
    out = f._finalize(
        [{"headline": "h", "raw_ts": "2026-10-24T04:00:00Z", "source": "S", "url": "u"}],
        "prov",
        limit=5,
    )
    item = out[0]
    assert set(item) == {"headline", "source", "url", "published_at", "provider"}
    assert isinstance(item["published_at"], pd.Timestamp)
    assert item["published_at"].tzinfo is not None
    assert item["provider"] == "prov"