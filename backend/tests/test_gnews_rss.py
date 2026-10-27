import pandas as pd

from src.streaming import gnews_rss as g

RSS = """<?xml version="1.0"?><rss><channel>
<item><title>JSW Steel jumps on Q2 profit - Mint</title><link>https://n/1</link>
<pubDate>Mon, 26 Oct 2026 04:12:00 GMT</pubDate><source url="x">Mint</source></item>
<item><title>JSW Steel jumps on Q2 profit - Mint</title><link>https://n/1b</link>
<pubDate>Mon, 26 Oct 2026 04:30:00 GMT</pubDate><source url="x">Mint</source></item>
<item><title>Steel prices rise worldwide - Reuters</title><link>https://n/2</link>
<pubDate>Sun, 25 Oct 2026 10:00:00 GMT</pubDate><source url="x">Reuters</source></item>
<item><title>JSW Steel old story - ET</title><link>https://n/3</link>
<pubDate>Mon, 19 Oct 2026 10:00:00 GMT</pubDate><source url="x">ET</source></item>
<item><title>No date here - ET</title><link>https://n/4</link><source url="x">ET</source></item>
</channel></rss>"""

SINCE = pd.Timestamp("2026-10-23T00:00:00Z")
UNTIL = pd.Timestamp("2026-10-26T12:00:00Z")


def test_parse_strips_source_suffix_and_drops_undated():
    items = g.parse_rss(RSS)
    assert [i["title"] for i in items][:2] == ["JSW Steel jumps on Q2 profit"] * 2
    assert all(i["title"] != "No date here" for i in items) and len(items) == 4
    assert items[0]["published_at"] == pd.Timestamp("2026-10-26T04:12:00Z")
    assert items[0]["source"] == "Mint"


def test_filter_window_name_and_dedupe():
    items, rep = g.filter_items(g.parse_rss(RSS), SINCE, UNTIL, "JSW Steel", "JSWSTEEL", 10)
    assert [i["headline"] for i in items] == ["JSW Steel jumps on Q2 profit"]
    assert rep["dropped_out_of_window"] == 1      # the 19 Oct story
    assert rep["dropped_no_title_match"] == 1     # generic steel story
    assert rep["dropped_dup"] == 1
    assert items[0]["provider"] == "google_news"


def test_window_is_half_open():
    raw = [{"title": "JSW Steel a", "source": "s", "url": None, "published_at": SINCE},
           {"title": "JSW Steel b", "source": "s", "url": None, "published_at": UNTIL}]
    items, _ = g.filter_items(raw, SINCE, UNTIL, "JSW Steel", None, 10)
    assert [i["headline"] for i in items] == ["JSW Steel b"]   # (since, until]


def test_build_query_pads_days():
    assert g.build_query("JSW Steel", SINCE, UNTIL) == '"JSW Steel" after:2026-10-22 before:2026-10-27'


class _Resp:
    def __init__(self, code, text=""): self.status_code, self.text = code, text


class _Http:
    def __init__(self, resp): self.resp, self.calls = resp, 0
    def get(self, *a, **k):
        self.calls += 1
        if isinstance(self.resp, Exception): raise self.resp
        return self.resp


def test_fetch_never_raises_and_reports(monkeypatch):
    monkeypatch.setattr(g, "MIN_INTERVAL_S", 0)
    g._cache.clear()
    assert g.fetch_news("T1", SINCE, UNTIL, session=_Http(_Resp(503)))[1] == {"error": "http_503"}
    assert g.fetch_news("T2", SINCE, UNTIL, session=_Http(RuntimeError("x")))[1]["error"].startswith("request_failed")
    items, rep = g.fetch_news("JSW Steel", SINCE, UNTIL, name="JSW Steel", session=_Http(_Resp(200, RSS)))
    assert len(items) == 1 and rep["kept"] == 1


def test_fetch_is_cached(monkeypatch):
    monkeypatch.setattr(g, "MIN_INTERVAL_S", 0)
    g._cache.clear()
    h = _Http(_Resp(200, RSS))
    g.fetch_news("JSW Steel", SINCE, UNTIL, name="JSW Steel", session=h)
    _, rep = g.fetch_news("JSW Steel", SINCE, UNTIL, name="JSW Steel", session=h)
    assert h.calls == 1 and rep.get("cached")