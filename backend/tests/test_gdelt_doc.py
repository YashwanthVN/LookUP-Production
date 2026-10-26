"""Offline tests for the GDELT DOC provider (synthetic fixtures; PROTOCOL rule 12)."""
import pandas as pd
import pytest

from src.streaming import gdelt_doc as g

SINCE = pd.Timestamp("2026-10-22T00:00:00", tz="UTC")
UNTIL = pd.Timestamp("2026-10-25T12:00:00", tz="UTC")


def art(title, seen="20261024T040000Z", domain="x.com"):
    return {"title": title, "seendate": seen, "domain": domain, "url": "http://x"}


def test_core_name():
    assert g.core_name("Reliance Industries Limited") == "Reliance Industries"
    assert g.core_name("Tata Consultancy Services Ltd.") == "Tata Consultancy Services"
    assert g.core_name("Castrol India Limited") == "Castrol India"
    assert g.core_name("") == ""


def test_parse_seendate():
    assert g.parse_seendate("20261024T043000Z") == pd.Timestamp("2026-10-24T04:30:00", tz="UTC")
    assert g.parse_seendate("garbage") is None
    assert g.parse_seendate(None) is None


def test_filter_keeps_only_company_titles_in_window():
    arts = [
        art("Reliance Industries posts record profit"),
        art("Oil prices fall on demand worries"),                     # no name
        art("Reliance Industries old story", seen="20261001T000000Z"),  # out of window
        art("Reliance Industries future story", seen="20261026T000000Z"),
        art("Reliance Industries bad time", seen="nonsense"),
        art(""),
        art("reliance industries POSTS RECORD PROFIT!"),              # dup after normalising
    ]
    out, rep = g.filter_articles(arts, "Reliance Industries", "RELIANCE", SINCE, UNTIL, 10)
    assert [d["headline"] for d in out] == ["Reliance Industries posts record profit"]
    assert rep["dropped_no_title_match"] == 1
    assert rep["dropped_out_of_window"] == 2
    assert rep["dropped_no_timestamp"] == 1
    assert rep["dropped_no_title"] == 1
    assert rep["dropped_dup"] == 1


def test_window_is_half_open():
    at_since = art("Reliance Industries a", seen="20261022T000000Z")
    at_until = art("Reliance Industries b", seen="20261025T120000Z")
    out, _ = g.filter_articles([at_since, at_until], "Reliance Industries", None, SINCE, UNTIL, 10)
    assert [d["headline"] for d in out] == ["Reliance Industries b"]


def test_symbol_match_is_whole_word():
    out, _ = g.filter_articles([art("TCS wins deal"), art("ETCSX rallies")], "Tata Consultancy",
                               "TCS", SINCE, UNTIL, 10)
    assert [d["headline"] for d in out] == ["TCS wins deal"]


def test_item_schema_and_order():
    out, _ = g.filter_articles(
        [art("Reliance Industries one", seen="20261023T000000Z"),
         art("Reliance Industries two", seen="20261024T000000Z")],
        "Reliance Industries", None, SINCE, UNTIL, 10)
    assert [d["headline"] for d in out][0].endswith("two")  # newest first
    assert set(out[0]) == {"headline", "source", "url", "published_at", "provider"}
    assert out[0]["published_at"].tzinfo is not None


class _Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._p, self.text = status, payload, text
    def json(self):
        if self._p is None:
            raise ValueError("no json")
        return self._p


class _Sess:
    def __init__(self, resp): self.resp, self.calls = resp, 0
    def get(self, *a, **k):
        self.calls += 1
        return self.resp


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(g, "MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(g, "RETRY_WAIT_S", 0.0)
    g._cache.clear()


def test_fetch_happy_path_and_cache():
    s = _Sess(_Resp(payload={"articles": [art("Reliance Industries wins")]}))
    items, rep = g.fetch_company_news("Reliance Industries Limited", SINCE, UNTIL, session=s)
    assert len(items) == 1 and s.calls == 1
    items2, rep2 = g.fetch_company_news("Reliance Industries Limited", SINCE, UNTIL, session=s)
    assert s.calls == 1 and rep2["cached"] is True and len(items2) == 1


def test_non_json_rate_limit_text_is_reported_not_raised():
    s = _Sess(_Resp(text="Please limit requests to one every 5 seconds"))
    items, rep = g.fetch_company_news("Reliance Industries", SINCE, UNTIL, session=s)
    assert items == [] and rep["error"] == "non_json_response"


def test_http_error_and_exception_are_reported():
    items, rep = g.fetch_company_news("A Corp X", SINCE, UNTIL, session=_Sess(_Resp(status=500)))
    assert items == [] and rep["error"] == "http_500"

    class Boom:
        def get(self, *a, **k): raise ConnectionError("down")
    items, rep = g.fetch_company_news("B Corp Y", SINCE, UNTIL, session=Boom())
    assert items == [] and rep["error"].startswith("request_failed")


def test_empty_articles_key_ok():
    items, rep = g.fetch_company_news("C Corp Z", SINCE, UNTIL, session=_Sess(_Resp(payload={})))
    assert items == [] and rep["raw"] == 0


def test_news_agent_adds_gdelt_for_indian_primary_only(monkeypatch):
    from src.agents import news_agent
    monkeypatch.setenv("GDELT_DOC_ENABLED", "1")
    calls = []
    def fake(ticker, since, until):
        calls.append(ticker)
        return ([{"headline": f"{ticker} gdelt story", "source": "d", "url": "u",
                  "published_at": until - pd.Timedelta(hours=3), "provider": "gdelt_doc"}],
                {"raw": 9, "kept": 1})
    class Empty:
        def fetch(self, *a, **k): return []
    monkeypatch.setattr(news_agent, "_gdelt_items", fake)
    monkeypatch.setattr(news_agent, "_make_fetcher", lambda: Empty())
    st = {"tickers": ["RELIANCE.NS", "BPCL.NS"], "as_of": UNTIL.isoformat()}
    out = news_agent.news_agent(st)["news_pool"]
    assert calls == ["RELIANCE.NS"]               # primary only, one call
    assert out[0]["is_primary"] and out[0]["provider"] == "gdelt_doc"
    d = news_agent.news_agent(st)["diagnostics"]
    assert d["gdelt"] == {"raw": 9, "kept": 1} and d["own_count"] == 1 and d["pool_size"] == 1

    calls.clear()
    st = {"tickers": ["AAPL", "MSFT"], "as_of": UNTIL.isoformat()}
    news_agent.news_agent(st)                      # US ticker, fetcher empty -> fallback fires
    assert calls == ["AAPL"]


def test_news_agent_respects_disable_flag(monkeypatch):
    from src.agents import news_agent
    monkeypatch.setenv("GDELT_DOC_ENABLED", "0")
    monkeypatch.setattr(news_agent, "_gdelt_items", lambda *a: pytest.fail("called"))
    class Empty:
        def fetch(self, *a, **k): return []
    monkeypatch.setattr(news_agent, "_make_fetcher", lambda: Empty())
    news_agent.news_agent({"tickers": ["RELIANCE.NS"], "as_of": UNTIL.isoformat()})


class _Seq:
    def __init__(self, *resps): self.resps, self.calls = list(resps), 0
    def get(self, *a, **k):
        self.calls += 1
        return self.resps.pop(0)


def test_429_is_retried_once_then_succeeds():
    s = _Seq(_Resp(status=429), _Resp(payload={"articles": [art("Reliance Industries wins")]}))
    items, rep = g.fetch_company_news("Reliance Industries", SINCE, UNTIL, session=s)
    assert s.calls == 2 and len(items) == 1


def test_429_twice_is_reported_and_not_cached():
    s = _Seq(_Resp(status=429), _Resp(status=429))
    items, rep = g.fetch_company_news("Reliance Industries", SINCE, UNTIL, session=s)
    assert items == [] and rep["error"] == "http_429" and s.calls == 2
    assert not g._cache