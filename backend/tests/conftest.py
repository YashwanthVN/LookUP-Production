import pytest


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Unit tests never touch the network: no GDELT, no Google News, no yfinance profile or prices."""
    monkeypatch.setenv("GDELT_DOC_ENABLED", "0")
    monkeypatch.setenv("GNEWS_ENABLED", "0")
    monkeypatch.delenv("LOOKUP_SKIP_MOVE_CHECK", raising=False)
    from src.agents import profile
    from src.pricing import move_check

    def fake_profile(t):
        base = t.split(".")[0]
        return {"name": f"{base} Corp", "summary": f"{base} Corp makes widgets and services."}

    monkeypatch.setattr(profile, "get_profile", fake_profile)
    monkeypatch.setattr(move_check, "peer_context", lambda *a, **k: [])
    monkeypatch.setattr(move_check, "check_move",
                        lambda t, *a, **k: {"status": "unavailable", "reason": "test"})