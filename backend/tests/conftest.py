import pytest


@pytest.fixture(autouse=True)
def _no_live_gdelt(monkeypatch):
    """Unit tests never touch the network."""
    monkeypatch.setenv("GDELT_DOC_ENABLED", "0")