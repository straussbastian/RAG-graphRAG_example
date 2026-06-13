import pytest

import graph_viz.config as config_mod
from graph_viz.config import normalize_lightrag_url


@pytest.mark.parametrize("raw,expected", [
    ("https://lightrag.apps.bastianstrauss.digital", "https://lightrag.apps.bastianstrauss.digital"),
    ("https://lightrag.apps.bastianstrauss.digital/", "https://lightrag.apps.bastianstrauss.digital"),
    ("  https://x.y/  ", "https://x.y"),
])
def test_normalize_strips_trailing_slash(raw, expected):
    assert normalize_lightrag_url(raw) == expected


def test_normalize_rejects_garbage():
    with pytest.raises(ValueError):
        normalize_lightrag_url("not-a-url")


def test_load_graph_settings_reads_env(monkeypatch):
    # Stub load_dotenv so the real .env does not leak into the test.
    monkeypatch.setattr(config_mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("LIGHTRAG_URL", "https://lr.example/")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "secret")
    s = config_mod.load_graph_settings()
    assert s.lightrag_url == "https://lr.example"  # trailing slash stripped
    assert s.lightrag_api_key == "secret"


def test_load_graph_settings_missing_raises(monkeypatch):
    monkeypatch.setattr(config_mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("LIGHTRAG_URL", raising=False)
    monkeypatch.delenv("LIGHTRAG_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        config_mod.load_graph_settings()
