"""LOKAL=true behavior: config defaults for the bundled local services, and the
Qdrant snapshot seeder (run against an in-process qdrant-client, no Docker)."""
import json

import pytest

import graph_viz.config as gcfg
import qdrant_viz.config as qcfg
from graph_viz.config import load_graph_settings
from qdrant_viz.config import is_local, load_settings


@pytest.mark.parametrize("val,expected", [
    ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
    ("false", False), ("0", False), ("", False), ("no", False),
])
def test_is_local_parsing(monkeypatch, val, expected):
    monkeypatch.setenv("LOKAL", val)
    assert is_local() is expected


def test_is_local_unset(monkeypatch):
    monkeypatch.delenv("LOKAL", raising=False)
    assert is_local() is False


def _stub_dotenv(monkeypatch):
    # Keep the real .env from leaking into the process env during the test.
    monkeypatch.setattr(qcfg, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(gcfg, "load_dotenv", lambda *a, **k: None)


def test_load_settings_local_defaults(monkeypatch):
    _stub_dotenv(monkeypatch)
    monkeypatch.setenv("LOKAL", "true")
    monkeypatch.setenv("MISTRAL_API_KEY", "m")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    s = load_settings()
    assert s.local is True
    assert s.qdrant_url == "http://qdrant:6333"  # no :443 forcing in local mode
    assert s.qdrant_api_key == ""
    assert s.mistral_api_key == "m"


def test_load_settings_local_uses_explicit_url_verbatim(monkeypatch):
    _stub_dotenv(monkeypatch)
    monkeypatch.setenv("LOKAL", "true")
    monkeypatch.setenv("MISTRAL_API_KEY", "m")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333/")
    s = load_settings()
    assert s.qdrant_url == "http://qdrant:6333"  # trailing slash trimmed, NOT normalized to :443


def test_load_settings_local_still_requires_mistral(monkeypatch):
    _stub_dotenv(monkeypatch)
    monkeypatch.setenv("LOKAL", "true")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        load_settings()


def test_load_settings_remote_still_requires_qdrant(monkeypatch):
    _stub_dotenv(monkeypatch)
    monkeypatch.delenv("LOKAL", raising=False)
    monkeypatch.setenv("MISTRAL_API_KEY", "m")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    with pytest.raises(RuntimeError):
        load_settings()


def test_load_graph_settings_local_defaults(monkeypatch):
    _stub_dotenv(monkeypatch)
    monkeypatch.setenv("LOKAL", "true")
    monkeypatch.delenv("LIGHTRAG_URL", raising=False)
    monkeypatch.delenv("LIGHTRAG_API_KEY", raising=False)
    s = load_graph_settings()
    assert s.lightrag_url == "http://lightrag:9621"
    assert s.lightrag_api_key == "local-demo-key"


def test_seed_qdrant_creates_and_is_idempotent(tmp_path):
    from qdrant_client import QdrantClient

    from qdrant_viz.seed import seed_qdrant

    seed = {
        "collection": "T",
        "dim": 3,
        "points": [
            {"id": 1, "vector": [0.1, 0.2, 0.3], "payload": {"content": "a"}},
            {"id": 2, "vector": [0.2, 0.1, 0.0], "payload": {"content": "b"}},
        ],
    }
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(seed), encoding="utf-8")

    client = QdrantClient(location=":memory:")
    assert seed_qdrant(client, "T", path) == 2
    assert seed_qdrant(client, "T", path) == 2  # idempotent — no re-insert / duplication
