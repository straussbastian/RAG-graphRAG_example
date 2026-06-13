"""Test wiring: point app.py at the small fixture instead of the real
data/points.json so live-query tests can resolve neighbor ids (a, b, c)
to known UMAP coordinates. Set before app is imported.

We also force a hermetic, network-free config: LOKAL=false (so importing app
never tries to reach/seed the local Qdrant container) plus dummy credentials, so
the suite runs without any real .env. python-dotenv's load_dotenv(override=False)
leaves these pre-set values untouched; tests that need other values use
monkeypatch (which reverts per-test)."""
import os
from pathlib import Path

_FIXTURE = Path(__file__).parent / "fixtures" / "points.json"
os.environ.setdefault("QDRANT_VIZ_POINTS", str(_FIXTURE))
os.environ.setdefault("LOKAL", "false")
os.environ.setdefault("MISTRAL_API_KEY", "test-key")
os.environ.setdefault("QDRANT_URL", "https://qdrant.test")
os.environ.setdefault("QDRANT_API_KEY", "test-key")
os.environ.setdefault("LIGHTRAG_URL", "https://lightrag.test")
os.environ.setdefault("LIGHTRAG_API_KEY", "test-key")
