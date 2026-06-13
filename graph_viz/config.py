"""Configuration: load LIGHTRAG_* secrets from .env.

Unlike Qdrant there is no port quirk — the ingress serves https on 443.
We only strip a trailing slash and sanity-check the scheme.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


# Defaults for the bundled LightRAG container (docker-compose service name + the
# fixed demo key wired into docker-compose.yml — local-only, never a real secret).
LOCAL_LIGHTRAG_URL = "http://lightrag:9621"
LOCAL_LIGHTRAG_API_KEY = "local-demo-key"


def is_local() -> bool:
    """True when LOKAL switches the app to the bundled local Docker services."""
    return os.environ.get("LOKAL", "").strip().lower() in {"1", "true", "yes", "on"}


def normalize_lightrag_url(raw: str) -> str:
    """Strip surrounding whitespace and a trailing slash; require http(s)."""
    raw = raw.strip().rstrip("/")
    if not raw.startswith(("http://", "https://")):
        raise ValueError(f"Invalid LIGHTRAG_URL: {raw!r}")
    return raw


@dataclass(frozen=True)
class GraphSettings:
    lightrag_url: str
    lightrag_api_key: str


def load_graph_settings(dotenv_path: str | None = None) -> GraphSettings:
    """Load and validate LightRAG settings from environment / .env.

    LOKAL=true → default to the bundled LightRAG container (``http://lightrag:9621``)
    with the fixed demo key, so no LIGHTRAG_* vars are required. Otherwise both
    LIGHTRAG_URL and LIGHTRAG_API_KEY are required (public ingress behavior).
    """
    load_dotenv(dotenv_path)
    if is_local():
        url = normalize_lightrag_url(os.environ.get("LIGHTRAG_URL", LOCAL_LIGHTRAG_URL))
        key = os.environ.get("LIGHTRAG_API_KEY", LOCAL_LIGHTRAG_API_KEY)
        return GraphSettings(lightrag_url=url, lightrag_api_key=key)
    try:
        url = normalize_lightrag_url(os.environ["LIGHTRAG_URL"])
        key = os.environ["LIGHTRAG_API_KEY"]
    except KeyError as exc:
        raise RuntimeError(f"Missing required env var: {exc.args[0]}") from exc
    return GraphSettings(lightrag_url=url, lightrag_api_key=key)
