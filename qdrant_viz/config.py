"""Configuration: load secrets from .env and normalize the Qdrant URL.

The public Qdrant ingress serves on 443; qdrant-client defaults to 6333 when
the URL carries no port, which times out. We always force an explicit :443.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from dotenv import load_dotenv


# Default endpoint of the bundled Qdrant container (docker-compose service name).
LOCAL_QDRANT_URL = "http://qdrant:6333"


def is_local() -> bool:
    """True when LOKAL switches the app to the bundled local Docker services."""
    return os.environ.get("LOKAL", "").strip().lower() in {"1", "true", "yes", "on"}


def normalize_qdrant_url(raw: str) -> str:
    """Strip an in-cluster :6333 and force an explicit https port (443)."""
    raw = raw.strip().rstrip("/").replace(":6333", "")
    parsed = urlparse(raw)
    scheme = parsed.scheme or "https"
    host = parsed.hostname
    if not host:
        raise ValueError(f"Invalid QDRANT_URL: {raw!r}")
    port = parsed.port or (443 if scheme == "https" else 80)
    return f"{scheme}://{host}:{port}"


@dataclass(frozen=True)
class Settings:
    qdrant_url: str
    qdrant_api_key: str
    mistral_api_key: str
    collection: str = "LizardDocu"
    embed_model: str = "mistral-embed"
    chat_model: str = "mistral-small-latest"
    local: bool = False


def load_settings(dotenv_path: str | None = None) -> Settings:
    """Load and validate settings from environment / .env.

    LOKAL=true → talk to the bundled local Qdrant container: the URL is used
    verbatim (no :443 forcing, default ``http://qdrant:6333``) and the API key is
    optional, so only MISTRAL_API_KEY is required. Otherwise the public ingress
    behavior is unchanged (QDRANT_URL/_API_KEY required, normalized to :443).
    """
    load_dotenv(dotenv_path)
    local = is_local()
    try:
        mkey = os.environ["MISTRAL_API_KEY"]
        if local:
            url = os.environ.get("QDRANT_URL", LOCAL_QDRANT_URL).strip().rstrip("/")
            qkey = os.environ.get("QDRANT_API_KEY", "")
        else:
            url = normalize_qdrant_url(os.environ["QDRANT_URL"])
            qkey = os.environ["QDRANT_API_KEY"]
    except KeyError as exc:
        raise RuntimeError(f"Missing required env var: {exc.args[0]}") from exc
    return Settings(
        qdrant_url=url, qdrant_api_key=qkey, mistral_api_key=mkey, local=local,
        chat_model=os.environ.get("MISTRAL_CHAT_MODEL", "mistral-small-latest"),
    )
