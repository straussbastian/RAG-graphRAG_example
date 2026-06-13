# syntax=docker/dockerfile:1
# ── Unified RAG ⇄ GraphRAG demo ──────────────────────────────────────────────
# FastAPI app serving both the Qdrant point-cloud view and the LightRAG graph
# view behind one shell. Dependencies are installed reproducibly from uv.lock.

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# 1) Dependency layer — cached unless pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) Application code, frontends (web/, graph_web/ + vendored libs & fonts),
#    and the prebuilt data (data/points.json + data/graph.json).
COPY . .

# Internal port; docker-compose publishes it on the host as 8888.
EXPOSE 8000

CMD [".venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
