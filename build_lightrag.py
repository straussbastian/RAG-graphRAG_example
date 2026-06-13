"""One-time author build: reconstruct the LizardDocu text from the 249 shipped
chunks and ingest it into the LOCAL LightRAG container so its knowledge graph is
pre-built and committed as lightrag_storage/. The LLM + embeddings run on Mistral
via the container's OpenAI-compatible binding (configured in docker-compose.yml).

Run with an empty lightrag_storage/ after starting the container:

    docker compose up -d lightrag
    uv run python build_lightrag.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

BASE = "http://localhost:9621"
HEADERS = {"X-API-Key": "local-demo-key"}


def reconstruct_text() -> str:
    """Concatenate the unique chunk contents (in scroll order) back into one doc.
    Exact chunk boundaries don't matter — LightRAG re-chunks before extraction."""
    pts = json.loads(Path("data/points.json").read_text(encoding="utf-8"))["points"]
    seen: set[str] = set()
    parts: list[str] = []
    for p in pts:
        c = (p.get("content") or "").strip()
        if c and c not in seen:
            seen.add(c)
            parts.append(c)
    return "\n\n".join(parts)


def main() -> None:
    text = reconstruct_text()
    print(f"Reconstructed {len(text)} chars from chunks; ingesting…")
    r = httpx.post(
        f"{BASE}/documents/text", headers=HEADERS,
        json={"text": text, "file_source": "LizardDocu.md"}, timeout=120,
    )
    r.raise_for_status()
    print("ingest accepted:", r.json().get("track_id"))

    deadline = time.time() + 60 * 45
    while time.time() < deadline:
        time.sleep(15)
        statuses = httpx.get(f"{BASE}/documents", headers=HEADERS, timeout=60).json().get("statuses", {})
        counts = {k: len(v) for k, v in statuses.items()}
        print("status:", counts)
        busy = counts.get("processing", 0) + counts.get("pending", 0)
        if busy == 0 and counts.get("processed", 0) >= 1:
            if counts.get("failed", 0):
                print("WARNING: some docs failed processing", file=sys.stderr)
            break
    else:
        print("WARNING: timed out waiting for processing", file=sys.stderr)

    g = httpx.get(
        f"{BASE}/graphs", params={"label": "*", "max_nodes": 5000, "max_depth": 6},
        headers=HEADERS, timeout=120,
    ).json()
    print(f"DONE — graph: {len(g.get('nodes', []))} nodes, {len(g.get('edges', []))} edges")


if __name__ == "__main__":
    main()
