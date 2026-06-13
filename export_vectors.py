"""One-time author build: export the LizardDocu vectors + payload from the
external Qdrant into data/qdrant_seed.json — the snapshot the bundled local
Qdrant container is seeded from at startup.

Run with the EXTERNAL Qdrant credentials present in .env (QDRANT_URL/_API_KEY);
the LOKAL flag is ignored here since this always targets the public cluster.
LLM-free: just a scroll + dump.

    uv run python export_vectors.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from qdrant_viz.config import normalize_qdrant_url
from qdrant_viz.data import fetch_points

OUT = Path("data/qdrant_seed.json")


def main() -> None:
    load_dotenv()
    url = normalize_qdrant_url(os.environ["QDRANT_URL"])
    client = QdrantClient(
        url=url, api_key=os.environ["QDRANT_API_KEY"],
        timeout=60, check_compatibility=False, prefer_grpc=False,
    )
    collection = os.environ.get("QDRANT_COLLECTION", "LizardDocu")
    records = fetch_points(client, collection)
    if not records:
        raise SystemExit(f"Collection {collection!r} is empty.")

    points = [
        {"id": r.id, "vector": list(r.vector), "payload": r.payload or {}}
        for r in records
    ]
    dim = len(points[0]["vector"])
    if dim != 1024:
        raise SystemExit(f"Expected 1024-dim vectors, got {dim}.")

    doc = {
        "collection": collection,
        "model": "mistral-embed",
        "dim": dim,
        "distance": "Cosine",
        "points": points,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(doc), encoding="utf-8")
    print(f"Wrote {OUT} — {len(points)} points, dim={dim}")


if __name__ == "__main__":
    main()
