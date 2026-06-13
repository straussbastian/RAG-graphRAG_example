"""Seed the local Qdrant collection from a committed snapshot file.

In LOKAL mode the bundled Qdrant container starts empty; on app startup we upsert
the LizardDocu vectors (exported once via ``export_vectors.py``) so live search
works without ever contacting the external cluster. Idempotent: if the collection
is already fully populated we leave it untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

SEED_PATH = Path("data/qdrant_seed.json")


def seed_qdrant(client: QdrantClient, collection: str, seed_path: Path = SEED_PATH) -> int:
    """Create ``collection`` and upsert the snapshot vectors if it isn't populated.

    Returns the point count after seeding. No-op (returns the existing count) when
    the collection already holds the full snapshot; rebuilds from scratch on a
    partial/mismatched count so the demo is deterministic.
    """
    doc = json.loads(seed_path.read_text(encoding="utf-8"))
    points = doc["points"]
    if not points:
        raise ValueError(f"Seed file {seed_path} has no points")
    dim = int(doc.get("dim") or len(points[0]["vector"]))

    if client.collection_exists(collection):
        existing = client.count(collection, exact=True).count
        if existing >= len(points):
            return existing
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(id=p["id"], vector=p["vector"], payload=p.get("payload") or {})
            for p in points
        ],
    )
    return client.count(collection, exact=True).count
