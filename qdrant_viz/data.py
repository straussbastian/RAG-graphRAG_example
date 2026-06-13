"""Fetch points from Qdrant and produce 3D coordinates + cluster colors.

Clustering runs on the UMAP 3D coordinates (not the raw 1024D vectors) so the
colors align with the visible groups the audience sees."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def fetch_points(client, collection: str) -> list:
    """Scroll the whole collection with vectors + payload. Returns qdrant Records."""
    records: list = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            limit=256,
            with_vectors=True,
            with_payload=True,
            offset=offset,
        )
        records.extend(batch)
        if offset is None:
            break
    return records


def reduce_3d(vectors: np.ndarray, seed: int = 42) -> dict:
    """Reduce (n, 1024) to two 3D coordinate sets. Returns umap, pca, umap_model."""
    import umap  # local import: numba init is slow

    reducer = umap.UMAP(n_components=3, random_state=seed)
    umap_xyz = reducer.fit_transform(vectors)
    pca_xyz = PCA(n_components=3, random_state=seed).fit_transform(vectors)
    return {
        "umap": np.asarray(umap_xyz, dtype=float),
        "pca": np.asarray(pca_xyz, dtype=float),
        "umap_model": reducer,
    }


def cluster_labels(coords: np.ndarray, k_range=range(4, 11), seed: int = 42):
    """K-Means on coords; pick k by best silhouette. Returns (labels, k)."""
    best_labels = None
    best_k = None
    best_score = -1.0
    for k in k_range:
        if k >= len(coords):
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(coords)
        score = silhouette_score(coords, km.labels_)
        if score > best_score:
            best_score, best_k, best_labels = score, k, km.labels_
    return np.asarray(best_labels), best_k


def build_points_json(ids, umap, pca, labels, contents, locs, titles, collection, model) -> dict:
    """Assemble the JSON-serializable point cloud shared by the static HTML
    and the live app. `titles` is the per-chunk section title (the closest
    thing to a source name), derived from the markdown frontmatter."""
    points = []
    for i in range(len(ids)):
        points.append({
            "id": str(ids[i]),
            "umap": [round(float(v), 4) for v in umap[i]],
            "pca": [round(float(v), 4) for v in pca[i]],
            "cluster": int(labels[i]),
            "content": contents[i],
            "loc": locs[i],
            "title": titles[i],
        })
    return {
        "collection": collection,
        "model": model,
        "reductions": ["umap", "pca"],
        "points": points,
    }
