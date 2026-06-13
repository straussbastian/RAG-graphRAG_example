"""Place a live query point in the existing 3D layout as a similarity-weighted
centroid of its true nearest neighbors, biased toward the top match. Robust:
no out-of-sample projection needed, and the point always lands among its
neighbors (the pedagogical point)."""
from __future__ import annotations

import numpy as np


def weighted_centroid(
    coords: np.ndarray, scores: np.ndarray, top_bias: float = 0.5
) -> np.ndarray:
    """coords: (k,3) neighbor positions. scores: (k,) similarity in [0,1].
    Returns a (3,) position. top_bias in [0,1] pulls toward the best match."""
    coords = np.asarray(coords, dtype=float)
    scores = np.asarray(scores, dtype=float)
    w = np.clip(scores, 0.0, None)
    if w.sum() <= 0:
        w = np.ones(len(coords))
    w = w / w.sum()
    centroid = (coords * w[:, None]).sum(axis=0)
    top = coords[int(np.argmax(scores))]
    return centroid * (1.0 - top_bias) + top * top_bias
