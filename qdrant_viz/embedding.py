"""Cosine similarity between two equal-length embedding vectors. Pure, no I/O."""
from __future__ import annotations

import numpy as np


def cosine_similarity(a, b) -> float:
    """Cosine of the angle between two vectors, in [-1, 1]. Returns 0.0 if either
    vector has zero magnitude."""
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))
