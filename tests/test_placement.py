import numpy as np
from qdrant_viz.placement import weighted_centroid


def test_query_lands_closest_to_top_match():
    coords = np.array([[10.0, 0, 0], [0, 0, 0], [0, 10.0, 0]])
    scores = np.array([0.9, 0.5, 0.4])  # first neighbor is the top match
    p = weighted_centroid(coords, scores)
    d_top = np.linalg.norm(p - coords[0])
    d_other = np.linalg.norm(p - coords[2])
    assert d_top < d_other


def test_result_within_bounding_box():
    coords = np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])
    scores = np.array([0.3, 0.6, 0.1])
    p = weighted_centroid(coords, scores)
    assert (p >= coords.min(axis=0) - 1e-9).all()
    assert (p <= coords.max(axis=0) + 1e-9).all()


def test_zero_scores_fall_back_to_mean():
    coords = np.array([[0.0, 0, 0], [2, 0, 0]])
    scores = np.array([0.0, 0.0])
    p = weighted_centroid(coords, scores, top_bias=0.0)
    assert np.allclose(p, [1.0, 0, 0])
