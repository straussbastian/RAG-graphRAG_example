import pytest

from qdrant_viz.embedding import cosine_similarity


def test_identical_vectors_one():
    v = [1.0, 2.0, 3.0, 4.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_orthogonal_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_minus_one():
    assert cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)


def test_zero_vector_returns_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
