import pytest
from qdrant_viz.config import normalize_qdrant_url


@pytest.mark.parametrize("raw,expected", [
    ("https://qdrant.apps.bastianstrauss.digital:6333", "https://qdrant.apps.bastianstrauss.digital:443"),
    ("https://qdrant.apps.bastianstrauss.digital", "https://qdrant.apps.bastianstrauss.digital:443"),
    ("https://qdrant.apps.bastianstrauss.digital:443", "https://qdrant.apps.bastianstrauss.digital:443"),
    ("https://qdrant.apps.bastianstrauss.digital:6333/", "https://qdrant.apps.bastianstrauss.digital:443"),
])
def test_normalize_strips_6333_and_forces_443(raw, expected):
    assert normalize_qdrant_url(raw) == expected


def test_normalize_rejects_garbage():
    with pytest.raises(ValueError):
        normalize_qdrant_url("not-a-url")
