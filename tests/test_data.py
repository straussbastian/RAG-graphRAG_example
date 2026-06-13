import json

import numpy as np
from qdrant_viz.data import reduce_3d, cluster_labels
from qdrant_viz.data import build_points_json


def _blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.1, size=(40, 1024)) + 5
    b = rng.normal(0, 0.1, size=(40, 1024)) - 5
    c = rng.normal(0, 0.1, size=(40, 1024))
    return np.vstack([a, b, c])


def test_reduce_3d_shapes():
    x = _blobs()
    out = reduce_3d(x)
    assert out["umap"].shape == (120, 3)
    assert out["pca"].shape == (120, 3)
    assert out["umap_model"] is not None


def test_cluster_labels_finds_three_blobs():
    x = _blobs()
    coords = reduce_3d(x)["umap"]
    labels, k = cluster_labels(coords, k_range=range(2, 7))
    assert labels.shape == (120,)
    assert k == 3
    # each blob's 40 rows should be mostly one label
    for start in (0, 40, 80):
        block = labels[start:start + 40]
        dominant = np.bincount(block).max()
        assert dominant >= 36


def test_build_points_json_schema(tmp_path):
    umap = np.array([[1.0, 2, 3], [4, 5, 6]])
    pca = np.array([[7.0, 8, 9], [10, 11, 12]])
    labels = np.array([0, 1])
    ids = ["a", "b"]
    contents = ["first chunk", "second chunk"]
    locs = ["1-10", "11-20"]
    titles = ["Zeiterfassung", "Urlaub"]
    doc = build_points_json(ids, umap, pca, labels, contents, locs, titles,
                            collection="LizardDocu", model="mistral-embed")
    assert doc["collection"] == "LizardDocu"
    assert len(doc["points"]) == 2
    p0 = doc["points"][0]
    assert p0["id"] == "a"
    assert p0["umap"] == [1.0, 2.0, 3.0]
    assert p0["pca"] == [7.0, 8.0, 9.0]
    assert p0["cluster"] == 0
    assert p0["content"] == "first chunk"
    assert p0["loc"] == "1-10"
    assert p0["title"] == "Zeiterfassung"
    json.loads(json.dumps(doc))
