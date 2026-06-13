from fastapi.testclient import TestClient
import app as app_module


def _make_client(monkeypatch):
    # Avoid real network: fake embedding + fake qdrant search.
    monkeypatch.setattr(app_module, "embed_text", lambda text, api_key, model="mistral-embed": [0.1] * 1024)

    class _Hit:
        def __init__(self, id, score):
            self.id, self.score = id, score

    class _FakeQdrant:
        def query_points(self, collection_name, query, limit, with_payload):
            class _R:
                points = [_Hit("a", 0.95), _Hit("b", 0.40)]
            return _R()

    monkeypatch.setattr(app_module, "_qdrant", _FakeQdrant())
    return TestClient(app_module.app)


def test_query_returns_point_and_neighbors(monkeypatch):
    client = _make_client(monkeypatch)
    r = client.get("/api/query", params={"q": "hello", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "hello"
    assert len(body["point"]["umap"]) == 3
    ids = [n["id"] for n in body["neighbors"]]
    assert ids == ["a", "b"]
    import numpy as np
    p = np.array(body["point"]["umap"])
    a = np.array([0.0, 0.0, 0.0]); b = np.array([10.0, 0.0, 0.0])
    assert np.linalg.norm(p - a) < np.linalg.norm(p - b)


def test_root_serves_frontend(monkeypatch):
    client = _make_client(monkeypatch)
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
