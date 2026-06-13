import pytest
from fastapi.testclient import TestClient

import app as m


def _client(monkeypatch):
    # fake embedder + minimal settings so the endpoint needs no real .env / network
    monkeypatch.setattr(
        m, "embed_texts",
        lambda texts, api_key, model="mistral-embed": [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )
    _fake_settings(monkeypatch)
    return TestClient(m.app)


def _fake_settings(monkeypatch):
    # monkeypatch (not plain assignment) so the global is restored after the
    # test — a bare fake here leaked into later test modules.
    monkeypatch.setattr(
        m, "_settings",
        type("S", (), {"mistral_api_key": "k", "embed_model": "mistral-embed"})(),
    )


def test_embed_returns_vectors_and_cosine(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/api/embed", json={"a": "hallo", "b": "welt"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["a"]["vector"]) == 3
    assert len(body["b"]["vector"]) == 3
    assert body["dims"] == 3
    assert body["cosine"] == pytest.approx(1.0)  # identical fake vectors


def test_embed_empty_input_400(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/api/embed", json={"a": "  ", "b": "x"})
    assert r.status_code == 400


def test_embed_matrix_nxn_and_lowest_pair(monkeypatch):
    vecs = {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0], "c": [1.0, 1.0, 0.0]}
    monkeypatch.setattr(m, "embed_texts", lambda texts, api_key, model="mistral-embed": [vecs[t] for t in texts])
    _fake_settings(monkeypatch)
    r = TestClient(m.app).post("/api/embed/matrix", json={"texts": ["a", "b", "c"]})
    assert r.status_code == 200
    d = r.json()
    assert d["labels"] == ["a", "b", "c"]
    assert len(d["matrix"]) == 3 and len(d["matrix"][0]) == 3
    assert d["matrix"][0][0] == pytest.approx(1.0)            # diagonal = self-similarity
    assert {d["min"]["i"], d["min"]["j"]} == {0, 1}           # a ⟂ b is the lowest pair
    assert d["min"]["value"] == pytest.approx(0.0)


def test_embed_matrix_needs_two(monkeypatch):
    _fake_settings(monkeypatch)
    r = TestClient(m.app).post("/api/embed/matrix", json={"texts": ["nur einer"]})
    assert r.status_code == 400
