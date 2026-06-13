import json

from fastapi.testclient import TestClient

import app as m


def _client(monkeypatch, tmp_path):
    doc = {
        "nodes": [
            {"id": "Betriebsmittel", "type": "concept", "description": "", "degree": 1},
            {"id": "Lizard", "type": "concept", "description": "", "degree": 1},
        ],
        "links": [{"source": "Betriebsmittel", "target": "Lizard", "description": ""}],
        "types": ["concept"],
    }
    gp = tmp_path / "graph.json"
    gp.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(m, "GRAPH_PATH", gp)

    class _FakeClient:
        def query_data(self, query, mode="mix", top_k=8):
            return {
                "data": {
                    "entities": [{"entity_name": "Betriebsmittel"}, {"entity_name": "Lizard"}],
                    "relationships": [{"src_id": "Betriebsmittel", "tgt_id": "Lizard"}],
                },
                "metadata": {
                    "query_mode": mode,
                    "keywords": {"high_level": ["Pruefauftrag"], "low_level": ["Betriebsmittel"]},
                },
            }

    m._graph_doc = doc
    m._graph_node_ids = {"Betriebsmittel", "Lizard"}
    m._graph_client = _FakeClient()
    return TestClient(m.app)


def test_query_returns_highlight(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/graph/query", json={"query": "pruefauftrag", "mode": "local"})
    assert r.status_code == 200
    b = r.json()
    assert set(b["highlight_nodes"]) == {"Betriebsmittel", "Lizard"}
    assert b["highlight_edges"] == [["Betriebsmittel", "Lizard"]]
    assert b["mode"] == "local"
    assert b["keywords"]["high_level"] == ["Pruefauftrag"]


def test_empty_query_returns_400(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/graph/query", json={"query": "   "})
    assert r.status_code == 400


def test_graph_view_serves_frontend(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.get("/graph")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_shell_at_root(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "GraphRAG" in r.text and "RAG" in r.text
