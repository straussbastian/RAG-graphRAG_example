"""Pipeline tab: retrieve (embed + qdrant with payload), answer (prompt build +
Mistral chat), graph (same question to LightRAG). All network faked."""
from fastapi.testclient import TestClient

import app as app_module
import graph_viz.client as client_mod
import qdrant_viz.mistral as mistral_mod
from graph_viz.client import LightRAGClient
from qdrant_viz.mistral import chat_complete


class _Hit:
    def __init__(self, id, score, payload):
        self.id, self.score, self.payload = id, score, payload


def _make_client(monkeypatch, captured=None):
    captured = captured if captured is not None else {}
    monkeypatch.setattr(app_module, "_settings", type("S", (), {
        "mistral_api_key": "k", "embed_model": "mistral-embed",
        "chat_model": "mistral-small-latest", "collection": "LizardDocu",
    })())
    monkeypatch.setattr(app_module, "_points", [{"id": "a"}, {"id": "b"}])
    monkeypatch.setattr(
        app_module, "embed_text",
        lambda text, api_key, model="mistral-embed": [0.25] * 1024,
    )

    class _FakeQdrant:
        def query_points(self, collection_name, query, limit, with_payload):
            captured["limit"] = limit
            captured["with_payload"] = with_payload
            class _R:
                points = [
                    _Hit("a", 0.91, {"content": "Chunk eins über Prüfungen.", "title": "Prüfung", "loc": "1-10"}),
                    _Hit("b", 0.80, {"content": "Chunk zwei über Wartung.", "title": "", "loc": "11-20"}),
                ]
            return _R()

    monkeypatch.setattr(app_module, "_qdrant", _FakeQdrant())
    return TestClient(app_module.app)


# ---------------- /api/pipeline/retrieve ----------------

def test_retrieve_returns_vector_and_chunks(monkeypatch):
    captured = {}
    c = _make_client(monkeypatch, captured)
    r = c.post("/api/pipeline/retrieve", json={"q": "Wer prüft Betriebsmittel?"})
    assert r.status_code == 200
    b = r.json()
    assert b["dims"] == 1024 and len(b["vector"]) == 1024
    assert b["collection"]
    assert b["total"] > 0
    assert [n["id"] for n in b["neighbors"]] == ["a", "b"]
    assert b["neighbors"][0]["content"] == "Chunk eins über Prüfungen."
    assert b["neighbors"][0]["score"] == 0.91
    assert b["neighbors"][0]["loc"] == "1-10"
    assert captured["with_payload"] is True
    assert captured["limit"] == 4  # talk-friendly default


def test_retrieve_empty_question_400(monkeypatch):
    c = _make_client(monkeypatch)
    assert c.post("/api/pipeline/retrieve", json={"q": "   "}).status_code == 400


# ---------------- /api/pipeline/answer ----------------

def test_answer_builds_prompt_with_chunks_and_question(monkeypatch):
    c = _make_client(monkeypatch)
    sent = {}

    def fake_chat(prompt, api_key, model):
        sent["prompt"], sent["model"] = prompt, model
        return "Die Prüfung übernimmt der Mitarbeiter."

    monkeypatch.setattr(app_module, "chat_complete", fake_chat)
    r = c.post("/api/pipeline/answer", json={
        "q": "Wer prüft Betriebsmittel?",
        "chunks": ["Chunk eins über Prüfungen.", "Chunk zwei über Wartung."],
    })
    assert r.status_code == 200
    b = r.json()
    assert b["answer"] == "Die Prüfung übernimmt der Mitarbeiter."
    assert b["model"] == sent["model"]
    assert b["prompt"] == sent["prompt"]  # UI shows exactly what was sent
    assert "Wer prüft Betriebsmittel?" in b["prompt"]
    assert "[1] Chunk eins über Prüfungen." in b["prompt"]
    assert "[2] Chunk zwei über Wartung." in b["prompt"]


def test_answer_requires_question_and_chunks(monkeypatch):
    c = _make_client(monkeypatch)
    assert c.post("/api/pipeline/answer", json={"q": "", "chunks": ["x"]}).status_code == 400
    assert c.post("/api/pipeline/answer", json={"q": "Frage?", "chunks": []}).status_code == 400


def test_answer_upstream_error_502(monkeypatch):
    c = _make_client(monkeypatch)

    def boom(prompt, api_key, model):
        raise RuntimeError("mistral down")

    monkeypatch.setattr(app_module, "chat_complete", boom)
    r = c.post("/api/pipeline/answer", json={"q": "Frage?", "chunks": ["x"]})
    assert r.status_code == 502
    assert "mistral down" in r.json()["error"]


# ---------------- /api/pipeline/graph ----------------

def test_graph_answer_uses_lightrag(monkeypatch):
    c = _make_client(monkeypatch)
    captured = {}

    class _FakeGraph:
        def query_answer(self, query, mode="mix"):
            captured["query"], captured["mode"] = query, mode
            return {"response": "GraphRAG sagt: der Mitarbeiter."}

    monkeypatch.setattr(app_module, "_graph_client", _FakeGraph())
    r = c.post("/api/pipeline/graph", json={"q": "Wer prüft Betriebsmittel?", "mode": "local"})
    assert r.status_code == 200
    b = r.json()
    assert b["answer"] == "GraphRAG sagt: der Mitarbeiter."
    assert b["mode"] == "local"
    assert captured["query"] == "Wer prüft Betriebsmittel?"
    assert captured["mode"] == "local"


def test_graph_answer_empty_400(monkeypatch):
    c = _make_client(monkeypatch)
    assert c.post("/api/pipeline/graph", json={"q": ""}).status_code == 400


# ---------------- /api/pipeline/graphdata ----------------

def _graphdata_fixture():
    """Shape as returned by LightRAG /query/data (verified live)."""
    return {
        "data": {
            "entities": [
                {"entity_name": f"E{i}", "entity_type": "person", "description": "…",
                 "source_id": "s", "file_path": "f.md", "created_at": "1"}
                for i in range(12)
            ],
            "relationships": [
                {"src_id": f"E{i}", "tgt_id": f"E{i + 1}", "description": "…",
                 "keywords": "k", "weight": "2.0", "source_id": "s", "file_path": "f.md"}
                for i in range(9)
            ],
            "chunks": [
                {"chunk_id": f"c{i}", "content": "X" * 500, "file_path": f"doc{i}.md",
                 "reference_id": str(i)}
                for i in range(20)
            ],
            "references": [],
        },
        "metadata": {
            "query_mode": "mix",
            "keywords": {"high_level": ["Prüfungsprozess"], "low_level": ["Mitarbeiter"]},
        },
    }


def test_graphdata_normalizes_and_caps(monkeypatch):
    c = _make_client(monkeypatch)
    captured = {}

    class _FakeGraph:
        def query_data(self, query, mode="mix", top_k=8):
            captured["query"], captured["mode"] = query, mode
            return _graphdata_fixture()

    monkeypatch.setattr(app_module, "_graph_client", _FakeGraph())
    r = c.post("/api/pipeline/graphdata", json={"q": "Wer prüft?", "mode": "mix"})
    assert r.status_code == 200
    b = r.json()
    assert captured["query"] == "Wer prüft?"
    # totals report the FULL retrieval, lists are capped for the UI
    assert b["totals"] == {"entities": 12, "relations": 9, "chunks": 20}
    assert len(b["entities"]) == 8
    assert b["entities"][0] == {"name": "E0", "type": "person"}
    assert len(b["relations"]) == 6
    assert b["relations"][0] == {"src": "E0", "tgt": "E1"}
    assert len(b["chunks"]) == 4
    assert b["chunks"][0]["file"] == "doc0.md"
    assert len(b["chunks"][0]["content"]) <= 240
    assert b["keywords"]["high_level"] == ["Prüfungsprozess"]
    assert b["mode"] == "mix"


def test_graphdata_empty_400_and_upstream_502(monkeypatch):
    c = _make_client(monkeypatch)
    assert c.post("/api/pipeline/graphdata", json={"q": " "}).status_code == 400

    class _Boom:
        def query_data(self, query, mode="mix", top_k=8):
            raise RuntimeError("lightrag down")

    monkeypatch.setattr(app_module, "_graph_client", _Boom())
    r = c.post("/api/pipeline/graphdata", json={"q": "Wer prüft?"})
    assert r.status_code == 502
    assert "lightrag down" in r.json()["error"]


# ---------------- view + helpers ----------------

def test_pipeline_view_serves_frontend(monkeypatch):
    c = _make_client(monkeypatch)
    r = c.get("/pipeline")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_shell_contains_pipeline_tab(monkeypatch):
    c = _make_client(monkeypatch)
    assert "/pipeline" in c.get("/").text


def test_chat_complete_calls_mistral_chat(monkeypatch):
    captured = {}

    class _Chat:
        def complete(self, model, messages):
            captured["model"], captured["messages"] = model, messages
            msg = type("M", (), {"content": "Antwort."})()
            choice = type("C", (), {"message": msg})()
            return type("R", (), {"choices": [choice]})()

    class _Client:
        def __init__(self, api_key):
            assert api_key == "k"
            self.chat = _Chat()

    monkeypatch.setattr(mistral_mod, "Mistral", _Client)
    out = chat_complete("Hallo?", api_key="k", model="mistral-small-latest")
    assert out == "Antwort."
    assert captured["model"] == "mistral-small-latest"
    assert captured["messages"] == [{"role": "user", "content": "Hallo?"}]


def test_lightrag_query_answer_posts_to_query(monkeypatch):
    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "ok"}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers)
        return _Resp()

    monkeypatch.setattr(client_mod.httpx, "post", fake_post)
    c = LightRAGClient("https://x.y", "secret")
    out = c.query_answer("hallo", mode="mix")
    assert out == {"response": "ok"}
    assert captured["url"] == "https://x.y/query"
    assert captured["json"] == {"query": "hallo", "mode": "mix"}
    assert captured["headers"] == {"X-API-Key": "secret"}
