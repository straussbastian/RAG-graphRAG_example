"""FastAPI demo: classic RAG (Qdrant point cloud) vs GraphRAG (LightRAG graph)
behind one shell with an iframe toggle.

- Qdrant view   : /qdrant  + /static       + /api/points + /api/query
- GraphRAG view : /graph   + /graph-static + /api/graph  + /api/graph/query
- Shell         : /        (RAG <-> GraphRAG toggle)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient

from qdrant_viz.config import load_settings
from qdrant_viz.embedding import cosine_similarity
from qdrant_viz.mistral import chat_complete, embed_text, embed_texts
from qdrant_viz.placement import weighted_centroid
from qdrant_viz.seed import seed_qdrant

from graph_viz.client import LightRAGClient
from graph_viz.config import load_graph_settings
from graph_viz.transform import build_highlight

# Override the points file (e.g. tests point at a small fixture) via env.
POINTS_PATH = Path(os.environ.get("QDRANT_VIZ_POINTS", "data/points.json"))
WEB_DIR = Path("web")
GRAPH_PATH = Path("data/graph.json")
GRAPH_WEB = Path("graph_web")
EMBED_WEB = Path("embed_web")
CHUNK_WEB = Path("chunk_web")
PIPELINE_WEB = Path("pipeline_web")

app = FastAPI(title="RAG vs GraphRAG Demo")


@app.middleware("http")
async def _no_store(request, call_next):
    # Live-demo: never let the browser cache the shell, frontends, or assets so
    # edits show up on a plain reload (iframes otherwise serve stale documents).
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


# --- Qdrant state ---
_settings = None
_qdrant = None
_points: list[dict] = []
_umap_by_id: dict[str, np.ndarray] = {}

# --- GraphRAG state ---
_graph_settings = None
_graph_client = None
_graph_doc: dict = {}
_graph_node_ids: set = set()


def _load_state(points_path: Path = POINTS_PATH) -> None:
    global _settings, _qdrant, _points, _umap_by_id
    _settings = load_settings()
    _qdrant = QdrantClient(
        url=_settings.qdrant_url, api_key=_settings.qdrant_api_key,
        timeout=60, check_compatibility=False, prefer_grpc=False,
    )
    if _settings.local:
        # Bundled Qdrant starts empty and may not be ready the instant the app
        # boots — retry briefly, then populate it once from the committed snapshot.
        last_exc: Exception | None = None
        for attempt in range(30):
            try:
                n = seed_qdrant(_qdrant, _settings.collection)
                print(f"[startup] local Qdrant ready: {n} points in {_settings.collection!r}")
                break
            except Exception as exc:  # qdrant not up yet, or transient
                last_exc = exc
                time.sleep(1)
        else:
            raise RuntimeError(f"local Qdrant never became ready: {last_exc}")
    doc = json.loads(points_path.read_text(encoding="utf-8"))
    _points = doc["points"]
    _umap_by_id = {p["id"]: np.array(p["umap"], dtype=float) for p in _points}


def _load_graph_state(graph_path: Path = GRAPH_PATH) -> None:
    global _graph_settings, _graph_client, _graph_doc, _graph_node_ids
    _graph_settings = load_graph_settings()
    _graph_client = LightRAGClient(_graph_settings.lightrag_url, _graph_settings.lightrag_api_key)
    _graph_doc = json.loads(graph_path.read_text(encoding="utf-8"))
    _graph_node_ids = {n["id"] for n in _graph_doc["nodes"]}


try:
    _load_state()
except Exception as exc:  # surfaced at /api/query, not at import
    print(f"[startup] qdrant state not loaded yet: {exc}")

try:
    _load_graph_state()
except Exception as exc:  # surfaced at /api/graph/query, not at import
    print(f"[startup] graph state not loaded yet: {exc}")


_SHELL = """<!doctype html><html lang="de"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RAG &times; GraphRAG</title>
<style>
@font-face{font-family:'Chakra Petch';src:url('/graph-static/fonts/ChakraPetch-700.woff2') format('woff2');font-weight:700;font-display:swap}
@font-face{font-family:'Sora';src:url('/graph-static/fonts/Sora-400.woff2') format('woff2');font-weight:400;font-display:swap}
:root{--cyan:#29e3ff;--violet:#a78bfa;--mint:#2af2a8}
*{box-sizing:border-box}
body{margin:0;height:100vh;display:flex;flex-direction:column;color:#eaf0ff;
 font-family:'Sora',system-ui,sans-serif;
 background:radial-gradient(900px 500px at 50% -10%,rgba(41,227,255,.10),transparent 60%),linear-gradient(160deg,#080b14,#04050a)}
.bar{display:flex;align-items:center;justify-content:center;padding:11px;position:relative;
 border-bottom:1px solid rgba(140,175,255,.12);background:rgba(4,5,10,.55);backdrop-filter:blur(10px)}
.bar .title{position:absolute;left:20px;font-family:'Chakra Petch',monospace;font-size:11px;
 letter-spacing:.34em;text-transform:uppercase;color:#8b97b4}
.seg{display:flex;background:rgba(12,17,30,.7);border:1px solid rgba(140,175,255,.14);border-radius:999px;padding:4px}
.seg button{font-family:'Chakra Petch',monospace;font-weight:700;font-size:12.5px;letter-spacing:.12em;
 text-transform:uppercase;cursor:pointer;border:0;padding:9px 24px;border-radius:999px;background:transparent;
 color:#8b97b4;transition:color .2s,background .3s,box-shadow .3s}
.seg button:hover{color:#eaf0ff}
.seg button.active{color:#04121a}
.seg button#b-rag.active{background:linear-gradient(135deg,var(--violet),var(--cyan));box-shadow:0 0 22px -6px var(--violet)}
.seg button#b-graph.active{background:linear-gradient(135deg,var(--cyan),var(--mint));box-shadow:0 0 22px -6px var(--cyan)}
.seg button#b-embed.active{background:linear-gradient(135deg,#fbbf24,#fb7185);box-shadow:0 0 22px -6px #fbbf24}
.seg button#b-chunk.active{background:linear-gradient(135deg,#a3e635,#34d399);box-shadow:0 0 22px -6px #a3e635}
.seg button#b-pipe.active{background:linear-gradient(135deg,#e879f9,#a78bfa);box-shadow:0 0 22px -6px #e879f9}
iframe{flex:1;border:0;width:100%;background:#04050a}
</style></head><body>
<div class="bar">
  <div class="title">RAG &times; GraphRAG</div>
  <div class="seg">
    <button id="b-chunk" class="active" onclick="show('/chunk','b-chunk')">Chunking</button>
    <button id="b-embed" onclick="show('/embed','b-embed')">Embedding</button>
    <button id="b-rag" onclick="show('/qdrant','b-rag')">Klassisches RAG</button>
    <button id="b-graph" onclick="show('/graph','b-graph')">GraphRAG</button>
    <button id="b-pipe" onclick="show('/pipeline','b-pipe')">Pipeline</button>
  </div>
</div>
<iframe id="view" src="/chunk" title="Ansicht"></iframe>
<script>
function show(u,id){document.getElementById('view').src=u;
  document.querySelectorAll('.seg button').forEach(function(b){b.classList.remove('active');});
  document.getElementById(id).classList.add('active');}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def shell() -> HTMLResponse:
    return HTMLResponse(_SHELL)


# ---------------- Qdrant (classic RAG) ----------------
@app.get("/qdrant")
def qdrant_view() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/points")
def points() -> FileResponse:
    return FileResponse(POINTS_PATH, media_type="application/json")


@app.get("/api/query")
def query(q: str = Query(..., min_length=1), k: int = 8) -> JSONResponse:
    try:
        vec = embed_text(q, api_key=_settings.mistral_api_key, model=_settings.embed_model)
        res = _qdrant.query_points(
            collection_name=_settings.collection, query=vec, limit=k, with_payload=False
        )
        hits = res.points
        coords = np.array([_umap_by_id[str(h.id)] for h in hits], dtype=float)
        scores = np.array([h.score for h in hits], dtype=float)
        placed = weighted_centroid(coords, scores)
        return JSONResponse({
            "query": q,
            "point": {"umap": [round(float(v), 4) for v in placed]},
            "neighbors": [{"id": str(h.id), "score": float(h.score)} for h in hits],
        })
    except Exception as exc:
        return JSONResponse(status_code=502, content={"error": f"{type(exc).__name__}: {exc}"})


# ---------------- LightRAG (GraphRAG) ----------------
@app.get("/graph")
def graph_view() -> FileResponse:
    return FileResponse(GRAPH_WEB / "index.html")


@app.get("/api/graph")
def api_graph() -> FileResponse:
    return FileResponse(GRAPH_PATH, media_type="application/json")


@app.post("/api/graph/query")
def api_graph_query(payload: dict = Body(...)) -> JSONResponse:
    q = (payload.get("query") or "").strip()
    mode = payload.get("mode", "mix")
    if not q:
        return JSONResponse(status_code=400, content={"error": "empty query"})
    try:
        resp = _graph_client.query_data(q, mode=mode)
        return JSONResponse(build_highlight(resp, _graph_node_ids))
    except Exception as exc:
        return JSONResponse(status_code=502, content={"error": f"{type(exc).__name__}: {exc}"})


# ---------------- Embedding live demo ----------------
@app.get("/embed")
def embed_view() -> FileResponse:
    return FileResponse(EMBED_WEB / "index.html")


@app.post("/api/embed")
def api_embed(payload: dict = Body(...)) -> JSONResponse:
    a = (payload.get("a") or "").strip()
    b = (payload.get("b") or "").strip()
    if not a or not b:
        return JSONResponse(status_code=400, content={"error": "empty input"})
    try:
        va, vb = embed_texts([a, b], api_key=_settings.mistral_api_key, model=_settings.embed_model)
        return JSONResponse({
            "a": {"vector": [round(float(x), 6) for x in va]},
            "b": {"vector": [round(float(x), 6) for x in vb]},
            "cosine": round(cosine_similarity(va, vb), 4),
            "dims": len(va),
        })
    except Exception as exc:
        return JSONResponse(status_code=502, content={"error": f"{type(exc).__name__}: {exc}"})


@app.post("/api/embed/matrix")
def api_embed_matrix(payload: dict = Body(...)) -> JSONResponse:
    """All-pairs cosine similarity for a handful of terms — the 'everything is
    connected' game. Returns the matrix plus the lowest off-diagonal pair."""
    texts = [t.strip() for t in (payload.get("texts") or []) if isinstance(t, str) and t.strip()][:8]
    if len(texts) < 2:
        return JSONResponse(status_code=400, content={"error": "need at least 2 texts"})
    try:
        vecs = embed_texts(texts, api_key=_settings.mistral_api_key, model=_settings.embed_model)
        n = len(texts)
        matrix = [[round(cosine_similarity(vecs[i], vecs[j]), 4) for j in range(n)] for i in range(n)]
        lo = {"i": 0, "j": 1, "value": 2.0}
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i][j] < lo["value"]:
                    lo = {"i": i, "j": j, "value": matrix[i][j]}
        return JSONResponse({"labels": texts, "matrix": matrix, "min": lo})
    except Exception as exc:
        return JSONResponse(status_code=502, content={"error": f"{type(exc).__name__}: {exc}"})


@app.get("/chunk")
def chunk_view() -> FileResponse:
    return FileResponse(CHUNK_WEB / "index.html")


# ---------------- Pipeline (classic RAG, step by step) ----------------
@app.get("/pipeline")
def pipeline_view() -> FileResponse:
    return FileResponse(PIPELINE_WEB / "index.html")


@app.post("/api/pipeline/retrieve")
def api_pipeline_retrieve(payload: dict = Body(...)) -> JSONResponse:
    """Embed the question and search Qdrant WITH payload: the pipeline view
    needs both the raw vector (fingerprint) and the real chunk texts."""
    q = (payload.get("q") or "").strip()
    k = int(payload.get("k") or 4)
    if not q:
        return JSONResponse(status_code=400, content={"error": "empty question"})
    try:
        vec = embed_text(q, api_key=_settings.mistral_api_key, model=_settings.embed_model)
        res = _qdrant.query_points(
            collection_name=_settings.collection, query=vec, limit=k, with_payload=True
        )
        return JSONResponse({
            "query": q,
            "vector": [round(float(x), 6) for x in vec],
            "dims": len(vec),
            "collection": _settings.collection,
            "total": len(_points),
            "neighbors": [{
                "id": str(h.id),
                "score": float(h.score),
                "content": (h.payload or {}).get("content", ""),
                "title": (h.payload or {}).get("title", ""),
                "loc": (h.payload or {}).get("loc", ""),
            } for h in res.points],
        })
    except Exception as exc:
        return JSONResponse(status_code=502, content={"error": f"{type(exc).__name__}: {exc}"})


def build_rag_prompt(question: str, chunks: list[str]) -> str:
    """The didactic centerpiece: retrieved chunks are simply pasted into the
    prompt. Returned to the UI verbatim so it shows exactly what was sent."""
    ctx = "\n\n".join(f"[{i + 1}] {c.strip()}" for i, c in enumerate(chunks))
    return (
        "Du bist ein Assistent für Fragen zu einer internen Unternehmens-Dokumentation.\n"
        "Beantworte die Frage ausschließlich anhand des folgenden Kontexts.\n"
        "Steht die Antwort nicht im Kontext, sage das offen.\n\n"
        f"### Kontext\n{ctx}\n\n"
        f"### Frage\n{question}"
    )


@app.post("/api/pipeline/answer")
def api_pipeline_answer(payload: dict = Body(...)) -> JSONResponse:
    q = (payload.get("q") or "").strip()
    chunks = [c.strip() for c in (payload.get("chunks") or []) if isinstance(c, str) and c.strip()]
    if not q or not chunks:
        return JSONResponse(status_code=400, content={"error": "need question and chunks"})
    prompt = build_rag_prompt(q, chunks)
    try:
        answer = chat_complete(prompt, api_key=_settings.mistral_api_key, model=_settings.chat_model)
        return JSONResponse({"prompt": prompt, "answer": answer, "model": _settings.chat_model})
    except Exception as exc:
        return JSONResponse(status_code=502, content={"error": f"{type(exc).__name__}: {exc}"})


@app.post("/api/pipeline/graphdata")
def api_pipeline_graphdata(payload: dict = Body(...)) -> JSONResponse:
    """What GraphRAG retrieves for the question: entities, relations, chunks.
    Totals report the full retrieval; the lists are capped for the UI."""
    q = (payload.get("q") or "").strip()
    mode = payload.get("mode", "mix")
    if not q:
        return JSONResponse(status_code=400, content={"error": "empty question"})
    try:
        resp = _graph_client.query_data(q, mode=mode)
        data = resp.get("data") or {}
        meta = resp.get("metadata") or {}
        ents = data.get("entities") or []
        rels = data.get("relationships") or []
        chunks = data.get("chunks") or []
        return JSONResponse({
            "mode": meta.get("query_mode", mode),
            "keywords": meta.get("keywords") or {"high_level": [], "low_level": []},
            "totals": {"entities": len(ents), "relations": len(rels), "chunks": len(chunks)},
            "entities": [{"name": e.get("entity_name", ""), "type": e.get("entity_type", "")}
                         for e in ents[:8]],
            "relations": [{"src": r.get("src_id", ""), "tgt": r.get("tgt_id", "")}
                          for r in rels[:6]],
            "chunks": [{"content": (c.get("content") or "")[:240], "file": c.get("file_path", "")}
                       for c in chunks[:4]],
        })
    except Exception as exc:
        return JSONResponse(status_code=502, content={"error": f"{type(exc).__name__}: {exc}"})


@app.post("/api/pipeline/graph")
def api_pipeline_graph(payload: dict = Body(...)) -> JSONResponse:
    """Same question, answered by LightRAG — the live GraphRAG comparison."""
    q = (payload.get("q") or "").strip()
    mode = payload.get("mode", "mix")
    if not q:
        return JSONResponse(status_code=400, content={"error": "empty question"})
    try:
        resp = _graph_client.query_answer(q, mode=mode)
        return JSONResponse({"answer": resp.get("response", ""), "mode": mode})
    except Exception as exc:
        return JSONResponse(status_code=502, content={"error": f"{type(exc).__name__}: {exc}"})


@app.post("/api/pipeline/graphprompt")
def api_pipeline_graphprompt(payload: dict = Body(...)) -> JSONResponse:
    """The REAL prompt LightRAG assembles for the question (only_need_prompt) —
    the GraphRAG counterpart to the classic RAG prompt shown in step 4."""
    q = (payload.get("q") or "").strip()
    mode = payload.get("mode", "mix")
    if not q:
        return JSONResponse(status_code=400, content={"error": "empty query"})
    try:
        resp = _graph_client.query_prompt(q, mode=mode)
        return JSONResponse({"prompt": resp.get("response", ""), "mode": mode})
    except Exception as exc:
        return JSONResponse(status_code=502, content={"error": f"{type(exc).__name__}: {exc}"})


# Static assets
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
app.mount("/graph-static", StaticFiles(directory=GRAPH_WEB), name="graph-static")
app.mount("/embed-static", StaticFiles(directory=EMBED_WEB), name="embed-static")
app.mount("/chunk-static", StaticFiles(directory=CHUNK_WEB), name="chunk-static")
app.mount("/pipeline-static", StaticFiles(directory=PIPELINE_WEB), name="pipeline-static")
