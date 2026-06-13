"""One-time build: pull vectors from Qdrant, reduce + cluster, and write
data/points.json, data/reducer.pkl, and a self-contained embeddings.html."""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from qdrant_client import QdrantClient

from qdrant_viz.config import load_settings
from qdrant_viz.data import (
    build_points_json,
    cluster_labels,
    fetch_points,
    reduce_3d,
)

DATA_DIR = Path("data")
HTML_OUT = Path("embeddings.html")
PALETTE = ["#ff6b6b", "#4ecdc4", "#ffe66d", "#a78bfa", "#06d6a0",
           "#f78c6b", "#5bc0eb", "#fb5607", "#8ac926", "#ff70a6"]


def _loc_str(payload: dict) -> str:
    loc = (payload.get("metadata") or {}).get("loc") or {}
    lines = loc.get("lines") or {}
    if "from" in lines and "to" in lines:
        return f"{lines['from']}-{lines['to']}"
    return ""


_TITLE_RE = re.compile(r'^\s*title:\s*["\']?(.+?)["\']?\s*$', re.MULTILINE)


def _extract_title(content: str) -> str | None:
    """Return the markdown frontmatter `title:` if the chunk opens a section.

    The payload has no filename; the document is a concatenation of markdown
    sections, each starting with a `---\ntitle: ...\n---` frontmatter block.
    Only look inside a leading frontmatter fence to avoid matching body text.
    """
    s = content.lstrip()
    if not s.startswith("---"):
        return None
    end = s.find("---", 3)
    block = s[3:end] if end != -1 else s[3:800]
    m = _TITLE_RE.search(block)
    return m.group(1).strip() if m else None


def _section_titles(contents: list[str]) -> list[str]:
    """Title for each chunk that opens with its own frontmatter, else "".

    The chunks' line numbers reset per source file, so there is no reliable
    global order to carry a title forward — we only trust a chunk's OWN
    frontmatter title and leave the rest blank rather than guess wrongly."""
    return [_extract_title(c) or "" for c in contents]


def _hover(content: str, loc: str, title: str = "") -> str:
    snippet = content.replace("\n", " ")[:160]
    suffix = "…" if len(content) > 160 else ""
    head = f"<b>{title}</b><br>" if title else ""
    tail = f"<br><i>Zeilen {loc}</i>" if loc else ""
    return f"{head}{snippet}{suffix}{tail}"


def main() -> None:
    settings = load_settings()
    client = QdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key,
        timeout=60, check_compatibility=False, prefer_grpc=False,
    )
    records = fetch_points(client, settings.collection)
    if not records:
        raise SystemExit(f"Collection {settings.collection!r} is empty.")

    ids = [r.id for r in records]
    vectors = np.array([r.vector for r in records], dtype=float)
    if vectors.shape[1] != 1024:
        raise SystemExit(f"Expected 1024-dim vectors, got {vectors.shape[1]}.")
    contents = [(r.payload or {}).get("content", "") for r in records]
    locs = [_loc_str(r.payload or {}) for r in records]
    titles = _section_titles(contents)
    n_titled = sum(1 for t in titles if t)

    print(f"Fetched {len(ids)} points. Reducing…")
    red = reduce_3d(vectors)
    labels, k = cluster_labels(red["umap"])
    print(f"Clusters: k={k} · {n_titled}/{len(ids)} chunks mapped to a section title")

    DATA_DIR.mkdir(exist_ok=True)
    doc = build_points_json(ids, red["umap"], red["pca"], labels, contents, locs, titles,
                            collection=settings.collection, model=settings.embed_model)
    (DATA_DIR / "points.json").write_text(json.dumps(doc), encoding="utf-8")
    with open(DATA_DIR / "reducer.pkl", "wb") as fh:
        pickle.dump(red["umap_model"], fh)

    _write_html(doc, k)
    print(f"Wrote {DATA_DIR/'points.json'}, {DATA_DIR/'reducer.pkl'}, {HTML_OUT}")


def _write_html(doc: dict, k: int) -> None:
    pts = doc["points"]
    colors = [PALETTE[p["cluster"] % len(PALETTE)] for p in pts]
    text = [_hover(p["content"], p["loc"], p.get("title", "")) for p in pts]

    def xyz(key):
        arr = np.array([p[key] for p in pts])
        return arr[:, 0], arr[:, 1], arr[:, 2]

    ux, uy, uz = xyz("umap")
    px, py, pz = xyz("pca")

    fig = go.Figure(go.Scatter3d(
        x=ux, y=uy, z=uz, mode="markers",
        marker=dict(size=4, color=colors, opacity=0.85),
        text=text, hoverinfo="text",
    ))
    fig.update_layout(
        template="plotly_dark",
        title=f"LizardDocu — {len(pts)} Chunks · {k} semantische Cluster (UMAP)",
        scene=dict(xaxis_title="", yaxis_title="", zaxis_title=""),
        margin=dict(l=0, r=0, t=40, b=0),
        updatemenus=[dict(
            type="buttons", direction="right", x=0.0, y=1.08, showactive=True,
            buttons=[
                dict(label="UMAP", method="restyle",
                     args=[{"x": [ux], "y": [uy], "z": [uz]}]),
                dict(label="PCA", method="restyle",
                     args=[{"x": [px], "y": [py], "z": [pz]}]),
            ],
        )],
    )
    fig.write_html(HTML_OUT, include_plotlyjs="inline", full_html=True)


if __name__ == "__main__":
    main()
