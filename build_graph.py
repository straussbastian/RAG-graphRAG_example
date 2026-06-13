"""One-time build: pull the LightRAG knowledge graph, transform it, and write
data/graph.json + a self-contained offline graph.html."""
from __future__ import annotations

import json
from pathlib import Path

from graph_viz.client import LightRAGClient
from graph_viz.config import load_graph_settings
from graph_viz.transform import build_graph_json

DATA_DIR = Path("data")
HTML_OUT = Path("graph.html")
VENDOR_JS = Path("graph_web/3d-force-graph.min.js")

PALETTE = {
    "concept": "#22d3ee", "artifact": "#fb7185", "method": "#fde047", "data": "#a78bfa",
    "content": "#34d399", "organization": "#38bdf8", "person": "#fb923c", "location": "#a3e635",
    "event": "#f472b6", "other": "#94a3b8", "UNKNOWN": "#64748b",
}

_OFFLINE_TEMPLATE = """<!doctype html><html lang="de"><head><meta charset="utf-8"/>
<title>LightRAG &middot; Wissensgraph</title>
<style>:root{--cyan:#29e3ff}*{box-sizing:border-box}
body{margin:0;overflow:hidden;color:#eaf0ff;font-family:system-ui,-apple-system,sans-serif;
 background:radial-gradient(1100px 760px at 78% -10%,rgba(41,227,255,.10),transparent 60%),radial-gradient(900px 640px at 14% 112%,rgba(124,92,255,.12),transparent 60%),linear-gradient(160deg,#080b14,#04050a)}
#graph{position:fixed;inset:0}
.vig{position:fixed;inset:0;pointer-events:none;box-shadow:inset 0 0 380px 60px rgba(0,0,0,.85)}
#hdr{position:fixed;top:16px;left:20px;z-index:10}
#hdr h1{margin:0;font-size:15px;letter-spacing:.16em;text-transform:uppercase;
 background:linear-gradient(90deg,#fff,var(--cyan));-webkit-background-clip:text;background-clip:text;color:transparent}
#hdr p{margin:4px 0 0;font-size:11px;letter-spacing:.06em;color:#8b97b4}</style>
</head><body>
<div id="graph"></div><div class="vig"></div>
<div id="hdr"><h1>LightRAG &middot; Wissensgraph</h1><p>Farbe = Entity-Typ &middot; Gr&ouml;&szlig;e = Verkn&uuml;pfungen &middot; Hover f&uuml;r Details</p></div>
<script>/*LIB*/</script>
<script>
const DATA = /*DATA*/; const PALETTE = /*PALETTE*/;
const THREE = window.THREE;
const colorOf = t => PALETTE[t] || '#9fb0d0';
function glowTex(){const s=128,c=document.createElement('canvas');c.width=c.height=s;
 const x=c.getContext('2d'),g=x.createRadialGradient(s/2,s/2,0,s/2,s/2,s/2);
 g.addColorStop(0,'rgba(255,255,255,1)');g.addColorStop(.18,'rgba(255,255,255,.85)');
 g.addColorStop(.42,'rgba(255,255,255,.30)');g.addColorStop(1,'rgba(255,255,255,0)');
 x.fillStyle=g;x.fillRect(0,0,s,s);return new THREE.CanvasTexture(c);}
const G = ForceGraph3D()(document.getElementById('graph'))
  .backgroundColor('#04050a').graphData(DATA).nodeVal(n => 1 + n.degree)
  .nodeLabel(n => `<div style="max-width:300px;padding:8px 11px;background:rgba(6,10,18,.92);border:1px solid ${colorOf(n.type)};border-radius:10px;color:#eaf0ff;font-family:system-ui,sans-serif"><b>${n.id}</b> <span style="color:${colorOf(n.type)}">&middot; ${n.type}</span><br><span style="color:#9fb0d0;font-size:12px">${(n.description||'').slice(0,180)}</span></div>`)
  .linkColor(() => 'rgba(125,165,225,0.12)').linkWidth(0.5).linkCurvature(0.22);
if (THREE && THREE.Sprite) { const TEX = glowTex();
  G.nodeThreeObjectExtend(false).nodeThreeObject(n => {
    const col = new THREE.Color(colorOf(n.type)), base = 2 + Math.cbrt(n.degree || 1) * 1.7, grp = new THREE.Group();
    grp.add(new THREE.Mesh(new THREE.SphereGeometry(base, 18, 18), new THREE.MeshBasicMaterial({ color: col })));
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: TEX, color: col, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: .85 }));
    sp.scale.setScalar(base * 6); grp.add(sp); return grp; });
} else { G.nodeColor(n => colorOf(n.type)); }
const ctr = G.controls(); if (ctr) { ctr.autoRotate = true; ctr.autoRotateSpeed = 0.42; }
G.onEngineStop(() => G.zoomToFit(900, 70));
</script></body></html>"""


def render_offline_html(doc: dict) -> str:
    """Inline the vendored library + graph data into a self-contained HTML page."""
    lib = VENDOR_JS.read_text(encoding="utf-8")
    return (
        _OFFLINE_TEMPLATE
        .replace("/*LIB*/", lib)
        .replace("/*DATA*/", json.dumps(doc, ensure_ascii=False))
        .replace("/*PALETTE*/", json.dumps(PALETTE, ensure_ascii=False))
    )


def main() -> None:
    settings = load_graph_settings()
    client = LightRAGClient(settings.lightrag_url, settings.lightrag_api_key)
    print("Fetching graph from LightRAG…")
    raw = client.get_graph(label="*", max_nodes=2000, max_depth=6)
    doc = build_graph_json(raw)
    if not doc["nodes"]:
        raise SystemExit("LightRAG graph is empty.")

    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "graph.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    HTML_OUT.write_text(render_offline_html(doc), encoding="utf-8")
    print(f"Wrote {DATA_DIR/'graph.json'} ({len(doc['nodes'])} nodes, "
          f"{len(doc['links'])} links) and {HTML_OUT}")


if __name__ == "__main__":
    main()
