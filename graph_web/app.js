/* ── GraphRAG · Luminous Knowledge Observatory ──────────────────────────────
   A living constellation of the LightRAG knowledge graph. Nodes are glowing
   orbs (core sphere + additive halo) sized by degree and coloured by entity
   type; relationships are curved strands. A query lights up the traversed
   subgraph, surges particles along its edges, and flies the camera in. */

const THREE = window.THREE;
const USE_GLOW = !!(THREE && THREE.Sprite && THREE.CanvasTexture);

const PALETTE = {
  concept: "#22d3ee", artifact: "#fb7185", method: "#fde047", data: "#a78bfa",
  content: "#34d399", organization: "#38bdf8", person: "#fb923c", location: "#a3e635",
  event: "#f472b6", other: "#94a3b8", UNKNOWN: "#64748b",
};
const colorOf = (t) => PALETTE[t] || "#9fb0d0";

const statusEl = document.getElementById("status");
const kwEl = document.getElementById("keywords");
const legendEl = document.getElementById("legend");
const hintEl = document.getElementById("hint");

let GRAPH = null, DATA = null;
let nodeById = {};
let hiddenTypes = new Set();
let hlNodes = new Set();
let hlEdges = new Set();
let flying = false, userTookOver = false, fitted = false;

const glowObjects = new Map();        // id -> { group, core, glow, base }
const idOf = (e) => (typeof e === "object" ? e.id : e);
const edgeKey = (s, t) => s + "\x01" + t;
const isHotLink = (l) => hlEdges.has(edgeKey(idOf(l.source), idOf(l.target)));

/* ── node glow object ─────────────────────────────────────────────────────── */
let GLOW_TEX = null;
function glowTexture() {
  if (GLOW_TEX) return GLOW_TEX;
  const s = 128, cv = document.createElement("canvas");
  cv.width = cv.height = s;
  const ctx = cv.getContext("2d");
  const g = ctx.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  g.addColorStop(0.0, "rgba(255,255,255,1)");
  g.addColorStop(0.18, "rgba(255,255,255,0.85)");
  g.addColorStop(0.42, "rgba(255,255,255,0.30)");
  g.addColorStop(1.0, "rgba(255,255,255,0)");
  ctx.fillStyle = g; ctx.fillRect(0, 0, s, s);
  GLOW_TEX = new THREE.CanvasTexture(cv);
  return GLOW_TEX;
}

function makeNodeObject(n) {
  const color = new THREE.Color(colorOf(n.type));
  const base = 2 + Math.cbrt(n.degree || 1) * 1.7;
  const group = new THREE.Group();

  const core = new THREE.Mesh(
    new THREE.SphereGeometry(base, 18, 18),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 1 })
  );
  group.add(core);

  const glow = new THREE.Sprite(new THREE.SpriteMaterial({
    map: glowTexture(), color, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, opacity: 0.85,
  }));
  glow.scale.setScalar(base * 6);
  group.add(glow);

  glowObjects.set(n.id, { group, core, glow, base });
  return group;
}

function nodeStateOf(n) {
  if (hiddenTypes.has(n.type)) return "hidden";
  if (hlNodes.size === 0) return "normal";
  return hlNodes.has(n.id) ? "hot" : "dim";
}

function styleGlowNodes() {
  glowObjects.forEach((o, id) => {
    const n = nodeById[id];
    const st = nodeStateOf(n);
    o.group.visible = st !== "hidden";
    if (st === "hidden") return;
    if (st === "hot") { o.core.material.opacity = 1; o.glow.material.opacity = 1; o.glow.scale.setScalar(o.base * 9); }
    else if (st === "dim") { o.core.material.opacity = 0.22; o.glow.material.opacity = 0.05; o.glow.scale.setScalar(o.base * 4); }
    else { o.core.material.opacity = 1; o.glow.material.opacity = 0.85; o.glow.scale.setScalar(o.base * 6); }
  });
}

/* fallback colouring when THREE isn't exposed */
function fallbackNodeColor(n) {
  const st = nodeStateOf(n);
  if (st === "hidden") return "rgba(0,0,0,0)";
  if (st === "dim") return "rgba(120,140,170,0.12)";
  return colorOf(n.type);
}

/* ── links ────────────────────────────────────────────────────────────────── */
function linkColor(l) {
  if (hlEdges.size === 0) return "rgba(150,190,245,0.42)";
  return isHotLink(l) ? "rgba(175,230,255,0.6)" : "rgba(120,150,210,0.05)";
}
const linkWidth = (l) => (isHotLink(l) ? 1.2 : 1.1);
const linkParticles = (l) => (isHotLink(l) ? 4 : 0);
function linkVisible(l) {
  const s = nodeById[idOf(l.source)], t = nodeById[idOf(l.target)];
  if (!s || !t) return true;
  return !hiddenTypes.has(s.type) && !hiddenTypes.has(t.type);
}

function refresh() {
  if (USE_GLOW) styleGlowNodes(); else GRAPH.nodeColor(fallbackNodeColor);
  GRAPH.linkColor(linkColor).linkWidth(linkWidth)
       .linkDirectionalParticles(linkParticles).linkVisibility(linkVisible);
}

/* ── HUD: legend, keywords, status ────────────────────────────────────────── */
function buildLegend() {
  legendEl.innerHTML = "";
  DATA.types.forEach((t) => {
    const b = document.createElement("button");
    b.className = "chip"; b.type = "button";
    b.innerHTML = `<span class="dot" style="background:${colorOf(t)};color:${colorOf(t)}"></span>${t}`;
    b.onclick = () => {
      if (hiddenTypes.has(t)) hiddenTypes.delete(t); else hiddenTypes.add(t);
      b.classList.toggle("off");
      refresh();
    };
    legendEl.appendChild(b);
  });
}

function showKeywords(k) {
  kwEl.innerHTML = "";
  (k.high_level || []).forEach((w) => {
    const c = document.createElement("span"); c.className = "kw hl"; c.textContent = w; kwEl.appendChild(c);
  });
  (k.low_level || []).forEach((w) => {
    const c = document.createElement("span"); c.className = "kw ll"; c.textContent = w; kwEl.appendChild(c);
  });
}

function setStatus(b) {
  const n = (v) => `<span class="num">${v}</span>`;
  const base = `${n(b.highlight_nodes.length)} Entities · ${n(b.highlight_edges.length)} Relationen · ${n(b.chunks)} Chunks · <b>${b.mode}</b>`;
  const naive = b.mode === "naive" || (b.highlight_nodes.length === 0 && b.highlight_edges.length === 0);
  statusEl.classList.toggle("naive", naive);
  if (naive) {
    statusEl.innerHTML = `${base}<br>Graph ungenutzt — reine Vektorsuche über Text-Chunks (= klassisches RAG)`;
  } else {
    const miss = b.missing && b.missing.length ? ` · ${n(b.missing.length)} nicht im Graph` : "";
    statusEl.innerHTML = base + miss;
  }
}

/* ── highlight + camera fly ───────────────────────────────────────────────── */
function applyHighlight(b) {
  hlNodes = new Set(b.highlight_nodes);
  hlEdges = new Set(b.highlight_edges.map(([s, t]) => edgeKey(s, t)));
  refresh();
  showKeywords(b.keywords || {});
  setStatus(b);
  flyToHighlighted();
}

function flyToHighlighted() {
  const ns = DATA.nodes.filter((n) => hlNodes.has(n.id) && typeof n.x === "number");
  if (!ns.length) return;
  const c = { x: 0, y: 0, z: 0 };
  ns.forEach((n) => { c.x += n.x; c.y += n.y; c.z += n.z; });
  c.x /= ns.length; c.y /= ns.length; c.z /= ns.length;
  let r = 0;
  ns.forEach((n) => { r = Math.max(r, Math.hypot(n.x - c.x, n.y - c.y, n.z - c.z)); });
  const dist = r * 2.4 + 110;
  const cam = GRAPH.cameraPosition();
  const len = Math.hypot(cam.x, cam.y, cam.z) || 1;
  flying = true;
  const ctr = GRAPH.controls(); if (ctr) ctr.autoRotate = false;
  setTimeout(() => GRAPH.cameraPosition(
    { x: c.x + (cam.x / len) * dist, y: c.y + (cam.y / len) * dist, z: c.z + (cam.z / len) * dist }, c, 1100), 250);
  setTimeout(() => { flying = false; if (ctr && !userTookOver) ctr.autoRotate = true; }, 1500);
}

/* ── animation: gentle pulse on the highlighted subgraph ──────────────────── */
function pulse() {
  if (USE_GLOW && hlNodes.size) {
    const t = performance.now() * 0.0042;
    const k = 1 + Math.sin(t) * 0.16;
    hlNodes.forEach((id) => {
      const o = glowObjects.get(id);
      if (o) o.glow.scale.setScalar(o.base * 9 * k);
    });
  }
  requestAnimationFrame(pulse);
}

/* ── init ─────────────────────────────────────────────────────────────────── */
async function init() {
  DATA = await (await fetch("/api/graph")).json();
  DATA.nodes.forEach((n) => { nodeById[n.id] = n; });

  GRAPH = ForceGraph3D()(document.getElementById("graph"))
    .backgroundColor("#04050a")
    .graphData(DATA)
    .nodeRelSize(2.4)
    .nodeVal((n) => 1 + n.degree)
    .nodeLabel((n) => `<div style="font-family:'Sora',sans-serif;max-width:300px;padding:8px 11px;background:rgba(6,10,18,0.92);border:1px solid ${colorOf(n.type)};border-radius:10px;box-shadow:0 8px 30px -8px #000;color:#eaf0ff">
        <b style="font-family:'Chakra Petch',monospace;letter-spacing:.04em">${n.id}</b>
        <span style="color:${colorOf(n.type)};font-size:11px"> · ${n.type}</span><br>
        <span style="color:#9fb0d0;font-size:12px">${(n.description || "").slice(0, 180)}${(n.description || "").length > 180 ? "…" : ""}</span></div>`)
    .linkColor(linkColor)
    .linkWidth(linkWidth)
    .linkOpacity(1)
    .linkCurvature(0.22)
    .linkDirectionalParticles(linkParticles)
    .linkDirectionalParticleSpeed(0.012)
    .linkDirectionalParticleWidth(2.4)
    .linkDirectionalParticleColor(() => "#c9f7ff")
    .linkVisibility(linkVisible)
    .onNodeClick((n) => {
      const dist = 160;
      const d = Math.hypot(n.x, n.y, n.z) || 1;
      GRAPH.cameraPosition({ x: n.x * (1 + dist / d), y: n.y * (1 + dist / d), z: n.z * (1 + dist / d) }, n, 900);
    });

  if (USE_GLOW) GRAPH.nodeThreeObject(makeNodeObject).nodeThreeObjectExtend(false);
  else GRAPH.nodeColor(fallbackNodeColor).nodeOpacity(0.95);

  // idle auto-rotation; pause while the user drags or during a camera fly
  const ctr = GRAPH.controls();
  if (ctr) {
    ctr.autoRotate = true; ctr.autoRotateSpeed = 0.42;
    // gentle intro spin; once the user grabs the view it stops for good (no snap-back)
    ctr.addEventListener("start", () => { userTookOver = true; ctr.autoRotate = false; });
  }
  GRAPH.onEngineStop(() => { if (!fitted) { fitted = true; GRAPH.zoomToFit(900, 70); } });

  buildLegend();
  pulse();
}

/* clickable example questions — graph-flavoured (entities + relations) */
const EXAMPLES = [
  "Wie hängen Prüfauftrag und Betriebsmittel zusammen?",
  "Welche Aufgabe hat ein Prüfer?",
  "Was passiert bei der Prüfung eines Feuerlöschers?",
  "Wie funktioniert die Zeiterfassung?",
  "Welche Rollen gibt es im System?",
];
const exWrap = document.getElementById("examples");
if (exWrap) EXAMPLES.forEach((q) => {
  const b = document.createElement("button");
  b.type = "button"; b.textContent = q;
  b.addEventListener("click", () => {
    document.getElementById("q").value = q;
    document.getElementById("qform").requestSubmit();
  });
  exWrap.appendChild(b);
});

document.getElementById("qform").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = document.getElementById("q").value.trim();
  if (!q) return;
  if (hintEl) hintEl.classList.add("hidden");
  const mode = document.getElementById("mode").value;
  statusEl.classList.remove("naive");
  statusEl.innerHTML = "Traversiere Graph…";
  try {
    const r = await fetch("/api/graph/query", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q, mode }),
    });
    const b = await r.json();
    if (!r.ok) { statusEl.innerHTML = `Fehler: ${b.error || r.status}`; return; }
    applyHighlight(b);
  } catch (err) {
    statusEl.innerHTML = `Fehler: ${err}`;
  }
});

document.getElementById("reset").addEventListener("click", () => {
  hlNodes = new Set(); hlEdges = new Set();
  kwEl.innerHTML = ""; statusEl.classList.remove("naive");
  statusEl.innerHTML = "Bereit. Graph geladen.";
  refresh();
  const ctr = GRAPH.controls(); if (ctr && !userTookOver) ctr.autoRotate = true;
  GRAPH.zoomToFit(900, 70);
});

init();
