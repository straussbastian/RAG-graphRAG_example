/* ── Classic RAG · Embedding Field ──────────────────────────────────────────
   Plotly point cloud of the LizardDocu embeddings. A query is embedded, its
   true nearest neighbours light up with glow halos and link lines, the rest of
   the field dims. Transparent Plotly background lets the CSS nebula show. Hover
   shows a glassy card identical in style to the GraphRAG node labels. */

const PALETTE = ["#22d3ee", "#fb7185", "#fde047", "#a78bfa", "#34d399",
                 "#fb923c", "#38bdf8", "#f472b6", "#a3e635", "#7c5cff"];
const plotDiv = document.getElementById("plot");
const statusEl = document.getElementById("status");
const hintEl = document.getElementById("hint");
let POINTS = [];
let idIndex = {};

const cluColor = (p) => PALETTE[p.cluster % PALETTE.length];

// glassy hover card — same look as the GraphRAG node tooltip
function cardHTML(p, head) {
  const color = cluColor(p);
  const title = p.title ? p.title : "Chunk";
  const loc = p.loc ? ` · Zeilen ${p.loc}` : "";
  const desc = (p.content || "").replace(/\n/g, " ").slice(0, 200);
  const ell = (p.content || "").length > 200 ? "…" : "";
  const h = head
    ? `<span style="font-family:'Chakra Petch',monospace;font-size:11px;color:${color}">${head}</span><br>`
    : "";
  return `<div style="font-family:'Sora',sans-serif;max-width:320px;padding:9px 12px;background:rgba(6,10,18,0.94);border:1px solid ${color};border-radius:10px;box-shadow:0 10px 34px -8px #000;color:#eaf0ff">
    ${h}<b style="font-family:'Chakra Petch',monospace;letter-spacing:.04em">${title}</b><span style="color:${color};font-size:11px">${loc}</span><br>
    <span style="color:#9fb0d0;font-size:12px;line-height:1.4">${desc}${ell}</span></div>`;
}

function cloudTrace(points, { size, opacity }) {
  return {
    type: "scatter3d", mode: "markers", hoverinfo: "none",
    x: points.map(p => p.umap[0]), y: points.map(p => p.umap[1]), z: points.map(p => p.umap[2]),
    customdata: points.map(p => cardHTML(p)),
    marker: { size, opacity, color: points.map(cluColor) },
  };
}

function haloTrace(points, { size, color, opacity }) {
  return {
    type: "scatter3d", mode: "markers", hoverinfo: "skip",
    x: points.map(p => p.umap[0]), y: points.map(p => p.umap[1]), z: points.map(p => p.umap[2]),
    marker: { size, opacity, color: color || points.map(cluColor), line: { width: 0 } },
  };
}

const axisHidden = { showgrid: false, zeroline: false, showline: false,
                     showticklabels: false, showbackground: false, title: "" };
const layout = {
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#eaf0ff", family: "Sora, system-ui, sans-serif" },
  showlegend: false, margin: { l: 0, r: 0, t: 0, b: 0 }, uirevision: "keep",
  scene: { xaxis: axisHidden, yaxis: axisHidden, zaxis: axisHidden, bgcolor: "rgba(0,0,0,0)" },
};
const CONFIG = { responsive: true, displayModeBar: false };

function baseCloud() {
  return [
    haloTrace(POINTS, { size: 34, opacity: 0.04 }),
    haloTrace(POINTS, { size: 18, opacity: 0.10 }),
    cloudTrace(POINTS, { size: 10, opacity: 0.95 }),
  ];
}

/* ── custom glass tooltip (matches the GraphRAG labels) ───────────────────── */
const tip = document.createElement("div");
tip.style.cssText = "position:fixed;z-index:50;pointer-events:none;opacity:0;transition:opacity .12s ease;max-width:340px";
document.body.appendChild(tip);
// 3D Plotly hover events don't carry reliable cursor coords, so track the mouse
// ourselves and place the card next to it.
let mx = 0, my = 0;
window.addEventListener("mousemove", (e) => { mx = e.clientX; my = e.clientY; });
function showTip(html) {
  tip.innerHTML = html;
  tip.style.left = Math.min(mx + 16, window.innerWidth - 350) + "px";
  tip.style.top = Math.min(my + 16, window.innerHeight - 130) + "px";
  tip.style.opacity = "1";
}
const hideTip = () => { tip.style.opacity = "0"; };

/* ── gentle auto-orbit; pauses on interaction or while hovering a point ───── */
let orbit = true, userTookOver = false, theta = 0.7, relayouting = false, resumeT = null;
const ORBIT_R = 1.9;
function autoOrbit() {
  if (orbit && !userTookOver && !relayouting && POINTS.length) {
    theta += 0.0016; relayouting = true;
    Plotly.relayout(plotDiv, { "scene.camera.eye": { x: ORBIT_R * Math.cos(theta), y: ORBIT_R * Math.sin(theta), z: 0.85 } })
      .then(() => { relayouting = false; }).catch(() => { relayouting = false; });
  }
  requestAnimationFrame(autoOrbit);
}
// hovering pauses the spin briefly (readable labels), then resumes smoothly
function pauseOrbitTemp() { if (userTookOver) return; orbit = false; clearTimeout(resumeT); resumeT = setTimeout(() => { orbit = true; }, 2500); }
// a real camera move (drag/zoom) stops the spin for good — the view stays put
function stopOrbit() { userTookOver = true; orbit = false; clearTimeout(resumeT); }

async function init() {
  const res = await fetch("/api/points");
  POINTS = (await res.json()).points;
  POINTS.forEach((p, i) => { idIndex[p.id] = i; });
  await Plotly.newPlot(plotDiv, baseCloud(), layout, CONFIG);
  plotDiv.on("plotly_hover", (e) => {
    const pt = e.points && e.points[0];
    if (pt && pt.customdata) { showTip(pt.customdata); pauseOrbitTemp(); }
  });
  plotDiv.on("plotly_unhover", hideTip);
  plotDiv.on("plotly_relayouting", stopOrbit);
  plotDiv.addEventListener("mousedown", stopOrbit);
  plotDiv.addEventListener("wheel", stopOrbit, { passive: true });
  autoOrbit();
}

function resetView() {
  document.getElementById("q").value = "";
  statusEl.innerHTML = "Bereit. Punktwolke geladen.";
  hideTip();
  Plotly.react(plotDiv, baseCloud(), layout, CONFIG);
}

function render(queryPoint, neighbors) {
  const nbset = new Set(neighbors.map(n => n.id));
  const background = POINTS.filter(p => !nbset.has(p.id));
  const nbPoints = neighbors.map(n => POINTS[idIndex[n.id]]);

  const bgTrace = cloudTrace(background, { size: 3, opacity: 0.10 });
  const nbHalo = haloTrace(nbPoints, { size: 22, opacity: 0.24 });
  const nbTrace = {
    type: "scatter3d", mode: "markers", hoverinfo: "none",
    x: nbPoints.map(p => p.umap[0]), y: nbPoints.map(p => p.umap[1]), z: nbPoints.map(p => p.umap[2]),
    customdata: neighbors.map((n, i) => cardHTML(nbPoints[i], `#${i + 1} · Score ${n.score.toFixed(3)}`)),
    marker: { size: 8, opacity: 1.0, color: nbPoints.map(cluColor) },
  };

  const lines = { type: "scatter3d", mode: "lines", hoverinfo: "skip",
    x: [], y: [], z: [], line: { color: "#7fe9ff", width: 2.5 }, opacity: 0.85 };
  nbPoints.forEach(p => {
    lines.x.push(queryPoint[0], p.umap[0], null);
    lines.y.push(queryPoint[1], p.umap[1], null);
    lines.z.push(queryPoint[2], p.umap[2], null);
  });

  const qHalo = { type: "scatter3d", mode: "markers", hoverinfo: "skip",
    x: [queryPoint[0]], y: [queryPoint[1]], z: [queryPoint[2]],
    marker: { size: 30, opacity: 0.28, color: "#a78bfa" } };
  const qtrace = { type: "scatter3d", mode: "markers+text", hoverinfo: "skip",
    x: [queryPoint[0]], y: [queryPoint[1]], z: [queryPoint[2]],
    text: ["✦ Query"], textposition: "top center",
    textfont: { color: "#fff", family: "Chakra Petch, monospace", size: 13 },
    marker: { size: 11, color: "#ffffff", symbol: "diamond", line: { color: "#a78bfa", width: 2 } } };

  Plotly.react(plotDiv, [bgTrace, nbHalo, lines, nbTrace, qHalo, qtrace], layout, CONFIG);
}

/* clickable example queries — so the audience sees what to ask */
const EXAMPLES = [
  "Wie funktioniert die Zeiterfassung?",
  "Was ist ein Prüfauftrag?",
  "Welche Rollen gibt es?",
  "Wie lege ich ein Betriebsmittel an?",
  "Listenansicht",
];
const exWrap = document.getElementById("examples");
if (exWrap) EXAMPLES.forEach((q) => {
  const b = document.createElement("button");
  b.type = "button"; b.textContent = q;
  b.addEventListener("click", () => {
    document.getElementById("q").value = q;
    document.getElementById("queryForm").requestSubmit();
  });
  exWrap.appendChild(b);
});

document.getElementById("resetBtn").addEventListener("click", resetView);

document.getElementById("queryForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = document.getElementById("q").value.trim();
  if (!q) return;
  if (hintEl) hintEl.classList.add("hidden");
  statusEl.innerHTML = "Embedding + Suche…";
  try {
    const res = await fetch(`/api/query?q=${encodeURIComponent(q)}&k=8`);
    const body = await res.json();
    if (!res.ok) { statusEl.innerHTML = `Fehler: ${body.error || res.status}`; return; }
    render(body.point.umap, body.neighbors);
    statusEl.innerHTML = `<span class="num">${body.neighbors.length}</span> Nachbarn · Top-Score <span class="num">${body.neighbors[0].score.toFixed(3)}</span>`;
  } catch (err) {
    statusEl.innerHTML = `Fehler: ${err}`;
  }
});

init();
