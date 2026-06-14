/* ── Pipeline · Klassisches RAG live ─────────────────────────────────────────
   Presenter-paced walk through one real RAG query. All API calls fire right
   after "Start" (retrieve + LightRAG in parallel, chat once retrieve lands),
   so the data is usually there before the presenter reveals its block. */

const $ = (id) => document.getElementById(id);
const statusEl = $("status");

/* fingerprint: same visual language as the embedding view (32×32, warm/cold) */
const GRID = 32, CELL = 11, SIZE = GRID * CELL;
const DARK = [8, 12, 20], WARM = [251, 191, 36], COLD = [41, 227, 255];
const lerp = (d, t, m) => Math.round(d + (t - d) * m);

function drawFingerprint(canvas, vec) {
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#05070d";
  ctx.fillRect(0, 0, SIZE, SIZE);
  let maxAbs = 0;
  for (const v of vec) maxAbs = Math.max(maxAbs, Math.abs(v));
  const n = Math.min(vec.length, GRID * GRID);
  for (let i = 0; i < n; i++) {
    const t = maxAbs > 0 ? Math.max(-1, Math.min(1, vec[i] / maxAbs)) : 0;
    const m = Math.abs(t), tg = t >= 0 ? WARM : COLD;
    ctx.fillStyle = `rgb(${lerp(DARK[0], tg[0], m)},${lerp(DARK[1], tg[1], m)},${lerp(DARK[2], tg[2], m)})`;
    ctx.fillRect((i % GRID) * CELL + 0.5, Math.floor(i / GRID) * CELL + 0.5, CELL - 1, CELL - 1);
  }
}

/* ── state ── */
const S = { q: "", step: 0, retrieve: null, answer: null, graphdata: null, graphprompt: null, graph: null, errors: {} };
const MAX_STEP = 8;

/* step n reveals: 1 question · 2 embedding · 3 vector search · 4 prompt ·
   5 classic answer · 6 graph retrieval · 7 graph prompt · 8 graph answer */
const REVEALS = {
  1: ["rail-top", "blk-q"],
  2: ["ar1", "blk-embed"],
  3: ["rail-classic", "blk-search"],
  4: ["ar2", "blk-prompt"],
  5: ["ar3", "blk-ans-classic"],
  6: ["rail-graph", "blk-graphretr"],
  7: ["ar4", "blk-graphprompt"],
  8: ["ar5", "blk-ans-graph"],
};

function setStatus(t) { statusEl.textContent = t; }

function renderDots() {
  const d = $("dots");
  d.innerHTML = "";
  for (let i = 1; i <= MAX_STEP; i++) {
    const s = document.createElement("span");
    if (i <= S.step) s.classList.add("on");
    d.appendChild(s);
  }
}

function markActive() {
  document.querySelectorAll(".block").forEach((b) => b.classList.remove("active"));
  const ids = REVEALS[S.step] || [];
  for (const id of ids) {
    const el = $(id);
    if (el && el.classList.contains("block")) el.classList.add("active");
  }
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const waiting = (msg) => `<div class="waiting">${esc(msg)}</div>`;
const errBox = (msg) => `<div class="err">${esc(msg)}</div>`;

/* answers come back as markdown — flatten it for the plain-text cards */
function plain(md) {
  return md
    .replace(/^#{1,6}\s*/gm, "")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*\n]+)\*/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/^---+\s*$/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/* typewriter for the answer cards */
function typeInto(el, text, speed = 9) {
  el.textContent = "";
  let i = 0;
  const tick = () => {
    el.textContent = text.slice(0, i);
    el.scrollTop = el.scrollHeight;
    if (i < text.length) { i = Math.min(text.length, i + 3); setTimeout(tick, speed); }
  };
  tick();
}

/* ── per-step renderers (re-run whenever data arrives) ── */
function renderStep(n) {
  if (n > S.step) return;
  if (n === 1) $("qtext").textContent = S.q;

  if (n === 2) {
    if (S.errors.retrieve) { $("dims").innerHTML = errBox(S.errors.retrieve); return; }
    if (!S.retrieve) { $("dims").innerHTML = waiting("Mistral embeddet…"); return; }
    drawFingerprint($("fp"), S.retrieve.vector);
    $("dims").textContent = `${S.retrieve.dims} Dimensionen`;
  }

  if (n === 3) {
    if (S.errors.retrieve) { $("chunks").innerHTML = errBox(S.errors.retrieve); return; }
    if (!S.retrieve) { $("chunks").innerHTML = waiting("Qdrant sucht…"); return; }
    $("coll").textContent = `Qdrant · ${S.retrieve.collection} · ${S.retrieve.total} Chunks`;
    $("chunks").innerHTML = S.retrieve.neighbors.map((nb, i) => `
      <div class="chunk" style="animation-delay:${i * 90}ms">
        <div class="ch-head"><span>#${i + 1}</span><span class="score">Score ${nb.score.toFixed(3)}</span>
          ${nb.loc ? `<span>Zeilen ${esc(nb.loc)}</span>` : ""}</div>
        <div class="ch-text">${esc(nb.content)}</div>
      </div>`).join("");
  }

  if (n === 4) {
    if (S.errors.answer) { $("prompt").innerHTML = errBox(S.errors.answer); return; }
    if (!S.answer) { $("prompt").innerHTML = waiting("Prompt wird gebaut…"); return; }
    renderPrompt(S.answer.prompt);
  }

  if (n === 5) {
    $("amodel").textContent = S.answer ? S.answer.model : "Mistral";
    const body = $("abody");
    if (S.errors.answer) { body.innerHTML = errBox(S.errors.answer); return; }
    if (!S.answer) { body.innerHTML = waiting("LLM generiert…"); return; }
    if (!body.dataset.typed) { body.dataset.typed = "1"; typeInto(body, plain(S.answer.answer)); }
  }

  if (n === 6) {
    const body = $("grbody");
    if (S.errors.graphdata) { body.innerHTML = errBox(S.errors.graphdata); return; }
    if (!S.graphdata) { body.innerHTML = waiting("LightRAG traversiert den Graphen…"); return; }
    renderGraphRetrieval(S.graphdata);
  }

  if (n === 7) {
    const el = $("gprompt");
    if (S.errors.graphprompt) { el.innerHTML = errBox(S.errors.graphprompt); return; }
    if (!S.graphprompt) { el.innerHTML = waiting("LightRAG baut den Prompt…"); return; }
    const p = S.graphprompt.prompt || "";
    $("gpstats").textContent =
      `${p.length.toLocaleString("de-DE")} Zeichen · ~${Math.round(p.length / 4000)}k Tokens`;
    el.textContent = p;  // full prompt, scrollable; textContent avoids any HTML injection
  }

  if (n === 8) {
    $("gmode").textContent = S.graph ? `LightRAG · ${S.graph.mode}` : "LightRAG";
    const body = $("gbody");
    if (S.errors.graph) { body.innerHTML = errBox(S.errors.graph); return; }
    if (!S.graph) { body.innerHTML = waiting("LightRAG generiert…"); return; }
    if (!body.dataset.typed) { body.dataset.typed = "1"; typeInto(body, plain(S.graph.answer)); }
  }
}

/* what GraphRAG loaded for the same question — counts tell the real story */
function renderGraphRetrieval(d) {
  $("gstats").textContent =
    `${d.totals.entities} Entitäten · ${d.totals.relations} Relationen · ${d.totals.chunks} Chunks`;
  const more = (shown, total, unit) =>
    total > shown ? `<span class="chip more">+${total - shown} ${unit}</span>` : "";
  const kws = (d.keywords.high_level || []).concat(d.keywords.low_level || []);
  $("grbody").innerHTML = `
    <div class="gr-col">
      <div class="gr-sub">Keywords</div>
      <div class="chips">${kws.map((k) => `<span class="chip kw">${esc(k)}</span>`).join("")}</div>
      <div class="gr-sub">Entitäten</div>
      <div class="chips">
        ${d.entities.map((e, i) => `<span class="chip" style="animation-delay:${i * 60}ms" title="${esc(e.type)}">${esc(e.name)}</span>`).join("")}
        ${more(d.entities.length, d.totals.entities, "weitere")}
      </div>
      <div class="gr-sub">Relationen</div>
      ${d.relations.slice(0, 4).map((r) => `<div class="rel"><b>${esc(r.src)}</b> ⟷ <b>${esc(r.tgt)}</b></div>`).join("")}
      ${d.totals.relations > 4 ? `<div class="rel">… +${d.totals.relations - 4} weitere</div>` : ""}
    </div>
    <div class="gr-col chunks-col">
      <div class="gr-sub">Chunks aus dem Graphen</div>
      ${d.chunks.map((c, i) => `
        <div class="gchunk" style="animation-delay:${i * 80}ms">
          <div class="gc-file">${esc(c.file)}</div>
          <div class="gc-text">${esc(c.content)}</div>
        </div>`).join("")}
      ${d.totals.chunks > d.chunks.length ? `<div class="rel">… +${d.totals.chunks - d.chunks.length} weitere Chunks</div>` : ""}
    </div>`;
}

/* colorize the REAL prompt string returned by the backend */
function renderPrompt(p) {
  const iCtx = p.indexOf("### Kontext");
  const iQ = p.indexOf("### Frage");
  const el = $("prompt");
  if (iCtx < 0 || iQ < 0) { el.textContent = p; return; }
  const sys = p.slice(0, iCtx).trim();
  const ctx = p.slice(iCtx, iQ).replace(/^### Kontext\n?/, "").trim();
  const q = p.slice(iQ).replace(/^### Frage\n?/, "").trim();
  const chunkHtml = ctx.split(/\n\n(?=\[\d+\] )/)
    .map((c) => `<span class="p-chunk">${esc(c)}</span>`).join("");
  el.innerHTML =
    `<span class="p-sys">${esc(sys)}</span>` +
    `<span class="p-head">### Kontext</span>${chunkHtml}` +
    `<span class="p-head">### Frage</span><span class="p-q">${esc(q)}</span>`;
}

function renderAll() { for (let n = 1; n <= S.step; n++) renderStep(n); }

/* ── stepping ── */
function advance() {
  if (!S.q || S.step >= MAX_STEP) return;
  S.step++;
  for (const id of REVEALS[S.step] || []) $(id).hidden = false;
  renderStep(S.step);
  renderDots();
  markActive();
  $("next").disabled = S.step >= MAX_STEP;
  const hints = ["", "Die Frage, wie der Nutzer sie stellt.", "1024 Zahlen — die Bedeutung der Frage.",
    "Nächste Nachbarn im Vektorraum.", "Kein Zauber: Chunks werden in den Prompt geklebt.",
    "Prompt → LLM → Antwort.", "Was GraphRAG stattdessen lädt: Entitäten, Relationen, Chunks.",
    "Der komplette Prompt, den LightRAG aus dem Graphen baut — viel größer als beim RAG.",
    "Dieselbe Frage, beantwortet über den Wissensgraphen."];
  setStatus(`Schritt ${S.step}/${MAX_STEP} · ${hints[S.step]}`);
}

function resetAll() {
  S.q = ""; S.step = 0; S.retrieve = null; S.answer = null; S.graphdata = null; S.graphprompt = null; S.graph = null; S.errors = {};
  Object.values(REVEALS).flat().forEach((id) => { $(id).hidden = true; });
  delete $("abody").dataset.typed; delete $("gbody").dataset.typed;
  $("abody").textContent = ""; $("gbody").textContent = ""; $("prompt").textContent = "";
  $("chunks").innerHTML = ""; $("grbody").innerHTML = ""; $("gstats").textContent = "";
  $("gprompt").textContent = ""; $("gpstats").textContent = "";
  $("next").disabled = true;
  document.querySelectorAll(".row").forEach((r) => { r.dataset.collapsed = "false"; });
  renderDots(); markActive();
  setStatus("Frage eingeben → Start. Dann Klick / Leertaste für den nächsten Schritt.");
  $("q").focus();
}

/* ── data orchestration: fire everything early, render as it lands ── */
async function start(q) {
  resetAll();
  S.q = q;
  advance(); // step 1: the question appears immediately

  const post = async (url, body) => {
    const r = await fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    return data;
  };

  // GraphRAG is slow (graph traversal + LLM) — fire both calls first, in parallel.
  post("/api/pipeline/graph", { q, mode: "mix" })
    .then((d) => { S.graph = d; renderAll(); })
    .catch((e) => { S.errors.graph = `LightRAG: ${e.message}`; renderAll(); });
  post("/api/pipeline/graphdata", { q, mode: "mix" })
    .then((d) => { S.graphdata = d; renderAll(); })
    .catch((e) => { S.errors.graphdata = `LightRAG: ${e.message}`; renderAll(); });
  post("/api/pipeline/graphprompt", { q, mode: "mix" })
    .then((d) => { S.graphprompt = d; renderAll(); })
    .catch((e) => { S.errors.graphprompt = `LightRAG: ${e.message}`; renderAll(); });

  try {
    S.retrieve = await post("/api/pipeline/retrieve", { q, k: 4 });
    renderAll();
  } catch (e) {
    S.errors.retrieve = `${e.message}`;
    S.errors.answer = "übersprungen — Retrieval fehlgeschlagen";
    renderAll();
    return;
  }

  post("/api/pipeline/answer", { q, chunks: S.retrieve.neighbors.map((n) => n.content) })
    .then((d) => { S.answer = d; renderAll(); })
    .catch((e) => { S.errors.answer = `${e.message}`; renderAll(); });
}

/* clickable example questions — fill the field and start the pipeline */
const EXAMPLES = [
  "Was ist ein Prüfauftrag?",
  "Wie funktioniert die Zeiterfassung?",
  "Welche Aufgabe hat ein Prüfer?",
  "Wie prüfe ich einen Feuerlöscher?",
];
const exWrap = $("examples");
if (exWrap) EXAMPLES.forEach((q) => {
  const b = document.createElement("button");
  b.type = "button"; b.textContent = q;
  b.addEventListener("click", () => { $("q").value = q; $("qform").requestSubmit(); });
  exWrap.appendChild(b);
});

/* ── wiring ── */
$("qform").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("q").value.trim();
  if (!q) { setStatus("Bitte eine Frage eingeben."); return; }
  start(q);
  $("q").blur(); // so space advances instead of typing
});
$("reset").addEventListener("click", resetAll);
$("next").addEventListener("click", advance);

/* collapse rows to make room — rail click or keys 1/2/3 */
function toggleRow(id) {
  const row = $(id);
  row.dataset.collapsed = row.dataset.collapsed === "true" ? "false" : "true";
}
document.querySelectorAll(".rail").forEach((rail) => {
  rail.addEventListener("click", (e) => { e.stopPropagation(); toggleRow(rail.closest(".row").id); });
});

/* stage click advances — but not when interacting with content or rails */
$("flow").addEventListener("click", (e) => {
  if (e.target.closest(".block") || e.target.closest(".rail")) return;
  advance();
});
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.code === "Space" || e.code === "ArrowRight" || e.code === "Enter") { e.preventDefault(); advance(); }
  if (e.key === "1") toggleRow("row-top");
  if (e.key === "2") toggleRow("row-classic");
  if (e.key === "3") toggleRow("row-graph");
});

renderDots();
