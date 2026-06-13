/* ── Embedding · Vektorraum-Live ────────────────────────────────────────────
   Embed two texts, draw each 1024-dim vector as a 32×32 "fingerprint" heatmap
   (cold = negative, warm = positive), and show their cosine similarity. */

const GRID = 32, CELL = 11, SIZE = GRID * CELL;   // 32×32 = 1024 cells, 352px
const DARK = [8, 12, 20], WARM = [251, 191, 36], COLD = [41, 227, 255];

const PRESETS = [
  { label: "Mitarbeiter ↔ Angestellter", a: "Mitarbeiter", b: "Angestellter" },
  { label: "Mitarbeiter ↔ Banane", a: "Mitarbeiter", b: "Banane" },
  { label: "Prüfung ↔ Wartung",
    a: "Wer führt die Prüfung eines Betriebsmittels durch?",
    b: "Welche Aufgabe hat ein Mitarbeiter bei der Wartung?" },
];

const statusEl = document.getElementById("status");
const lerp = (d, t, m) => Math.round(d + (t - d) * m);

function cellColor(v, maxAbs) {
  const t = maxAbs > 0 ? Math.max(-1, Math.min(1, v / maxAbs)) : 0;
  const m = Math.abs(t), tg = t >= 0 ? WARM : COLD;
  return `rgb(${lerp(DARK[0], tg[0], m)},${lerp(DARK[1], tg[1], m)},${lerp(DARK[2], tg[2], m)})`;
}

// Agreement cells emphasise SIGN (warm = agree, cool = disagree) with a
// brightness floor + perceptual scaling, so identical inputs read as a clearly
// all-warm field even though each dimension's contribution is tiny.
function agreeColor(v, maxAbs) {
  const s = maxAbs > 0 ? Math.max(-1, Math.min(1, v / maxAbs)) : 0;
  const m = 0.42 + 0.58 * Math.sqrt(Math.abs(s)), tg = s >= 0 ? WARM : COLD;
  return `rgb(${lerp(DARK[0], tg[0], m)},${lerp(DARK[1], tg[1], m)},${lerp(DARK[2], tg[2], m)})`;
}

function drawFingerprint(canvas, vec) {
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#05070d";
  ctx.fillRect(0, 0, SIZE, SIZE);
  let maxAbs = 0;
  for (const v of vec) maxAbs = Math.max(maxAbs, Math.abs(v));
  const n = Math.min(vec.length, GRID * GRID);
  for (let i = 0; i < n; i++) {
    ctx.fillStyle = cellColor(vec[i], maxAbs);
    ctx.fillRect((i % GRID) * CELL + 0.5, Math.floor(i / GRID) * CELL + 0.5, CELL - 1, CELL - 1);
  }
}

// Calibrated to the real mistral-embed band for German text (~0.66 floor → 1.0).
function wordFor(c) {
  if (c >= 0.92) return "nahezu identisch";
  if (c >= 0.83) return "sehr ähnlich";
  if (c >= 0.75) return "verwandt";
  if (c >= 0.68) return "schwach verwandt";
  return "kaum verwandt";
}

const GAUGE_LO = 0.6, GAUGE_HI = 1.0;
function setCosine(c) {
  document.getElementById("cosval").textContent = c.toFixed(2);
  const pct = Math.max(0, Math.min(1, (c - GAUGE_LO) / (GAUGE_HI - GAUGE_LO))) * 100;
  document.getElementById("gmark").style.left = pct + "%";
  document.getElementById("cosword").textContent = wordFor(c);
}

// Element-wise agreement of the two unit-normalised vectors. Warm = same sign
// (drives similarity up), cool = opposite. The sum of all cells IS the cosine.
function drawAgreement(canvas, a, b) {
  const ctx = canvas.getContext("2d"), S = canvas.width;
  ctx.fillStyle = "#05070d";
  ctx.fillRect(0, 0, S, S);
  let na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) na += a[i] * a[i];
  for (let i = 0; i < b.length; i++) nb += b[i] * b[i];
  na = Math.sqrt(na) || 1; nb = Math.sqrt(nb) || 1;
  const n = Math.min(a.length, b.length, GRID * GRID), prod = new Array(n);
  let maxAbs = 0;
  for (let i = 0; i < n; i++) { prod[i] = (a[i] / na) * (b[i] / nb); maxAbs = Math.max(maxAbs, Math.abs(prod[i])); }
  const cell = S / GRID;
  for (let i = 0; i < n; i++) {
    ctx.fillStyle = agreeColor(prod[i], maxAbs);
    ctx.fillRect((i % GRID) * cell, Math.floor(i / GRID) * cell, cell, cell);
  }
}

async function compare() {
  const a = document.getElementById("ta").value.trim();
  const b = document.getElementById("tb").value.trim();
  if (!a || !b) { statusEl.textContent = "Bitte beide Texte ausfüllen."; return; }
  statusEl.textContent = "Embedde…";
  try {
    const r = await fetch("/api/embed", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ a, b }),
    });
    const body = await r.json();
    if (!r.ok) { statusEl.textContent = "Fehler: " + (body.error || r.status); return; }
    drawFingerprint(document.getElementById("ca"), body.a.vector);
    drawFingerprint(document.getElementById("cb"), body.b.vector);
    drawAgreement(document.getElementById("cc"), body.a.vector, body.b.vector);
    document.getElementById("dimA").textContent = body.dims + " Dimensionen";
    document.getElementById("dimB").textContent = body.dims + " Dimensionen";
    setCosine(body.cosine);
    statusEl.textContent = `${body.dims}-dim Vektoren · Cosine ${body.cosine.toFixed(3)}`;
  } catch (err) {
    statusEl.textContent = "Fehler: " + err;
  }
}

document.getElementById("cmp").addEventListener("click", compare);

const presetsEl = document.getElementById("presets");
PRESETS.forEach((p) => {
  const b = document.createElement("button");
  b.className = "preset"; b.type = "button"; b.textContent = p.label;
  b.onclick = () => {
    document.getElementById("ta").value = p.a;
    document.getElementById("tb").value = p.b;
    compare();
  };
  presetsEl.appendChild(b);
});

/* ── mode toggle (Paar-Vergleich ↔ Verbindungs-Spiel) ─────────────────────── */
document.querySelectorAll(".modes button").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll(".modes button").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    const game = b.dataset.mode === "game";
    document.getElementById("pair-mode").hidden = game;
    document.getElementById("game-mode").hidden = !game;
    document.getElementById("presets").style.display = game ? "none" : "";
    statusEl.textContent = game ? "Begriffe eingeben → Verbinden." : "Zwei Texte eingeben → Vergleichen.";
  });
});

/* ── connection game: all-pairs cosine matrix ─────────────────────────────── */
const GAME_DEFAULTS = ["Banane", "Steuererklärung", "Gitarre", "Vulkan", "Schnürsenkel"];
const ginputs = document.getElementById("ginputs");
function addTerm(val = "") {
  if (ginputs.children.length >= 8) return;
  const inp = document.createElement("input");
  inp.type = "text"; inp.className = "gterm"; inp.value = val; inp.placeholder = "Begriff…";
  ginputs.appendChild(inp);
}
GAME_DEFAULTS.forEach((w) => addTerm(w));
document.getElementById("addterm").addEventListener("click", () => addTerm());

function matrixColor(v) {
  const LO = 0.55, HI = 1.0;
  const t = Math.max(0, Math.min(1, (v - LO) / (HI - LO)));
  const lo = [18, 36, 64], hi = [251, 191, 36];
  return `rgb(${lerp(lo[0], hi[0], t)},${lerp(lo[1], hi[1], t)},${lerp(lo[2], hi[2], t)})`;
}
const mkCell = (text, cls) => { const d = document.createElement("div"); d.className = cls; d.textContent = text; return d; };
const short = (s) => (s.length > 12 ? s.slice(0, 11) + "…" : s);
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function renderMatrix(d) {
  const M = document.getElementById("matrix");
  M.innerHTML = "";
  const L = d.labels, N = L.length;
  M.style.gridTemplateColumns = `minmax(72px,120px) repeat(${N}, minmax(46px,1fr))`;
  M.appendChild(mkCell("", "mcorner"));
  L.forEach((l) => { const c = mkCell(short(l), "mhead mcol"); c.title = l; M.appendChild(c); });
  for (let i = 0; i < N; i++) {
    const rh = mkCell(short(L[i]), "mhead mrow"); rh.title = L[i]; M.appendChild(rh);
    for (let j = 0; j < N; j++) {
      if (i === j) { M.appendChild(mkCell("–", "mself")); continue; }
      const v = d.matrix[i][j];
      const isMin = d.min && ((i === d.min.i && j === d.min.j) || (i === d.min.j && j === d.min.i));
      const c = mkCell(v.toFixed(2), "mcell" + (isMin ? " mmin" : ""));
      c.style.background = matrixColor(v);
      c.style.color = v > 0.82 ? "#1a1000" : "#eaf0ff";
      M.appendChild(c);
    }
  }
  document.getElementById("gnote").innerHTML =
    `Niedrigster Wert: <b>${d.min.value.toFixed(2)}</b> zwischen „${esc(L[d.min.i])}" und „${esc(L[d.min.j])}".<br>` +
    `Selbst die fremdesten Begriffe bleiben verbunden — eine echte <b>0</b> gibt es nicht.`;
}

async function playGame() {
  const texts = [...document.querySelectorAll(".gterm")].map((i) => i.value.trim()).filter(Boolean);
  if (texts.length < 2) { statusEl.textContent = "Mindestens 2 Begriffe eingeben."; return; }
  statusEl.textContent = "Embedde…";
  try {
    const r = await fetch("/api/embed/matrix", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts }),
    });
    const d = await r.json();
    if (!r.ok) { statusEl.textContent = "Fehler: " + (d.error || r.status); return; }
    renderMatrix(d);
    statusEl.textContent = `${d.labels.length} Begriffe · niedrigster Cosine ${d.min.value.toFixed(3)}`;
  } catch (err) {
    statusEl.textContent = "Fehler: " + err;
  }
}
document.getElementById("play").addEventListener("click", playGame);
