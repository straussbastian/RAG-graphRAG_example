/* ── Chunking · Text → Chunks ───────────────────────────────────────────────
   Visualise how a document is split before it is embedded. Live controls for
   splitter type, chunk size, and overlap; the text is re-rendered with each
   chunk in its own colour and overlap regions hatched. */

const PALETTE = ["#22d3ee", "#fb7185", "#fde047", "#a78bfa", "#34d399",
                 "#fb923c", "#38bdf8", "#f472b6", "#a3e635", "#7c5cff"];
const col = (i) => PALETTE[i % PALETTE.length];
const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

const ta = document.getElementById("ta");
const out = document.getElementById("out");
const sizeEl = document.getElementById("size"), sizeVal = document.getElementById("sizeVal");
const ovlEl = document.getElementById("ovl"), ovlVal = document.getElementById("ovlVal");
const splitterEl = document.getElementById("splitter");
const statsEl = document.getElementById("stats");

function render(text, chunks) {
  const pts = new Set([0, text.length]);
  chunks.forEach((c) => { pts.add(c.start); pts.add(c.end); });
  const sorted = [...pts].sort((a, b) => a - b);
  let html = "";
  for (let s = 0; s < sorted.length - 1; s++) {
    const p = sorted[s], q = sorted[s + 1];
    if (p >= q) continue;
    const cov = [];
    for (let ci = 0; ci < chunks.length; ci++) {
      if (chunks[ci].start <= p && chunks[ci].end >= q) cov.push(ci);
    }
    const seg = esc(text.slice(p, q));
    if (cov.length === 0) {
      html += `<span class="seg">${seg}</span>`;
    } else if (cov.length === 1) {
      const c = col(cov[0]);
      html += `<span class="seg" style="background:${c}2e;box-shadow:inset 0 -2px 0 ${c}">${seg}</span>`;
    } else {
      const c1 = col(cov[0]), c2 = col(cov[cov.length - 1]);
      html += `<span class="seg ovl" title="Overlap (in zwei Chunks)" style="background:repeating-linear-gradient(45deg, ${c1}66 0 6px, ${c2}66 6px 12px)">${seg}</span>`;
    }
  }
  out.innerHTML = html;
}

function stats(chunks) {
  const sizes = chunks.map((c) => c.end - c.start);
  const sum = sizes.reduce((a, b) => a + b, 0);
  const avg = chunks.length ? Math.round(sum / chunks.length) : 0;
  const mn = chunks.length ? Math.min(...sizes) : 0;
  const mx = chunks.length ? Math.max(...sizes) : 0;
  const n = (v) => `<span class="num">${v}</span>`;
  statsEl.innerHTML = `${n(chunks.length)} Chunks · Ø ${n(avg)} · min ${n(mn)} · max ${n(mx)} Zeichen`;
}

function run() {
  const text = ta.value;
  const size = Math.max(1, +sizeEl.value);
  ovlEl.max = Math.max(0, size - 1);
  const overlap = Math.min(+ovlEl.value, size - 1);
  ovlEl.value = overlap;
  sizeVal.textContent = size;
  ovlVal.textContent = overlap;
  const fn = CHUNKERS[splitterEl.value] || CHUNKERS.recursive;
  const chunks = fn(text, size, overlap);
  render(text, chunks);
  stats(chunks);
}

[sizeEl, ovlEl, splitterEl, ta].forEach((el) => el.addEventListener("input", run));

fetch("/chunk-static/example.txt")
  .then((r) => (r.ok ? r.text() : ""))
  .then((t) => { if (t && !ta.value.trim()) ta.value = t; run(); })
  .catch(run);
