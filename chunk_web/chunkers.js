/* ── chunkers.js ─────────────────────────────────────────────────────────────
   Pure text splitters. Each returns an array of {start, end} ranges over the
   original text (consecutive ranges may overlap by `overlap` characters).
   No DOM access → unit-testable in Node. */

(function (global) {
  // Fixed character windows: cut anywhere, step = size - overlap.
  function characterSplit(text, size, overlap) {
    const ranges = [];
    const step = Math.max(1, size - overlap);
    for (let i = 0; i < text.length; i += step) {
      const end = Math.min(i + size, text.length);
      ranges.push({ start: i, end });
      if (end >= text.length) break;
    }
    return ranges.length ? ranges : [{ start: 0, end: text.length }];
  }

  // Recursively split [s,e) on a separator hierarchy until each atom is <= size
  // (or no separators remain). Returns contiguous {start,end} atoms covering the
  // whole span. The separator stays attached to the preceding atom.
  function atomize(text, s, e, seps, size) {
    if (e - s <= size || seps.length === 0) return [{ start: s, end: e }];
    const sep = seps[0], rest = seps.slice(1), out = [];
    if (sep === "") {
      for (let i = s; i < e; i += size) out.push({ start: i, end: Math.min(i + size, e) });
      return out;
    }
    const sub = text.slice(s, e);
    let idx, last = 0;
    while ((idx = sub.indexOf(sep, last)) !== -1) {
      pushAtoms(out, text, s + last, s + idx + sep.length, rest, size);
      last = idx + sep.length;
    }
    if (last < sub.length) pushAtoms(out, text, s + last, e, rest, size);
    return out.length ? out : [{ start: s, end: e }];
  }
  function pushAtoms(out, text, s, e, seps, size) {
    if (e <= s) return;
    if (e - s <= size || seps.length === 0) { out.push({ start: s, end: e }); return; }
    for (const a of atomize(text, s, e, seps, size)) out.push(a);
  }

  // Merge contiguous atoms into chunks <= size, with `overlap` chars carried over.
  function mergeAtoms(atoms, size, overlap) {
    const chunks = [];
    let i = 0;
    while (i < atoms.length) {
      const start = atoms[i].start;
      let end = atoms[i].end, j = i + 1;
      while (j < atoms.length && atoms[j].end - start <= size) { end = atoms[j].end; j++; }
      chunks.push({ start, end });
      if (j >= atoms.length) break;
      const target = end - overlap;
      let k = j;
      while (k > i + 1 && atoms[k - 1].start >= target) k--;
      i = k;
    }
    return chunks.length ? chunks : [{ start: 0, end: 0 }];
  }

  const recursiveSplit = (text, size, overlap) =>
    mergeAtoms(atomize(text, 0, text.length, ["\n\n", "\n", " ", ""], size), size, overlap);

  const markdownSplit = (text, size, overlap) =>
    mergeAtoms(atomize(text, 0, text.length,
      ["\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ", "```", "\n\n", "\n", " ", ""], size),
      size, overlap);

  function sentenceSplit(text, size, overlap) {
    const atoms = [];
    let start = 0, m;
    const re = /[.!?]+[\s]+|\n{2,}/g;
    while ((m = re.exec(text)) !== null) {
      atoms.push({ start, end: m.index + m[0].length });
      start = m.index + m[0].length;
    }
    if (start < text.length) atoms.push({ start, end: text.length });
    return mergeAtoms(atoms.length ? atoms : [{ start: 0, end: text.length }], size, overlap);
  }

  global.CHUNKERS = {
    character: characterSplit,
    recursive: recursiveSplit,
    markdown: markdownSplit,
    sentence: sentenceSplit,
  };
})(typeof window !== "undefined" ? window : globalThis);
