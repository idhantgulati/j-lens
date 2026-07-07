/* J-Lens Visualizer — all state, rendering, and API calls. Zero dependencies. */

const API = window.JLENS_API;

const PIN_COLORS = ["#ffd166", "#6ee7b7", "#7dd3fc", "#f0a8ff", "#fda4af", "#fbbf24"];
const RANK_BG = [
  [1,        "rgba(255,209,102,0.95)", false],
  [10,       "rgba(255,178,82,0.60)",  false],
  [100,      "rgba(224,122,95,0.38)",  true],
  [1000,     "rgba(146,94,120,0.28)",  true],
  [Infinity, "rgba(90,97,114,0.10)",   true],
];

const S = {
  resp: null,
  prompt: "",
  mode: "jlens",       // "jlens" | "logit_lens"
  gridMode: "argmax",  // "argmax" | "rank"
  sel: { layer: 24, pos: 0 },
  pins: [],            // {tokenId, text, color, ranks, loading}
  activePin: -1,
};

const $ = (id) => document.getElementById(id);

// ---------- small utils ----------

function esc(s) {
  return s.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}
function showTok(s) {
  if (s === "") return "∅";
  const cleaned = s.replaceAll("\n", "↵").replaceAll("\t", "⇥");
  return cleaned.trim() === "" ? "·".repeat(Math.min(cleaned.length, 3)) : cleaned;
}
function fmtRank(r) {
  return r < 1000 ? String(r) : (r / 1000).toFixed(r < 10000 ? 1 : 0) + "k";
}
function fmtProb(p) {
  return p >= 0.1 ? p.toFixed(2) : p >= 0.001 ? p.toFixed(3) : p.toExponential(0);
}
function rankStyle(r) {
  for (const [lim, bg, dim] of RANK_BG) if (r <= lim) return [bg, dim];
  return RANK_BG[RANK_BG.length - 1].slice(1);
}
function bandClass(layer) {
  const [w0, w1] = S.resp.workspace_band;
  if (layer > w1) return "row-motor";
  if (layer >= w0) return "row-workspace";
  return "row-sensory";
}
function layerLabel(layer) {
  const n = S.resp.n_layers;
  if (layer === n - 1) return `L${layer} · output`;
  return `L${layer}`;
}

// Top-k [strIdx...] + probs at (layer, pos) for the current mode; layer n-1 = model.
function topkAt(layer, pos) {
  const r = S.resp;
  if (layer === r.n_layers - 1) return [r.model.topk[pos], r.model.probs[pos]];
  const li = r.lens_layers.indexOf(layer);
  return [r[S.mode].topk[li][pos], r[S.mode].probs[li][pos]];
}
// Pinned-token rank at (layer, pos) for the current mode.
function rankAt(pin, layer, pos) {
  const r = S.resp;
  if (layer === r.n_layers - 1) return pin.ranks.model_ranks[pos];
  const li = r.lens_layers.indexOf(layer);
  return pin.ranks[S.mode === "jlens" ? "jlens_ranks" : "logit_lens_ranks"][li][pos];
}
function cellTitle(layer, pos) {
  const [tk, pr] = topkAt(layer, pos);
  const lines = tk.slice(0, 5).map((si, i) => `${i + 1}. ${JSON.stringify(S.resp.strings[si])}  ${fmtProb(pr[i])}`);
  return `${layerLabel(layer)} @ pos ${pos} ${JSON.stringify(S.resp.strings[S.resp.prompt_tokens[pos]])}\n${lines.join("\n")}`;
}

// ---------- warmup ----------

async function warmup() {
  const badge = $("status-badge");
  badge.className = "badge badge-warming";
  badge.textContent = "waking GPU…";
  const slow = setTimeout(() => { badge.textContent = "waking GPU… (~1 min, scales from zero)"; }, 3000);
  try {
    const r = await fetch(`${API}/warmup`);
    if (!r.ok) throw new Error(await r.text());
    clearTimeout(slow);
    badge.className = "badge badge-warm";
    badge.textContent = "GPU ready";
  } catch (e) {
    clearTimeout(slow);
    badge.className = "badge badge-error";
    badge.textContent = "backend unreachable";
    console.error(e);
  }
}

// ---------- analyze ----------

async function analyze() {
  const prompt = $("prompt").value;
  if (!prompt.trim()) return;
  const btn = $("analyze-btn");
  btn.disabled = true;
  btn.textContent = "computing…";
  $("error-banner").classList.add("hidden");
  try {
    const r = await fetch(`${API}/api/analyze`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt, top_k: 10 }),
    });
    if (!r.ok) {
      const detail = (await r.json().catch(() => ({}))).detail;
      throw new Error(detail || `request failed (${r.status})`);
    }
    S.resp = await r.json();
    S.prompt = prompt;
    S.pins = [];
    S.activePin = -1;
    S.gridMode = "argmax";
    const P = S.resp.prompt_tokens.length;
    const [w0, w1] = S.resp.workspace_band;
    S.sel = { layer: Math.round((w0 + w1) / 2), pos: P - 1 };
    $("results").classList.remove("hidden");
    $("examples").classList.add("hidden");
    renderAll();
    $("status-badge").className = "badge badge-warm";
    $("status-badge").textContent = "GPU ready";
  } catch (e) {
    const banner = $("error-banner");
    banner.textContent = `Analyze failed: ${e.message}`;
    banner.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze";
  }
}

// ---------- pinning ----------

async function pinToken({ tokenId = null, tokenStr = null }) {
  const existing = S.pins.findIndex((p) => tokenId !== null && p.tokenId === tokenId);
  if (existing >= 0) { S.activePin = existing; renderPins(); renderGrid(); renderCharts(); return; }
  const pin = {
    tokenId,
    text: tokenStr ?? S.resp.strings[S.resp.token_ids.indexOf(tokenId)] ?? "",
    color: PIN_COLORS[S.pins.length % PIN_COLORS.length],
    ranks: null,
    loading: true,
  };
  S.pins.push(pin);
  S.activePin = S.pins.length - 1;
  renderPins();
  try {
    const r = await fetch(`${API}/api/rank`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        request_id: S.resp.request_id,
        prompt: S.prompt,
        token_id: tokenId,
        token_str: tokenStr,
      }),
    });
    if (!r.ok) {
      const detail = (await r.json().catch(() => ({}))).detail;
      throw new Error(detail || `rank failed (${r.status})`);
    }
    const data = await r.json();
    pin.ranks = data;
    pin.tokenId = data.token_id;
    pin.text = data.token_text;
    pin.loading = false;
    $("rank-legend").classList.remove("hidden");
    document.querySelector('#grid-mode [data-grid="rank"]').disabled = false;
    if (S.gridMode === "argmax") setGridMode("rank");
    renderPins();
    renderGrid();
    renderCharts();
  } catch (e) {
    S.pins = S.pins.filter((p) => p !== pin);
    S.activePin = S.pins.length - 1;
    renderPins();
    const banner = $("error-banner");
    banner.textContent = `Pin failed: ${e.message}`;
    banner.classList.remove("hidden");
  }
}

function unpin(i) {
  S.pins.splice(i, 1);
  if (S.activePin >= S.pins.length) S.activePin = S.pins.length - 1;
  if (!S.pins.length) {
    $("rank-legend").classList.add("hidden");
    document.querySelector('#grid-mode [data-grid="rank"]').disabled = true;
    if (S.gridMode === "rank") setGridMode("argmax");
  }
  renderPins();
  renderGrid();
  renderCharts();
}

// ---------- rendering ----------

function renderAll() {
  renderPrediction();
  renderPins();
  renderMeta();
  renderGrid();
  renderByLayer();
  renderByPos();
  renderCharts();
}

function renderPrediction() {
  const r = S.resp;
  const P = r.prompt_tokens.length;
  $("prediction-list").innerHTML = r.model.topk[P - 1].slice(0, 5).map((si, i) =>
    `<button class="pred" data-si="${si}" title="pin this token">` +
    `<span>${esc(showTok(r.strings[si]))}</span><span class="p">${fmtProb(r.model.probs[P - 1][i])}</span></button>`
  ).join("");
}

function renderPins() {
  $("pins").innerHTML = S.pins.map((p, i) =>
    `<span class="pin-chip ${p.loading ? "loading" : ""} ${i === S.activePin ? "active-pin" : ""}"` +
    ` style="border-color:${p.color};color:${p.color}" data-pin="${i}">` +
    `${esc(showTok(p.text))}${p.loading ? " …" : ""}` +
    `<button class="x" data-unpin="${i}" aria-label="unpin">×</button></span>`
  ).join("");
}

function renderMeta() {
  const r = S.resp;
  const t = r.timing_ms;
  $("meta").textContent =
    `${r.prompt_tokens.length} tokens${r.truncated ? " (truncated!)" : ""}\n` +
    `forward ${t.forward} ms · readout ${t.readout} ms\nid ${r.request_id}`;
}

function setGridMode(m) {
  S.gridMode = m;
  document.querySelectorAll("#grid-mode .seg-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.grid === m));
  renderGrid();
}

function gridTitle() {
  const modeName = S.mode === "jlens" ? "J-lens" : "logit lens";
  if (S.gridMode === "rank" && S.pins[S.activePin]) {
    return `rank of ${JSON.stringify(S.pins[S.activePin].text)} · ${modeName} · layer × position`;
  }
  return `argmax · ${modeName} · layer × position`;
}

function renderGrid() {
  const r = S.resp;
  if (!r) return;
  $("grid-title").textContent = gridTitle();
  const P = r.prompt_tokens.length;
  const layers = [r.n_layers - 1, ...[...r.lens_layers].reverse()];
  const pin = S.gridMode === "rank" ? S.pins[S.activePin] : null;
  const usePin = pin && pin.ranks;
  $("grid").className = usePin ? "rank-mode" : "";

  const rows = layers.map((L) => {
    const cls = `${bandClass(L)}${L === S.sel.layer ? " sel-row" : ""}`;
    const cells = [];
    for (let p = 0; p < P; p++) {
      const selCol = p === S.sel.pos ? " sel-col" : "";
      const selected = L === S.sel.layer && p === S.sel.pos ? " selected" : "";
      const title = esc(cellTitle(L, p));
      if (usePin) {
        const rank = rankAt(pin, L, p);
        const [bg, dim] = rankStyle(rank);
        cells.push(`<td class="cell${selCol}${selected}${dim ? " rank-dim" : ""}" data-l="${L}" data-p="${p}"` +
          ` style="background-color:${bg}" title="${title}\nrank ${rank}">${fmtRank(rank)}</td>`);
      } else {
        const [tk] = topkAt(L, p);
        cells.push(`<td class="cell${selCol}${selected}" data-l="${L}" data-p="${p}" title="${title}">` +
          `${esc(showTok(r.strings[tk[0]]))}</td>`);
      }
    }
    return `<tr class="${cls}"><th class="layer-label">${layerLabel(L)}</th>${cells.join("")}</tr>`;
  });

  const axis = r.prompt_tokens.map((si, p) =>
    `<td class="axis-tok${p === S.sel.pos ? " sel-col" : ""}" data-axis="${p}" title="position ${p}">` +
    `${esc(showTok(r.strings[si]))}</td>`).join("");
  rows.push(`<tr class="axis-row"><th class="layer-label">prompt →</th>${axis}</tr>`);
  $("grid").innerHTML = rows.join("");
}

function renderByLayer() {
  const r = S.resp;
  const pos = S.sel.pos;
  $("by-layer-title").textContent =
    `by layer · pos ${pos} ${JSON.stringify(r.strings[r.prompt_tokens[pos]])}`;
  const layers = [r.n_layers - 1, ...[...r.lens_layers].reverse()];
  $("by-layer").innerHTML = layers.map((L) => {
    const [tk, pr] = topkAt(L, pos);
    const band = bandClass(L) === "row-workspace" ? "in-workspace" : bandClass(L) === "row-motor" ? "in-motor" : "";
    const toks = tk.map((si, i) =>
      `<button class="tok${i === 0 ? " rank1" : ""}" data-si="${si}" title="pin ${esc(JSON.stringify(r.strings[si]))}">` +
      `${esc(showTok(r.strings[si]))}<span class="p">${fmtProb(pr[i])}</span></button>`).join("");
    return `<div class="stack-row ${band}${L === S.sel.layer ? " is-selected" : ""}">` +
      `<span class="rl" data-layer="${L}" title="select this layer">${layerLabel(L)}</span>` +
      `<span class="toks">${toks}</span></div>`;
  }).join("");
  const sel = $("by-layer").querySelector(".is-selected");
  if (sel) sel.scrollIntoView({ block: "nearest" });
}

function renderByPos() {
  const r = S.resp;
  const L = S.sel.layer;
  const P = r.prompt_tokens.length;
  $("by-pos-title").textContent = `by position · ${layerLabel(L)}`;
  const rows = [];
  for (let p = 0; p < P; p++) {
    const [tk, pr] = topkAt(L, p);
    const toks = tk.map((si, i) =>
      `<button class="tok${i === 0 ? " rank1" : ""}" data-si="${si}" title="pin ${esc(JSON.stringify(r.strings[si]))}">` +
      `${esc(showTok(r.strings[si]))}<span class="p">${fmtProb(pr[i])}</span></button>`).join("");
    rows.push(`<div class="stack-row${p === S.sel.pos ? " is-selected" : ""}">` +
      `<span class="rl" data-pos="${p}" title="select this position">${p} ${esc(showTok(r.strings[r.prompt_tokens[p]]).slice(0, 8))}</span>` +
      `<span class="toks">${toks}</span></div>`);
  }
  $("by-pos").innerHTML = rows.join("");
  const sel = $("by-pos").querySelector(".is-selected");
  if (sel) sel.scrollIntoView({ block: "nearest" });
}

// ---------- charts ----------

const CH = { w: 460, h: 170, l: 34, r: 8, t: 10, b: 22 };
const RMAX = Math.log10(160000);

function chartFrame(xTicks, xLabel, bandX) {
  const iw = CH.w - CH.l - CH.r, ih = CH.h - CH.t - CH.b;
  const parts = [];
  if (bandX) {
    parts.push(`<rect class="bandrect" x="${bandX[0]}" y="${CH.t}" width="${bandX[1] - bandX[0]}" height="${ih}"/>`);
  }
  for (const rv of [1, 10, 100, 1000, 10000, 100000]) {
    const y = CH.t + ih - (Math.log10(rv) / RMAX) * ih;
    parts.push(`<line class="gridline" x1="${CH.l}" y1="${y}" x2="${CH.w - CH.r}" y2="${y}"/>`);
    parts.push(`<text x="${CH.l - 4}" y="${y + 3}" text-anchor="end">${rv >= 1000 ? rv / 1000 + "k" : rv}</text>`);
  }
  for (const [xv, lbl] of xTicks) {
    if (xv > CH.w - 60) continue; // keep clear of the axis label
    parts.push(`<text x="${xv}" y="${CH.h - 8}" text-anchor="middle">${lbl}</text>`);
  }
  parts.push(`<text x="${CH.w - CH.r}" y="${CH.h - 8}" text-anchor="end">${xLabel}</text>`);
  return parts;
}

function rankY(r) {
  const ih = CH.h - CH.t - CH.b;
  return CH.t + ih - (Math.log10(Math.max(r, 1)) / RMAX) * ih;
}

function renderCharts() {
  const r = S.resp;
  if (!r) return;
  const ready = S.pins.filter((p) => p.ranks);
  const iw = CH.w - CH.l - CH.r;

  // rank vs layer at selected position
  {
    const n = r.n_layers;
    const x = (L) => CH.l + (L / (n - 1)) * iw;
    const [w0, w1] = r.workspace_band;
    const ticks = [0, 8, 16, 24, 31].map((L) => [x(L), `L${L}`]);
    const parts = chartFrame(ticks, "layer →", [x(w0), x(w1)]);
    for (const pin of ready) {
      const pts = r.lens_layers.map((L, li) => `${x(L)},${rankY(rankAt(pin, L, S.sel.pos))}`);
      pts.push(`${x(n - 1)},${rankY(pin.ranks.model_ranks[S.sel.pos])}`);
      parts.push(`<polyline class="rankline" stroke="${pin.color}" points="${pts.join(" ")}"/>`);
    }
    if (!ready.length) parts.push(`<text class="empty-note" x="${CH.l + 10}" y="${CH.t + 16}">pin a token to trace its rank across layers</text>`);
    $("chart-layer").innerHTML = parts.join("");
  }

  // rank vs position at selected layer
  {
    const P = r.prompt_tokens.length;
    const x = (p) => CH.l + (P > 1 ? (p / (P - 1)) * iw : iw / 2);
    const step = Math.max(1, Math.round(P / 6));
    const ticks = [];
    for (let p = 0; p < P; p += step) ticks.push([x(p), String(p)]);
    const parts = chartFrame(ticks, "position →", null);
    for (const pin of ready) {
      const pts = [];
      for (let p = 0; p < P; p++) pts.push(`${x(p)},${rankY(rankAt(pin, S.sel.layer, p))}`);
      parts.push(`<polyline class="rankline" stroke="${pin.color}" points="${pts.join(" ")}"/>`);
    }
    if (!ready.length) parts.push(`<text class="empty-note" x="${CH.l + 10}" y="${CH.t + 16}">pin a token to trace its rank across positions</text>`);
    $("chart-pos").innerHTML = parts.join("");
  }
}

// ---------- selection ----------

function select({ layer = null, pos = null }) {
  if (layer !== null) S.sel.layer = layer;
  if (pos !== null) S.sel.pos = pos;
  renderGrid();
  renderByLayer();
  renderByPos();
  renderCharts();
}

// ---------- events ----------

$("analyze-btn").addEventListener("click", analyze);
$("prompt").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) analyze();
});

document.querySelectorAll("#examples .example").forEach((b) =>
  b.addEventListener("click", () => { $("prompt").value = b.textContent; analyze(); }));

document.querySelectorAll("#mode-toggle .seg-btn").forEach((b) =>
  b.addEventListener("click", () => {
    S.mode = b.dataset.mode;
    document.querySelectorAll("#mode-toggle .seg-btn").forEach((x) =>
      x.classList.toggle("active", x === b));
    if (S.resp) { renderGrid(); renderByLayer(); renderByPos(); renderCharts(); }
  }));

document.querySelectorAll("#grid-mode .seg-btn").forEach((b) =>
  b.addEventListener("click", () => { if (!b.disabled) setGridMode(b.dataset.grid); }));

$("grid").addEventListener("click", (e) => {
  const cell = e.target.closest("td.cell");
  if (cell) return select({ layer: +cell.dataset.l, pos: +cell.dataset.p });
  const axis = e.target.closest("td.axis-tok");
  if (axis) return select({ pos: +axis.dataset.axis });
});

for (const id of ["by-layer", "by-pos"]) {
  $(id).addEventListener("click", (e) => {
    const tok = e.target.closest(".tok");
    if (tok) return pinToken({ tokenId: S.resp.token_ids[+tok.dataset.si] });
    const rl = e.target.closest(".rl");
    if (rl && rl.dataset.layer !== undefined) return select({ layer: +rl.dataset.layer });
    if (rl && rl.dataset.pos !== undefined) return select({ pos: +rl.dataset.pos });
  });
}

$("prediction-list").addEventListener("click", (e) => {
  const pred = e.target.closest(".pred");
  if (pred) pinToken({ tokenId: S.resp.token_ids[+pred.dataset.si] });
});

$("pins").addEventListener("click", (e) => {
  const x = e.target.closest("[data-unpin]");
  if (x) return unpin(+x.dataset.unpin);
  const chip = e.target.closest("[data-pin]");
  if (chip) {
    S.activePin = +chip.dataset.pin;
    renderPins();
    if (S.gridMode === "rank") renderGrid();
  }
});

$("pin-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const v = $("pin-input").value;
  if (!v || !S.resp) return;
  $("pin-input").value = "";
  pinToken({ tokenStr: v });
});

warmup();
