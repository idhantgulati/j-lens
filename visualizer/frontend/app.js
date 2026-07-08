/* J-Lens Visualizer — all state, rendering, and API calls. */

import { inject } from "@vercel/analytics";

inject();

const API = window.JLENS_API;

// Pin palettes tuned per theme so chips and chart lines stay readable.
const PIN_PALETTES = {
  light: ["#b3561b", "#2e7d52", "#2d5e9d", "#8c4f96", "#c05746", "#5c6b48"],
  dark:  ["#d9a441", "#7fb886", "#8ab4dd", "#c79ecf", "#d98079", "#a8b48a"],
};
const RANK_CLASSES = [
  [1, "rank-r1"], [10, "rank-r10"], [100, "rank-r100"], [1000, "rank-r1k"], [Infinity, "rank-r10k"],
];

const S = {
  resp: null,
  params: null,        // prompt params of the current result (for rank/intervene reuse)
  mode: "jlens",       // "jlens" | "logit_lens"
  gridMode: "argmax",  // "argmax" | "rank"
  chat: "raw",         // "raw" | "chat"
  sel: { layer: 24, pos: 0 },
  pins: [],            // {tokenId, text, colorIdx, ranks, loading}
  activePin: -1,
  ready: false,
  ivKind: "swap",      // "swap" | "steer"
};

// Prompt parameters as currently configured in the controls.
function promptParams() {
  return {
    prompt: $("prompt").value,
    chat: S.chat === "chat",
    system_prompt: $("system-prompt").value || null,
    prefill: $("prefill").value || null,
    max_new_tokens: +$("gen-len").value,
  };
}

const $ = (id) => document.getElementById(id);

// ---------- theme ----------

function theme() { return document.documentElement.dataset.theme === "dark" ? "dark" : "light"; }
function pinColor(p) { return PIN_PALETTES[theme()][p.colorIdx % 6]; }

$("theme-toggle").addEventListener("click", () => {
  const next = theme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("jlens-theme", next);
  if (S.resp) { renderPins(); renderCharts(); }
});

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
  return p < 0.005 ? "<.01" : p.toFixed(2);
}
function rankClass(r) {
  for (const [lim, cls] of RANK_CLASSES) if (r <= lim) return cls;
  return "rank-r10k";
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

// ---------- status + cold start ----------

function setStatus(state, text) {
  const el = $("status");
  el.className = `status is-${state}`;
  $("status-text").textContent = text;
}

function setPatience(msg) {
  const note = $("patience-note");
  if (!msg) { note.classList.add("hidden"); return; }
  note.textContent = msg;
  note.classList.remove("hidden");
}

async function warmup() {
  setStatus("warming", "starting up");
  const slow = setTimeout(() => {
    setStatus("warming", "warming up");
    setPatience("The GPU behind this page sleeps when nobody is around, and it takes about a minute to wake. Feel free to write your prompt in the meantime — it will run as soon as things are ready.");
  }, 2500);
  try {
    const r = await fetch(`${API}/warmup`);
    if (!r.ok) throw new Error(await r.text());
    clearTimeout(slow);
    S.ready = true;
    setStatus("ready", "ready");
    setPatience(null);
  } catch (e) {
    clearTimeout(slow);
    setStatus("error", "backend unreachable");
    setPatience("The backend could not be reached. Reloading the page usually fixes it; if not, the demo may be down for the moment.");
    console.error(e);
  }
}

// ---------- analyze ----------

async function analyze() {
  const params = promptParams();
  if (!params.prompt.trim()) return;
  const btn = $("analyze-btn");
  btn.disabled = true;
  btn.textContent = "reading…";
  $("error-banner").classList.add("hidden");
  const slow = setTimeout(() => {
    if (!S.ready) setPatience("Still waking the GPU — your prompt is queued and will run the moment it's up.");
  }, 2500);
  const t0 = performance.now();
  try {
    const r = await fetch(`${API}/api/analyze`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ ...params, top_k: 10 }),
    });
    if (!r.ok) {
      const detail = (await r.json().catch(() => ({}))).detail;
      throw new Error(detail || `request failed (${r.status})`);
    }
    S.resp = await r.json();
    S.resp.client_ms = Math.round(performance.now() - t0);
    S.params = params;
    S.pins = [];
    S.activePin = -1;
    S.gridMode = "argmax";
    const P = S.resp.prompt_tokens.length;
    const [w0, w1] = S.resp.workspace_band;
    S.sel = { layer: Math.round((w0 + w1) / 2), pos: P - 1 };
    $("results").classList.remove("hidden");
    $("examples").classList.add("hidden");
    renderAll();
    S.ready = true;
    setStatus("ready", "ready");
  } catch (e) {
    const banner = $("error-banner");
    banner.textContent = `Analyze failed: ${e.message}`;
    banner.classList.remove("hidden");
  } finally {
    clearTimeout(slow);
    setPatience(null);
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
    colorIdx: S.pins.length,
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
        ...S.params,
        request_id: S.resp.request_id,
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
  renderCompletion();
  renderPrediction();
  renderPins();
  renderMeta();
  renderGrid();
  renderJSpace();
  renderByLayer();
  renderByPos();
  renderCharts();
}

function isGen(p) {
  return S.resp.gen_start !== undefined && p >= S.resp.gen_start;
}

// Neuronpedia-style transcript: chat turns as bubbles (user right, assistant
// left), template tokens in faint mono, ↵ for newlines, every token clickable.
function renderCompletion() {
  const r = S.resp;
  const card = $("completion-card");
  const isChat = !!(S.params && S.params.chat);
  if (!isChat && !r.completion) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");

  const isSpecial = (s) => /^<\|[^|]*\|>$/.test(s) || /^<\/?think>$/.test(s.trim());
  const tokHtml = (p) => {
    const s = r.strings[r.prompt_tokens[p]];
    const prev = p > 0 ? r.strings[r.prompt_tokens[p - 1]] : "";
    const cls = ["tr-tok"];
    if (isSpecial(s) || prev === "<|im_start|>") cls.push("tr-special"); // roles ride the im_start
    if (isGen(p)) cls.push("tr-gen");
    if (p === S.sel.pos) cls.push("tr-sel");
    const breaks = "<br>".repeat((s.match(/\n/g) || []).length);
    return `<button class="${cls.join(" ")}" data-pos="${p}" title="position ${p}${isGen(p) ? " (generated)" : ""}">` +
      `${esc(showTok(s))}</button>${breaks}`;
  };

  const bubbles = [];
  let cur = null;
  const close = () => { if (cur) { bubbles.push(`<div class="tr-bubble tr-${cur.role}">${cur.parts.join("")}</div>`); cur = null; } };
  for (let p = 0; p < r.prompt_tokens.length; p++) {
    const s = r.strings[r.prompt_tokens[p]];
    if (s === "<|im_start|>") {
      close();
      const role = (r.strings[r.prompt_tokens[p + 1]] || "").trim();
      cur = { role: ["user", "assistant", "system"].includes(role) ? role : "assistant", parts: [] };
    }
    if (!cur) cur = { role: "assistant", parts: [] }; // raw mode: one left-aligned bubble
    cur.parts.push(tokHtml(p));
  }
  close();
  $("completion").innerHTML = bubbles.join("");
  $("completion-label").innerHTML = isChat
    ? 'transcript <span class="hint">&mdash; the exact token sequence the lens reads; click any token to inspect it below</span>'
    : 'prompt + model output <span class="hint">&mdash; greedy continuation highlighted; click any token to inspect it below</span>';
}

function renderPrediction() {
  const r = S.resp;
  const P = r.prompt_tokens.length;
  $("prediction-label").textContent = r.completion
    ? "model prediction · token after the output" : "model prediction · next token";
  $("prediction-list").innerHTML = r.model.topk[P - 1].slice(0, 5).map((si, i) =>
    `<button class="pred" data-si="${si}" title="pin this token">` +
    `<span>${esc(showTok(r.strings[si]))}</span><span class="p">${fmtProb(r.model.probs[P - 1][i])}</span></button>`
  ).join("");
}

// ---------- J-Space aggregate (neuronpedia-style count view) ----------

function renderJSpace() {
  const r = S.resp;
  if (!r) return;
  const data = r[S.mode];
  const L = r.lens_layers.length;
  const P = r.prompt_tokens.length;
  const [w0, w1] = r.workspace_band;
  const stats = new Map(); // strIdx -> {total, perLayer: Int32Array}
  for (let li = 0; li < L; li++) {
    if (r.lens_layers[li] < w0 || r.lens_layers[li] > w1) continue; // workspace cells only
    const layerTop = data.topk[li];
    for (let p = 0; p < P; p++) {
      for (const si of layerTop[p]) {
        let s = stats.get(si);
        if (!s) { s = { total: 0, perLayer: new Int32Array(L) }; stats.set(si, s); }
        s.total++;
        s.perLayer[li]++;
      }
    }
  }
  // Special/template and pure-punctuation tokens hold real workspace info (turn
  // structure, formatting state) but drown out content words — shown on demand.
  const showAll = $("js-show-all").checked;
  const isContent = (si) => /\p{L}|\p{N}/u.test(r.strings[si]) && !/<.*>/.test(r.strings[si]);
  const rows = [...stats.entries()]
    .filter(([si]) => showAll || isContent(si))
    .sort((a, b) => b[1].total - a[1].total).slice(0, 40);
  const maxCell = Math.max(1, ...rows.map(([, s]) => Math.max(...s.perLayer)));
  $("jspace").innerHTML = rows.map(([si, s]) => {
    const strip = Array.from(s.perLayer, (c, li) =>
      `<i style="opacity:${c ? (0.15 + 0.85 * c / maxCell).toFixed(2) : 0}" title="L${r.lens_layers[li]}: ${c}×"></i>`
    ).join("");
    return `<div class="js-row">` +
      `<button class="tok js-tok" data-si="${si}" title="pin ${esc(JSON.stringify(r.strings[si]))}">${esc(showTok(r.strings[si]))}</button>` +
      `<span class="js-count">${s.total}</span>` +
      `<span class="js-strip" aria-hidden="true">${strip}</span>` +
      `<span class="js-actions">` +
      `<button class="mini-btn" data-iv="steer" data-si="${si}">steer</button>` +
      `<button class="mini-btn" data-iv="swap" data-si="${si}">swap</button>` +
      `</span></div>`;
  }).join("") || `<p class="panel-hint js-empty">no readout data</p>`;
}

function renderPins() {
  $("pins").innerHTML = S.pins.map((p, i) =>
    `<span class="pin-chip ${p.loading ? "loading" : ""} ${i === S.activePin ? "active-pin" : ""}"` +
    ` style="border-color:${pinColor(p)};color:${pinColor(p)}" data-pin="${i}">` +
    `${esc(showTok(p.text))}${p.loading ? " …" : ""}` +
    `<button class="x" data-unpin="${i}" aria-label="unpin">×</button></span>`
  ).join("");
}

function renderMeta() {
  const r = S.resp;
  const secs = ((r.client_ms ?? r.timing_ms.total) / 1000).toFixed(1);
  $("meta").textContent =
    `${r.prompt_tokens.length} tokens${r.truncated ? " (truncated)" : ""} · ${secs} s`;
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
        cells.push(`<td class="cell ${rankClass(rank)}${selCol}${selected}" data-l="${L}" data-p="${p}"` +
          ` title="${title}\nrank ${rank}">${fmtRank(rank)}</td>`);
      } else {
        const [tk] = topkAt(L, p);
        cells.push(`<td class="cell${selCol}${selected}" data-l="${L}" data-p="${p}" title="${title}">` +
          `${esc(showTok(r.strings[tk[0]]))}</td>`);
      }
    }
    return `<tr class="${cls}"><th class="layer-label">${layerLabel(L)}</th>${cells.join("")}</tr>`;
  });

  const axis = r.prompt_tokens.map((si, p) =>
    `<td class="axis-tok${p === S.sel.pos ? " sel-col" : ""}${isGen(p) ? " is-gen" : ""}" data-axis="${p}"` +
    ` title="position ${p}${isGen(p) ? " (generated)" : ""}">` +
    `${esc(showTok(r.strings[si]))}</td>`).join("");
  rows.push(`<tr class="axis-row"><th class="layer-label">${r.completion ? "prompt + output →" : "prompt →"}</th>${axis}</tr>`);
  $("grid").style.minWidth = `${96 + 80 * P}px`;
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
    rows.push(`<div class="stack-row${p === S.sel.pos ? " is-selected" : ""}${isGen(p) ? " is-gen-row" : ""}">` +
      `<span class="rl" data-pos="${p}" title="select this position${isGen(p) ? " (generated token)" : ""}">${p} ${esc(showTok(r.strings[r.prompt_tokens[p]]).slice(0, 8))}</span>` +
      `<span class="toks">${toks}</span></div>`);
  }
  $("by-pos").innerHTML = rows.join("");
  const sel = $("by-pos").querySelector(".is-selected");
  if (sel) sel.scrollIntoView({ block: "nearest" });
}

// ---------- charts ----------

const CH = { h: 170, l: 34, r: 8, t: 10, b: 22 };
const RMAX = Math.log10(160000);

// viewBox width tracks the rendered width so nothing stretches.
function chartWidth(svg) {
  const w = Math.max(Math.round(svg.clientWidth) || 460, 240);
  svg.setAttribute("viewBox", `0 0 ${w} ${CH.h}`);
  return w;
}

function chartFrame(w, xTicks, xLabel, bandX) {
  const ih = CH.h - CH.t - CH.b;
  const parts = [];
  if (bandX) {
    parts.push(`<rect class="bandrect" x="${bandX[0]}" y="${CH.t}" width="${bandX[1] - bandX[0]}" height="${ih}"/>`);
  }
  for (const rv of [1, 10, 100, 1000, 10000, 100000]) {
    const y = CH.t + ih - (Math.log10(rv) / RMAX) * ih;
    parts.push(`<line class="gridline" x1="${CH.l}" y1="${y}" x2="${w - CH.r}" y2="${y}"/>`);
    parts.push(`<text x="${CH.l - 4}" y="${y + 3}" text-anchor="end">${rv >= 1000 ? rv / 1000 + "k" : rv}</text>`);
  }
  for (const [xv, lbl] of xTicks) {
    if (xv > w - 60) continue; // keep clear of the axis label
    parts.push(`<text x="${xv}" y="${CH.h - 8}" text-anchor="middle">${lbl}</text>`);
  }
  parts.push(`<text x="${w - CH.r}" y="${CH.h - 8}" text-anchor="end">${xLabel}</text>`);
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

  // rank vs layer at selected position
  {
    const w = chartWidth($("chart-layer"));
    const iw = w - CH.l - CH.r;
    const n = r.n_layers;
    const x = (L) => CH.l + (L / (n - 1)) * iw;
    const [w0, w1] = r.workspace_band;
    const ticks = [0, 8, 16, 24, 31].map((L) => [x(L), `L${L}`]);
    const parts = chartFrame(w, ticks, "layer →", [x(w0), x(w1)]);
    for (const pin of ready) {
      const pts = r.lens_layers.map((L, li) => `${x(L)},${rankY(rankAt(pin, L, S.sel.pos))}`);
      pts.push(`${x(n - 1)},${rankY(pin.ranks.model_ranks[S.sel.pos])}`);
      parts.push(`<polyline class="rankline" stroke="${pinColor(pin)}" points="${pts.join(" ")}"/>`);
    }
    if (!ready.length) parts.push(`<text class="empty-note" x="${CH.l + 10}" y="${CH.t + 16}">pin a token to trace its rank across layers</text>`);
    $("chart-layer").innerHTML = parts.join("");
  }

  // rank vs position at selected layer
  {
    const w = chartWidth($("chart-pos"));
    const iw = w - CH.l - CH.r;
    const P = r.prompt_tokens.length;
    const x = (p) => CH.l + (P > 1 ? (p / (P - 1)) * iw : iw / 2);
    const step = Math.max(1, Math.round(P / 6));
    const ticks = [];
    for (let p = 0; p < P; p += step) ticks.push([x(p), String(p)]);
    const parts = chartFrame(w, ticks, "position →", null);
    for (const pin of ready) {
      const pts = [];
      for (let p = 0; p < P; p++) pts.push(`${x(p)},${rankY(rankAt(pin, S.sel.layer, p))}`);
      parts.push(`<polyline class="rankline" stroke="${pinColor(pin)}" points="${pts.join(" ")}"/>`);
    }
    if (!ready.length) parts.push(`<text class="empty-note" x="${CH.l + 10}" y="${CH.t + 16}">pin a token to trace its rank across positions</text>`);
    $("chart-pos").innerHTML = parts.join("");
  }
}

// ---------- selection ----------

function select({ layer = null, pos = null }) {
  if (layer !== null) S.sel.layer = layer;
  if (pos !== null) S.sel.pos = pos;
  renderCompletion(); // keep the transcript's selected-token highlight in sync
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
  b.addEventListener("click", () => {
    $("prompt").value = b.textContent;
    const chat = b.dataset.chat === "chat" ? "chat" : "raw";
    document.querySelector(`#chat-toggle [data-chat="${chat}"]`).click();
    analyze();
  }));

document.querySelectorAll("#mode-toggle .seg-btn").forEach((b) =>
  b.addEventListener("click", () => {
    S.mode = b.dataset.mode;
    document.querySelectorAll("#mode-toggle .seg-btn").forEach((x) =>
      x.classList.toggle("active", x === b));
    if (S.resp) { renderGrid(); renderJSpace(); renderByLayer(); renderByPos(); renderCharts(); }
  }));

// ---------- run config (chat mode, generation length, advanced) ----------

document.querySelectorAll("#chat-toggle .seg-btn").forEach((b) =>
  b.addEventListener("click", () => {
    S.chat = b.dataset.chat;
    document.querySelectorAll("#chat-toggle .seg-btn").forEach((x) =>
      x.classList.toggle("active", x === b));
    document.body.classList.toggle("chat-mode", S.chat === "chat");
  }));

$("advanced-toggle").addEventListener("click", () => {
  const adv = $("advanced");
  const open = adv.classList.toggle("hidden");
  $("advanced-toggle").textContent = open ? "more options" : "fewer options";
  $("advanced-toggle").setAttribute("aria-expanded", String(!open));
});

// ---------- interventions ----------

function openIntervene(kind, tokenText) {
  S.ivKind = kind;
  document.querySelectorAll("#intervene-kind .seg-btn").forEach((x) =>
    x.classList.toggle("active", x.dataset.kind === kind));
  $("iv-target-wrap").classList.toggle("hidden", kind !== "swap");
  $("iv-strength-wrap").classList.toggle("hidden", kind !== "steer");
  $("intervene-title").textContent = kind === "swap"
    ? "swap · exchange two workspace concepts" : "steer · inject a concept into the workspace";
  if (tokenText !== undefined) $("iv-source").value = tokenText.trim();
  $("iv-results").classList.add("hidden");
  $("intervene-panel").classList.remove("hidden");
  $("intervene-panel").scrollIntoView({ block: "nearest", behavior: "smooth" });
  (kind === "swap" && $("iv-source").value ? $("iv-target") : $("iv-source")).focus();
}

async function runIntervention() {
  const btn = $("iv-run");
  const source = $("iv-source").value.trim();
  if (!source || !S.params) return;
  if (S.ivKind === "swap" && !$("iv-target").value.trim()) { $("iv-target").focus(); return; }
  btn.disabled = true;
  btn.textContent = "running…";
  $("error-banner").classList.add("hidden");
  try {
    const body = {
      ...S.params,
      max_new_tokens: Math.max(16, S.params.max_new_tokens || 0),
      kind: S.ivKind,
      source,
      target: S.ivKind === "swap" ? $("iv-target").value.trim() : null,
      strength: +$("iv-strength").value,
      layer_lo: +$("iv-lo").value,
      layer_hi: +$("iv-hi").value,
    };
    const r = await fetch(`${API}/api/intervene`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const detail = (await r.json().catch(() => ({}))).detail;
      throw new Error(detail || `intervention failed (${r.status})`);
    }
    const d = await r.json();
    const fmtTop = (top) => top.map((t, i) =>
      `<span class="tok${i === 0 ? " rank1" : ""}">${esc(showTok(t.token))}<span class="p">${fmtProb(t.prob)}</span></span>`).join("");
    const fmtSide = (side) =>
      `<p class="iv-gen">${esc(side.text) || "<i>(empty)</i>"}</p><div class="iv-top">${fmtTop(side.top)}</div>`;
    $("iv-default").innerHTML = fmtSide(d.default);
    $("iv-modified").innerHTML = fmtSide(d.modified);
    $("iv-mod-label").textContent = d.kind === "swap"
      ? `swapped ${d.pairs.map(([a, b]) => `${JSON.stringify(a)} → ${JSON.stringify(b)}`).join(", ")} · L${d.layers[0]}–${d.layers[1]}`
      : `steered ${JSON.stringify(d.token)} @ ${d.strength} · L${d.layers[0]}–${d.layers[1]}`;
    $("iv-results").classList.remove("hidden");
  } catch (e) {
    const banner = $("error-banner");
    banner.textContent = `Intervention failed: ${e.message}`;
    banner.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.textContent = "Run intervention";
  }
}

document.querySelectorAll("#intervene-kind .seg-btn").forEach((b) =>
  b.addEventListener("click", () => openIntervene(b.dataset.kind)));
$("intervene-close").addEventListener("click", () => $("intervene-panel").classList.add("hidden"));
$("iv-run").addEventListener("click", runIntervention);
$("iv-strength").addEventListener("input", () => { $("iv-strength-val").textContent = $("iv-strength").value; });
for (const id of ["iv-source", "iv-target"]) {
  $(id).addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); runIntervention(); } });
}

$("jspace").addEventListener("click", (e) => {
  const iv = e.target.closest(".mini-btn");
  if (iv) return openIntervene(iv.dataset.iv, S.resp.strings[+iv.dataset.si]);
  const tok = e.target.closest(".js-tok");
  if (tok) return pinToken({ tokenId: S.resp.token_ids[+tok.dataset.si] });
});

$("completion").addEventListener("click", (e) => {
  const t = e.target.closest(".tr-tok");
  if (t) select({ pos: +t.dataset.pos });
});

$("js-show-all").addEventListener("change", () => { if (S.resp) renderJSpace(); });

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

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (S.resp) renderCharts(); }, 150);
});

$("pin-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const v = $("pin-input").value;
  if (!v || !S.resp) return;
  $("pin-input").value = "";
  pinToken({ tokenStr: v });
});

warmup();
