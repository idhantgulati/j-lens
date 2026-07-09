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
const LIMITS = { max_sequence: 320, max_new: 128, max_chars: 4000 };

const S = {
  resp: null,
  params: null,        // prompt params of the current result (for rank/intervene reuse)
  mode: "jlens",       // "jlens" | "logit_lens"
  gridMode: "argmax",  // "argmax" | "rank"
  chat: "raw",         // "raw" | "chat"
  chatTurns: [],       // prior {role, content} turns in chat mode
  sel: { layer: 24, pos: 0 },
  pins: [],            // {tokenId, text, colorIdx, ranks, loading}
  activePin: -1,
  ready: false,
  ivKind: "swap",      // "swap" | "steer"
};

// Prompt parameters as currently configured in the controls.
function promptParams() {
  const p = {
    prompt: $("prompt").value,
    chat: S.chat === "chat",
    system_prompt: $("system-prompt").value || null,
    prefill: $("prefill").value || null,
    max_new_tokens: +$("gen-len").value,
  };
  if (S.chat === "chat" && S.chatTurns.length) {
    p.messages = S.chatTurns.map((t) => ({ role: t.role, content: t.content }));
  }
  return p;
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
// Exact integer for hover tooltips (grid cells keep the compact "24k" form).
function fmtRankExact(r) {
  return String(r);
}
function fmtProb(p) {
  return p < 0.005 ? "<.01" : p.toFixed(2);
}
// Leading top-k rows when the pin sits outside the list. Keep this short so the
// pin±1 window + both ··· markers stay on-screen (a head of 10 was clipping).
function tipHeadLimit(rank) {
  if (rank == null) return 5;
  if (rank >= 1000) return 5;
  if (rank >= 100) return 6;
  return 5; // ranks 11–99
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
function scrollStackToRow(stackEl, rowEl) {
  if (!stackEl || !rowEl) return;
  const rowTop = rowEl.offsetTop;
  const rowBottom = rowTop + rowEl.offsetHeight;
  const viewTop = stackEl.scrollTop;
  const viewBottom = viewTop + stackEl.clientHeight;
  if (rowTop < viewTop) stackEl.scrollTop = rowTop;
  else if (rowBottom > viewBottom) stackEl.scrollTop = rowBottom - stackEl.clientHeight;
}

// Pinned-token rank at (layer, pos) for the current mode.
function rankAt(pin, layer, pos) {
  const r = S.resp;
  if (layer === r.n_layers - 1) return pin.ranks.model_ranks[pos];
  const li = r.lens_layers.indexOf(layer);
  return pin.ranks[S.mode === "jlens" ? "jlens_ranks" : "logit_lens_ranks"][li][pos];
}
// ±neighbor window from /api/rank (token strings + probs around the pin).
function neighborsAt(pin, layer, pos) {
  if (!pin?.ranks) return null;
  const r = S.resp;
  if (layer === r.n_layers - 1) return pin.ranks.model_neighbors?.[pos] ?? null;
  const li = r.lens_layers.indexOf(layer);
  const key = S.mode === "jlens" ? "jlens_neighbors" : "logit_lens_neighbors";
  return pin.ranks[key]?.[li]?.[pos] ?? null;
}
function pinStateForSi(si) {
  if (!S.resp) return null;
  const tokenId = S.resp.token_ids[si];
  const i = S.pins.findIndex((p) => p.tokenId === tokenId && p.ranks);
  if (i < 0) return null;
  return { index: i, hidden: !!S.pins[i].chartHidden };
}
function tracedMarkup(si) {
  const ps = pinStateForSi(si);
  if (!ps) return { cls: "", style: "" };
  const color = pinColor(S.pins[ps.index]);
  return {
    cls: ps.hidden ? " is-traced-hidden" : " is-traced",
    style: ` style="--trace-color:${color}"`,
  };
}
function traceTipFoot(si) {
  const ps = pinStateForSi(si);
  if (!ps) return "click to trace on chart";
  return ps.hidden ? "click to show on chart" : "click to hide from chart";
}
function tipAccentFor(el) {
  if (el.matches(".chart-point")) {
    const pin = S.pins[+el.dataset.pin];
    return pin ? pinColor(pin) : null;
  }
  if (el.matches(".stack-table td.tok-cell, .pred, .js-tok, .js-row")) {
    const si = +el.dataset.si;
    if (Number.isNaN(si)) return null;
    const ps = pinStateForSi(si);
    if (ps && !ps.hidden) return pinColor(S.pins[ps.index]);
  }
  if (el.matches(".pin-chip")) {
    const pin = S.pins[+el.dataset.pin];
    return pin && !pin.chartHidden ? pinColor(pin) : null;
  }
  return null;
}

function findPin({ tokenId = null, tokenStr = null }) {
  if (tokenId !== null) {
    const byId = S.pins.findIndex((p) => p.tokenId === tokenId);
    if (byId >= 0) return byId;
  }
  if (tokenStr) {
    const byText = S.pins.findIndex((p) => p.text === tokenStr);
    if (byText >= 0) return byText;
  }
  return -1;
}

// ---------- rich hover tooltips ----------

const tipEl = (() => {
  const el = document.createElement("div");
  el.id = "jlens-tip";
  el.className = "jlens-tip hidden";
  el.setAttribute("role", "tooltip");
  document.body.appendChild(el);
  return el;
})();
let tipHideTimer = null;
let tipTarget = null;

function bandName(layer) {
  const [w0, w1] = S.resp.workspace_band;
  if (layer > w1) return "motor";
  if (layer >= w0) return "workspace";
  return "sensory";
}
function bandBadge(layer) {
  const b = bandName(layer);
  return `<span class="tip-badge is-${b}">${b}</span>`;
}
function modeLabel() {
  return S.mode === "jlens" ? "J-lens" : "logit lens";
}
function tipTopkList(tk, pr, { limit = 10, highlightSi = null, trailingGap = false } = {}) {
  const r = S.resp;
  const rows = tk.slice(0, limit).map((si, i) => {
    const top = i === 0 ? " is-top" : "";
    const hi = highlightSi === si ? " is-pin" : "";
    return `<li class="${(top + hi).trim()}">` +
      `<span class="tip-rank">${i + 1}</span>` +
      `<span class="tip-tok">${esc(showTok(r.strings[si]))}</span>` +
      `<span class="tip-prob">${fmtProb(pr[i])}</span></li>`;
  }).join("");
  return `<ol class="tip-list">${rows}${trailingGap ? tipGapRow() : ""}</ol>`;
}
function tipGapRow() {
  return `<li class="is-gap" aria-hidden="true"><span class="tip-gap">···</span></li>`;
}
function tipTokenRow(rank, text, prob, { pin = false, top = false } = {}) {
  const cls = [top ? "is-top" : "", pin ? "is-pin" : ""].filter(Boolean).join(" ");
  const probStr = prob == null ? "—" : fmtProb(prob);
  return `<li class="${cls}">` +
    `<span class="tip-rank">${fmtRankExact(rank)}</span>` +
    `<span class="tip-tok">${esc(showTok(text))}</span>` +
    `<span class="tip-prob">${probStr}</span></li>`;
}
// Rank-heatmap hover: top head ··· pin±1 ···
// Only collapse to the plain top-k list when the pin's rank is actually in 1..k.
function tipRankAwareList(tk, pr, pin, layer, pos) {
  const r = S.resp;
  const rank = rankAt(pin, layer, pos);
  const k = tk.length;

  // Pin is inside the analyze top-k by rank → show that list, highlight it, trail ···.
  if (rank >= 1 && rank <= k) {
    const highlightSi = tk.find((si) => r.token_ids[si] === pin.tokenId) ?? null;
    return tipTopkList(tk, pr, { highlightSi, trailingGap: true });
  }

  const neigh = neighborsAt(pin, layer, pos);
  // Prefer a ±1 window around the pin (trim older ±2 payloads).
  let winRows;
  if (neigh?.tokens?.length) {
    const pinIdx = neigh.tokens.findIndex((_, i) => neigh.start + i === rank);
    const center = pinIdx >= 0 ? pinIdx : Math.floor(neigh.tokens.length / 2);
    const lo = Math.max(0, center - 1);
    const hi = Math.min(neigh.tokens.length - 1, center + 1);
    winRows = [];
    for (let i = lo; i <= hi; i++) {
      winRows.push({
        rank: neigh.start + i,
        text: neigh.tokens[i],
        prob: neigh.probs[i],
        pin: neigh.start + i === rank,
        top: false,
      });
    }
  } else {
    winRows = [{ rank, text: pin.text, prob: null, pin: true, top: false }];
  }
  const winStart = winRows[0].rank;

  // Never let the head reach the pin window — always keep a ··· gap between them.
  const headN = Math.min(tipHeadLimit(rank), k, Math.max(0, winStart - 2));
  const headRows = [];
  for (let i = 0; i < headN; i++) {
    headRows.push({
      rank: i + 1,
      text: r.strings[tk[i]],
      prob: pr[i],
      pin: false,
      top: i === 0,
    });
  }

  let html = headRows.map((row) =>
    tipTokenRow(row.rank, row.text, row.prob, { pin: row.pin, top: row.top })).join("");
  // Leading ··· whenever the pin window isn't contiguous with the head.
  if (headRows.length) html += tipGapRow();
  else if (winStart > 1) html += tipGapRow();
  html += winRows.map((row) =>
    tipTokenRow(row.rank, row.text, row.prob, { pin: row.pin })).join("");
  // Always trail with ··· — the vocab continues after the pin window.
  html += tipGapRow();
  return `<ol class="tip-list">${html}</ol>`;
}
function tipHead(text) { return `<p class="tip-head">${text}</p>`; }
// Each part is a flex child so badge/text gaps stay even (no join-spaces + margin-right).
function tipMeta(...parts) {
  const html = parts.filter((p) => p != null && p !== "").map((p) =>
    `<span class="tip-meta-part">${p}</span>`).join("");
  return `<p class="tip-meta">${html}</p>`;
}
function tipFoot(text) { return `<p class="tip-foot">${text}</p>`; }

function positionTip(x, y) {
  tipEl.classList.remove("hidden");
  // Measure natural size first, then only clamp/scroll if it won't fit the viewport.
  tipEl.classList.remove("is-scrollable");
  tipEl.style.maxHeight = "";
  const pad = 14;
  const margin = 10;
  const natural = tipEl.getBoundingClientRect();
  const maxH = Math.max(160, window.innerHeight - 2 * margin);
  const needsScroll = natural.height > maxH;
  tipEl.classList.toggle("is-scrollable", needsScroll);
  tipEl.style.maxHeight = needsScroll ? `${maxH}px` : "";

  const rect = tipEl.getBoundingClientRect();
  let left = x + pad;
  let top = y + pad;
  if (left + rect.width > window.innerWidth - margin) {
    left = Math.max(margin, x - rect.width - pad);
  }
  if (top + rect.height > window.innerHeight - margin) {
    top = Math.max(margin, y - rect.height - pad);
  }
  if (top + rect.height > window.innerHeight - margin) top = margin;
  tipEl.style.left = `${left}px`;
  tipEl.style.top = `${top}px`;
}
function showTip(html, x, y, opts = {}) {
  const options = opts === true ? {} : opts;
  clearTimeout(tipHideTimer);
  tipEl.innerHTML = html;
  tipEl.classList.toggle("is-accented", !!options.accent);
  if (options.accent) tipEl.style.setProperty("--tip-accent", options.accent);
  else tipEl.style.removeProperty("--tip-accent");
  tipEl.classList.remove("is-scrollable");
  tipEl.style.maxHeight = "";
  requestAnimationFrame(() => positionTip(x, y));
}
function hideTip() {
  tipHideTimer = setTimeout(() => {
    tipEl.classList.add("hidden");
    tipEl.classList.remove("is-accented", "is-scrollable");
    tipEl.style.removeProperty("--tip-accent");
    tipEl.style.maxHeight = "";
    tipTarget = null;
  }, 60);
}

function tipGridCell(L, p) {
  const r = S.resp;
  const [tk, pr] = topkAt(L, p);
  const ctx = esc(JSON.stringify(r.strings[r.prompt_tokens[p]]));
  const pin = S.gridMode === "rank" ? S.pins[S.activePin] : null;
  const metaParts = [`context token ${ctx}`, bandBadge(L), modeLabel()];
  let foot = "click to select";
  let list = tipTopkList(tk, pr);
  if (pin?.ranks) {
    const rank = rankAt(pin, L, p);
    metaParts.push(
      `<span class="tip-badge is-rank">rank ${fmtRankExact(rank)}</span>`,
      `for ${esc(JSON.stringify(pin.text))}`,
    );
    foot = `pinned rank · ${foot}`;
    list = tipRankAwareList(tk, pr, pin, L, p);
  }
  return tipHead(`${layerLabel(L)} @ pos ${p}`) +
    tipMeta(...metaParts) +
    list +
    tipFoot(foot);
}
function tipTokCell(si, rank, prob, layer, pos) {
  const r = S.resp;
  const tok = esc(JSON.stringify(r.strings[si]));
  const where = layer !== undefined
    ? `${layerLabel(layer)} @ pos ${pos}`
  : pos !== undefined
    ? `pos ${pos} · ${layerLabel(S.sel.layer)}`
    : `pos ${S.sel.pos}`;
  return tipHead(tok) +
    tipMeta(`${bandBadge(layer ?? S.sel.layer)}${where} · ${modeLabel()}`) +
    `<dl class="tip-kv">` +
    `<dt>rank</dt><dd>#${rank}</dd>` +
    `<dt>prob</dt><dd>${fmtProb(prob)}</dd>` +
    `</dl>` +
    tipFoot(traceTipFoot(si));
}
function tipAxisPos(p) {
  const r = S.resp;
  const tok = esc(JSON.stringify(r.strings[r.prompt_tokens[p]]));
  const gen = isGen(p) ? " · generated token" : "";
  const [tk, pr] = topkAt(S.sel.layer, p);
  return tipHead(`position ${p}`) +
    tipMeta(`token ${tok}${gen}`, `${bandBadge(S.sel.layer)}${layerLabel(S.sel.layer)} readout`) +
    tipTopkList(tk, pr, { limit: 5 }) +
    tipFoot("click to select position");
}
function tipLayerLabel(L) {
  const r = S.resp;
  const [w0, w1] = r.workspace_band;
  const band = bandName(L);
  let region = band === "workspace" ? `workspace band (L${w0}–L${w1})` :
    band === "motor" ? "motor layers (next-token prediction)" : "sensory layers (noisy readouts)";
  const [tk, pr] = topkAt(L, S.sel.pos);
  const ctx = esc(JSON.stringify(r.strings[r.prompt_tokens[S.sel.pos]]));
  return tipHead(layerLabel(L)) +
    tipMeta(`${bandBadge(L)}${region}`, `top readout at pos ${S.sel.pos} (${ctx}) · ${modeLabel()}`) +
    tipTopkList(tk, pr, { limit: 5 }) +
    tipFoot("click to select layer");
}
function tipTranscriptPos(p) {
  const r = S.resp;
  const tok = esc(JSON.stringify(r.strings[r.prompt_tokens[p]]));
  const gen = isGen(p) ? `<span class="tip-badge is-motor">generated</span>` : "";
  const [tk, pr] = topkAt(r.n_layers - 1, p);
  return tipHead(`position ${p}`) +
    tipMeta(`token ${tok}`, gen) +
    tipMeta(`model next-token prediction from here`) +
    tipTopkList(tk, pr, { limit: 5 }) +
    tipFoot("click to inspect position");
}
function tipPred(si, prob, rank) {
  const r = S.resp;
  return tipHead(esc(JSON.stringify(r.strings[si]))) +
    tipMeta(`model output · pos ${S.sel.pos}`, `#${rank} · prob ${fmtProb(prob)}`) +
    tipFoot(traceTipFoot(si));
}
function tipJRow(row) {
  const r = S.resp;
  const si = +row.dataset.si;
  const total = +row.dataset.total;
  const tok = esc(JSON.stringify(r.strings[si]));
  const [w0, w1] = r.workspace_band;
  const layers = row.dataset.layers.split(",").map((x) => +x);
  const counts = row.dataset.counts.split(",").map((x) => +x);
  // Compact tip: only layers where the token actually appears.
  const hits = layers.map((L, i) => [L, counts[i]]).filter(([, c]) => c > 0);
  const strip = hits.slice(0, 18).map(([L, c]) =>
    `<span class="has-count">L${L} · ${c}×</span>`).join("");
  const more = hits.length > 18 ? `<span>+${hits.length - 18} more</span>` : "";
  return tipHead(tok) +
    tipMeta(`appears in <strong>${total}</strong> workspace top-10 cells`, `L${w0}–L${w1} · ${modeLabel()}`) +
    (strip ? `<div class="tip-strip">${strip}${more}</div>` : "") +
    tipFoot(pinStateForSi(si) ? traceTipFoot(si) : "click to pin · steer/swap on row hover");
}
function tipJSeg(layer, count, si) {
  const r = S.resp;
  const tok = esc(JSON.stringify(r.strings[si]));
  const band = bandBadge(layer);
  if (!count) {
    return tipHead(`L${layer}`) +
      tipMeta(band, `token ${tok} not in top-10 at any position on this layer`) +
      tipFoot("darker bars = more frequent");
  }
  return tipHead(`L${layer} · ${count}×`) +
    tipMeta(band, `token ${tok} in top-10 at ${count} position${count === 1 ? "" : "s"} on this layer`) +
    tipFoot("part of the layer distribution");
}
function tipPinChip(pin, i) {
  const color = pinColor(pin);
  const traceBadge = `<span class="tip-trace-badge" style="--trace-accent:${color}">trace ${i + 1}</span>`;
  const state = pin.loading ? "loading ranks…" :
    pin.chartHidden ? "hidden from charts" :
    pin.ranks ? `traced across ${S.resp.n_layers} layers` : "no rank data";
  return tipHead(esc(JSON.stringify(pin.text))) +
    tipMeta(traceBadge, state) +
    tipFoot(pin.chartHidden
      ? "click token to show on chart"
      : i === S.activePin ? "active for rank heatmap · click token to hide" : "click for rank heatmap");
}
function tipChartPoint(el) {
  const pinIdx = +el.dataset.pin;
  const pin = S.pins[pinIdx];
  if (!pin) return null;
  const color = pinColor(pin);
  const rank = +el.dataset.rank;
  const tok = esc(JSON.stringify(pin.text));
  const traceBadge = `<span class="tip-trace-badge" style="--trace-accent:${color}">trace ${pinIdx + 1}</span>`;
  if (el.dataset.chart === "layer") {
    const L = +el.dataset.layer;
    const ctx = esc(JSON.stringify(S.resp.strings[S.resp.prompt_tokens[S.sel.pos]]));
    return tipHead(tok) +
      tipMeta(traceBadge, `${bandBadge(L)}${layerLabel(L)} @ pos ${S.sel.pos}`, `context ${ctx} · ${modeLabel()}`) +
      `<dl class="tip-kv"><dt>rank</dt><dd class="tip-rank-val">#${fmtRankExact(rank)}</dd></dl>`;
  }
  const p = +el.dataset.pos;
  const ctx = esc(JSON.stringify(S.resp.strings[S.resp.prompt_tokens[p]]));
  return tipHead(tok) +
    tipMeta(traceBadge, `${bandBadge(S.sel.layer)}${layerLabel(S.sel.layer)} @ pos ${p}`, `context ${ctx} · ${modeLabel()}`) +
    `<dl class="tip-kv"><dt>rank</dt><dd class="tip-rank-val">#${fmtRankExact(rank)}</dd></dl>`;
}

function findTipTarget(el) {
  if (!el?.closest) return null;
  return el.closest(
    "#grid td.cell, #grid td.axis-tok, #grid th.layer-label, " +
    ".stack-table td.tok-cell, .stack-table th.layer-label, " +
    ".by-pos-table th.pos-label, .by-pos-table th.tok-label, " +
    ".tr-tok, .pred, .js-row, .js-seg, .pin-chip, #prediction-list .pred, .chart-point"
  );
}

function buildTip(el) {
  if (!S.resp) return null;
  if (el.matches("#grid td.cell")) {
    return tipGridCell(+el.dataset.l, +el.dataset.p);
  }
  if (el.matches("#grid td.axis-tok")) {
    return tipAxisPos(+el.dataset.axis);
  }
  if (el.matches("#grid th.layer-label, .stack-table th.layer-label")) {
    const L = el.dataset.layer !== undefined ? +el.dataset.layer :
      (el.textContent.match(/L(\d+)/) || [])[1];
    if (L !== undefined && !Number.isNaN(+L)) return tipLayerLabel(+L);
    return null;
  }
  if (el.matches(".stack-table td.tok-cell")) {
    const si = +el.dataset.si;
    const rank = +el.dataset.rank;
    const prob = +el.dataset.prob;
    const layer = el.dataset.layer !== undefined ? +el.dataset.layer : undefined;
    const pos = el.dataset.pos !== undefined ? +el.dataset.pos : undefined;
    return tipTokCell(si, rank, prob, layer, pos);
  }
  if (el.matches(".by-pos-table th.pos-label, .by-pos-table th.tok-label")) {
    return tipAxisPos(+el.dataset.pos);
  }
  if (el.matches(".tr-tok")) {
    return tipTranscriptPos(+el.dataset.pos);
  }
  if (el.matches(".pred")) {
    const si = +el.dataset.si;
    const rank = +el.dataset.rank;
    const prob = +el.dataset.prob;
    return tipPred(si, prob, rank);
  }
  if (el.matches(".js-seg")) {
    const si = +el.dataset.si;
    if (si < 0) {
      return tipHead(`L${el.dataset.layer}`) +
        tipMeta(bandBadge(+el.dataset.layer), "layer axis on the distribution track");
    }
    return tipJSeg(+el.dataset.layer, +el.dataset.count, si);
  }
  if (el.matches(".js-row") && el.dataset.si !== undefined) {
    return tipJRow(el);
  }
  if (el.matches(".pin-chip")) {
    const pin = S.pins[+el.dataset.pin];
    if (pin) return tipPinChip(pin, +el.dataset.pin);
  }
  if (el.matches(".chart-point")) {
    return tipChartPoint(el);
  }
  return null;
}

document.addEventListener("mouseover", (e) => {
  const el = findTipTarget(e.target);
  if (!el || !S.resp) return;
  if (el === tipTarget) {
    positionTip(e.clientX, e.clientY);
    return;
  }
  const html = buildTip(el);
  if (!html) return;
  tipTarget = el;
  showTip(html, e.clientX, e.clientY, {
    accent: tipAccentFor(el),
  });
});

document.addEventListener("mousemove", (e) => {
  if (!tipTarget || tipEl.classList.contains("hidden")) return;
  if (findTipTarget(e.target) === tipTarget) positionTip(e.clientX, e.clientY);
});
document.addEventListener("mouseout", (e) => {
  const from = findTipTarget(e.target);
  const to = findTipTarget(e.relatedTarget);
  if (from && from !== to) hideTip();
});
tipEl.addEventListener("mouseenter", () => clearTimeout(tipHideTimer));
tipEl.addEventListener("mouseleave", hideTip);

// Top-k [strIdx...] + probs at (layer, pos) for the current mode; layer n-1 = model.
function topkAt(layer, pos) {
  const r = S.resp;
  if (layer === r.n_layers - 1) return [r.model.topk[pos], r.model.probs[pos]];
  const li = r.lens_layers.indexOf(layer);
  return [r[S.mode].topk[li][pos], r[S.mode].probs[li][pos]];
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
    setPatience("The GPU behind this page sleeps when idle and takes about a minute to wake. Feel free to type your prompt in the meantime.");
  }, 2500);
  try {
    const r = await fetch(`${API}/warmup`);
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    if (data.limits) Object.assign(LIMITS, data.limits);
    clearTimeout(slow);
    S.ready = true;
    setStatus("ready", "ready");
    setPatience(null);
    updateBudgetHint();
  } catch (e) {
    clearTimeout(slow);
    setStatus("error", "backend unreachable");
    setPatience("The backend could not be reached. Reloading the page usually fixes it; if not, the demo may be down for the moment.");
    console.error(e);
  }
}

// ---------- analyze ----------

function hasAnalyzablePrompt(params) {
  if ((params.prompt || "").trim()) return true;
  // Chat mode: history alone is enough — Analyze runs on the current thread
  // after × removals, including consecutive user turns / empty compose box.
  return params.chat && (params.messages || []).some((m) => (m.content || "").trim());
}

async function analyze() {
  const params = promptParams();
  if (!hasAnalyzablePrompt(params)) {
    const banner = $("error-banner");
    banner.textContent = S.chat === "chat"
      ? "Enter a message, or keep at least one turn in the conversation."
      : "Enter a prompt to analyze.";
    banner.classList.remove("hidden");
    return;
  }
  const btn = $("analyze-btn");
  btn.disabled = true;
  btn.textContent = "reading…";
  $("error-banner").classList.add("hidden");
  const slow = setTimeout(() => {
    if (!S.ready) setPatience("Still waking the GPU · your prompt is queued and will run when it's up.");
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
    const hasGen = S.resp.gen_start !== undefined && S.resp.gen_start < P;
    S.sel = { layer: Math.round((w0 + w1) / 2), pos: hasGen ? S.resp.gen_start : P - 1 };
    $("results").classList.remove("hidden");
    $("examples").classList.add("hidden");
    maybeKeepChatTurns();
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
    btn.innerHTML = 'Analyze <kbd class="key-hint">⏎</kbd>';
  }
}

// ---------- pinning ----------

function refreshPinViews() {
  renderPins();
  renderGrid();
  renderCharts();
  renderByLayer();
  renderByPos();
}

function togglePinChart(i) {
  const pin = S.pins[i];
  if (!pin?.ranks) return;
  pin.chartHidden = !pin.chartHidden;
  S.activePin = i;
  refreshPinViews();
}

async function pinToken({ tokenId = null, tokenStr = null, ensure = false }) {
  const existing = findPin({ tokenId, tokenStr });
  if (existing >= 0) {
    const pin = S.pins[existing];
    if (pin.loading) { S.activePin = existing; renderPins(); return; }
    if (pin.ranks) {
      if (ensure) {
        pin.chartHidden = false;
        S.activePin = existing;
        refreshPinViews();
        return;
      }
      togglePinChart(existing);
      return;
    }
  }
  const pin = {
    tokenId,
    text: tokenStr ?? S.resp.strings[S.resp.token_ids.indexOf(tokenId)] ?? "",
    colorIdx: S.pins.length,
    ranks: null,
    loading: true,
    chartHidden: false,
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
    pin.chartHidden = false;
    $("rank-legend").classList.remove("hidden");
    document.querySelector('#grid-mode [data-grid="rank"]').disabled = false;
    if (S.gridMode === "argmax") setGridMode("rank");
    refreshPinViews();
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
  renderByLayer();
  renderByPos();
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
    return `<button class="${cls.join(" ")}" data-pos="${p}">` +
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
    ? 'transcript <span class="hint">&mdash; exact token sequence; click any token to inspect</span>'
    : 'prompt + model output <span class="hint">&mdash; greedy continuation highlighted; click any token to inspect</span>';
}

function renderPrediction() {
  const r = S.resp;
  const pos = S.sel.pos;
  const isLast = pos === r.prompt_tokens.length - 1;
  $("prediction-label").textContent = isLast && r.completion
    ? "model prediction · token after the output"
    : `model prediction · next token after pos ${pos}`;
  $("prediction-list").innerHTML = r.model.topk[pos].slice(0, 5).map((si, i) => {
    const trace = tracedMarkup(si);
    return `<button class="pred${trace.cls}"${trace.style} data-si="${si}" data-rank="${i + 1}" data-prob="${r.model.probs[pos][i]}">` +
      `<span>${esc(showTok(r.strings[si]))}</span><span class="p">${fmtProb(r.model.probs[pos][i])}</span></button>`;
  }).join("");
}

// ---------- J-Space aggregate (neuronpedia-style count view) ----------

function renderJSpace() {
  const r = S.resp;
  if (!r) return;
  const data = r[S.mode];
  const L = r.lens_layers.length;
  const P = r.prompt_tokens.length;
  const [w0, w1] = r.workspace_band;
  const allIndices = [...Array(L).keys()];
  const wsIndices = allIndices.filter((li) => {
    const layer = r.lens_layers[li];
    return layer >= w0 && layer <= w1;
  });
  // Count every lens layer for the distribution track; rank by workspace total.
  const stats = new Map(); // strIdx -> {total, perLayer: Int32Array}
  for (let li = 0; li < L; li++) {
    const inWs = r.lens_layers[li] >= w0 && r.lens_layers[li] <= w1;
    const layerTop = data.topk[li];
    for (let p = 0; p < P; p++) {
      for (const si of layerTop[p]) {
        let s = stats.get(si);
        if (!s) { s = { total: 0, perLayer: new Int32Array(L) }; stats.set(si, s); }
        s.perLayer[li]++;
        if (inWs) s.total++;
      }
    }
  }
  // Special/template and pure-punctuation tokens hold real workspace info (turn
  // structure, formatting state) but drown out content words — shown on demand.
  const showAll = $("js-show-all").checked;
  const isContent = (si) => /\p{L}|\p{N}/u.test(r.strings[si]) && !/<.*>/.test(r.strings[si]);
  const rows = [...stats.entries()]
    .filter(([si, s]) => s.total > 0 && (showAll || isContent(si)))
    .sort((a, b) => b[1].total - a[1].total).slice(0, 40);
  const maxCell = Math.max(1, ...rows.map(([, s]) => Math.max(0, ...s.perLayer)));
  const lo = r.lens_layers[0];
  const hi = r.lens_layers[L - 1];
  // Workspace band as a fraction of the full lens-layer track.
  const bandLeft = ((w0 - lo) / Math.max(1, hi - lo + 1)) * 100;
  const bandWidth = ((w1 - w0 + 1) / Math.max(1, hi - lo + 1)) * 100;

  const seg = (c, li, si) => {
    const layer = r.lens_layers[li];
    const inWs = layer >= w0 && layer <= w1;
    if (!c) {
      return `<i class="js-seg is-zero${inWs ? "" : " is-out"}" data-layer="${layer}" data-count="0" data-si="${si}"></i>`;
    }
    const t = c / maxCell;
    const op = (0.22 + 0.78 * Math.sqrt(t)).toFixed(2);
    return `<i class="js-seg${inWs ? "" : " is-out"}" style="--seg-op:${op}" data-layer="${layer}" data-count="${c}" data-si="${si}"></i>`;
  };

  const axisTicks = allIndices.map((li) => {
    const layer = r.lens_layers[li];
    const edge = layer === lo || layer === hi || layer === w0 || layer === w1;
    const tick = edge || layer % 4 === 0;
    return `<i class="js-seg is-label${tick ? " has-label" : ""}" data-layer="${layer}" data-count="0" data-si="-1">` +
      `${tick ? `<span>L${layer}</span>` : ""}</i>`;
  }).join("");

  const trackStyle = `--band-left:${bandLeft.toFixed(2)}%; --band-width:${bandWidth.toFixed(2)}%;`;
  const colHead =
    `<div class="js-colhead" aria-hidden="true">` +
    `<span class="js-tok-col">token</span>` +
    `<span class="js-count-col">count</span>` +
    `<span class="js-dist-col">distribution by layer` +
    `<span class="js-band-tag">workspace L${w0}–${w1}</span></span>` +
    `<span class="js-actions-col"></span></div>`;
  const legend =
    `<div class="js-axis" aria-hidden="true">` +
    `<span class="js-tok js-tok-ghost"></span><span class="js-count"></span>` +
    `<span class="js-track js-track-axis" style="${trackStyle}">${axisTicks}</span>` +
    `<span class="js-actions"></span></div>`;

  const body = rows.map(([si, s]) => {
    const strip = allIndices.map((li) => seg(s.perLayer[li], li, si)).join("");
    // Tip / hover still focuses on workspace activity.
    const layerData = wsIndices.map((li) => r.lens_layers[li]).join(",");
    const countData = wsIndices.map((li) => s.perLayer[li]).join(",");
    const rowAttrs =
      `data-si="${si}" data-total="${s.total}" data-layers="${layerData}" data-counts="${countData}"`;
    const trace = tracedMarkup(si);
    return `<div class="js-row" ${rowAttrs}>` +
      `<button class="tok js-tok${trace.cls}"${trace.style} data-si="${si}">${esc(showTok(r.strings[si]))}</button>` +
      `<span class="js-count">${s.total}</span>` +
      `<span class="js-track" style="${trackStyle}" aria-hidden="true">${strip}</span>` +
      `<span class="js-actions">` +
      `<button class="mini-btn" data-iv="steer" data-si="${si}">steer</button>` +
      `<button class="mini-btn" data-iv="swap" data-si="${si}">swap</button>` +
      `</span></div>`;
  }).join("");

  $("jspace").innerHTML = rows.length
    ? `<div class="js-sticky-head">${colHead}${legend}</div>${body}`
    : `<p class="panel-hint js-empty">no readout data</p>`;
}

function renderPins() {
  $("pins").innerHTML = S.pins.map((p, i) =>
    `<span class="pin-chip ${p.loading ? "loading" : ""} ${p.chartHidden ? "chart-hidden" : ""} ${i === S.activePin ? "active-pin" : ""}"` +
    ` style="border-color:${pinColor(p)};color:${pinColor(p)}" data-pin="${i}">` +
    `${esc(showTok(p.text))}${p.loading ? " …" : ""}` +
    `<button class="x" data-unpin="${i}" aria-label="unpin">×</button></span>`
  ).join("");
}

function renderMeta() {
  const r = S.resp;
  const secs = ((r.client_ms ?? r.timing_ms.total) / 1000).toFixed(1);
  const hasGen = r.gen_start !== undefined && r.gen_start < r.prompt_tokens.length;
  const budget = r.token_budget;
  const tokStr = budget
    ? `${budget.prompt} prompt + ${budget.generated} gen (${budget.max} max)`
    : hasGen
      ? `${r.gen_start} prompt + ${r.prompt_tokens.length - r.gen_start} gen`
      : `${r.prompt_tokens.length} tokens`;
  $("meta").textContent = `${tokStr} · ${secs} s${r.truncated ? " · truncated" : ""}`;
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
      if (usePin) {
        const rank = rankAt(pin, L, p);
        cells.push(`<td class="cell ${rankClass(rank)}${selCol}${selected}" data-l="${L}" data-p="${p}">` +
          `${fmtRank(rank)}</td>`);
      } else {
        const [tk] = topkAt(L, p);
        cells.push(`<td class="cell${selCol}${selected}" data-l="${L}" data-p="${p}">` +
          `${esc(showTok(r.strings[tk[0]]))}</td>`);
      }
    }
    return `<tr class="${cls}"><th class="layer-label" data-layer="${L}">${layerLabel(L)}</th>${cells.join("")}</tr>`;
  });

  const axis = r.prompt_tokens.map((si, p) =>
    `<td class="axis-tok${p === S.sel.pos ? " sel-col" : ""}${isGen(p) ? " is-gen" : ""}" data-axis="${p}">` +
    `${esc(showTok(r.strings[si]))}</td>`).join("");
  rows.push(`<tr class="axis-row"><th class="layer-label">${r.completion ? "prompt + output →" : "prompt →"}</th>${axis}</tr>`);
  $("grid").style.minWidth = `${96 + 80 * P}px`;
  $("grid").innerHTML = rows.join("");
}

// One top-k row as grid-style <td> cells (rank order left to right).
function tokCells(tk, pr, { layer = null, pos = null } = {}) {
  const r = S.resp;
  return tk.map((si, i) => {
    const trace = tracedMarkup(si);
    const attrs = [
      `data-si="${si}"`, `data-rank="${i + 1}"`, `data-prob="${pr[i]}"`,
      layer !== null ? `data-layer="${layer}"` : "",
      pos !== null ? `data-pos="${pos}"` : "",
    ].filter(Boolean).join(" ");
    return `<td class="tok-cell${i === 0 ? " rank1" : ""}${trace.cls}"${trace.style} ${attrs}>` +
      `${esc(showTok(r.strings[si]))}<span class="p">${fmtProb(pr[i])}</span></td>`;
  }).join("");
}

function stackTable(rows, k) {
  return `<table class="stack-table" style="min-width:${96 + 104 * k}px">${rows.join("")}</table>`;
}

function stackTableByPos(rows, k) {
  const head = `<tr class="stack-head">` +
    `<th class="pos-label">pos</th><th class="tok-label">token</th>` +
    `<th class="stack-head-rest" colspan="${k}"></th></tr>`;
  return `<table class="stack-table by-pos-table" style="min-width:${48 + 104 + 104 * k}px">` +
    head + rows.join("") + `</table>`;
}

function renderByLayer(autoScroll = false) {
  const r = S.resp;
  const pos = S.sel.pos;
  $("by-layer-title").textContent =
    `by layer · pos ${pos} ${JSON.stringify(r.strings[r.prompt_tokens[pos]])}`;
  const layers = [r.n_layers - 1, ...[...r.lens_layers].reverse()];
  let k = 0;
  const rows = layers.map((L) => {
    const [tk, pr] = topkAt(L, pos);
    k = Math.max(k, tk.length);
    return `<tr class="${bandClass(L)}${L === S.sel.layer ? " is-selected" : ""}">` +
      `<th class="layer-label" data-layer="${L}">${layerLabel(L)}</th>${tokCells(tk, pr, { layer: L, pos })}</tr>`;
  });
  $("by-layer").innerHTML = stackTable(rows, k);
  const sel = $("by-layer").querySelector(".is-selected");
  if (sel && autoScroll) scrollStackToRow($("by-layer"), sel);
}

function renderByPos(autoScroll = false) {
  const r = S.resp;
  const L = S.sel.layer;
  const P = r.prompt_tokens.length;
  $("by-pos-title").textContent = `by position · ${layerLabel(L)}`;
  const rows = [];
  let k = 0;
  for (let p = 0; p < P; p++) {
    const [tk, pr] = topkAt(L, p);
    k = Math.max(k, tk.length);
    rows.push(`<tr class="${p === S.sel.pos ? "is-selected" : ""}${isGen(p) ? " is-gen-row" : ""}">` +
      `<th class="pos-label" data-pos="${p}">${p}</th>` +
      `<th class="tok-label" data-pos="${p}">` +
      `${esc(showTok(r.strings[r.prompt_tokens[p]]))}</th>${tokCells(tk, pr, { pos: p })}</tr>`);
  }
  $("by-pos").innerHTML = stackTableByPos(rows, k);
  const sel = $("by-pos").querySelector(".is-selected");
  if (sel && autoScroll) scrollStackToRow($("by-pos"), sel);
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

function chartTrace(pin, pinIdx, points, hitMeta) {
  const color = pinColor(pin);
  const linePts = points.map(([cx, cy]) => `${cx},${cy}`).join(" ");
  const dots = points.map(([cx, cy], i) => {
    const m = hitMeta[i];
    const attrs = [
      `class="chart-point"`, `cx="${cx}"`, `cy="${cy}"`, `r="6"`,
      `data-pin="${pinIdx}"`, `data-chart="${m.chart}"`, `data-rank="${m.rank}"`,
      m.layer !== undefined ? `data-layer="${m.layer}"` : "",
      m.pos !== undefined ? `data-pos="${m.pos}"` : "",
    ].filter(Boolean).join(" ");
    return `<circle ${attrs} style="--point-color:${color}"/>`;
  }).join("");
  return `<polyline class="rankline" stroke="${color}" points="${linePts}"/>${dots}`;
}

function renderCharts() {
  const r = S.resp;
  if (!r) return;
  const traced = S.pins.filter((p) => p.ranks);
  const visible = traced.filter((p) => !p.chartHidden);

  // rank vs layer at selected position
  {
    const w = chartWidth($("chart-layer"));
    const iw = w - CH.l - CH.r;
    const n = r.n_layers;
    const x = (L) => CH.l + (L / (n - 1)) * iw;
    const [w0, w1] = r.workspace_band;
    const ticks = [0, 8, 16, 24, 31].map((L) => [x(L), `L${L}`]);
    const parts = chartFrame(w, ticks, "layer →", [x(w0), x(w1)]);
    for (const pin of visible) {
      const pinIdx = S.pins.indexOf(pin);
      const points = [];
      const meta = [];
      for (const L of r.lens_layers) {
        const rank = rankAt(pin, L, S.sel.pos);
        points.push([x(L), rankY(rank)]);
        meta.push({ chart: "layer", layer: L, rank });
      }
      const outL = n - 1;
      const outRank = pin.ranks.model_ranks[S.sel.pos];
      points.push([x(outL), rankY(outRank)]);
      meta.push({ chart: "layer", layer: outL, rank: outRank });
      parts.push(chartTrace(pin, pinIdx, points, meta));
    }
    if (!visible.length) {
      const msg = traced.length
        ? "all traces hidden · click a pinned token to show"
        : "pin a token to trace its rank across layers";
      parts.push(`<text class="empty-note" x="${CH.l + 10}" y="${CH.t + 16}">${msg}</text>`);
    }
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
    for (const pin of visible) {
      const pinIdx = S.pins.indexOf(pin);
      const points = [];
      const meta = [];
      for (let p = 0; p < P; p++) {
        const rank = rankAt(pin, S.sel.layer, p);
        points.push([x(p), rankY(rank)]);
        meta.push({ chart: "pos", pos: p, rank });
      }
      parts.push(chartTrace(pin, pinIdx, points, meta));
    }
    if (!visible.length) {
      const msg = traced.length
        ? "all traces hidden · click a pinned token to show"
        : "pin a token to trace its rank across positions";
      parts.push(`<text class="empty-note" x="${CH.l + 10}" y="${CH.t + 16}">${msg}</text>`);
    }
    $("chart-pos").innerHTML = parts.join("");
  }
}

// ---------- selection ----------

function select({ layer = null, pos = null }) {
  if (layer !== null) S.sel.layer = layer;
  if (pos !== null) S.sel.pos = pos;
  renderCompletion(); // keep the transcript's selected-token highlight in sync
  renderPrediction();
  renderGrid();
  renderByLayer(true);
  renderByPos(true);
  renderCharts();
}

// ---------- events ----------

$("analyze-btn").addEventListener("click", analyze);
$("prompt").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); analyze(); }
});

document.querySelectorAll("#examples .example").forEach((b) =>
  b.addEventListener("click", () => {
    $("prompt").value = b.dataset.multiturn ? "Name a different one in one word." : b.textContent;
    const chat = b.dataset.chat === "chat" ? "chat" : "raw";
    document.querySelector(`#chat-toggle [data-chat="${chat}"]`).click();
    if (b.dataset.multiturn) {
      try { S.chatTurns = JSON.parse(b.dataset.multiturn); } catch { S.chatTurns = []; }
      renderChatHistory();
    } else {
      S.chatTurns = [];
      renderChatHistory();
    }
    updateBudgetHint();
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
    $("chat-history-panel").classList.toggle("hidden", S.chat !== "chat");
    updateChatControls();
    $("prompt").placeholder = S.chat === "chat"
      ? "Your message (latest user turn)…"
      : "Type a prompt…";
    updateBudgetHint();
  }));

// ---------- chat multi-turn + token budget ----------

function estPromptTokens() {
  let n = Math.ceil($("prompt").value.length / 3.2);
  n += Math.ceil(($("system-prompt").value || "").length / 3.2);
  n += Math.ceil(($("prefill").value || "").length / 3.2);
  for (const t of S.chatTurns) n += Math.ceil(t.content.length / 3.2);
  if (S.chat === "chat") n += 24 + S.chatTurns.length * 18; // template overhead
  return n;
}

function refineGenOptions(room) {
  const sel = $("gen-len");
  const maxGen = Math.max(0, Math.min(LIMITS.max_new, room));
  let picked = +sel.value;
  for (const opt of sel.options) {
    const v = +opt.value;
    opt.disabled = v > 0 && v > maxGen;
  }
  if (picked > maxGen) {
    const allowed = [...sel.options].filter((o) => !o.disabled).map((o) => +o.value);
    picked = allowed.length ? Math.max(...allowed) : 0;
    sel.value = String(picked);
  }
  return maxGen;
}

function updateBudgetHint() {
  const hint = $("budget-hint");
  const est = estPromptTokens();
  const room = Math.max(0, LIMITS.max_sequence - est);
  const avail = refineGenOptions(room);
  if (est > LIMITS.max_sequence) {
    hint.textContent = `over ${LIMITS.max_sequence} token limit (~${est} estimated)`;
  } else if (est > LIMITS.max_sequence - 24) {
    hint.textContent = `~${est} / ${LIMITS.max_sequence} tokens · up to ${avail} new`;
  } else {
    hint.textContent = `${LIMITS.max_sequence} token limit · up to ${avail} new`;
  }
  hint.classList.toggle("is-warn", est > LIMITS.max_sequence);
}

function renderChatHistory() {
  const el = $("chat-history");
  const roleLabel = { user: "You", assistant: "Assistant" };
  if (!S.chatTurns.length) {
    el.innerHTML = '<p class="chat-history-empty">no prior turns · check &ldquo;keep history&rdquo; to build a thread</p>';
    updateChatControls();
    return;
  }
  el.innerHTML = S.chatTurns.map((t, i) =>
    `<div class="chat-turn tr-${t.role}">` +
    `<span class="chat-turn-role">${roleLabel[t.role] || t.role}</span>` +
    `<span class="chat-turn-text">${esc(t.content)}</span>` +
    `<button type="button" class="chat-turn-rm" data-i="${i}" aria-label="remove this turn">×</button></div>`
  ).join("");
  updateChatControls();
}

function updateChatControls() {
  const inChat = S.chat === "chat";
  $("prompt-chat-actions").classList.toggle("hidden", !inChat);
  const hasTurns = S.chatTurns.length > 0;
  const clearBtn = $("chat-clear");
  clearBtn.disabled = !hasTurns;
  const count = $("chat-turn-count");
  if (inChat && hasTurns) {
    count.textContent = `${S.chatTurns.length} turn${S.chatTurns.length === 1 ? "" : "s"}`;
    count.classList.remove("hidden");
  } else {
    count.textContent = "";
    count.classList.add("hidden");
  }
}

function clearChatTurns() {
  S.chatTurns = [];
  renderChatHistory();
  updateBudgetHint();
}

function maybeKeepChatTurns() {
  if (S.chat !== "chat" || !$("chat-keep").checked) return;
  // Only append when the compose box contributed a new user turn. Re-analyzing
  // an edited history with an empty compose must not duplicate turns.
  const userMsg = (S.params?.prompt || "").trim();
  if (!userMsg) return;
  S.chatTurns.push({ role: "user", content: userMsg });
  if (S.resp?.completion) S.chatTurns.push({ role: "assistant", content: S.resp.completion });
  $("prompt").value = "";
  renderChatHistory();
  updateBudgetHint();
}

$("chat-clear").addEventListener("click", clearChatTurns);
$("chat-history").addEventListener("click", (e) => {
  const rm = e.target.closest(".chat-turn-rm");
  if (!rm) return;
  S.chatTurns.splice(+rm.dataset.i, 1);
  renderChatHistory();
  updateBudgetHint();
});
for (const id of ["prompt", "system-prompt", "prefill", "gen-len"]) {
  $(id).addEventListener("input", updateBudgetHint);
  $(id).addEventListener("change", updateBudgetHint);
}

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
    btn.textContent = "run intervention";
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
    const tok = e.target.closest("td.tok-cell");
    if (tok) return pinToken({ tokenId: S.resp.token_ids[+tok.dataset.si] });
    const rl = e.target.closest("th.layer-label");
    if (rl && rl.dataset.layer !== undefined) return select({ layer: +rl.dataset.layer });
    const posTh = e.target.closest("th.pos-label, th.tok-label");
    if (posTh && posTh.dataset.pos !== undefined) return select({ pos: +posTh.dataset.pos });
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

$("pin-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const raw = $("pin-input").value;
  if (!raw || !S.resp) return;
  $("pin-input").value = "";
  const parts = raw.split(",").map((s) => s.trim()).filter(Boolean);
  for (const tokenStr of parts) await pinToken({ tokenStr, ensure: true });
});

warmup();
renderChatHistory();
updateBudgetHint();
