"""J-Lens Visualizer backend — Modal app serving per-layer Jacobian-lens readouts.

One GPU class loads Qwen3.5-4B plus the released neuronpedia n=1000 lens once per
container and serves a FastAPI app. Endpoints:

    GET  /healthz       liveness, no GPU work
    GET  /warmup        touches the GPU container so page load hides the cold start
    POST /api/analyze   {prompt, top_k, chat?, messages?, system_prompt?, prefill?,
                        max_new_tokens?} -> greedy completion + per-(layer,
                        position) top-k over prompt AND generated tokens, for
                        J-lens, logit-lens, and the model's own output
    POST /api/rank      {request_id, ...same prompt params, token_id|token_str}
                        -> the token's rank at every (layer, position), plus a
                        ±RANK_NEIGHBOR_WINDOW token/prob window around that rank
                        for rank-heatmap hover context
    POST /api/intervene {...prompt params, kind: swap|steer, source, target?,
                        strength?, layer_lo?, layer_hi?} -> default vs modified
                        greedy generations (paper §2.5 workspace interventions)

Chat mode wraps messages in Qwen's chat template with enable_thinking=False
(main.ipynb verbal-report protocol); prefill text is appended to the template.
`messages` is the conversation as currently edited (× may leave consecutive user
turns); `prompt` is an optional extra latest user message. Prompt + generation
share a single MAX_SEQUENCE token budget.
Swap follows the notebook exactly: single-token surface-form pairs (leading
space + case variants), bisector reflection at alpha=1 over the workspace band
(layers 12-28). Steer adds strength * mean_residual_norm * unit lens vector.

Deploy from the repo root:  modal deploy visualizer/backend/app.py
"""

import os
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import modal

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
LENS_REVISION = "qwen-n1000"

MAX_CHARS = 4000
MAX_SEQUENCE = 320   # total tokens analyzed (prompt + greedy generation)
MAX_NEW = 128        # max generation tokens per request (clamped to remaining budget)
MAX_K = 10
RANK_NEIGHBOR_WINDOW = 1  # ±tokens around the pinned rank for hover context
CACHE_SIZE = 8  # cached activations, ~40 MB each
WORKSPACE_BAND = [12, 28]  # main.ipynb: WORKSPACE = range(round(0.38*n), round(0.92*n))
NORM_PROMPTS = [  # generic contexts for mean_residual_norms (each > 17 tokens)
    "The history of science is full of ideas that seemed absurd at first and later became foundations of entire fields of study.",
    "When you travel to a new city, the first thing you notice is usually the rhythm of the streets and the way people move through them.",
    "Cooking a good meal requires attention to timing, temperature, and the order in which ingredients are combined in the pan.",
    "The committee reviewed the proposal carefully before deciding whether the project should receive funding for another year.",
]

REPO_ROOT = Path(__file__).parent.parent.parent

app = modal.App("j-lens-visualizer")
vol = modal.Volume.from_name("j-lens")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "huggingface_hub",
        "hf-transfer",
        "fastapi[standard]",
        "supabase",
    )
    .env({"HF_HOME": "/vol/j-lens/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(REPO_ROOT / "jlens.py", "/root/jlens.py")
    .add_local_file(REPO_ROOT / "interventions.py", "/root/interventions.py")
)


def pair_token_ids(tok, source, target):
    """All single-token surface-form pairs (evals.py protocol): ' w', ' w.lower()', 'w', 'w.lower()'."""
    pairs = []
    for f in (lambda w: " " + w, lambda w: " " + w.lower(), lambda w: w, lambda w: w.lower()):
        a = tok.encode(f(source), add_special_tokens=False)
        b = tok.encode(f(target), add_special_tokens=False)
        if len(a) == 1 and len(b) == 1 and (a[0], b[0]) not in pairs:
            pairs.append((a[0], b[0]))
    return pairs


def single_token_id(tok, word):
    """First single-token surface form of a word, preferring the leading-space form."""
    for f in (lambda w: " " + w, lambda w: " " + w.lower(), lambda w: w, lambda w: w.lower()):
        ids = tok.encode(f(word), add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/vol/j-lens": vol},
    scaledown_window=300,
    max_containers=2,
    timeout=600,
    secrets=[modal.Secret.from_name("supabase-jlens")],
)
@modal.concurrent(max_inputs=4)
class Server:
    @modal.enter()
    def load(self):
        import torch
        import jlens
        import interventions as iv

        self.torch = torch
        self.jlens = jlens
        self.iv = iv
        self.model, self.tok = jlens.load_model(MODEL_NAME)
        self.lens = jlens.JLens.from_pretrained(LENS_REPO, LENS_FILE, revision=LENS_REVISION)
        # CPU view for the intervention builders — lens.vectors() computes on CPU,
        # so it needs CPU-resident J (the hooks move edit vectors to GPU themselves).
        self.lens_cpu = jlens.JLens(self.lens.J, n_prompts=self.lens.n_prompts)
        # Pre-move J to the GPU so transport()'s .to(device) is a no-op per request.
        self.lens.J = {l: j.to("cuda") for l, j in self.lens.J.items()}
        self.final = len(self.model.model.layers) - 1
        self.gpu_lock = threading.Lock()
        self.cache = OrderedDict()  # request_id -> {"ids", "acts", "gen_start"}
        # Steering scale: mean residual norm per lens layer over generic prompts.
        self.mean_norms = iv.mean_residual_norms(self.model, self.tok, NORM_PROMPTS, self.lens.layers)
        self.sb = None
        if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"):
            try:
                from supabase import create_client

                self.sb = create_client(
                    os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"]
                )
            except Exception as e:
                print(f"supabase init failed, logging disabled: {e}")

    # ---- prompt rendering --------------------------------------------------

    def _chat_messages(self, req):
        """Build chat-template message list: optional system, prior turns, optional latest user.

        `messages` is the conversation as currently edited (× may leave consecutive
        user turns). `prompt` is appended as a final user turn only when non-empty,
        so Analyze can run on history alone after the compose box was cleared.
        """
        msgs = []
        if getattr(req, "system_prompt", None) and req.system_prompt.strip():
            msgs.append({"role": "system", "content": req.system_prompt.strip()})
        for m in getattr(req, "messages", None) or []:
            role = m.role if hasattr(m, "role") else m.get("role")
            content = (m.content if hasattr(m, "content") else m.get("content") or "").strip()
            if not content:
                continue
            if role not in ("user", "assistant"):
                raise ValueError(f"invalid message role: {role}")
            msgs.append({"role": role, "content": content})
        latest = (req.prompt or "").strip()
        if latest:
            msgs.append({"role": "user", "content": latest})
        if not msgs or not any(m["role"] == "user" for m in msgs):
            raise ValueError("chat needs at least one user message")
        return msgs

    def _render_text(self, req):
        """Final string fed to the model. Raw mode rstrips (a trailing space becomes
        the readout token and wrecks baselines); the chat template is used verbatim.

        After × edits the history may end on either role. Open a fresh assistant
        turn when the last message is from the user, or when a prefill is set.
        """
        if not getattr(req, "chat", False):
            return req.prompt.rstrip()
        msgs = self._chat_messages(req)
        prefill = getattr(req, "prefill", None) or ""
        open_assistant = msgs[-1]["role"] == "user" or bool(prefill)
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=open_assistant, enable_thinking=False
        )
        if prefill:
            text += prefill
        return text

    def _prompt_token_count(self, req):
        return len(self.tok(self._render_text(req)).input_ids)

    def _gen_budget(self, req, prompt_len):
        """How many new tokens we can generate within MAX_SEQUENCE."""
        room = max(0, MAX_SEQUENCE - prompt_len)
        requested = max(0, getattr(req, "max_new_tokens", 0) or 0)
        return min(requested, MAX_NEW, room)

    # ---- compute -----------------------------------------------------------

    def _forward_full(self, req):
        """Render, optionally generate greedily, then one recorded forward over the
        full sequence. Returns (ids [1,T+G], acts fp16 cpu, gen_start)."""
        text = self._render_text(req)
        full_len = self._prompt_token_count(req)
        truncated = full_len > MAX_SEQUENCE
        ids = self.jlens.encode(self.model, self.tok, text, max_len=MAX_SEQUENCE)
        gen_start = ids.shape[1]
        n_new = self._gen_budget(req, gen_start)
        with self.torch.no_grad():
            if n_new > 0:
                ids = self.model.generate(
                    ids, max_new_tokens=n_new, do_sample=False,
                    pad_token_id=self.tok.eos_token_id,
                )
            want = [*self.lens.layers, self.final]
            with self.jlens.record_residuals(self.model, want) as rec:
                self.model.model(input_ids=ids, use_cache=False)
                acts = {l: rec.acts[l][0].detach().half().cpu() for l in want}
        return ids, acts, gen_start

    def _layer_logits(self, acts, layer, use_jacobian):
        """[P, vocab] fp32 logits on GPU for one layer; caller discards after use."""
        h = acts[layer].to("cuda", self.torch.float32)
        z = self.lens.transport(h, layer) if use_jacobian else h
        return self.jlens.unembed(self.model, z).float()

    def _rank_and_neighbors(self, lg, tid, window=RANK_NEIGHBOR_WINDOW):
        """1-based ranks plus a ±window token window around the pinned token.

        For each position returns neighbors as
        ``{"start": lo, "tokens": [...], "probs": [...]}`` covering ranks
        ``[max(1, R-window), min(V, R+window)]``. Above/below neighbors are the
        tokens whose logits sit just better/worse than the pin (exact rank
        adjacency aside from logit ties).
        """
        torch = self.torch
        P, V = lg.shape
        ranks = (1 + (lg > lg[:, tid : tid + 1]).sum(-1)).int()
        probs = lg.softmax(-1)
        pv = lg[:, tid]
        w = max(0, int(window))
        if w == 0:
            return ranks, [
                {
                    "start": int(ranks[p]),
                    "tokens": [self.tok.decode([tid])],
                    "probs": [round(float(probs[p, tid]), 4)],
                }
                for p in range(P)
            ]

        above_lg = lg.masked_fill(lg <= pv.unsqueeze(1), float("inf"))
        above_vals, above_idx = torch.topk(above_lg, w, dim=-1, largest=False)
        below_lg = lg.masked_fill(lg >= pv.unsqueeze(1), float("-inf"))
        below_vals, below_idx = below_lg.topk(w, dim=-1)

        neighbors = []
        pin_text = self.tok.decode([tid])
        for p in range(P):
            R = int(ranks[p])
            lo = max(1, R - w)
            hi = min(V, R + w)
            items = []
            for i in range(w):
                r = R - 1 - i
                if r < lo:
                    break
                if not torch.isfinite(above_vals[p, i]):
                    break
                t = int(above_idx[p, i])
                items.append((r, self.tok.decode([t]), float(probs[p, t])))
            items.reverse()
            items.append((R, pin_text, float(probs[p, tid])))
            for i in range(w):
                r = R + 1 + i
                if r > hi:
                    break
                if not torch.isfinite(below_vals[p, i]):
                    break
                t = int(below_idx[p, i])
                items.append((r, self.tok.decode([t]), float(probs[p, t])))
            neighbors.append({
                "start": items[0][0],
                "tokens": [t for _, t, _ in items],
                "probs": [round(pr, 4) for _, _, pr in items],
            })
        return ranks, neighbors

    def _generate(self, ids, max_new_tokens, edits=()):
        """Greedy continuation under optional edits; returns decoded new text."""
        with self.torch.no_grad(), self.iv.apply_edits(self.model, edits):
            out = self.model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def _next_top5(self, ids, edits=()):
        """Top-5 next-token strings+probs at the last position under edits."""
        with self.torch.no_grad(), self.iv.apply_edits(self.model, edits):
            out = self.model(input_ids=ids, use_cache=False)
        probs, idx = out.logits[0, -1].float().softmax(-1).topk(5)
        return [
            {"token": self.tok.decode([t]), "prob": round(p, 4)}
            for t, p in zip(idx.tolist(), probs.tolist())
        ]

    def _cache_put(self, rid, ids, acts, gen_start):
        self.cache[rid] = {"ids": ids.cpu(), "acts": acts, "gen_start": gen_start}
        while len(self.cache) > CACHE_SIZE:
            self.cache.popitem(last=False)

    def _log(self, row):
        if self.sb is None:
            return

        def insert():
            try:
                self.sb.table("jlens_requests").insert(row).execute()
            except Exception as e:
                print(f"supabase insert failed: {e}")

        threading.Thread(target=insert, daemon=True).start()

    # ---- web app -------------------------------------------------------------

    @modal.asgi_app(label="jlens-api")
    def web(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.gzip import GZipMiddleware
        from pydantic import BaseModel

        api = FastAPI(title="J-Lens Visualizer API")
        api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
        api.add_middleware(GZipMiddleware, minimum_size=1000)

        class ChatMessage(BaseModel):
            role: str
            content: str

        class PromptParams(BaseModel):
            prompt: str
            chat: bool = False
            messages: list[ChatMessage] | None = None
            system_prompt: str | None = None
            prefill: str | None = None
            max_new_tokens: int = 0

        class AnalyzeReq(PromptParams):
            top_k: int = MAX_K

        class RankReq(PromptParams):
            request_id: str
            token_id: int | None = None
            token_str: str | None = None

        class IntervReq(PromptParams):
            kind: str = "swap"  # "swap" | "steer"
            source: str = ""
            target: str | None = None
            strength: float = 0.02
            layer_lo: int = WORKSPACE_BAND[0]
            layer_hi: int = WORKSPACE_BAND[1]

        def check_prompt(req):
            prompt = (req.prompt or "").strip()
            messages = req.messages or []
            has_history = any((m.content or "").strip() for m in messages)
            if not prompt and not (req.chat and has_history):
                raise HTTPException(400, "empty prompt")
            total = len(req.prompt or "")
            for m in messages:
                if m.role not in ("user", "assistant"):
                    raise HTTPException(400, f"invalid message role: {m.role}")
                total += len(m.content)
            if len(messages) > 24:
                raise HTTPException(400, "too many prior turns (max 24)")
            if total > MAX_CHARS:
                raise HTTPException(413, f"prompt too long (max {MAX_CHARS} chars)")

        @api.get("/healthz")
        def healthz():
            return {"ok": True}

        @api.get("/warmup")
        def warmup():
            # Reaching this handler means @modal.enter (model load) already ran.
            return {
                "status": "warm",
                "model": MODEL_NAME,
                "lens_layers": [self.lens.layers[0], self.lens.layers[-1]],
                "n_layers": self.final + 1,
                "version": 3,
                "limits": {"max_sequence": MAX_SEQUENCE, "max_new": MAX_NEW, "max_chars": MAX_CHARS},
            }

        @api.post("/api/analyze")
        def analyze(req: AnalyzeReq):
            check_prompt(req)
            k = max(1, min(req.top_k, MAX_K))
            t0 = time.time()
            try:
                truncated = self._prompt_token_count(req) > MAX_SEQUENCE
                with self.gpu_lock:
                    ids, acts, gen_start = self._forward_full(req)
                    t_fwd = time.time()
                    topks = {}
                    for name, use_j in [("jlens", True), ("logit_lens", False)]:
                        per = []
                        for l in self.lens.layers:
                            probs, idx = self._layer_logits(acts, l, use_j).softmax(-1).topk(k)
                            per.append((idx.cpu(), probs.cpu()))
                        topks[name] = per
                    m_probs, m_idx = (
                        self._layer_logits(acts, self.final, False).softmax(-1).topk(k)
                    )
                    model_top = (m_idx.cpu(), m_probs.cpu())
                    t_readout = time.time()
                rid = uuid.uuid4().hex[:12]
                self._cache_put(rid, ids, acts, gen_start)
                resp = self._build_response(rid, ids, topks, model_top, truncated)
                resp["gen_start"] = gen_start
                resp["token_budget"] = {
                    "max": MAX_SEQUENCE,
                    "prompt": gen_start,
                    "generated": ids.shape[1] - gen_start,
                }
                resp["completion"] = (
                    self.tok.decode(ids[0, gen_start:], skip_special_tokens=True)
                    if ids.shape[1] > gen_start else ""
                )
                resp["timing_ms"] = {
                    "forward": round((t_fwd - t0) * 1000),
                    "readout": round((t_readout - t_fwd) * 1000),
                    "total": round((time.time() - t0) * 1000),
                }
                self._log({
                    "endpoint": "analyze", "prompt": req.prompt[:2000],
                    "n_tokens": ids.shape[1], "top_k": k,
                    "duration_ms": round((time.time() - t0) * 1000), "status": "ok",
                })
                return resp
            except HTTPException:
                raise
            except Exception as e:
                self._log({"endpoint": "analyze", "prompt": req.prompt[:2000],
                           "status": "error", "error": str(e)[:500]})
                raise HTTPException(500, f"analyze failed: {e}")

        @api.post("/api/rank")
        def rank(req: RankReq):
            tid = req.token_id
            if tid is None:
                enc = self.tok.encode(req.token_str or "", add_special_tokens=False)
                if not enc:
                    raise HTTPException(400, "token_str tokenized to nothing")
                tid = enc[0]
            if not (0 <= tid < self.model.config.vocab_size):
                raise HTTPException(400, f"token_id out of range: {tid}")
            t0 = time.time()
            try:
                entry = self.cache.get(req.request_id)
                recomputed = False
                with self.gpu_lock:
                    if entry is None:  # container recycled or a different replica
                        check_prompt(req)
                        ids, acts, gen_start = self._forward_full(req)
                        self._cache_put(req.request_id, ids, acts, gen_start)
                        entry, recomputed = self.cache[req.request_id], True
                    acts = entry["acts"]
                    out = {
                        "jlens_ranks": [], "logit_lens_ranks": [],
                        "jlens_neighbors": [], "logit_lens_neighbors": [],
                    }
                    for l in self.lens.layers:
                        for name, use_j in [("jlens", True), ("logit_lens", False)]:
                            lg = self._layer_logits(acts, l, use_j)
                            ranks, neighbors = self._rank_and_neighbors(lg, tid)
                            out[f"{name}_ranks"].append(ranks.cpu().tolist())
                            out[f"{name}_neighbors"].append(neighbors)
                    mlg = self._layer_logits(acts, self.final, False)
                    model_ranks, model_neighbors = self._rank_and_neighbors(mlg, tid)
                    model_ranks = model_ranks.cpu().tolist()
                self._log({"endpoint": "rank", "prompt": req.prompt[:2000],
                           "n_tokens": len(model_ranks),
                           "duration_ms": round((time.time() - t0) * 1000), "status": "ok"})
                return {"token_id": tid, "token_text": self.tok.decode([tid]),
                        "recomputed": recomputed, "model_ranks": model_ranks,
                        "model_neighbors": model_neighbors,
                        "neighbor_window": RANK_NEIGHBOR_WINDOW, **out}
            except HTTPException:
                raise
            except Exception as e:
                self._log({"endpoint": "rank", "prompt": req.prompt[:2000],
                           "status": "error", "error": str(e)[:500]})
                raise HTTPException(500, f"rank failed: {e}")

        @api.post("/api/intervene")
        def intervene(req: IntervReq):
            check_prompt(req)
            source = (req.source or "").strip()
            if not source:
                raise HTTPException(400, "source token required")
            lo = max(self.lens.layers[0], min(req.layer_lo, self.lens.layers[-1]))
            hi = max(lo, min(req.layer_hi, self.lens.layers[-1]))
            layers = [l for l in self.lens.layers if lo <= l <= hi]
            n_new = max(1, self._gen_budget(req, self.jlens.encode(
                self.model, self.tok, self._render_text(req), max_len=MAX_SEQUENCE
            ).shape[1]))
            t0 = time.time()
            try:
                if req.kind == "swap":
                    target = (req.target or "").strip()
                    if not target:
                        raise HTTPException(400, "swap needs a target token")
                    pairs = pair_token_ids(self.tok, source, target)
                    if not pairs:
                        raise HTTPException(
                            400,
                            f"'{source}' and '{target}' have no matching single-token "
                            "surface forms — try shorter/more common words",
                        )
                    with self.gpu_lock:
                        edits = self.iv.swap_edits(self.lens_cpu, self.model, layers, pairs, alpha=1.0)
                    detail = {"pairs": [[self.tok.decode([a]), self.tok.decode([b])] for a, b in pairs]}
                elif req.kind == "steer":
                    tid = single_token_id(self.tok, source)
                    if tid is None:
                        raise HTTPException(400, f"'{source}' is not a single token in any surface form")
                    strength = max(-0.2, min(req.strength, 0.2))
                    with self.gpu_lock:
                        edits = self.iv.steer_edits(
                            self.lens_cpu, self.model, layers, tid, strength, self.mean_norms
                        )
                    detail = {"token": self.tok.decode([tid]), "strength": strength}
                else:
                    raise HTTPException(400, f"unknown kind: {req.kind}")

                text = self._render_text(req)
                with self.gpu_lock:
                    ids = self.jlens.encode(self.model, self.tok, text, max_len=MAX_SEQUENCE)
                    default_text = self._generate(ids, n_new)
                    default_top = self._next_top5(ids)
                    modified_text = self._generate(ids, n_new, edits)
                    modified_top = self._next_top5(ids, edits)
                self._log({"endpoint": f"intervene-{req.kind}", "prompt": req.prompt[:2000],
                           "n_tokens": ids.shape[1],
                           "duration_ms": round((time.time() - t0) * 1000), "status": "ok"})
                return {
                    "kind": req.kind, "layers": [layers[0], layers[-1]], **detail,
                    "default": {"text": default_text, "top": default_top},
                    "modified": {"text": modified_text, "top": modified_top},
                    "timing_ms": round((time.time() - t0) * 1000),
                }
            except HTTPException:
                raise
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._log({"endpoint": f"intervene-{req.kind}", "prompt": req.prompt[:2000],
                           "status": "error", "error": str(e)[:500]})
                raise HTTPException(500, f"intervene failed: {e}")

        return api

    # ---- response assembly ---------------------------------------------------

    def _build_response(self, rid, ids, topks, model_top, truncated):
        prompt_ids = ids[0].tolist()
        uniq = list(dict.fromkeys(
            prompt_ids
            + [t for per in topks.values() for idx, _ in per for t in idx.flatten().tolist()]
            + model_top[0].flatten().tolist()
        ))
        id2idx = {tid: i for i, tid in enumerate(uniq)}

        def pack(per):
            return {
                "topk": [[[id2idx[t] for t in row] for row in idx.tolist()] for idx, _ in per],
                "probs": [[[round(p, 4) for p in row] for row in pr.tolist()] for _, pr in per],
            }

        return {
            "request_id": rid,
            "truncated": truncated,
            "n_layers": self.final + 1,
            "lens_layers": self.lens.layers,
            "workspace_band": WORKSPACE_BAND,
            "strings": [self.tok.decode([t]) for t in uniq],
            "token_ids": uniq,
            "prompt_tokens": [id2idx[t] for t in prompt_ids],
            "jlens": pack(topks["jlens"]),
            "logit_lens": pack(topks["logit_lens"]),
            "model": {
                "topk": [[id2idx[t] for t in row] for row in model_top[0].tolist()],
                "probs": [[round(p, 4) for p in row] for row in model_top[1].tolist()],
            },
        }
