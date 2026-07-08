"""J-Lens Visualizer backend — Modal app serving per-layer Jacobian-lens readouts.

One GPU class loads Qwen3.5-4B plus the released neuronpedia n=1000 lens once per
container and serves a FastAPI app. Endpoints:

    GET  /healthz       liveness, no GPU work
    GET  /warmup        touches the GPU container so page load hides the cold start
    POST /api/analyze   {prompt, top_k} -> per-(layer, position) top-k for J-lens,
                        logit-lens, and the model's own output
    POST /api/rank      {request_id, prompt, token_id|token_str} -> the token's rank
                        at every (layer, position), for pin heatmaps

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
MAX_TOKENS = 200
MAX_K = 10
CACHE_SIZE = 8  # cached activations, ~32 MB each
WORKSPACE_BAND = [12, 29]  # from main.ipynb: round(0.38*n)..round(0.92*n)-1

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
)


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

        self.torch = torch
        self.jlens = jlens
        self.model, self.tok = jlens.load_model(MODEL_NAME)
        self.lens = jlens.JLens.from_pretrained(LENS_REPO, LENS_FILE, revision=LENS_REVISION)
        # Pre-move J to the GPU so transport()'s .to(device) is a no-op per request.
        self.lens.J = {l: j.to("cuda") for l, j in self.lens.J.items()}
        self.final = len(self.model.model.layers) - 1
        self.gpu_lock = threading.Lock()
        self.cache = OrderedDict()  # request_id -> {"ids": [1,T] cpu, "acts": {layer: [T,d] fp16 cpu}}
        self.sb = None
        if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"):
            try:
                from supabase import create_client

                self.sb = create_client(
                    os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"]
                )
            except Exception as e:
                print(f"supabase init failed, logging disabled: {e}")

    # ---- compute ---------------------------------------------------------

    def _forward(self, prompt):
        """One forward pass; fp16 CPU activations for every lens layer + final."""
        ids = self.jlens.encode(self.model, self.tok, prompt, max_len=MAX_TOKENS)
        want = [*self.lens.layers, self.final]
        with self.torch.no_grad(), self.jlens.record_residuals(self.model, want) as rec:
            self.model.model(input_ids=ids, use_cache=False)
            acts = {l: rec.acts[l][0].detach().half().cpu() for l in want}
        return ids, acts

    def _layer_logits(self, acts, layer, use_jacobian):
        """[P, vocab] fp32 logits on GPU for one layer; caller discards after use."""
        h = acts[layer].to("cuda", self.torch.float32)
        z = self.lens.transport(h, layer) if use_jacobian else h
        return self.jlens.unembed(self.model, z).float()

    def _cache_put(self, rid, ids, acts):
        self.cache[rid] = {"ids": ids.cpu(), "acts": acts}
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

    # ---- web app ---------------------------------------------------------

    @modal.asgi_app(label="jlens-api")
    def web(self):
        import torch
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.gzip import GZipMiddleware
        from pydantic import BaseModel

        api = FastAPI(title="J-Lens Visualizer API")
        api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
        api.add_middleware(GZipMiddleware, minimum_size=1000)

        class AnalyzeReq(BaseModel):
            prompt: str
            top_k: int = MAX_K

        class RankReq(BaseModel):
            request_id: str
            prompt: str
            token_id: int | None = None
            token_str: str | None = None

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
            }

        @api.post("/api/analyze")
        def analyze(req: AnalyzeReq):
            prompt = req.prompt
            if not prompt.strip():
                raise HTTPException(400, "empty prompt")
            if len(prompt) > MAX_CHARS:
                raise HTTPException(413, f"prompt too long (max {MAX_CHARS} chars)")
            k = max(1, min(req.top_k, MAX_K))
            t0 = time.time()
            try:
                truncated = len(self.tok(prompt).input_ids) > MAX_TOKENS
                with self.gpu_lock:
                    ids, acts = self._forward(prompt)
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
                self._cache_put(rid, ids, acts)
                resp = self._build_response(rid, ids, topks, model_top, truncated)
                resp["timing_ms"] = {
                    "forward": round((t_fwd - t0) * 1000),
                    "readout": round((t_readout - t_fwd) * 1000),
                    "total": round((time.time() - t0) * 1000),
                }
                self._log({
                    "endpoint": "analyze", "prompt": prompt[:2000],
                    "n_tokens": ids.shape[1], "top_k": k,
                    "duration_ms": round((time.time() - t0) * 1000), "status": "ok",
                })
                return resp
            except HTTPException:
                raise
            except Exception as e:
                self._log({"endpoint": "analyze", "prompt": prompt[:2000],
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
                        if not req.prompt.strip() or len(req.prompt) > MAX_CHARS:
                            raise HTTPException(400, "cache miss and no valid prompt to recompute")
                        ids, acts = self._forward(req.prompt)
                        self._cache_put(req.request_id, ids, acts)
                        entry, recomputed = self.cache[req.request_id], True
                    acts = entry["acts"]
                    out = {"jlens_ranks": [], "logit_lens_ranks": []}
                    for l in self.lens.layers:
                        for name, use_j in [("jlens_ranks", True), ("logit_lens_ranks", False)]:
                            lg = self._layer_logits(acts, l, use_j)
                            r = 1 + (lg > lg[:, tid : tid + 1]).sum(-1)
                            out[name].append(r.int().cpu().tolist())
                    mlg = self._layer_logits(acts, self.final, False)
                    model_ranks = (1 + (mlg > mlg[:, tid : tid + 1]).sum(-1)).int().cpu().tolist()
                self._log({"endpoint": "rank", "prompt": req.prompt[:2000],
                           "n_tokens": len(model_ranks),
                           "duration_ms": round((time.time() - t0) * 1000), "status": "ok"})
                return {"token_id": tid, "token_text": self.tok.decode([tid]),
                        "recomputed": recomputed, "model_ranks": model_ranks, **out}
            except HTTPException:
                raise
            except Exception as e:
                self._log({"endpoint": "rank", "prompt": req.prompt[:2000],
                           "status": "error", "error": str(e)[:500]})
                raise HTTPException(500, f"rank failed: {e}")

        return api

    # ---- response assembly -------------------------------------------------

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
