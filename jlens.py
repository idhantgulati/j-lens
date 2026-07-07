"""Minimal Jacobian lens (Gurnee et al. 2026, transformer-circuits.pub/2026/workspace).

The lens reads out a residual-stream activation h_l by transporting it into the
final-layer basis with the corpus-averaged Jacobian J_l = E[dh_final / dh_l],
then decoding with the model's own unembedding:

    lens_l(h) = unembed(J_l @ h)

Estimator (paper A.7): per prompt, one forward pass caching every layer's
residual, then one backward pass per output dimension (batched dim_batch at a
time), injecting a one-hot cotangent at every valid target position. The
gradient at source position t is then sum_{t'>=t} dz[t']/dh_l[t]; we mean over
source positions, then over prompts. Written for Qwen-style HF decoders.
"""

import math

import torch

SKIP_FIRST = 16  # early positions are attention sinks; excluded from the average


def load_model(name, dtype=torch.bfloat16, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(name)
    return model, tok


def encode(model, tok, text, max_len=None):
    ids = tok(text, return_tensors="pt", truncation=max_len is not None, max_length=max_len)
    return ids.input_ids.to(model.device)


def unembed(model, h):
    """Final norm + LM head on a [..., d_model] tensor."""
    w = model.lm_head.weight
    return model.lm_head(model.model.norm(h.to(dtype=w.dtype, device=w.device)))


class record_residuals:
    """Forward hooks capturing block outputs at the given layer indices.

    Tensors are not detached, so they can be passed to autograd.grad. If
    graph_root is set, that layer's output is marked requires_grad so the
    retained graph starts there (all params are frozen).
    """

    def __init__(self, model, at, graph_root=None):
        self.blocks = model.model.layers
        self.at = sorted(set(at) | ({graph_root} if graph_root is not None else set()))
        self.graph_root = graph_root
        self.acts = {}
        self._handles = []

    def _hook(self, idx):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if idx == self.graph_root:
                h.requires_grad_(True)
            self.acts[idx] = h

        return hook

    def __enter__(self):
        for i in self.at:
            self._handles.append(self.blocks[i].register_forward_hook(self._hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()


def jacobian_for_prompt(model, tok, prompt, source_layers, target_layer=-1,
                        dim_batch=8, max_len=128, skip_first=SKIP_FIRST):
    """Per-prompt J_l estimate for each source layer: fp32 CPU [d, d] tensors."""
    n_layers = len(model.model.layers)
    target_layer = target_layer % n_layers
    d = model.config.hidden_size

    ids = encode(model, tok, prompt, max_len=max_len)
    T = ids.shape[1]
    if T <= skip_first + 1:
        raise ValueError(f"prompt too short ({T} tokens)")
    valid = torch.arange(skip_first, T - 1, device=model.device)

    js = {l: torch.zeros(d, d) for l in source_layers}
    n_passes = math.ceil(d / dim_batch)
    with record_residuals(model, [*source_layers, target_layer],
                          graph_root=min(source_layers)) as rec, torch.enable_grad():
        model.model(input_ids=ids.expand(dim_batch, -1), use_cache=False)
        z = rec.acts[target_layer]  # [dim_batch, T, d]
        sources = [rec.acts[l] for l in source_layers]
        cot = torch.zeros_like(z)
        rows = torch.arange(dim_batch, device=z.device)
        for p in range(n_passes):
            lo = p * dim_batch
            n = min(dim_batch, d - lo)
            cot.zero_()
            cot[rows[:n, None], valid[None, :], lo + rows[:n, None]] = 1.0
            grads = torch.autograd.grad(z, sources, grad_outputs=cot,
                                        retain_graph=p < n_passes - 1)
            for l, g in zip(source_layers, grads):
                js[l][lo:lo + n] = g[:n, valid, :].float().mean(dim=1).cpu()
    return js


def load_wikitext(n_prompts, min_chars=600):
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="train", streaming=True)
    out = []
    for rec in ds:
        if len(rec["text"].strip()) >= min_chars:
            out.append(rec["text"])
            if len(out) == n_prompts:
                break
    return out


def fit(model, tok, prompts, layers=None, target_layer=-1, dim_batch=8,
        max_len=128, checkpoint=None, log_every=10):
    """Average jacobian_for_prompt over prompts -> JLens. Resumes from checkpoint."""
    import os

    n_layers = len(model.model.layers)
    target_layer = target_layer % n_layers
    layers = list(range(target_layer)) if layers is None else sorted(layers)
    d = model.config.hidden_size

    total, done = {l: torch.zeros(d, d) for l in layers}, 0
    if checkpoint and os.path.exists(checkpoint):
        state = torch.load(checkpoint, weights_only=True)
        total, done = state["sum"], state["done"]
        print(f"resuming from {checkpoint}: {done}/{len(prompts)} prompts")
    for i, prompt in enumerate(prompts):
        if i < done:
            continue
        try:
            js = jacobian_for_prompt(model, tok, prompt, layers,
                                     target_layer=target_layer,
                                     dim_batch=dim_batch, max_len=max_len)
        except ValueError as e:
            print(f"skipping prompt {i}: {e}")
        else:
            for l in layers:
                total[l] += js[l]
        done = i + 1
        if checkpoint and done % log_every == 0:
            torch.save({"sum": total, "done": done}, checkpoint)
        if done % log_every == 0:
            print(f"{done}/{len(prompts)}")
    if checkpoint:
        torch.save({"sum": total, "done": done}, checkpoint)
    return JLens({l: total[l] / max(done, 1) for l in layers}, n_prompts=done)


class JLens:
    """Per-layer J_l matrices plus readout and lens-vector helpers."""

    def __init__(self, jacobians, n_prompts=0):
        self.J = {l: j.float() for l, j in jacobians.items()}
        self.layers = sorted(self.J)
        self.n_prompts = n_prompts

    def __repr__(self):
        return (f"JLens(layers={self.layers[0]}..{self.layers[-1]}, "
                f"n_prompts={self.n_prompts})")

    def save(self, path):
        torch.save({"J": {l: j.half() for l, j in self.J.items()},
                    "n_prompts": self.n_prompts}, path)

    @classmethod
    def load(cls, path):
        s = torch.load(path, map_location="cpu", weights_only=True)
        return cls(s["J"], n_prompts=s.get("n_prompts", 0))

    @classmethod
    def from_pretrained(cls, repo, filename, revision=None):
        """Load a released lens from the HF Hub (e.g. neuronpedia/jacobian-lens)."""
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo, filename, revision=revision)
        s = torch.load(path, map_location="cpu", weights_only=True)
        return cls(s["J"], n_prompts=s.get("n_prompts", 0))

    def transport(self, h, layer):
        J = self.J[layer].to(device=h.device, dtype=torch.float32)
        return h.float() @ J.T

    @torch.no_grad()
    def readout(self, model, tok, prompt, layers=None, positions=None,
                use_jacobian=True, max_len=1024):
        """Lens logits at (layers x positions) plus the model's own logits.

        Returns (lens_logits: {layer: [P, vocab]}, model_logits [P, vocab], ids).
        use_jacobian=False skips the transport (logit-lens baseline).
        """
        layers = self.layers if layers is None else list(layers)
        final = len(model.model.layers) - 1
        ids = encode(model, tok, prompt, max_len=max_len)
        with record_residuals(model, [*layers, final]) as rec:
            model.model(input_ids=ids, use_cache=False)
            acts = {l: rec.acts[l][0].detach() for l in {*layers, final}}
        sel = (lambda h: h) if positions is None else (lambda h: h[list(positions)])
        out = {}
        for l in layers:
            h = sel(acts[l])
            out[l] = unembed(model, self.transport(h, l) if use_jacobian else h).float().cpu()
        model_logits = unembed(model, sel(acts[final])).float().cpu()
        return out, model_logits, ids

    def vectors(self, model, layer, token_ids):
        """J-lens vectors: rows of W_U diag(g) J_l for the given tokens, [n, d].

        (Paper: rows of W_U J_l; we fold in the final RMSNorm's elementwise
        scale g so the vectors match the actual readout path.)
        """
        w = model.lm_head.weight[list(token_ids)].float().cpu()  # [n, d]
        g = model.model.norm.weight.float().cpu()
        return (w * g) @ self.J[layer]

    def decompose(self, model, layer, h, k=16, n_candidates=512):
        """Sparse nonnegative pursuit of h against this layer's lens vectors.

        Greedy: candidates are the top-n_candidates tokens by lens logit; each
        step adds the atom with the largest positive correlation to the
        residual and refits coefficients by nonneg least squares. Returns
        (token_ids [k], coefs [k], reconstruction [d]).
        """
        h = h.detach().float().cpu()
        logits = unembed(model, self.transport(h, layer)).float().cpu()
        cand = logits.topk(n_candidates).indices
        V = self.vectors(model, layer, cand.tolist())
        V = V / V.norm(dim=1, keepdim=True)  # unit atoms; coefs absorb scale
        picked, coef = [], torch.zeros(0)
        resid = h.clone()
        for _ in range(k):
            corr = V @ resid
            corr[picked] = -torch.inf
            i = int(corr.argmax())
            if corr[i] <= 0:
                break
            picked.append(i)
            A = V[picked].T  # [d, m]
            coef = torch.linalg.lstsq(A, h).solution.clamp(min=0)
            resid = h - A @ coef
        return cand[picked], coef, h - resid
