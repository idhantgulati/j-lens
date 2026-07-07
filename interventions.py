"""Residual-stream interventions with J-lens vectors (paper §2.5).

An edit is (layer, fn) where fn maps the block's output hidden states
[B, T, d] -> [B, T, d]. Edits run as forward hooks during a normal forward or
generate, so "clamped at every position" is the default behaviour.
"""

import torch

from jlens import encode, record_residuals


def apply_edits(model, edits):
    """Context manager registering the edits as forward hooks."""

    class _ctx:
        def __enter__(ctx):
            ctx.handles = []
            for layer, fn in edits:
                def hook(module, args, output, fn=fn):
                    h = output[0] if isinstance(output, tuple) else output
                    h2 = fn(h)
                    return (h2, *output[1:]) if isinstance(output, tuple) else h2
                ctx.handles.append(model.model.layers[layer].register_forward_hook(hook))
            return ctx

        def __exit__(ctx, *exc):
            for handle in ctx.handles:
                handle.remove()

    return _ctx()


def swap_edits(lens, model, layers, pairs, alpha=1.0):
    """Coordinate swap: exchange each pair's lens coordinates, leaving the rest
    of the activation unchanged (paper §2.5 / §3.1: subtract the projection
    onto the source lens vector, add an equal-magnitude projection onto the
    target's, and vice versa).

    Coordinates are inner products with unit-normed lens vectors rather than
    pinv(V) h: same-category lens vectors are highly correlated (cos 0.5-0.75
    on Qwen3.5-4B), which makes the pseudoinverse coordinates of §2.5's
    formulation ill-conditioned at this scale.

    pairs: [(source_token_id, target_token_id), ...], applied at every
    position of the given layers.
    """
    edits = []
    for layer in layers:
        ids = [i for pair in pairs for i in pair]
        V = lens.vectors(model, layer, ids)
        V = V / V.norm(dim=1, keepdim=True)  # [2m, d]
        perm = [i + 1 if i % 2 == 0 else i - 1 for i in range(len(ids))]

        def fn(h, V=V, perm=perm):
            Vd = V.to(device=h.device, dtype=torch.float32)
            c = h.float() @ Vd.T  # [B, T, 2m]
            return (h.float() + alpha * (c[..., perm] - c) @ Vd).to(h.dtype)

        edits.append((layer, fn))
    return edits


def ablate_edits(lens, model, layers, token_ids):
    """Project out each token's (unit-normed) lens direction at every position."""
    edits = []
    for layer in layers:
        V = lens.vectors(model, layer, token_ids)
        V = V / V.norm(dim=1, keepdim=True)  # [n, d]

        def fn(h, V=V):
            Vd = V.to(device=h.device, dtype=torch.float32)
            hf = h.float()
            return (hf - (hf @ Vd.T) @ Vd).to(h.dtype)

        edits.append((layer, fn))
    return edits


def steer_edits(lens, model, layers, token_id, strength, mean_norms):
    """h += strength * mean_norms[layer] * unit lens vector, every position."""
    edits = []
    for layer in layers:
        v = lens.vectors(model, layer, [token_id])[0]
        v = v / v.norm()

        def fn(h, v=v, s=strength * mean_norms[layer]):
            return (h.float() + s * v.to(h.device)).to(h.dtype)

        edits.append((layer, fn))
    return edits


@torch.no_grad()
def mean_residual_norms(model, tok, prompts, layers, max_len=128):
    """Mean residual-stream norm per layer over some prompts (steering scale)."""
    norms = {l: [] for l in layers}
    for p in prompts:
        ids = encode(model, tok, p, max_len=max_len)
        with record_residuals(model, layers) as rec:
            model.model(input_ids=ids, use_cache=False)
        for l in layers:
            norms[l].append(rec.acts[l][0].float().norm(dim=-1).mean().item())
    return {l: sum(v) / len(v) for l, v in norms.items()}


@torch.no_grad()
def logits_with(model, tok, prompt, edits=(), position=-1):
    """Next-token logits at `position` under the given edits."""
    ids = encode(model, tok, prompt)
    with apply_edits(model, edits):
        out = model(input_ids=ids, use_cache=False)
    return out.logits[0, position].float().cpu()


@torch.no_grad()
def generate_with(model, tok, prompt, edits=(), max_new_tokens=30):
    """Greedy generation under the given edits."""
    ids = encode(model, tok, prompt)
    with apply_edits(model, edits):
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
