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
    """Coordinate swap (paper §2.5): exchange the two tokens' lens coordinates,
    leaving the component outside span{v_s, v_t} unchanged.

    For unit-normed vectors the pinv-coordinate swap h + V(sigma(c) - c) is
    exactly a reflection across the pair's bisector hyperplane,
    h - 2 (h.u) u with u ∝ v_s - v_t, which we use because it stays
    well-conditioned when v_s and v_t are highly correlated (cos 0.5-0.75 for
    same-category tokens on Qwen3.5-4B). alpha scales the exchanged amount
    (alpha=1 is the exact swap; alpha=2 the paper's "double strength").

    pairs: [(source_token_id, target_token_id), ...], applied at every
    position of the given layers.
    """
    edits = []
    for layer in layers:
        us = []
        for s, t in pairs:
            V = lens.vectors(model, layer, [s, t])
            V = V / V.norm(dim=1, keepdim=True)
            u = V[0] - V[1]
            us.append(u / u.norm())
        U = torch.stack(us)  # [m, d]

        def fn(h, U=U):
            Ud = U.to(device=h.device, dtype=torch.float32)
            hf = h.float()
            for u in Ud:  # sequential reflections (pair directions overlap)
                hf = hf - 2 * alpha * (hf @ u)[..., None] * u
            return hf.to(h.dtype)

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
