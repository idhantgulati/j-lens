"""Lens evaluations: the six pass@k sets of paper §A.6, plus workspace-band
metrics (§4.1). Prompt data is downloaded from the official companion repo
(anthropics/jacobian-lens) into data/ on first use.
"""

import json
import math
import os
import urllib.request

import torch

RAW = "https://raw.githubusercontent.com/anthropics/jacobian-lens/main/data"
EVALS = ["multihop", "multilingual", "order-ops", "poetry", "typo", "association"]

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_OPS = {
    "multiplication": ["*", "×", "times", "multiply", "multiplication", "product"],
    "addition": ["+", "plus", "add", "addition", "sum"],
    "subtraction": ["-", "−", "minus", "subtract", "subtraction", "difference"],
    "division": ["/", "÷", "divided", "divide", "division", "quotient"],
}


def fetch(kind, name):
    """Download one JSON from the companion repo (cached under data/)."""
    os.makedirs("data", exist_ok=True)
    fn = f"lens-eval-{name}.json" if kind == "evaluations" else f"{name}.json"
    path = os.path.join("data", fn)
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"{RAW}/{kind}/{fn}", path)
    return json.load(open(path))


def _number_word(s):
    n = int(s)
    if n < 20:
        return _ONES[n]
    if n < 100 and n % 10 == 0:
        return _TENS[n // 10 - 2]
    if n < 100:
        return None  # "twenty-one" is multi-token anyway
    return None


def synonyms(word):
    """Surface variants of an intermediate (order-ops: digit/word + symbol/word)."""
    out = [word]
    if word.isdigit() and (w := _number_word(word)):
        out.append(w)
    out += _OPS.get(word.lower(), [])
    return out


def token_ids_of(tok, word):
    """Single-token ids for a word's surface forms (leading space, case)."""
    ids = set()
    for w in synonyms(word):
        for f in {w, w.lower(), w.capitalize(), " " + w, " " + w.lower(), " " + w.capitalize()}:
            t = tok.encode(f, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
    return sorted(ids)


def pair_token_ids(tok, source, target):
    """Token-id pairs for a swap, matched by surface form (leading space and
    case must agree within a pair, else the swap mixes unrelated directions)."""
    pairs = []
    for f in (lambda w: " " + w, lambda w: " " + w.lower(), lambda w: w,
              lambda w: w.lower()):
        a = tok.encode(f(source), add_special_tokens=False)
        b = tok.encode(f(target), add_special_tokens=False)
        if len(a) == 1 and len(b) == 1 and (a[0], b[0]) not in pairs:
            pairs.append((a[0], b[0]))
    return pairs


def rank_of(logits, ids):
    """Best (1-indexed) rank of any of `ids` in a [vocab] logit vector."""
    if not ids:
        return None
    best = logits[ids].max()
    return int((logits > best).sum().item()) + 1


def readout_position(tok, name, prompt):
    """Paper conventions: poetry reads at the newline ending line 1; all other
    sets read at the final prompt token."""
    if name != "poetry":
        return -1
    ids = tok(prompt, add_special_tokens=True).input_ids
    nl = [i for i, t in enumerate(ids) if "\n" in tok.decode([t])]
    return nl[-1] if nl else -1


def pass_at_k(model, tok, lens, name, ks=(1, 2, 5, 10, 25, 100), use_jacobian=True,
              layers=None):
    """pass@k per §A.6: an intermediate is recovered at k if its min-over-layers
    lens rank <= k. Returns (pass@k dict, normalized AUC over log k)."""
    items = fetch("evaluations", name)["items"]
    hits = {k: [] for k in ks}
    for item in items:
        prompt = item["prompt"].rstrip()  # a trailing space would become the readout token
        pos = readout_position(tok, name, prompt)
        out, _, _ = lens.readout(model, tok, prompt, layers=layers,
                                 positions=[pos], use_jacobian=use_jacobian)
        for word in item["intermediates"]:
            ids = token_ids_of(tok, word)
            if not ids:
                continue
            r = min(rank_of(out[l][0], ids) for l in out)
            for k in ks:
                hits[k].append(r <= k)
    curve = {k: sum(v) / len(v) for k, v in hits.items()}
    xs = [math.log(k) for k in ks]
    ys = [curve[k] for k in ks]
    auc = sum((ys[i] + ys[i + 1]) / 2 * (xs[i + 1] - xs[i]) for i in range(len(ks) - 1))
    return curve, auc / (xs[-1] - xs[0])


@torch.no_grad()
def band_metrics(model, tok, lens, prompts, top_k=10, max_len=128):
    """Per-layer workspace signatures (§4.1) over some prompts:
    excess kurtosis of lens logits, top-k agreement with the model's next
    token, and top-1 autocorrelation (adjacent positions) vs a shuffled null.
    """
    stats = {l: {"kurt": [], "agree": [], "auto": [], "auto0": []} for l in lens.layers}
    for p in prompts:
        outs, model_logits, ids = lens.readout(model, tok, p, max_len=max_len)
        next_tok = model_logits.argmax(dim=-1)  # model's top-1 at each position
        for l, lg in outs.items():
            z = (lg - lg.mean(dim=-1, keepdim=True)) / lg.std(dim=-1, keepdim=True)
            stats[l]["kurt"].append(((z ** 4).mean(dim=-1) - 3).mean().item())
            top = lg.topk(top_k, dim=-1).indices
            stats[l]["agree"].append((top == next_tok[:, None]).any(-1).float().mean().item())
            t1 = lg.argmax(dim=-1)
            same = (t1[:-1] == t1[1:]).float().mean().item()
            perm = t1[torch.randperm(len(t1))]
            stats[l]["auto"].append(same)
            stats[l]["auto0"].append((perm[:-1] == perm[1:]).float().mean().item())
    agg = lambda l, k: sum(stats[l][k]) / len(stats[l][k])
    return {l: {"kurtosis": agg(l, "kurt"), "next_token_agree": agg(l, "agree"),
                "autocorr": agg(l, "auto") - agg(l, "auto0")} for l in lens.layers}
