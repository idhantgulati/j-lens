# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A minimal, faithful reimplementation of the Jacobian lens and J-space methods from "Verbalizable Representations Form a Global Workspace in Language Models" (Gurnee et al., 2026, transformer-circuits.pub/2026/workspace), on Qwen3.5-4B, plus a deployed web visualizer (`visualizer/`). The research code has nothing beyond the paper's core methods — no packaging, no CLI, no config system. Deliberately out of scope: tuned-lens baseline, SAE/κ analyses, broadcast-head analyses, template/oracle lens, ablation task battery.

The untracked `knowledge/` directory (present locally, gitignored) is the authoritative spec: `knowledge/implementation.md` has the full design, empirical bring-up findings, and verified results; `knowledge/paper-notes.md` has paper notes. Read `implementation.md` before making changes. `paper/` holds a local copy of the paper itself.

## Running

There are no tests, lint, or build. Dependencies: `pip install torch transformers datasets huggingface_hub matplotlib`.

Fitting and evaluation need a GPU. Dev flow: write code locally → commit and push to https://github.com/idhantgulati/j-lens → pull and run on the Modal box (volume `j-lens` for HF caches and lens checkpoints, treated as scratch) → fix locally, repeat. The Modal SSH address changes per box; check memory or ask the user.

`main.ipynb` is the entry point: fit/load lens → readout → workspace band → pass@k evals → interventions → J-space decomposition. Keep notebook cells short (~10 lines); anything longer belongs in the modules.

## Architecture

- `jlens.py` — the method. `fit()` estimates $J_\ell = \mathbb{E}[\partial h_{\text{final}} / \partial h_\ell]$ per the paper's §A.7 estimator (one forward per prompt with the prompt replicated `dim_batch` times, one-hot cotangents, batched backward passes on one retained graph; fp32 CPU accumulation, checkpoint/resume). `JLens` holds `{layer: J}` with `readout()` (lens logits via `unembed(J @ h)`; `use_jacobian=False` gives the logit-lens baseline), `vectors()` (rows of $W_U\,\mathrm{diag}(g)\,J_\ell$ — the final RMSNorm scale $g$ is folded in so vectors match the readout path), `decompose()` (greedy nonneg matching pursuit), and `from_pretrained()` (released lens on HF Hub `neuronpedia/jacobian-lens`). `record_residuals` is the shared forward-hook context manager for capturing block outputs.
- `interventions.py` — residual-stream edits as forward hooks: `swap_edits`, `ablate_edits`, `steer_edits` build `(layer, fn)` edit lists; `apply_edits` registers them; `logits_with`/`generate_with` run under them. Depends on `jlens.py` for `encode`/`record_residuals` and lens vectors.
- `evals.py` — the six §A.6 pass@k sets and §4.1 workspace-band metrics. Eval JSON downloads from the companion repo `anthropics/jacobian-lens` into `data/` (cached, gitignored).

Model access is Qwen-specific, no layout autodetection: blocks at `model.model.layers`, final norm `model.model.norm`, unembed = `lm_head(norm(·))`.

## Visualizer (`visualizer/`)

Interactive per-layer readout explorer (paper's Figure 5 plus chat mode, generation, pinned-token rank heatmaps, J-Space decomposition, swap/steer interventions). `visualizer/README.md` is the authoritative doc — architecture, full API schema, deploy, Supabase logging, cost guardrails. Live: frontend https://j-lens-visualizer.vercel.app, API https://order-evaluation--jlens-api.modal.run.

- **Frontend** (`visualizer/frontend/`) — static vanilla JS, no framework, no build step: `index.html` + `app.js` + `style.css`; the API base URL lives in `config.js`. Local dev: `cd visualizer && npm run dev` (serves on :3000 against the live Modal API).
- **Backend** (`visualizer/backend/app.py`) — Modal GPU class (L40S) exposing FastAPI: loads Qwen3.5-4B bf16 + the released n=1000 lens once per container, scales to zero after 5 idle min. Endpoints: `/api/analyze` (per-layer top-k), `/api/rank` (pinned-token ranks; activations LRU-cached per `request_id`), `/api/intervene` (swap/steer), plus `/healthz` and `/warmup` (frontend pings it on page load to hide cold start).
- **Deploy**: backend `modal deploy visualizer/backend/app.py` from the repo root (the image ships root-level `jlens.py`); frontend `cd visualizer/frontend && vercel deploy --prod --yes`. Both also push-to-deploy on `main`: Vercel is git-connected (root dir `visualizer/frontend`), and `.github/workflows/deploy-backend.yml` redeploys Modal when `visualizer/backend/**` or `jlens.py` change — so **editing `jlens.py` redeploys the backend**.
- Smoke test after backend changes: `curl $API/warmup` then POST `/api/analyze` with a short prompt (see visualizer/README.md).
- **Before every commit that touches the visualizer, bump the version number** in both places it appears: the footer `version-line` in `visualizer/frontend/index.html` (also update its "last updated" date) and `version` in `visualizer/package.json` — keep the two in sync.

## Non-obvious constraints (learned during bring-up; details in knowledge/implementation.md)

- **Swap is implemented as a bisector reflection**, not the paper's pinv-coordinate form: the pinv version is ill-conditioned at 4B scale (same-category lens vectors have cosine 0.5–0.75). For unit vectors the reflection $h - 2(h\cdot\hat u)\hat u$, $\hat u \propto \hat v_s - \hat v_t$, is the identical operation. With this form α=1 is the exact swap and α=2 overshoots — don't "double strength."
- Swap token pairs must match surface form (leading space and case must agree within a pair) — use `evals.pair_token_ids`.
- Eval prompts need `.rstrip()` — a trailing space becomes the readout/generation token and wrecks baselines.
- Qwen3.5-4B's chat template opens a `<think>` block by default; pass `enable_thinking=False` for verbal-report protocols.
- Qwen3.5-4B is a hybrid linear-attention model — fitting is slow (~100 s/prompt at max_len=128) and `dim_batch=8` fits a 40 GB A100 (`dim_batch=16` does not).
- The first 16 positions are skipped during fitting (attention sinks, `SKIP_FIRST`).
- Expected numbers when validating changes: n=100 lens matches the released n=1000 lens at per-layer cosine ≥0.97 above layer 4; workspace band ≈ L12–28; multihop pass@k AUC ≈ 0.39 (J-lens) vs 0.24 (logit lens). Swap success at 4B is weak (~13/81) — that's expected, not a bug.

## Repo hygiene

- Tracked artifacts: the notebook, the three modules, README, and `visualizer/` (frontend, backend, its README, deploy workflow). `paper/`, `knowledge/`, `data/`, `*.pt`, and `visualizer/.env` (Supabase keys) stay untracked.
- Commits: small, plain messages ("add fitting", "notebook: swap demo"); no attribution footers.
