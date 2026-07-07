# J-Lens Visualizer

Web app for exploring per-layer Jacobian-lens readouts on Qwen3.5-4B, mirroring the
interactive Figure 5 of Gurnee et al. 2026 (transformer-circuits.pub/2026/workspace).
Type a prompt; see, for every (layer × position), the top-k tokens the lens reads out
of the residual stream — J-lens vs logit-lens — plus pinned-token rank heatmaps and
rank-trajectory charts.

- **Live frontend:** https://j-lens-visualizer.vercel.app (static, Vercel)
- **API:** https://order-evaluation--jlens-api.modal.run (Modal, GPU)

## Architecture

```
Browser (Vercel static site)  ──CORS──►  Modal app "j-lens-visualizer"
  frontend/{index.html,app.js,           backend/app.py: @app.cls GPU L40S,
  style.css,config.js}                   FastAPI via @modal.asgi_app(label="jlens-api")
                                         Qwen3.5-4B bf16 + neuronpedia n=1000 lens
                                         Volume "j-lens" (HF_HOME=/vol/j-lens/hf)
```

The GPU class loads the model and lens once per container (`@modal.enter`), keeps the
J matrices on the GPU, and **scales to zero** after 5 idle minutes (`scaledown_window=300`).
The frontend pings `GET /warmup` on page load so the container boots while the user
types (cold start ≈ 20 s thanks to the volume-cached weights; warm requests ≈ 0.3–0.5 s).
Concurrency: `@modal.concurrent(max_inputs=4)` per container with a GPU lock serializing
compute; Modal autoscales up to `max_containers=2` under load — requests queue beyond that.

Per request, one forward pass captures residuals at layers 0–31
(`jlens.record_residuals`); per-layer logits are computed on-GPU
(`unembed(transport(h, l))` for J-lens, `unembed(h)` for logit-lens) and only top-k
arrays leave the GPU. Activations (~32 MB fp16) are LRU-cached per `request_id` so
`/api/rank` (pinning) recomputes a token's rank at every (layer, position) in ~0.3 s;
on cache miss it re-runs the forward from the prompt (`"recomputed": true`).

## API

- `GET /healthz` — liveness, no GPU.
- `GET /warmup` — boots/touches the GPU container.
- `POST /api/analyze` `{prompt, top_k}` → `strings` (dedup token table),
  `prompt_tokens`, `jlens`/`logit_lens` `{topk, probs}` as `[31][P][k]` indices into
  `strings`, `model` `{topk, probs}` `[P][k]`, `workspace_band`, `request_id`,
  `timing_ms`. Prompt capped at 4000 chars / 200 tokens; `top_k` clamped to [1, 10].
- `POST /api/rank` `{request_id, prompt, token_id|token_str}` →
  `jlens_ranks`/`logit_lens_ranks` `[31][P]`, `model_ranks` `[P]` (1-based).

## Deploy

```bash
# backend (from the repo root; ships ../jlens.py into the image)
modal deploy Visualizer/backend/app.py
modal app logs j-lens-visualizer          # inspect requests/timings

# frontend (config.js holds the API URL)
cd Visualizer/frontend && vercel deploy --prod --yes
```

Smoke tests:

```bash
API=https://order-evaluation--jlens-api.modal.run
curl $API/warmup
curl -X POST $API/api/analyze -H 'content-type: application/json' \
  -d '{"prompt":"Fact: The currency used in the country shaped like a boot is the","top_k":10}'
```

## Push-to-deploy

Both halves deploy automatically on `git push` to `main`:

- **Frontend** — the Vercel project is git-connected to `idhantgulati/j-lens` with
  root directory `Visualizer/frontend`; pushes to `main` go to production, other
  branches get preview URLs.
- **Backend** — `.github/workflows/deploy-backend.yml` runs
  `modal deploy Visualizer/backend/app.py` when `Visualizer/backend/**` or `jlens.py`
  change (GitHub secrets `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`).

## Supabase logging (optional)

The backend logs each request (endpoint, prompt, timing, status) to Supabase if the
`supabase-jlens` Modal secret has real values; otherwise logging is silently disabled.

1. Create a Supabase project, then run in its SQL editor:

```sql
create table if not exists jlens_requests (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  endpoint text not null,
  prompt text,
  n_tokens int,
  top_k int,
  duration_ms real,
  status text,
  error text
);
alter table jlens_requests enable row level security;  -- service key bypasses RLS
```

2. Copy `.env.example` to `.env`, fill `SUPABASE_URL` + `SUPABASE_SERVICE_KEY`, then:

```bash
cd Visualizer && set -a && source .env && set +a
modal secret create supabase-jlens SUPABASE_URL="$SUPABASE_URL" \
  SUPABASE_SERVICE_KEY="$SUPABASE_SERVICE_KEY" --force
modal deploy backend/app.py   # restart containers so they pick up the secret
```

## Cost guardrails

`max_containers=2` (≤ 2 × L40S at ~$1.95/hr only while active), 5-min scale-down to
zero, 200-token prompt cap, top-k ≤ 10. GPU headroom: model + lens use ~11 GB of 48 GB;
switch `gpu="L40S"` to `"A10"` in `backend/app.py` for a cheaper (slower) fallback.
