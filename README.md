# j-lens

Minimal implementation of the Jacobian lens and J-space methods from
[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
(Gurnee et al., 2026), on Qwen3.5-4B.

- `jlens.py` — fitting the lens ($J_\ell = \mathbb{E}[\partial h_{\text{final}} / \partial h_\ell]$), readout, lens vectors, sparse J-space decomposition
- `interventions.py` — steering, ablation, and lens-coordinate swaps
- `evals.py` — the §A.6 pass@k evaluations and §4.1 workspace-band metrics
- `main.ipynb` — the full run: fit/load → read out → evaluate → intervene

Eval prompt data is downloaded from the official companion repo
([anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)) on first use.

```
pip install torch transformers datasets huggingface_hub matplotlib
```
