# lsa-rocm

Experiments on Latent Sparse Attention (LSA) key retrieval on AMD ROCm. The question, models, gates and
agent batches are in [docs/plan.md](docs/plan.md). Metric definitions are in [docs/metrics.md](docs/metrics.md).

Status: planning. No experiments run yet. See `FINDINGS.md` (not created until there is something to report).

Layout: `docs/` plan and definitions, `src/lsa_rocm/` shared code, `tests/`, `experiments/NNN-name/` one folder per
experiment (hypothesis, command, result, conclusion), `results/` run JSON, `third_party/` upstream pinned and unmodified,
`agents/reports/` agent reports.
