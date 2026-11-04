# Plan: Latent Sparse Attention retrieval on AMD ROCm

Status: draft, 2026-11-03. Nothing below has been run yet.

## Question

Does LSA-style key retrieval (cluster the keys, search centroids, gather the top keys) find the keys
a model actually attends to, on the full-attention layers of current models, and is it cheaper than
exhaustive scoring on an AMD GPU (R9700)?

Not in scope: porting upstream serving code, SGLang, CUDA/CUTLASS kernels, training or distillation.

## Models

| Role | Model | Why | Confirmed |
|---|---|---|---|
| Primary | Ornith 1.5 9B (local, `/ai/models/llm/Ornith-1.5-9B`) | Qwen3.5-family hybrid, 8 of 32 layers full attention (16/4 heads, hd 256). Same layout the LSA paper evaluates (Qwen3.8-27B). No download needed. | config read, B0d confirmed on disk |
| Full-attention model | `Nanbeige/Nanbeige4.3-33B` | 52 layers, 56/8 GQA, hd 128, Oct 2026, dense. bf16 is 66 GB, so quantized or on the second GPU pair. | no hybrid markers in config, B0b verifies |
| Scale (hybrid) | Qwen3.8-27B (quantized) | the paper's own model, 16 of 64 layers full attention | config read |
| Harness debugging | Qwen3.5-0.8B (local) | tiny, fast, same architecture as Ornith | B0d confirmed on disk |
| Later / control | gpt-oss-20b | half sliding-window, attention sinks | not downloaded |

Dropped: `granite-switch-4.1-8b`. It is a full-attention model but not an interesting one, and its adapter
mechanism changes K/V. Its B0a report is kept in `agents/reports/` as a record.

Rule: only models created after 2026-05-03 as primaries. gpt-oss-20b (Aug 2025) is allowed as a control only.
The LSA paper itself uses hybrid and MLA models (Qwen3.8-27B, GLM-5.3-Flash), so hybrids are in scope from the
start. A pure full-attention model (Nanbeige) is the less-tested case and the one that decides whether the
method generalises.

## Gates

Thresholds are proposals. Confirm them after B0c shows what the paper reports.

- **Gate 1 (after B2):** on Ornith and then Nanbeige, at a budget of 10% of keys scored, tree search captures >= 0.90 of the true
  attention mass on a majority of full-attention layers at 32K. Otherwise stop and write up the negative result.
- **Gate 2 (after B3):** centroid scoring plus gather is faster than exhaustive scoring on the R9700 at 32K or above.
  Otherwise the method is accuracy-only on this hardware, and that is the finding.
- **Gate 3:** only then run the remaining models (B4) and the CPU-memory KV idea.

## Rules for agents

1. Work only inside the paths named in your task. Do not edit `docs/metrics.md` or `docs/plan.md`.
2. Do not download anything over 1 GB. Write the command and size in your report instead.
3. GPU jobs run on the R9700 only. B0d records the device index. Until then, no GPU jobs.
4. Every run writes a JSON file under `results/` containing: git commit, model id and revision, torch and ROCm versions,
   device name, seed, config. No hand-copied numbers.
5. Write a report to `agents/reports/<task-id>.md`: what you did, exact commands, what passed or failed, open questions.
   Report failures plainly. Do not claim a test passes without showing its output.
6. Do not touch `third_party/` except to clone and pin (B0c).

## Batches

Run batches in order. Tasks inside a batch are independent and run in parallel.
I review every report before the next batch starts.

### B0 Verify and probe (read-only, no GPU)

| ID | Task | Output |
|---|---|---|
| B0a | (done, then dropped: Granite-switch is no longer a subject. Report kept as a record.) | `agents/reports/B0a.md` |
| B0b | Same for Nanbeige4.3-33B. Also state size on disk at bf16 and which quants exist. | `agents/reports/B0b.md` |
| B0c | Read the upstream LSA repo and paper. Report: models and context lengths they evaluated, exact `lsa.core` API, what the indexer takes as input (pre- or post-RoPE keys, per head or shared), and CPU-only install steps. Clone and pin it into `third_party/lsa` at a recorded commit. | `agents/reports/B0c.md`, `third_party/lsa` |
| B0d | Probe the machine: ROCm and PyTorch versions, which GPU index is the R9700, free VRAM on each, transformers version, which of the models above are already in `/ai/huggingface`, and download sizes for those that are not. Do not download. | `docs/env.md`, `agents/reports/B0d.md` |

### B1 Foundations (after B0)

| ID | Task | Output |
|---|---|---|
| B1a | Implement the metrics in `docs/metrics.md` as pure functions with unit tests on hand-made tensors where the answer is known. | `src/lsa_rocm/metrics.py`, `tests/test_metrics.py` |
| B1b | Wrap `lsa.core` as the CPU oracle. Tests: deterministic small inputs where flat search with a full budget returns exactly the exhaustive top-k. | `src/lsa_rocm/oracle.py`, `tests/test_oracle.py` |
| B1c | Q/K dumper: forward hooks on full-attention layers only. For each such layer save keys as cached (state which side of RoPE) and the queries of the last N positions. Test on Qwen3.5-0.8B first, then Ornith at 4K. | `src/lsa_rocm/dump.py`, `results/dumps/` (gitignored) |
| B1d | Evaluation prompts: real long text at 4K, 32K and 128K tokens from datasets already on disk, as a manifest with source, token count and a hash. | `experiments/data/manifest.json` |

### B2 Real-KV recall, Ornith 1.5 9B (after B1, Gate 1)

| ID | Task | Output |
|---|---|---|
| B2a | At 4K then 32K: run exhaustive, flat cluster, and Ward-tree search on the dumped tensors. Sweep budget, cluster count and tree depth. Report metrics per layer and per KV head. | `experiments/002-real-kv-recall/`, results JSON |
| B2b | Ablate: pre-RoPE versus post-RoPE keys, and queries from the last 1, 8 and 64 positions. | same folder |
| B2c | Repeat B2a on Nanbeige4.3-33B (full attention, quantized or bf16 split across the two R9700s), gated on B0b confirming its attention design. | `experiments/002c-fullattn/` |

I write up the conclusion and decide Gate 1.

### B3 GPU timing (after Gate 1 passes)

| ID | Task | Output |
|---|---|---|
| B3a | Profile clustering, centroid scoring, selection and gather separately on the R9700 at 4K, 32K and 128K, against exhaustive GPU scoring. Include memory use. | `experiments/003-gpu-timing/` |

### B4 Scale and controls (after Gate 2, or after Gate 1 if timing is skipped)

| ID | Task | Output |
|---|---|---|
| B4a | Repeat B2 on Qwen3.8-27B (quantized, full-attention layers only, the paper's own model). Check the quant's effect on keys against a bf16 subset. | `experiments/004-scale/` |
| B4b | Optional: Qwen3.5-0.8B and 4B, to see whether the result changes with scale inside one family. | `experiments/005-small/` |
| B4c | Optional: gpt-oss-20b control, handling attention sinks in the metric. | `experiments/006-oss-control/` |

### B5 Write-up

`FINDINGS.md`: each claim linked to an experiment folder. Negative results included.

## Open questions

- Are cached keys compared before or after RoPE? B0c and B2b decide.
- Does the metric use true post-softmax attention mass or top-k overlap? `docs/metrics.md` defines both. Mass is primary.
- Pre-RoPE versus post-RoPE keys: the LSA indexer uses pre-RoPE position-free dims shared per KV group (B0c draft). B2b tests this.
