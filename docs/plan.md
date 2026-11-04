# Plan: Latent Sparse Attention retrieval on AMD ROCm

Status: revised 2026-11-03, after reading the paper (see [paper-notes.md](paper-notes.md)). Foundations are built and
one rough probe exists. The recipe-faithful experiments have not been run.

## What the source claims

The LSA paper (arXiv:2610.42147) and an author's announcement post (X, 2026-11-02) claim: at 10M tokens the key
search takes about 98% of the attention layer's time, and LSA makes sparse attention up to 22.7x and full attention
up to 72.4x faster per layer, while matching full-attention accuracy on benchmarks up to 10M tokens. Those results
use distilled indexers, H200 and B300 GPUs, SGLang, and end-task accuracy (RULER and others).

## What this project can and cannot test

| Can | Cannot |
|---|---|
| Retrieval quality of the paper's training-free recipe on full-attention layers, at 4K to 128K | The headline speedups (10M tokens, H200/B300, SGLang, context parallelism) |
| Whether it holds on models the paper did not use (Ornith 9B, Nanbeige 33B) | End-task accuracy at 1M+ tokens |
| Per-layer behaviour, and how mass grows with budget | Distilled projections (D-LSA), unless we run a distillation |
| The cost of each indexing stage on an AMD GPU, including synthetic 10M-token index timing | Retrieval quality at 10M (the real model's KV cache does not fit anywhere we own) |
| Retrieval quality at 1M, and about 2M on the Strix Halo, as stretch goals (YaRN-extended, so out of distribution) | Any claim about end-task behaviour at those lengths |

**Question:** does the paper's training-free recipe retrieve the keys that matter on the full-attention layers of
current models at the contexts we can run, and what do its stages cost on an R9700?

Not in scope: porting the upstream serving code, CUDA/CUTLASS kernels, end-task benchmarks, training.

## Where we are

| Done | Evidence |
|---|---|
| Models checked from config and code | Ornith 9B (hybrid, 8 of 32 layers full), Nanbeige4.3-33B (plain Llama, 52 full layers, verified against the Hub) |
| CPU reference indexer runs | `lsa.core` on CPU torch, 7 oracle tests |
| Metrics and tests | 24 tests, `src/lsa_rocm/metrics.py` |
| Dumper at 8K, 16K, 32K | independent recompute cosine 0.999999, 32K in 35 s, 23.5 GiB on one R9700 |
| Coherent prompts, 4K to 131K | one repo per prompt, exact token counts, passes acceptance |
| Harness 002 (per-KV-head, raw dims) | passes acceptance. Not the paper's recipe, see below. |
| Rough probe, experiment 000 | Ornith 4K: mass 0.60 vs recency 0.44 vs random 0.05, exact top-k 0.83. **Preliminary, off-recipe.** |

## Models

| Role | Model | Notes |
|---|---|---|
| Primary | Ornith 1.5 9B (local) | Qwen3.5-family hybrid, 16/4 heads, hd 256, rotary_dim 64. Close to the paper's Qwen-3.8-27B layout. |
| Full-attention case | Nanbeige4.3-33B | Plain Llama, 56/8 heads, hd 128, RoPE theta 120M. Outside the paper. bf16 is 66 GB, so a quant or offload is needed (spike B2.4). |
| Paper's model | Qwen3.8-27B, quantized | Spike B4: find a usable local copy or quant, confirm loading. |
| Debugging | Qwen3.5-0.8B (local) | Same architecture as Ornith, tiny. |
| Control | gpt-oss-20b | Sliding window plus sinks. Optional. |

Dropped: granite-switch (its adapters alter K/V, and it is not an interesting case).

## The recipe we test ("paper mode")

Training-free D-LSA flat and tree, following paper Section 3 and Table 2, scaled to our contexts.

- One **layer-shared** index per query. Keys: RoPE-free dims (dims after `rotary_dim`) of every KV head concatenated,
  projected to `d_idx = 64` by the top principal directions of that layer's keys. Queries: the same dims, summed
  over query heads, same projection. L2-normalise both.
- `n_pool = 4`, average bucket 8 and maximum 16 keys, Ward clustering in cluster blocks, 4 sinks.
- Window equals one cluster block (the paper uses 512). At 4K that is 12.5% of the context, so 4K runs use a smaller
  block and say so. Ablation over {64, 128, 512}.
- Budget `k` swept over {256, 512, 1024, 2048, 4096} with `p = k / 4` (the paper uses k = 4096, p = 1024).
- Flat and tree. Tree parameters from the paper's Table 2, reduced where the context is too short for the levels.

**Controls, matched in selection size:** random, most recent keys, exact top-k (upper bound), and a **block-sparse
baseline** that scores blocks of 4 and 8 keys on the raw keys (the paper's own baseline family, QSA and HiP style).
A method only counts as working if it beats recency and the block-sparse baseline.

## Metrics

Definitions are in [metrics.md](metrics.md). Primary: **retrieval efficiency**, mass captured divided by the exact
top-k mass at the same selection size, plus the paper's own target: head-averaged attention over the keys outside
the sink and window, and the share of the top 2048 such keys recovered. Always report per layer, never only a mean.

## Gates (pre-registered)

These were written on 2026-11-03 **before** the paper-mode sweep was run. Change them only with a dated note here.

- **Gate 1 (retrieval quality).** At 32K on coherent prompts, on Ornith, paper-mode flat retrieval at k <= 2048 reaches
  retrieval efficiency >= 0.70 on at least half of the full-attention layers, and beats both recency and the block-sparse
  baseline on at least half of them. If not, stop and write up the negative result. Then repeat on Nanbeige; a different
  answer there is a finding.
- **Gate 2 (cost on AMD).** At 32K and 128K, centroid scoring plus gather costs less than exhaustive scoring on the
  R9700. Report the per-stage breakdown (clustering, scoring, selection, gather) either way. The paper says search dominates
  at 10M tokens. We report where it dominates at our lengths, which will differ.
- **Gate 3.** Only then run the remaining models (B4) and the CPU-memory KV idea on the Strix Halo node.

**Change note, 2026-11-03 (before any paper-mode result existed):** long context is the point of the experiment, so
the long-context work (chunked-prefill dumper, 128K and 1M dumps, synthetic 1M and 10M index timing, the 2M Strix Halo
spike) is no longer blocked by Gate 1. Gate 1 stays as the pre-registered reporting criterion at 32K. A short-context
result does not predict long-context behaviour: at 32K a k=4096 budget is 12.5% of the keys, at 1M it is 0.4%.

**Gate 1 outcome (2026-11-03, Ornith 9B, prose-32768, one prompt, preliminary): FAILED as pre-registered.** Flat retrieval
at k=2048 reached efficiency >= 0.70 on 0 of 8 layers (best 0.69, mean 0.53), and did not beat the block-sparse baseline
(mean 0.55). The second criterion was mis-specified: the paper claims parity with block-sparse at a cheaper index, which is
what we see. The original gate stays on record. It is not rewritten after the fact. Remaining checks before any conclusion:
more prompts (code, technical), Nanbeige, and long context, where the budget share is tiny and the tree matters.

## Rules for agents

1. Work only inside the paths named in your task. Do not edit `docs/plan.md`, `docs/metrics.md` or `agents/verify/`.
2. No downloads over 1 GB. Write the command and size in your report instead.
3. GPU jobs use device 1 only (`HIP_VISIBLE_DEVICES=1`). Check VRAM first: Vulkan jobs on this machine do not show
   in the process list. The wrapper waits for device 1 to be idle.
4. Every run writes JSON under `results/` with git commit, model and revision, torch and ROCm versions, device, seed
   and config. No hand-copied numbers.
5. Acceptance is decided by `agents/verify/<id>` run by the wrapper (`agents/run_agent.sh`), not by the agent. The
   verify scripts recompute results independently from the dumps. An agent's own "DONE" counts for nothing.
6. On a denied tool call or a network failure, write `STATUS: BLOCKED: <reason>` and stop. Never read shell history or
   opencode logs.
7. Tasks are sized for about 15 minutes. Open-ended surveys fail (B1f). Give agents concrete sources and parameters.
8. DeepSeek is the default backend with one retry on network errors, then `longcat/LongCat-2.5-Preview`.

## Batches

| ID | Task | Status |
|---|---|---|
| B0a-d | Verify models, read the repo, probe the machine | done |
| B1a-f | Metrics, oracle, dumper (scaled to 32K), prompts | done (B1d stitched prompts superseded by B1f coherent ones) |
| B2.0 | Harness 002, per-KV-head raw dims | done, off-recipe, kept as an ablation |
| **B2.1** | **Paper mode** in `experiments/002-real-kv-recall/run.py`: layer-shared PCA index, paper parameters scaled, new metrics | next |
| **B2.2** | Block-sparse baseline control (blocks of 4 and 8 on raw keys) | next, parallel to B2.1 |
| B2.3 | Run Ornith at 4K and 32K, prose, code and technical prompts. I read the numbers and decide Gate 1. | after B2.1, B2.2 |
| B2.4 | Spike: load Nanbeige (FP8 compressed-tensors, else bf16 with CPU offload) on ROCm and dump 32K | parallel, may need a download |
| B3 | GPU stage timing on the R9700 at 4K, 32K, 128K (needs a 128K dump, expected to fit now) | after Gate 1 |
| B3b | Index-stage timing at 1M and 10M with synthetic keys (64-dim index vectors, 10M is about 1.3 GB). Measures cost only, not quality. Tests the claim that search dominates at 10M. | after B2.1, not gated |
| B1g | **Chunked-prefill dumper** (needed for 128K and 1M) and the 128K Ornith dump | **next, parallel** |
| B2.5 | Stretch: Ornith 1M-token dump and retrieval-quality run. Needs chunked prefill and the model split over both R9700s (KV 32 GiB), YaRN factor 4 beyond the 262K native context. | after B1g, not gated |
| B2.6 | Stretch: Ornith about 2M tokens on the Strix Halo node (KV 64 GiB plus 18 GiB weights inside its ~112 GiB GPU-addressable RAM). Needs a ROCm torch for gfx1151, chunked prefill, YaRN factor 8. 3M does not fit. | after B2.5, spike first, not gated |
| B4 | Qwen3.8-27B quantized spike and run, then small models and gpt-oss control | after Gate 2 |
| B5 | `FINDINGS.md`, each claim linked to an experiment | last |

## Status update, 2026-11-04 (overnight run)

Done: B2.1 paper mode, B2.2 block-sparse baseline, B1g chunked dumper, B2.5a KV offload and YaRN, B2.3 Ornith runs on prose, code and technical text at 32K and 128K,
B2.5 long-context dumps and analysis at 512K (prose, code) and 917K (code; 1,048,576 tokens was not reachable from a single source), B2.4 Nanbeige at 32K (with weights partly offloaded to CPU),
B3 and B3b stage timing on the R9700 and CPU, including synthetic 1M and 10M keys. Results and caveats: [FINDINGS.md](FINDINGS.md).
Gate outcomes are recorded there and not rewritten here: Gate 1 fails on prose and code and passes on the technical docs at 32K; Gate 2 is not met as written at 32K and 128K (the gather is a fixed cost) and is met at 10M.

Not done: B2.6 (about 2M tokens on the Strix Halo node, needs a ROCm torch for gfx1151), GPU clustering stage timings, tech 512K and 1M prompts (not enough single-source text),
a cross-prompt PCA test, a second model at long context, and any end-task accuracy measurement.

## Open questions

- Does the PCA basis from one prompt generalise to another, or must it be fitted per prompt? The paper fits it per layer.
  We test fitting on one prompt and applying to another.
- Cached keys are post-RoPE, the paper's index uses RoPE-free dims. Dumps already contain the position-free part
  (dims after `rotary_dim` are unrotated), so no pre-RoPE dump is needed for Qwen-family models. Nanbeige rotates all
  dims, so it has no RoPE-free part. How the index is built there is a design choice to make and document (B2.4).
- Whether mass captured predicts end-task accuracy. We cannot test it here. We say so in the findings.
