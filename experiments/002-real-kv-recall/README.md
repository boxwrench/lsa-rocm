# 002-real-kv-recall

**Status:** running

## Hypothesis
LSA-style key retrieval (cluster the keys, search centroids, gather the top keys) finds the keys a
model actually attends to on the full-attention layers of a real cached KV, so at a small budget it
captures most of the true attention mass, and beating random / recency controls. If flat and tree
capture no more mass than those controls, or much less than exhaustive scoring at the same selection
size, the hypothesis fails.

## Setup
Models: Ornith 1.5 9B (`/ai/models/llm/Ornith-1.5-9B`, Qwen3.5-family, 8 of 32 layers full attention,
16 query / 4 KV heads, head dim 256, rotary dim 64, scaling 0.0625) and Qwen3.5-0.8B as a harness
check. Dumps: `results/dumps/ornith-1.5-9b-4096` (4096 tokens, 8 queries) and
`results/dumps/qwen3.5-0.8b-2048` (2048 tokens, 8 queries); see each `meta.json` for the prompt hash.
Layers: every full-attention layer of the dump (`--layers all`), all KV heads. Queries: the last
`--n-queries` positions of the dump.

Index modes: `full` (all 256 dims of the post-RoPE keys) and `posfree` (dims `[64:]`, the position-free
part). Index keys and the summed group query are unit rows; the index query is the sum of the KV group's
query heads, shared across the group. Ground truth is `metrics.attention_weights` with the layer's own
scaling and causal mask, for every query head in the group. Metrics per `docs/metrics.md`:
`mass_captured` (primary), `mass_excluding_sinks`, `recall_at_k` (k = selection size), `output_error`,
`budget_fraction`; means over the group's query heads and the n queries. Methods: `flat`, `tree`
(`src/lsa_rocm/oracle.py`), and controls matched in selection size per query (`random`, `recency`,
`exhaustive`), each tagged with `matched_to` = `flat` or `tree`. `sink_size=4`, `window_size=128`.

Device: CPU only (`agents/scratch/venv-lsa-min`). Seed: 0. Metric definitions:
[`docs/metrics.md`](../../docs/metrics.md).

## Command
Quick harness check (one flat config, one tree config, one KV head per layer):

```
PYTHONPATH=src:third_party/lsa/src agents/scratch/venv-lsa-min/bin/python \
    experiments/002-real-kv-recall/run.py --dump results/dumps/qwen3.5-0.8b-2048 \
    --out results/b2_0-quick.json --layers 3 --max-kv-heads 1 --n-queries 4 --quick
```

Full sweep (`bucket_size` in {8,16}, `top_p` in {4,8,16}, `top_k` in {32,64,128}; oracle defaults
elsewhere):

```
PYTHONPATH=src:third_party/lsa/src agents/scratch/venv-lsa-min/bin/python \
    experiments/002-real-kv-recall/run.py --dump results/dumps/ornith-1.5-9b-4096 \
    --out results/b2_0-ornith-4096.json --layers all --n-queries 8
```

## Result
**Preliminary: one model, one prompt, 16 queries.** Ornith 1.5 9B, coherent prose, 32768 tokens, all 8 full-attention layers, paper-mode (training-free, layer-shared PCA index, d_idx 64, n_cluster = window = 512, p = k/4). Retrieval efficiency = captured head-averaged attention mass outside sink and window divided by the exact top-k at the same selection size (mean over layers). Source: `results/b2_1-summary-ornith-prose32768.json`.

| k (share of 32768 keys) | flat | tree | block-sparse | recency | layers with flat >= 0.70 | flat > recency | flat > block-sparse |
|---|---|---|---|---|---|---|---|
| 512 (1.6%) | 0.32 | 0.32 | 0.35 | 0.15 | 0/8 | 7/8 | 0/8 |
| 1024 (3.1%) | 0.42 | 0.42 | 0.45 | 0.28 | 0/8 | 7/8 | 0/8 |
| 2048 (6.2%) | 0.53 | 0.53 | 0.55 | 0.39 | 0/8 | 7/8 | 0/8 |
| 4096 (12.5%) | 0.64 | 0.64 | 0.66 | 0.41 | 4/8 | 7/8 | 0/8 |

Tree equals flat at this length. At k=2048 the best layer reaches 0.69 and none reach 0.70. The block-sparse baseline is slightly better than LSA on every layer. Recency beats flat on the first full-attention layer (layer 3).

### 128K tokens (same model, prompt and settings)

Ornith 1.5 9B, coherent prose, 131072 tokens (chunked-prefill dump, self-check cosine 0.99999911), window 512. Source: `results/b2_1-summary-ornith-prose131072.json`.

| k (share of 131072 keys) | flat | tree | block-sparse | recency | layers with flat >= 0.70 | flat > recency | flat > block-sparse |
|---|---|---|---|---|---|---|---|
| 512 (0.39%) | 0.26 | 0.26 | 0.28 | 0.12 | 0/8 | 6/8 | 0/8 |
| 2048 (1.56%) | 0.43 | 0.43 | 0.46 | 0.29 | 0/8 | 6/8 | 0/8 |
| 4096 (3.12%) | 0.52 | 0.52 | 0.55 | 0.43 | 1/8 | 6/8 | 0/8 |

Efficiency falls gently as the budget share shrinks: at k=4096 the mean is 0.64 at 32K (12.5% of keys) and 0.52 at 128K (3.1%). Tree still equals flat: with 4096 open buckets needed to form a level, the hierarchy barely exists below about 1M tokens. Recency beats flat on the first full-attention layer (layer 3) at every budget. The block-sparse baseline stays slightly ahead of flat.

### ornith-code32768: 32768 tokens

Dump `ornith-code-coh-32768` (self-check cosine 0.9999988079071045), tree preset `scaled`, window 512, mean retrieval efficiency over layers. Source: `results/b2_1-summary-ornith-code32768.json`.

| k (share of keys) | flat | tree | block-sparse | recency | layers flat >= 0.70 | flat > recency | flat > block-sparse |
|---|---|---|---|---|---|---|---|
| 512 (1.56%) | 0.34 | 0.34 | 0.35 | 0.11 | 0/8 | 8/8 | 2/8 |
| 2048 (6.25%) | 0.54 | 0.54 | 0.55 | 0.20 | 0/8 | 8/8 | 2/8 |
| 4096 (12.50%) | 0.65 | 0.65 | 0.65 | 0.22 | 3/8 | 8/8 | 4/8 |

## Conclusion
As pre-registered, **Gate 1 fails** on this run: efficiency is below 0.70 on every layer at k=2048, and LSA does not beat the block-sparse baseline. Two readings, both preliminary: (a) the paper's claim is parity with block-sparse at a much cheaper index, and parity is what we see, so the 'beats block-sparse' criterion was mis-specified; (b) the efficiency criterion is a real shortfall at 32K for this model, though the paper's own budget regime is 1M and beyond. Diagnostics: the projected 64-dim shared index space reaches 68 to 80% of the ideal at k=2048 and clustering costs roughly a third more; raising d_idx to 768 changes efficiency by under 0.02.

## Caveats
