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

## Conclusion

## Caveats
