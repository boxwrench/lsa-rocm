# 004: end-to-end decode-step attention on real KV (R9700)

**Status:** first results, preliminary. One layer (11) of Ornith 1.5 9B, real dumped K/V/Q, code prompts, one card, KV resident in VRAM (no CPU offload), bf16, k = 2048 plus 4 sinks and a 512 window.
Prompted by an outside review that asked whether cheaper indexing turns into faster usable attention.

A decode step is select -> gather K/V -> attention for one query (all 16 query heads, 4 KV heads). Index building (clustering) is excluded and per-step index maintenance is not modelled.
Timings are GPU-synced medians of 30 runs. Error is the relative L2 difference of the layer's attention output against full attention, averaged over the last 8 queries
(sparse paths) or the last query only (unmasked full paths).

| keys | full, SDPA (ms) | full, batched matmul (ms) | block-sparse ms (err) | LSA flat ms (err) | full (bmm) / LSA |
|---|---|---|---|---|---|
| 32,768 | 2.79 | 0.47 | 0.47 (0.088) | 1.17 (0.104) | 0.4x |
| 131,072 | 10.25 | 1.87 | 0.55 (0.129) | 1.17 (0.141) | 1.6x |
| 524,288 | 38.99 | 6.89 | 0.85 (0.166) | 1.33 (0.181) | 5.2x |

## What this says
- The choice of full-attention baseline matters a lot. PyTorch SDPA with few query rows per KV head runs at about 55 GB/s here; a plain batched-matmul decode runs near 300 GB/s. Against SDPA the sparse paths look 30-40x faster at 512K; against the matmul baseline it is 5x.
- At 32K full attention (0.47 ms) is as cheap as block-sparse and cheaper than LSA (1.17 ms): sparse attention does not pay at short context.
- At 128K LSA is 1.6x faster than full, at 512K 5.2x. Block-sparse is faster than our LSA at every length (0.47 to 0.85 ms vs 1.2 to 1.3 ms): our LSA selection (centroid scoring, bucket expansion with a mask and cumulative count, then gather) costs more than scoring 4-key block means, and the extra work is not paid back by the cheaper index on 512K keys.
- Output error is not small: 0.10 to 0.18 relative L2 at k = 2048, growing with context, and slightly higher for LSA than block-sparse. This matches the retrieval-efficiency numbers (about 0.5) and says nothing about end-task accuracy.

## Limits
One layer, one query position for timing, one prompt type, YaRN beyond 262K, selection implemented with plain torch ops (no fused kernel), KV in VRAM. In real serving the KV of 1M tokens sits in CPU RAM, where the gather of the selected keys over PCIe is the cost that matters; that is not measured here.
Tree search is not measured (no nodes-visited counts yet). Not an end-to-end tokens-per-second number.

Reproduce: `HIP_VISIBLE_DEVICES=1 python experiments/004-e2e-attention/bench.py --dump results/dumps/<name> --layer 11 --out results/004-<name>-L11.json`
