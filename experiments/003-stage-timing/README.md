# 003-stage-timing

**Status:** CPU baseline done. GPU (R9700) run pending: it needs GPU 1 while the long-context dumps are running.

## Hypothesis
Scoring the bucket centroids is much cheaper than scoring every key, and the gap grows with context. The paper reports
that search dominates an attention layer's time at 10M tokens (their baselines score every block on 128 to 256 dims).

## Setup
`bench_stages.py`: synthetic unit-norm index vectors from a Gaussian mixture, d_idx 64, n_pool 4, bucket 8 pool keys (32 keys),
cluster block 512, k 4096, p 1024, 32 queries. Stages: centroid scoring plus top-p, exhaustive scoring of every pool key plus top-k,
gather of k rows from a [1M, 2048] bf16 KV table, and Ward clustering with the `lsa.core` reference (timed on 16 blocks, scaled).
**Synthetic keys: this measures cost only, never retrieval quality.** CPU numbers come from this machine's Ryzen 7 9800X3D
under load from other jobs.

## Command
`python bench_stages.py --device cpu --out results/003-timing-cpu.json` (CPU venv with `lsa` on the path).

## Result (CPU, from `results/003-timing-cpu.json`)

| keys | buckets | centroid scoring | exhaustive scoring | ratio | gather (32 q x 4096 keys) | clustering, all blocks (reference, once) |
|---|---|---|---|---|---|---|
| 32,768 | 1,024 | 1.05 ms | 0.95 ms | 0.9x | 66 ms | 0.0 s |
| 131,072 | 4,096 | 0.47 ms | 1.95 ms | 4.1x | 69 ms | 0.2 s |
| 1,048,576 | 32,768 | 1.57 ms | 15.54 ms | 9.9x | 66 ms | 1.2 s |
| 10,485,760 | 327,680 | 14.07 ms | 181.41 ms | 12.9x | 65 ms | 12.3 s |

## Conclusion (preliminary)
On CPU, scoring centroids is about 10x cheaper than scoring every pool key at 1M tokens and about 13x at 10M, and the ratio grows with context.
At 10M tokens the exhaustive scan costs about three times the gather on CPU, so search is the largest stage but not 98% of the total here.
The paper's 98% figure is for its GPU kernels, with wider index vectors and a different baseline. The GPU numbers are what matter and are still to come.

## Caveats
CPU only, synthetic keys, one query batch size, the clustering reference is a Python reference implementation, not a kernel.
The gather is measured on a 1M-row table and assumed independent of context length.
