# 003-stage-timing

**Status:** CPU and GPU (AMD Radeon AI PRO R9700, one card) runs done. Clustering is timed with the CPU reference only.

## Hypothesis
Scoring the bucket centroids is much cheaper than scoring every key, and the gap grows with context. The paper reports that search
dominates an attention layer's time at 10M tokens (their baselines score every block on 128 to 256 dims).

## Setup
`bench_stages.py`: synthetic unit-norm index vectors from a Gaussian mixture, d_idx 64, n_pool 4, bucket 8 pool keys (32 keys),
cluster block 512, k 4096, p 1024, 32 queries, fp32 scoring. Stages: centroid scoring plus top-p, exhaustive scoring of every pool key plus top-k,
gather of k rows from a [1M, 2048] bf16 KV table (measured on a 1M-row table, assumed independent of context length), and Ward clustering
with the `lsa.core` reference (timed on 16 blocks of 512 keys, scaled to all blocks). **Synthetic keys: this measures cost only, never retrieval quality.**

## Command
`python bench_stages.py --device cpu|cuda --out results/003-timing-<dev>.json` (GPU run with `HIP_VISIBLE_DEVICES=1`, ROCm torch 2.12.0+rocm7.14.0).

## Result: GPU (AMD Radeon AI PRO R9700), from `results/003-timing-gpu.json`

| keys | buckets | centroid scoring | exhaustive scoring | ratio | gather (32 q x 4096 keys) | centroid + gather | exhaustive + gather |
|---|---|---|---|---|---|---|---|
| 32,768 | 1,024 | 0.102 ms | 0.165 ms | 1.6x | 2.20 ms | 2.31 ms | 2.37 ms |
| 131,072 | 4,096 | 0.110 ms | 0.288 ms | 2.6x | 2.20 ms | 2.31 ms | 2.48 ms |
| 1,048,576 | 32,768 | 0.279 ms | 0.688 ms | 2.5x | 2.20 ms | 2.48 ms | 2.89 ms |
| 10,485,760 | 327,680 | 0.837 ms | 6.254 ms | 7.5x | 2.20 ms | 3.04 ms | 8.46 ms |

## Result: CPU (Ryzen 7 9800X3D, under load from other jobs), from `results/003-timing-cpu.json`

| keys | buckets | centroid scoring | exhaustive scoring | ratio | gather (32 q x 4096 keys) | centroid + gather | exhaustive + gather |
|---|---|---|---|---|---|---|---|
| 32,768 | 1,024 | 1.053 ms | 0.953 ms | 0.9x | 65.51 ms | 66.57 ms | 66.47 ms |
| 131,072 | 4,096 | 0.472 ms | 1.949 ms | 4.1x | 68.54 ms | 69.01 ms | 70.49 ms |
| 1,048,576 | 32,768 | 1.574 ms | 15.541 ms | 9.9x | 66.38 ms | 67.95 ms | 81.92 ms |
| 10,485,760 | 327,680 | 14.072 ms | 181.409 ms | 12.9x | 65.22 ms | 79.29 ms | 246.63 ms |

## Clustering cost (CPU reference, done once per cluster block as keys arrive)

| keys | Ward clustering, all blocks, once (reference on CPU) |
|---|---|
| 32,768 | 0.0 s |
| 131,072 | 0.2 s |
| 1,048,576 | 1.2 s |
| 10,485,760 | 12.3 s |

## Conclusion (preliminary)
- Centroid scoring is cheaper than exhaustive scoring and the gap grows with context: on the GPU 2.6x at 131K, 2.5x at 1M and 7.5x at 10M keys.
- On the GPU the gather (about 2.2 ms for 32 queries x 4096 keys) is a floor that does not shrink with context. At 1M tokens it is as large as the whole saving, so centroid plus gather
  is only about 1.2x cheaper than exhaustive plus gather; at 10M it is about 2.8x.
- The paper's claim that search is 98% of layer time at 10M refers to its own kernels and baselines. Here exhaustive scoring is the largest stage at 10M but this table does not include attention itself, so it
  neither confirms nor contradicts that figure.

## Caveats
Synthetic keys, one query batch size (32), fp32 scoring (production kernels use lower precision), the clustering reference is a Python implementation and was not run on the GPU,
the gather is measured on a 1M-row table, and this is a single card. Index vectors at 10M tokens are about 1.3 GB, which fits easily.
