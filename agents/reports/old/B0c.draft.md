# B0c — Upstream LSA repo and paper

Status: DRAFT (in progress). Task: read https://github.com/Prefrontal-AI/latent-sparse-attention and its paper.
`third_party/lsa` is reused as cloned; pinned commit **97b82d695b74ca341965a16bbc2a951c8b0c567a**
(remote `https://github.com/Prefrontal-AI/latent-sparse-attention`).

## Paper

arXiv 2610.42147, "Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention"
(Kubicek et al., 2026). Abstract fetched 2026-11-03 (HTTP 200 from arxiv.org/abs/2610.42147).

Claims from the abstract (verified text, quoted in full in the final report):
- D-LSA applied by distillation to pre-trained models; N-LSA trained from scratch on next-token loss.
- GLM-5.3-Flash at 10M tokens: D-LSA 22.7x prefill / 4.6x decode on one attention layer.
- Qwen3.8-27B at 10M: D-LSA 72.4x prefill / 42.5x decode vs full attention.
- N-LSA matches full attention, 1.6x per-layer training speedup at 128k.

## Models and context lengths evaluated (from repo configs/README, not yet the paper body)

- **GLM-5.3-Flash** (MLA, NoPE, 45 layers, 4096 model dims): D-LSA flat/tree/early-budget, block-sparse
  baselines, accuracy RULER 4k/32k/512k/1M/2M/5M/10M, PaperTrail to 10M, kernel + production timings
  128k→10M.
- **Qwen3.8-27B** (GQA 32/4? — verify, `full_attention` layers): D-LSA flat/tree/PCA, QSA native baseline,
  RULER 4k/32k (README) and 1M–5M configs, YaRN factor 40 to 10,485,760.
- **Native comparison**: Qwen3.5-0.8B-gpt2 hybrid, 24 layers, pre-train at 4k on 100B FineWeb-Edu tokens,
  decay at 128k on 12B LongDocs tokens; arms Full/SWA/NSA/PISA/N-LSA; ~2,560 keys/query in decay.
- Kernel harness contexts: `[131072, 262144, 524288, 1048576, 2097152, 5242880, 10485760]`.

## Exact `lsa.core` API

Public `__all__` of `src/lsa/core/__init__.py`:
`Buckets, Hierarchy, LatentMetric, LearnedProjection, RotaryLayout, Shapes, build_buckets,
build_hierarchy, position_free, project, regroup, regroup_latent, regroup_partial, read,
read_hierarchy, read_hierarchy_counted, rownorm, shape_buckets, topk_attention`.

Signatures being transcribed with full args next; key ones:
- `build_buckets(index_keys, pool_size, bucket_size, bucket_cap, cluster_block_size, cluster_algorithm="ward") -> Buckets`
- `read(index_queries, positions, buckets, top_p, top_k, sink_size, window_size, shapes=None, deviations=0.0, early_budget=None) -> (selected int32, lengths int32)`
- `build_hierarchy(index_keys, buckets, top_level, node_bucket_size, node_cluster_block_size, node_cluster_algorithm="ward") -> Hierarchy`
- `read_hierarchy(...)` / `read_hierarchy_counted(...)`.

## Indexer input

- Indexer default input is **key and query**, `model.indexer.input: key_and_query`, `query_heads: all`.
- Input is **pre-RoPE, position-free dims**: `position_free(q, k, rotary)` drops the rotated dims from
  *pre-RoPE* heads. For Qwen3.5/3.8 the rotation is a 64-dim prefix of a 256-dim head; the server's fused
  norm-and-rope passes the other 192 dims through untouched, so post-rope cached keys equal pre-rope on those
  dims. Index keys are read out of the KV pool slot at `offset=rotary_dim`, `width=head_dim-rotary_dim`.
- **Shared, not per head**: GQA collapses query heads into their KV group sum and concatenates KV heads:
  index vectors are `[B, T, Hkv*dh]` unit rows (`regroup`/`regroup_partial`). MLA models use
  `regroup_latent` + `LatentMetric` on the compressed latent (512 dims for GLM's 16,384).
- The exact identity is about summed logits, not summed per-head softmax.

## CPU-only install / what fails without CUDA

Being tested with `uv` next; pyproject pins `torch==2.13.0` from a `cu129` index and
`nvidia-cutlass-dsl==4.6.2` as a required dependency — likely a CUDA-only wheel. Record exact failures.

Draft written; refining below with full signatures and exact commands/output.
