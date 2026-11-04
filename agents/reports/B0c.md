# B0c — Upstream LSA repo and paper

Status: DRAFT (in progress). Task: read https://github.com/Prefrontal-AI/latent-sparse-attention and its paper.
`third_party/lsa` is cloned and pinned at commit **97b82d695b74ca341965a16bbc2a951c8b0c567a**
(remote `https://github.com/Prefrontal-AI/latent-sparse-attention`, `git log -1`: "Add issue and pull request
templates", 2026-11-03 15:29:24 +0000). Working tree clean.

## Paper

arXiv 2610.42147, "Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention"
(Kubicek, Caciolai, Peng, Sumant, Hoover, Greenlee, Battleday, Urban, Hoefler, Whittington; Prefrontal AI /
ETH Zurich; 30 Oct 2026). Abstract and introduction fetched 2026-11-03; full text cached at
`agents/scratch/paper.txt` (`agents/scratch/arxiv-full.html`). The abstract was read in full, not summarized:

> Agentic workflows are increasingly shifting towards long-horizon tasks ... at ultra-long context (above 1M
> tokens), full attention is prohibitively expensive ... We address this with Latent Sparse Attention (LSA) ...
> keys are hierarchically clustered in a learned latent space, and queries score the hierarchy of clusters
> instead of scoring every key ... We propose two variants, Distilled-LSA (D-LSA) ... and Native-LSA (N-LSA) ...
> On GLM-5.3-Flash (Qwen-3.8-27B) at 10M tokens, D-LSA speeds up one attention layer by 22.7x (72.4x) in prefill
> and 4.6x (42.5x) in decode over its original sparse attention (full attention), while keeping similar accuracy
> across a variety of tasks. N-LSA matches the accuracy of full attention with a training speedup of 1.6x per
> layer at 128k tokens.

Introduction adds: D-LSA only trains small per-layer projection matrices on a frozen pre-trained model;
N-LSA trains from scratch on next-token loss; evaluations are on RULER and a new PaperTrail benchmark, and LSA
is integrated into SGLang with context parallelism and H200/B300 kernels.

## Models and context lengths the paper evaluates

(full text section 4.1 and appendix; not just repo configs)

- **GLM-5.3-Flash** (`zai-org/GLM-5.3-Flash`, 45 layers, 313.3B text params, fp8; MLA quadratic layers with no
  rotary, `qk_rope_head_dim` 0). D-LSA distilled. Long-context accuracy on RULER 4k→10M (13 tasks, 100 prompts
  per task up to 1M and 20 beyond) and PaperTrail 4k→10M. Baselines: its own block-sparse (native) indexer,
  QSA, PISA, block-sparse tree. Performance on one attention layer from 128k to 10M (chunk 32,768).
- **Qwen3.8-27B** (`Qwen/Qwen3.8-27b`, 64 layers, 48 Gated DeltaNet + full-attention layers). D-LSA distilled;
  baseline full attention. RULER 4k→10M and PaperTrail 4k→10M; YaRN factor 40 (42 for PaperTrail at 10M).
- **N-LSA native model**: ~0.5B params, 24 layers, every fourth layer attention (8 query heads, 1 KV head of
  128 dims, no positional encoding), rest Gated DeltaNet. Pre-train 100B FineWeb-Edu tokens at 4k, decay 12B
  LongDocs tokens at 128k. Compared against full attention, SWA, NSA, PISA. RULER 4k, 8k, 16k, 32k, 64k, 128k.
- Other accuracy benchmarks on the two big models: MMLU-Pro (2,000 q), GPQA-Diamond, AIME25 (4 repeats),
  MATH500 (100), LongBench v2 (60 q).
- Context lengths, explicitly: 4k and 32k (RULER quick cells in the README), then 128k, 512k, 1M, 2M, 5M, 10M
  for accuracy; performance 128k→10M; kernel harness list `[131072, 262144, 524288, 1048576, 2097152, 5242880,
  10485760]`; prefill scaling extrapolated to 100M.

## Exact `lsa.core` API

`src/lsa/core/__init__.py` exports: `Buckets, Hierarchy, LatentMetric, LearnedProjection, RotaryLayout, Shapes,
build_buckets, build_hierarchy, position_free, project, regroup, regroup_latent, regroup_partial, read,
read_hierarchy, read_hierarchy_counted, rownorm, shape_buckets, topk_attention`.

Signatures and argument meanings to be transcribed below (draft: see source).

## Indexer input

- Indexer input is **key and query** (`model.indexer.input: key_and_query`, `query_heads: all`).
- Input is **pre-RoPE, position-free dims** via `position_free(q, k, rotary)`. For Qwen3.5/3.8 the rotation is a
  64-dim prefix of a 256-dim head, so the remaining 192 dims are untouched by RoPE.
- **Shared per KV group, not per head**: `regroup` sums query heads within each GQA group and concatenates KV
  heads, giving `[B, T, Hkv*dh]` unit rows. MLA models use `regroup_latent` + `LatentMetric` on the compressed
  latent (512 dims for GLM's 16,384).
- The identity is about summed logits, not summed per-head softmax.

## CPU-only install / what fails without CUDA

See the supervisor-verified section at the end.

## Does `lsa.core` work on CPU torch?

Yes, see the supervisor-verified section at the end.


## Supervisor-verified addendum (agent run did not finish; these facts were checked by hand)

- The agent runs for this task failed four times on network errors and a blocked log read. Status: PARTIAL, completed by the supervisor.
- A CPU-only virtualenv with `torch 2.13.0+cpu` imports `lsa.core` and runs `build_buckets` and `read` unmodified. No CUDA, no cutlass needed for the reference path.
- Verified signatures: `build_buckets(index_keys, pool_size, bucket_size, bucket_cap, cluster_block_size, cluster_algorithm="ward")`;
  `read(index_queries, positions, buckets, top_p, top_k, sink_size, window_size, shapes=None, deviations=0.0, early_budget=None) -> (selected, lengths)`;
  `build_hierarchy(index_keys, buckets, top_level, node_bucket_size, node_cluster_block_size, node_cluster_algorithm="ward")`;
  `read_hierarchy(index_queries, positions, buckets, hierarchy, beam_width, start_nodes, top_p, top_k, sink_size, window_size, page_size, growth_buckets, ...)`.
- Conventions found by running it: keys are `[N, D]` with no batch dimension; `pool_size` must divide `bucket_size`, which must divide `cluster_block_size`; `pool_size=1` works.
- `read` returns selected ids `[Nq, top_k + window_size + sink_size]` padded with -1, plus the number of valid ids per query.

STATUS: PARTIAL (agent) / facts above verified by supervisor
