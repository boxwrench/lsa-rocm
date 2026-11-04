# Notes on the LSA paper, and how this project differs

Source: Kubicek, Caciolai, Peng et al., "Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention",
arXiv:2610.42147 (v2, 2026-11-03), CC BY 4.0. Code: https://github.com/Prefrontal-AI/latent-sparse-attention.
Read from the arXiv HTML version (method, setup, accuracy and appendix tables). These notes paraphrase it. Section
numbers refer to the paper. Nothing here is our result.

## What the method does (Sections 3.1 to 3.3)

- **Index keys and queries.** Every key is L2-normalised, then every `n_pool` consecutive keys are summed into one
  pool key (n_pool = 4).
- **Clustering.** Keys are clustered in cluster blocks of `n_cluster` consecutive keys (512). Within a block, pool keys
  are merged bottom-up with Ward's method into buckets averaging 8 keys (max 16). A bucket's centroid is its
  normalised key sum. Buckets never change once their block is closed. The Tree variant repeats this to build levels.
- **Search.** The index query is L2-normalised and scored against every closed bucket's centroid. The top `p` buckets
  are expanded, best first, until `k` long-range keys are selected. Every query also attends to `n_sink` sink keys (4)
  and a window of `n_w` recent keys. The window equals one cluster block (512), so it covers every unclustered key.
- **One index list per query, shared by all heads.** The index query is the sum over query heads, so a single list
  of keys serves every head of the layer (Eq. 1). This is not one index per KV head.
- **D-LSA (distilled).** Learned projections W_K, W_Q map keys and queries to `d_idx = 64` dimensions, trained by
  distillation against the frozen model's head-averaged attention (Eq. 2). On Qwen-3.8-27B (RoPE on a quarter of each
  head) the projections read only the RoPE-free part.
- **D-LSA (Training-Free).** The same pipeline with the leading `d_idx` principal directions of the layer's keys as
  both projections. No training. The paper reports it stays close to D-LSA (Flat) on Qwen-3.8-27B and trails it on
  GLM-5.3-Flash.
- **Reported settings (Table 2, Qwen-3.8-27B, bucket-based):** k = 4096, p = 1024, sink 4, window 512, d_idx 64,
  n_pool 4, n_cluster 512. Tree adds node buckets of 4, 4096 open buckets per level, beam 2048.
- **What the paper measures:** end-task accuracy (MMLU-Pro, GPQA, AIME25, MATH500, LongBench v2, RULER up to 10M
  tokens, PaperTrail) and kernel and serving speed. It does not report a per-layer attention-mass metric.

## How this project differs, stated plainly

1. **Different measure.** We measure how much true attention mass the selected keys capture, per layer. That is a
   proxy for the paper's end-task accuracy, not a replacement. A good mass number does not prove accuracy is kept.
2. **Small scale.** Our contexts are 4K to 128K on 9B to 33B models. The paper's budgets (k = 4096, window 512)
   are comparable to our whole 4K prompt, so we scale them down and say so.
3. **Training-free only.** We have no distillation. The closest comparison is the paper's D-LSA (Training-Free).
4. **My first probe (experiment 000) did not follow this recipe.** It built one raw-dimension index per KV head,
   with a window of 128 and a 64-key block, rather than one layer-shared PCA index. Treat its numbers as a rough
   indication only. The recipe-faithful version is the next experiment ("paper mode" in 002).
5. **Models.** The paper's own full-attention-layer model is a hybrid (Qwen-3.8-27B). Ornith 9B is a smaller relative.
   Nanbeige (plain Llama, all layers full attention) is outside the paper's evaluation, so results there are new.

## Metric changes this implies

- Add the paper's own target: head-averaged attention over the keys outside the sink and window, and how much of it
  the `k` long-range keys capture. Report it next to the per-head mass.
- Use one layer-shared index (concatenate KV heads' RoPE-free dims, project with a PCA basis, sum queries over heads).
- Set window = cluster block and scale `k`, `p`, and the block size with context length; record the scaling.
- Keep the random, recency and exact top-k controls matched to selection size.
