# Fidelity: how this project's LSA compares with the paper's

Checked 2026-11-05 against the arXiv HTML of Kubicek et al. (2610.42147; the text I could read ended inside Appendix E.4, so a few details below are marked unknown). This describes the **training-free flat** variant used in `experiments/005-needle` and `006-long-context`. Nothing here is a claim about the paper's results.

## Where we match the paper
| Item | Paper | Here |
|---|---|---|
| Index keys | L2-normalised, 4 consecutive keys summed into a pool key | same (the authors' own `lsa.core` reference builds the buckets) |
| Index queries | queries summed over heads, projected, L2-normalised; one index list shared by all heads | same |
| Projection | training-free: leading d_idx = 64 principal directions of the layer's keys, same for keys and queries | same, fitted by eigendecomposition of the key covariance |
| Key dims | RoPE-free dims (on Qwen-3.8-27B, a quarter of each head is rotated) | dims after `rotary_dim`; Ornith also rotates a quarter of each head (partial_rotary_factor 0.25) |
| Clustering | blocks of 512 keys, Ward, buckets of average 8 and at most 16 keys, only completed blocks, buckets never change | same, via `lsa.core.build_buckets` |
| Recent keys | the incomplete block is covered by a 512-key window | window 544 (512 plus slack) plus the new tokens |
| Selection | dot product of unit centroid and unit query, top-p buckets, expanded best-first until k long-range keys, plus 4 sinks and the window | same |
| Layers | only the global (full-attention) layers | only Ornith's 8 full-attention layers; its 24 linear-attention layers are untouched |

**Direct check:** on real Ornith keys (`experiments/005-needle/check_selection_vs_reference.py`) my GPU selection logic picks exactly the same long-range keys as `lsa.core.read` (100% overlap, k = 2048 and k = 128, 12 queries each), with no sink keys counted twice.

## Where we differ
| Item | Paper | Here | Effect |
|---|---|---|---|
| Budget | k = 4096 and p = 1024 on Qwen (GLM: 2048 below 1M); the paper says smaller budgets hold accuracy less well past 1M | k = 2048 (p = 512), and tiny budgets of 8 to 256 as stress tests | The tiny budgets are far outside the paper's regime. Failures there say little about the paper's setting. |
| Prefill | the index is used during prefill too (sparse prefill, 32K-token chunks) | the haystack is prefilled densely; only the question and answer tokens use the sparse path | Our prefix is more accurate than the paper's would be, which favours LSA. The "sparse question" runs are the closest to the paper's practice. |
| Indexer | headline results use distilled projections (D-LSA); training-free is the weaker variant | training-free only | Expect lower quality than the paper's headline. |
| PCA fit data | not stated in the part I read | fitted on the prompt's own keys | A cross-prompt check cost about 0.01 to 0.02 in retrieval efficiency (experiment 002). |
| Model | Qwen-3.8-27B (and GLM-5.3-Flash) | Ornith 1.5 9B, a smaller hybrid | A weaker base model, especially at 1M. |
| Tree | tree for cost; decoding uses flat search | flat only in the needle tests | Consistent with the paper's decoding. Tree quality tested separately (matches flat in mean). |
| Evaluation | RULER, LongBench, etc. | single planted facts with distractors | Ours is a much narrower test. |
| Long context | Qwen reaches 10M with YaRN | YaRN x4 to 1M | Similar mechanism, much weaker model. |
| Hardware | H200 / B300, SGLang, fused kernels | R9700, plain PyTorch ops | Irrelevant to accuracy, relevant only to speed claims, which we do not make. |

## What the 1M results do and do not say
- At 1M the exact full-attention reference found the right code in only 3 of 10 questions and answered with a distractor's code in 6. Its failures are mostly at middle depths (tokens 475K to 870K) and at 80K and 277K. That is the base model's behaviour on this task at 1M with YaRN x4, independent of LSA. The cause is untested: it could be model size, the YaRN extrapolation, the 20 similar distractors, or all three.
- That does not clear LSA. At a 128-key budget LSA scored 1 of 10 and block-sparse 7 of 10 on the same cache. Where the reference is reliable (128K: full attention 10 of 10 and 6 of 6), LSA matched it at budgets of 128 keys and up and fell behind block-sparse below that.
- A fair test of "does LSA preserve accuracy at 1M" needs a setting where the full-attention reference is reliable at that length: fewer or no distractors, a stronger model, or a model with native long context. Not done.
