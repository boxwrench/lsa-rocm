# Metrics

Setup: for one layer, one KV head group, and a query q, the true attention weights over N cached keys are
`a = softmax(q K^T / sqrt(d))`. A retrieval method returns a set S of key indices under a budget B = |S|.
All metrics are computed per (layer, head), then aggregated. Always report the distribution, not only the mean.

| Metric | Definition | Role |
|---|---|---|
| Mass captured | `sum_{i in S} a_i` | Primary. Measures what matters for the output. |
| Recall@k | `|S ∩ topk(a, k)| / k` | Secondary. Penalises missing near-ties too much on its own. |
| Output error | `||o_full - o_S||_2 / ||o_full||_2`, where `o_S` is attention restricted to S and renormalised | Checks mass against the real output. |
| Budget | `B / N`, and number of centroids scored | Cost axis. Report every metric against it. |
| Index cost | time and memory to build the clusters | Reported separately from query cost. |

Rules:
- Attention sinks (first few tokens) are always included in S and reported separately so they do not inflate mass captured.
- The true top-k and mass use the same scaling and masking as the model.
- GQA: queries sharing a KV head are evaluated against that head's keys. State whether the budget is per query head or per KV head.
- Aggregate by layer and by head. A mean over all heads hides the easy and hard heads, which is the interesting result.
