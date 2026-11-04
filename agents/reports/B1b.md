# B1b: CPU correctness oracle over `lsa.core`

Status: done, 2026-11-03. Commit `b75251933d49c46dbbc33bfa508cbf3edc55a6c1`.

## What was built

| File | Contents |
|---|---|
| `src/lsa_rocm/oracle.py` | `OracleConfig` dataclass; `build_flat_index`, `flat_select`, `build_tree_index`, `tree_select`, `exhaustive_select`; `FlatIndex` / `TreeIndex` wrappers. |
| `tests/test_oracle.py` | 7 tests, runnable directly (writes `results/b1b-oracle-tests.json`) or under pytest. |
| `results/b1b-oracle-tests.json` | machine-written run record: config, per-test status, recall table, containment. |

The wrapper is thin: `build_flat_index` calls `lsa.core.build_buckets`, `flat_select` calls
`lsa.core.read`, `build_tree_index` calls `build_buckets` + `lsa.core.build_hierarchy`, `tree_select`
calls `lsa.core.read_hierarchy`. The installed `lsa` 0.1.0 in `venv-lsa-min` is byte-identical to
`third_party/lsa/src/lsa` (`diff -q .../lsa/core/reference.py` reports `SAME`), so the oracle wraps the
pinned source. Keys/queries are `[N, D]` / `[Nq, D]` float32 unit rows; positions `[Nq]` int; selections
are the repo format `selected [Nq, S] int32` (-1 padded) + `lengths [Nq] int32`.

Valid argument values were taken from `third_party/lsa/src/lsa/core/{reference,hierarchy}.py` and
`tests/unit/core/test_{reference,hierarchy}.py`: `pool_size | bucket_size | cluster_block_size`,
`bucket_cap >= 2 * bucket_size`, `node_bucket_size` a power of two from 2 dividing
`node_cluster_block_size`, `page_size % cluster_block_size == 0`, `top_p/top_k/window_size >= 1`,
`sink_size >= 0`, positions non-negative. `growth_buckets=0` grows at every page end (upstream
`test_hierarchy.py`). The test config is `N=8192`, `D=32`, 32 clusters, `pool=4`, `bucket=8`, `cap=16`,
`block=64`, `top_level=4`, `node_bucket=4`, `node_cluster_block=16`, `beam_width=8`, `page=512`;
the recall budget is `top_p=32`, `top_k=256`, `sink=4`, `window=128`.

`exhaustive_select(q, k, pos, top_k, sink_size, window_size)` ranks the same long-range candidates
`[sink_size, max(0, pos+1-window_size))` by true dot product and appends the same window and sinks as
`expand_buckets`, so all three methods draw from the same candidate universe. Recall is measured on the
long-range part only (window and sinks are common to every method and would inflate the score).

## Commands and output

The named CPU venv has no `pytest` and no `pip` (`No module named pytest`, `No module named pip`), so
the test file has a direct runner; it also collects under pytest. Both were run.

```
$ cd /ai/github/lsa-rocm && PYTHONPATH=/ai/github/lsa-rocm/src \
    agents/scratch/venv-lsa-min/bin/python -I tests/test_oracle.py
PASS test_determinism_same_seed_identical_selections (0.060s)
PASS test_selected_indices_are_valid_and_causal (0.022s)
PASS test_sinks_and_window_are_always_included (0.003s)
clustered: flat recall 0.972, tree recall 0.971, chance 0.0325 (top_p=32, top_k=256, 64 queries)
PASS test_clustered_recall_is_clearly_above_chance (0.039s)
random:     flat recall 0.143, tree recall 0.086, chance 0.0325 (top_p=32, top_k=256, 64 queries)
PASS test_random_keys_recall_far_below_clustered (0.061s)
PASS test_full_top_p_and_full_top_k_covers_every_causal_key (0.019s)
full top_p (1024 buckets), top_k=256: exhaustive top-k containment mean 0.976, min 0.461, max 1.000
PASS test_full_top_p_with_a_limited_top_k_does_not_contain_the_exhaustive_top_k (0.034s)

recall table (long-range top-256 vs exhaustive):
        keys     flat     tree   chance
   clustered    0.972    0.971   0.0325
      random    0.143    0.086   0.0325

wrote /ai/github/lsa-rocm/results/b1b-oracle-tests.json
```

```
$ cd /ai/github/lsa-rocm && agents/scratch/venv-lsa/bin/python -I -m pytest tests/test_oracle.py -v
tests/test_oracle.py::test_determinism_same_seed_identical_selections PASSED [ 14%]
tests/test_oracle.py::test_selected_indices_are_valid_and_causal PASSED  [ 28%]
tests/test_oracle.py::test_sinks_and_window_are_always_included PASSED   [ 42%]
tests/test_oracle.py::test_clustered_recall_is_clearly_above_chance PASSED [ 57%]
tests/test_oracle.py::test_random_keys_recall_far_below_clustered PASSED [ 71%]
tests/test_oracle.py::test_full_top_p_and_full_top_k_covers_every_causal_key PASSED [ 85%]
tests/test_oracle.py::test_full_top_p_with_a_limited_top_k_does_not_contain_the_exhaustive_top_k PASSED [100%]
============================== 7 passed in 0.75s ===============================
```

`py_compile` on both files passes. `results/b1b-oracle-tests.json` records `"tests_run": 7,
"failures": 0, "successful": true`.

## Recall table (from `results/b1b-oracle-tests.json`)

| keys | flat recall | tree recall | chance |
|---|---|---|---|
| 32 separated Gaussian clusters (sigma 0.15) | 0.972 | 0.971 | 0.0325 |
| random unit rows | 0.143 | 0.086 | 0.0325 |

64 queries, long-range top-256 against the exhaustive top-256; chance is `top_k / n_candidates`.
Clustered recall is ~30x chance; random is ~4x / ~2.6x chance. The clustered-versus-random gap is the
point of the table.

## Full-budget containment: actual behaviour

The claim "when `top_p` covers every bucket, flat selection contains the exhaustive top-k" is **false
in general**, and the test reports the measured behaviour instead of forcing it:

- `top_p = 1024` (every bucket), `top_k = 256`: containment mean **0.976**, min **0.461**, max 1.000.
- `top_p = 1024`, `top_k = N` (a full long-range budget): the flat selection covers every causal key
  exactly once, so containment is 1.0 (test passes with `held.sort() == arange(position + 1)`).

Why: `expand_buckets` takes the first `top_k` members in bucket-centroid order (position order within a
bucket), not the `top_k` highest-scoring keys. With every bucket picked, the first `top_k` members are
the first `top_k / bucket_size` buckets' keys; a key with a higher dot product in a lower-centroid
bucket is missed. A full long-range budget is what makes the selection cover every causal key. The
`top_k=256` case is close to 1 only because the top 32 buckets are pure (cluster purity 1.0), but the
last few positions of the query's own block are in the window and two non-cluster keys enter the
exhaustive top-256.

## Acceptance

- [x] `src/lsa_rocm/oracle.py` implements `build_flat_index`, `flat_select`, `build_tree_index`,
      `tree_select`, `exhaustive_select` over `lsa.core` (shown above).
- [x] `tests/test_oracle.py` covers determinism, valid/causal indices, sinks+window inclusion,
      clustered recall above chance, random contrast, and the full-budget containment behaviour.
- [x] Tests pass with shown output (direct runner on `venv-lsa-min`, and pytest on `venv-lsa`).
- [x] Short recall table on clustered versus random keys (above, from the JSON).

## Open questions

- `lsa.core.read`'s causal rule is block-based (a query cannot read its own open block except through
  the window). The synthetic queries sit in the last page with `window_size >= cluster_block_size`, so
  the exhaustive baseline and the flat candidate universe coincide; on real KV the open block can cap
  recall and B2 should report it.
- The `exhaustive_select` baseline is the true dot-product top-k, not the softmax-attention top-k of
  `docs/metrics.md`. For unit rows and a single query the ranking is the same; B2 should use the
  metrics module for mass captured.

STATUS: DONE
