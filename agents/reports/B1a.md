# B1a — retrieval metrics (`src/lsa_rocm/metrics.py`)

Task: implement the metrics defined in `docs/metrics.md` as pure-torch functions, with unit tests on
hand-made tensors whose answers are known exactly.

Status: DONE. 24/24 unit tests pass on the CPU venv (`torch 2.13.0+cpu`), plus a smoke integration with
the real `lsa.core.read` output.

## What was built

`src/lsa_rocm/metrics.py` — pure torch, no numpy, float32, device agnostic, no batch dimension
(queries are `[Nq, D]`, keys `[N, D]`). Selection format is the repo-wide one returned by
`lsa.core.read`: `selected [Nq, S]` int with `lengths [Nq]` giving the valid prefix of each row
(padding, including copies of key 0, is ignored; `-1` padding also ignored because only the first
`lengths[i]` entries are read).

| Function | Signature | Returns |
|---|---|---|
| `attention_weights` | `(q [Nq,D], k [N,D], scale, q_pos=None)` | float32 `[Nq,N]` softmax; with `q_pos`, key `j` allowed iff `j <= q_pos[i]` (inclusive), future keys weight 0; a fully-masked row is all zeros (no NaN) |
| `mass_captured` | `(weights, selected, lengths, sink_size=0)` | dict of per-query `[Nq]` tensors: `total_mass`, `mass_excluding_sinks`, `sink_mass` |
| `recall_at_k` | `(weights, selected, lengths, k)` | `|S ∩ topk(a,k)| / min(k,N)`, float32 `[Nq]` |
| `output_error` | `(weights, v [N,Dv], selected, lengths)` | `||o_full - o_S|| / ||o_full||`, `o_S` renormalised over S; zero mass -> error 1; zero `o_full` -> error 0 |
| `budget_fraction` | `(lengths, n_keys)` | `lengths / n_keys`, float32 `[Nq]` |

Tie-breaking in the true top-k is documented in the module and function docstrings: **stable descending
sort by weight, so equal weights keep ascending key index order** (i.e. `argsort(-w, stable=True)`).

Sinks: `mass_captured(..., sink_size=s)` reports sink weight and non-sink weight separately so sinks do
not inflate the captured non-sink mass, per `docs/metrics.md`.

## Tests

`tests/test_metrics.py` — `unittest` (pytest is not installed in the CPU venv, so no extra install was
needed). The test file inserts `src/` on `sys.path` itself, so it runs without `PYTHONPATH`.

Exact-answer cases included:

- softmax equals a hand-computed softmax; `scale` applied; causal mask excludes future keys and includes
  the boundary key `j == q_pos`; fully-masked query row is zeros.
- selecting all keys -> mass 1.0, recall 1.0, output error 0.
- missing heavy keys -> computed mass (e.g. weights `[0.5,0.2,0.2,0.1]`, select keys 1,2 -> 0.4).
- padding beyond `lengths` ignored; `-1` padding ignored (as produced by `lsa.core`).
- sink mass reported separately and excluded from `mass_excluding_sinks`.
- recall hit / miss / half; ties broken by ascending index (all-equal weights -> top-2 = keys 0,1);
  `k > N` clamped.
- output error on a known case (`[0.5,0.5]`, select one of two orthogonal `v` rows -> 1.0) and a
  renormalisation case; empty selection -> error 1.
- `budget_fraction` basic values and validation.

## Commands and output

Run with the CPU venv:

```
cd /ai/github/lsa-rocm
/ai/github/lsa-rocm/agents/scratch/venv-lsa-min/bin/python -I -m unittest discover -s tests -v
```

Output (full, from `agents/scratch/b1a_final.txt`):

```
test_causal_mask_excludes_future_keys (test_metrics.AttentionWeightsTest...) ... ok
test_causal_mask_inclusive_boundary (test_metrics.AttentionWeightsTest...) ... ok
test_fully_masked_query_gives_zero_row (test_metrics.AttentionWeightsTest...) ... ok
test_input_validation (test_metrics.AttentionWeightsTest...) ... ok
test_matches_manual_softmax (test_metrics.AttentionWeightsTest...) ... ok
test_scale_is_applied (test_metrics.AttentionWeightsTest...) ... ok
test_basic (test_metrics.BudgetFractionTest...) ... ok
test_validation (test_metrics.BudgetFractionTest...) ... ok
test_all_keys_selected_gives_mass_one (test_metrics.MassCapturedTest...) ... ok
test_empty_selection_gives_zero (test_metrics.MassCapturedTest...) ... ok
test_minus_one_padding_from_lsa_core_is_ignored (test_metrics.MassCapturedTest...) ... ok
test_missing_heavy_keys_gives_computed_mass (test_metrics.MassCapturedTest...) ... ok
test_padding_beyond_length_is_ignored (test_metrics.MassCapturedTest...) ... ok
test_sink_mass_reported_separately (test_metrics.MassCapturedTest...) ... ok
test_all_keys_selected_gives_zero_error (test_metrics.OutputErrorTest...) ... ok
test_error_formula_on_a_known_case (test_metrics.OutputErrorTest...) ... ok
test_padding_is_ignored (test_metrics.OutputErrorTest...) ... ok
test_renormalisation_is_used (test_metrics.OutputErrorTest...) ... ok
test_zero_selection_gives_error_one (test_metrics.OutputErrorTest...) ... ok
test_all_keys_selected_gives_one (test_metrics.RecallAtKTest...) ... ok
test_k_larger_than_n_is_clamped (test_metrics.RecallAtKTest...) ... ok
test_padding_is_ignored (test_metrics.RecallAtKTest...) ... ok
test_ties_broken_by_ascending_index (test_metrics.RecallAtKTest...) ... ok
test_top_k_miss (test_metrics.RecallAtKTest...) ... ok

----------------------------------------------------------------------
Ran 24 tests in 0.003s

OK
```

(The CPU venv prints one benign `Failed to initialize NumPy` warning from torch itself; numpy is not a
dependency of this module.)

Machine-written result file: `results/b1a-metrics-tests.json` (generated by
`agents/scratch/run_b1a.py`, no hand-copied numbers). Key fields:

```
task=B1a  git_commit=b75251933d49c46dbbc33bfa508cbf3edc55a6c1
torch=2.13.0+cpu  tests_run=24  failures=0  errors=0  skipped=0  successful=true
```

## Integration with `lsa.core.read`

`agents/scratch/run_b1a_integration.py` builds real buckets (32 unit-norm keys, d=8, pool_size=1,
bucket_size=8, bucket_cap=16, cluster_block_size=8), calls `lsa.core.read(...)` with top_p=2, top_k=8,
sink_size=1, window_size=4, and feeds `selected, lengths` straight into every metric. Output is written
to `results/b1a-lsa-core-integration.json`; all sanity checks pass (`mass in [0,1]`, `recall <= 1`,
`error >= 0`). This confirms the selection-format contract with B1b/B1c and that `lsa.core`'s `-1`
padding is handled.

## Open questions / notes

- `recall_at_k` normalises by `min(k, N)`, so recall is still 1.0 when `k > N` and all keys are
  selected. Documented in the docstring.
- Duplicate indices in a selection would be double-counted by `mass_captured`/`output_error`; selections
  from `lsa.core` are distinct, and the docstring says so. If a future selector can repeat keys, dedupe
  first.
- `attention_weights` takes an explicit `scale` (per `docs/metrics.md`, use the model's own scaling, not
  an assumed `1/sqrt(d)`); the dumper is responsible for passing it.

STATUS: DONE
