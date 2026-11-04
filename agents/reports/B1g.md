# B1g: chunked-prefill dumper, 32K and 128K Ornith dumps

DRAFT 2026-11-03. Report path: agents/reports/B1g.md.

## What changed

`src/lsa_rocm/dump.py` gained `--prefill-chunk N` (default 0 = single pass, unchanged).
With chunking the prompt is run through the model in chunks of N tokens with
`use_cache=True` and a `DynamicCache`; `logits_to_keep=1` keeps per-chunk logits tiny.
Each chunk's post-RoPE K and V are the trailing `chunk_len` rows of the cached K/V
seen by the attention interface, and are appended to per-layer CPU buffers. Queries
of every chunk are appended and only the last `--n-queries` positions are kept.
`selfcheck_module_out` is the last position of the last chunk, exactly as in
single-pass. `prefill_chunk`, `chunked_prefill` and `num_chunks` were added to
meta.json; all earlier fields are kept. `sdpa` is the attention implementation.

Validation of the chunked path on Qwen3.5-0.8B (1024 tokens, chunk 256 vs single
pass, same prompt): keys/values/queries/selfcheck_module_out cosines 0.99989-0.99999
per layer. The residual difference is the linear-attention layers' chunked vs
single-pass recurrence, not the full-attention capture.

## Runs

- 32K: Ornith-1.5-9B, `experiments/data/texts/prose-32768.txt`, `--prefill-chunk 8192`,
  `results/dumps/ornith-chunked-32768`. Reference single-pass:
  `results/dumps/ornith-1.5-9b-32768`. Status: running.
- 128K: same model, `experiments/data/texts/prose-131072.txt`, chunk 8192,
  `results/dumps/ornith-chunked-131072`. Status: pending.

## Results

Pending. Numbers will be filled from the JSON/meta files written by the runs and
by the acceptance script output; no hand-copied numbers.

STATUS: DRAFT
