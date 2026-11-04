# B1e — Long-context Q/K/V dumper (sdpa)

Task: make `src/lsa_rocm/dump.py` scale past the full-attention-matrix (eager) path,
capture only what the retrieval analysis needs, self-check the last query against the
module's own output, and dump Ornith-1.5-9B at 8K/16K/32K from `prose-32768.txt`.

## What changed in `src/lsa_rocm/dump.py`

- New `--attn {sdpa,eager}` option, default `sdpa`. `eager` keeps the old
  full-attention-matrix path for the small-model tests; `sdpa` is the default and is
  what the long-context runs use.
- The model is loaded with `attn_implementation=args.attn`; the wrapper is registered
  under that same key in `transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS`
  (for `eager`, the module's own `eager_attention_forward` is recovered first, as in
  B1c). The wrapper calls the original interface and captures, per full-attention
  layer: post-RoPE `keys_post_rope [Hkv,T,D]`, `values [Hkv,T,D]`,
  `queries_post_rope [Hq,Q,D]` for the last `--n-queries` positions, `positions [Q]`,
  the module's own `scaling`, and the attention interface's output for the last query
  position `[Hq,D]` (before the output gate / o_proj), saved as
  `selfcheck_module_out` in every layer file.
- Pre-RoPE keys are still captured by wrapping `apply_rotary_pos_emb` (kept for the
  B2b ablation); `rotary_dim` is recorded per layer.
- `attention_mask=None` is passed to the model: the prompt is truncated to exactly
  `--max-tokens` tokens and never padded, so SDPA can use `is_causal=True` and its
  flash kernel instead of materialising a mask.
- meta.json additions: `max_tokens`, `prompt_file` (basename), `prompt_file_path`,
  `prompt_sha256`, `peak_vram_gib` (`torch.cuda.max_memory_allocated`, reset before
  the forward), `elapsed_s` (forward wall time), `attention_impl` (read back from
  `model.config._attn_implementation`), top-level `self_check_layer` and
  `self_check_cosine`. The old `attention_implementation` and nested `self_check`
  keys are kept for compatibility.
- The tokenizer still truncates to the first `--max-tokens` tokens
  (`truncation=True, max_length=--max-tokens`).

## Environment

- GPU: physical index 1 (`HIP_VISIBLE_DEVICES=1`), AMD Radeon AI PRO R9700, 34.2 GB.
  `rocm-smi` before the runs: GPU 1 at 0% use / 59.9 MB used. Devices 0 and 2 unused.
- `/ai/environments/comfyui-rocm714/bin/python`: torch 2.12.0+rocm7.14.0,
  ROCm 7.14.60850, transformers 5.16.1.
- Probe of SDPA at 32K (q [1,16,32768,256], k/v [1,4,32768,256], `enable_gqa=True`):
  flash kernel, 0.34 s, peak 0.63 GiB. Math fallback OOMs (needs a 64 GiB matrix).

## Validation

CPU probe, Qwen3.5-0.8B, 512 tokens, `--attn sdpa` (default), n-queries 4:

```
SELF-CHECK layer 3: cosine(last-query attention output) = 1.00000000 (threshold 0.999)
WROTE agents/scratch/probe-sdpa-cpu/meta.json (42.21 MiB of tensors across 6 files)
{"full_attention_layer_indices": [3, 7, 11, 15, 19, 23], "self_check_cosine": 1.0, "dump_mib": 42.205}
```

meta fields confirmed: `max_tokens=512`, `prompt_file=prose-4096.txt`,
`prompt_sha256=3662ed89…` (matches the manifest), `attention_impl=sdpa`,
`self_check_layer=3`, `self_check_cosine=1.0`.

## Ornith-1.5-9B runs (GPU 1)

Commands (one at a time, `n-queries 32`, prompt `experiments/data/texts/prose-32768.txt`):

```
HIP_VISIBLE_DEVICES=1 PYTHONPATH=/ai/github/lsa-rocm/src \
  /ai/environments/comfyui-rocm714/bin/python -m lsa_rocm.dump \
  --model /ai/models/llm/Ornith-1.5-9B \
  --prompt-file experiments/data/texts/prose-32768.txt \
  --max-tokens <L> --n-queries 32 --device cuda \
  --out results/dumps/ornith-1.5-9b-<L>
```

Results: PENDING (runs in progress).

## Acceptance

PENDING.

STATUS: PARTIAL — code changed and CPU-validated; Ornith 8K/16K/32K dumps not yet on disk.
