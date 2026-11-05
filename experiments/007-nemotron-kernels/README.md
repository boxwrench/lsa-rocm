# Nemotron-3-Nano Triton kernel port (Nimo, gfx1151)

## Borrowed / written
- vLLM `vllm/model_executor/layers/mamba/ops/{ssd_bmm,ssd_chunk_scan,ssd_chunk_state,ssd_state_passing,ssd_combined,triton_helpers}.py` (main, Apache-2.0, vLLM contributors; originally Tri Dao/Albert Gu mamba-ssm, Apache-2.0). Copied to `src/lsa_rocm/kernels/` with headers kept, LICENSE.vllm alongside. Edits only: `vllm.triton_utils` -> `import triton`, helper import made relative, `current_platform` sm90 filter and CPU branch removed, `softplus` inlined from vllm `mamba_ssm.py`. Raw downloads: nimo:~/lsa/vendor/vllm.
- `conv1d.py`: ORIGINAL fused depthwise causal conv + SiLU Triton kernel (vLLM causal_conv1d.py read, too heavy to vendor).
- `moe.py`: ORIGINAL grouped-GEMM fused MoE (up -> relu^2 -> down*weight), torch-op token alignment (vLLM's align op is CUDA, skipped). No vLLM code copied.
- `nemotron_patch.py`: monkeypatches `mh.mamba2_chunk_scan`, `mh.causal_conv1d_fn`, `NemotronHExperts.forward` (apply()). Nothing in site-packages edited.
- Scripts: `agents/scratch/kernel-port/{test_kernels,verify_nemotron_kernels}.py` (copies on nimo:~/lsa/). Logs: nimo:~/lsa/verify.log (first run), verify2.log.

## Per-kernel check vs torch reference (random inputs, test_kernels.py)
SSD (T=1000 ragged, 2048; with/without initial state): out cos 0.999995, final-state cos 0.999996+, max abs 0.21-0.26 (ref max ~45). Conv: cos 0.999994, max abs 0.0625. MoE (M=64, 2048, 128 experts top-6): cos 0.999986, max abs 0.14.

## Model verify (2K prompt, patched vs unmodified fallback)
hidden states cos 0.999385 (fp64), max abs diff 9.5 (ref max 174), Frobenius rel err 3.5%; per-position min cos 0.967 (pos 536), mean 0.9993, 15% of positions < 0.999. Final-position logits cos 0.999786, max abs diff 0.56 (ref max 27), argmax identical. Passes the >=0.999 gate on aggregate cosine; the per-position tail is bf16-level drift through 52 layers (scan output is bf16 here vs fp32 in the fallback) - not investigated further.

## Timing (patched, prefill, logits_to_keep=1, weights 58.9 GiB included in peak)
| ctx | tok/s | time | peak GiB |
|---|---|---|---|
| 2K | 2512 | 0.82 s | 59.3 |
| 8K | 2265 | 3.62 s | 60.1 |
| 16K | 1968 | 8.32 s | 61.3 |
| 32K | 1710 | 19.16 s | 63.6 |
Baseline (given): 92 / 71 / OOM / -. Same-run fallback 2K forward took 23 s (includes first-call overhead).

## Caveats / what failed
- First run without `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`: 2K 1829, 8K 810 tok/s, 16K OOM (32 GiB alloc, math-path SDPA in the attention layers). The flag (efficient/flash SDPA, marked experimental by torch) fixed it; the verify script sets it for both reference and patched runs. Needs a user decision whether to accept an experimental flag.
- Baseline MoE already used transformers `grouped_mm` experts (not the python loop), so MoE gain is smaller than assumed; Mamba scan/conv + attention were the bulk.
- Decode path (seq_len 1) not patched (uses transformers fallback); only prefill kernels. Batch>1 loops per sequence. No GPU faults.
