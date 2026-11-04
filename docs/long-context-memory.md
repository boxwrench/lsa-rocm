# Possibility: LSA plus quantized KV plus CPU-resident cache

**Status: a hypothesis with back-of-envelope numbers. Nothing here is measured except where it links to an experiment.**

## The idea
Two techniques act on different costs of long context:
- **Quantized KV** (8-bit or 4-bit) shrinks what is *stored* per token.
- **LSA-style sparse decoding** shrinks what is *read* per decode step: sinks, a recent window, and k keys chosen by a small index, instead of the whole cache.

Alone, quantization stretches what fits in VRAM about 4x but every step still reads the whole cache. Alone, LSA does not shrink the stored cache.
Together, the quantized cache can live in CPU RAM (or Strix Halo unified memory) and only the selected keys cross the bus each step, so the context limit is set by system memory, not VRAM.
The index stays at full precision and separate from the stored K/V: the LSA index is a 64-dim vector per pooled key (about 1.3 GB at 10M tokens). A reference point: the public `vllm-htr` project (a vLLM fork for Strix Halo) states that its sparse-attention indexer keeps its own precision while main K/V is quantized, the same split.

## Estimates (Ornith 1.5 9B: 8 full-attention layers, 4 KV heads, head dim 256)
KV per token: bf16 32 KiB, 8-bit about 16 KiB, 4-bit about 8.5 KiB (with scale and zero-point). One R9700 has 32 GB, weights about 18 GB, leaving about 10 GiB for KV.

| Setup | KV lives in | Context limit (by capacity) |
|---|---|---|
| bf16, full attention | VRAM | about 330K tokens |
| 8-bit, full attention | VRAM | about 650K |
| 4-bit, full attention | VRAM | about 1.2M |
| 8-bit + LSA | CPU RAM (186 GB on Nautilus) | about 11M |
| 4-bit + LSA | CPU RAM | about 20M |
| 4-bit KV on Strix Halo (121 GB unified) | unified memory | about 14M (a ceiling, not a target) |

Per decode step the selected keys (about 2.6K at k=2048 plus window and sinks) are about 83 MB in bf16 or about 20 MB at 4-bit across the 8 layers: roughly 0.4 to 1.7 ms of PCIe time at 50 GB/s. This is bandwidth arithmetic, not a benchmark.

## What supports it so far (this repo)
- Sparse decoding found a planted fact at every tested depth at 32K, 10 of 10, same as full attention ([experiments/005-needle](../experiments/005-needle/)). 128K: see that directory's results.
- On real keys, a sparse decode step was 5x cheaper than a fair full-attention step at 512K ([experiments/004-e2e-attention](../experiments/004-e2e-attention/README.md)). Output error was 0.10 to 0.18 relative L2 at k=2048.
- A chunked-prefill dumper reached 917K tokens with KV in CPU RAM ([FINDINGS.md](FINDINGS.md)).

## What is not shown
- The combination itself: no quantized-KV gather, no dequantizing kernel, no copy/compute overlap. The 4-bit noise and the selection error stack, unmeasured.
- Prefill is still dense: 128K takes about 2 minutes on one R9700 and cost grows faster than linearly, so multi-million-token prompts take hours. Sparse prefill is untested.
- Accuracy at long context beyond one planted fact (several facts, aggregation, reasoning). Context past the native 262K uses YaRN.
- The `vllm-htr` claims (2M to 4M tokens on Strix Halo) come from its own page: its qualification profiles are "capacity-qualified, not filled-window benchmarks", its quality figure is next-token agreement with a bf16 cache over 495 positions, and I found no concurrency numbers. Different model (Qwen3.8-Flash). Not verified or reproduced here.

## Experiments that would test it, in order
1. Needle test at 128K and 512K with sparse decoding (running, 005); then several needles, smaller k, block-sparse as a comparison.
2. KV quantization applied to the cached K/V in the needle runner (simulated: quantize then dequantize), to measure the combined error before writing kernels.
3. Offloaded gather: selected keys from pinned CPU memory, timed against the 004 numbers.
4. A real corpus with facts the model cannot know.
5. Sparse prefill.
