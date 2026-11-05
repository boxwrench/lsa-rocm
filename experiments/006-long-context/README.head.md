# 006: needle test at 128K and 1M with the KV cache in CPU RAM

**Status: preliminary.** Ornith 1.5 9B, one haystack per length, 6 questions at 128K and 10 at 1M, 20 distractor facts about other vaults, string-match scoring. 1M tokens = code-coh-917504 (exllamav3 source) followed by prose-coh-524288, truncated, with YaRN x4 beyond the native 262K.

## What this does
`run_big.py` prefills the haystack once (dense attention, chunked), keeps the K/V of the 8 full-attention layers in CPU RAM, and saves the whole cache so a GPU fault cannot cost the prefill. Each question is then answered from copies of the cache with attention on those 8 layers done one of these ways:
- `full`: exact attention, streaming the entire CPU cache through the GPU in chunks (online softmax). This is the reference.
- `lsa:QUANT:K`: sinks + the last 544 prefix keys + the new tokens + K keys chosen by the layer-shared PCA(64) flat index; only those keys are copied from CPU RAM to the GPU.
- `bsp:QUANT:K`: same, with block-sparse selection (means of 4-key blocks).
- QUANT `int4rht` is a simulated rotated 4-bit cache (quantize then dequantize on read).
"Sparse question" means the question's own tokens also go through the config's attention. "Dense question" (`--question-full`) reads the question exactly and only the answer tokens use the config.

## How to read the tables
Wrong answers are split into: a distractor's code (the model found a vault note but the wrong one), another target's code, or neither (no note found or garbled). Correct counts are against the planted truth, not against the full-attention answer.

## Measured facts
- Prefill of 1,048,576 tokens took 4316 s (72 min) on one R9700 with the cache offloaded to RAM, chunk 1024. A two-GPU split run hit a GPU memory fault at about 820K tokens after 28 min. A flash-kernel prefill path (`src/lsa_rocm/fastattn.py`) cut a 128K prefill from 121 s to 83 s; the 1M two-GPU run was on pace for about 40 min.
- A bug in an early version (cache state of the linear-attention layers not saved) invalidated the first 128K stage; those results were discarded, and the fixed pipeline reproduces in-process answers when reloaded from disk.

## Reading the 256K native-length results (written 2026-11-05)
At 262,144 tokens with no YaRN (the model's native limit, 20 distractors, 10 questions) exact full attention scored 10 of 10, so the reference is reliable there and the comparison is meaningful. With the question read through sparse attention, LSA matched it at budgets of 512 keys and up (10 of 10) and fell to 3 of 10 at 128 keys, while block-sparse at 128 keys scored 9 of 10. The 1M failures below are therefore due to the YaRN extension beyond 262K, not to the 9B being unable to resolve these facts.

## Reading the 1M results (written 2026-11-05)
At 1M the test cannot separate the methods. With the question read through sparse attention, sparse configs scored 1 to 7 of 10 against 3 for exact attention; with the question read exactly, every config scored 2 or 3 of 10, including exact full attention. The 9B is close to unreliable here (YaRN x4 beyond its native 262K, 20 similar distractors), so the earlier edge of the sparse paths over the reference is best read as noise. 128K is the length where the reference is reliable and the comparison is meaningful.

## Limits
One haystack per length, 6 to 10 questions per cell: differences of one or two answers are noise. At 1M the exact reference itself is wrong most of the time (YaRN extrapolation plus similar distractors), so "close to full attention" is not the right bar there. Mixed code and prose haystack. Hybrid model: the 24 linear-attention layers also carry information and are not sparsified. No speed claims: the 1M question stage timings are dominated by streaming the exact reference over PCIe.
