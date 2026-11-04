# 005: fact-at-depth (needle) test with sparse decoding

**Status: preliminary.** Ornith 1.5 9B (a hybrid: 8 full-attention layers, 24 linear-attention layers), prose haystack, a planted fact ("the access code for vault N is CODE") and a question at the end, greedy decoding of 16 tokens, scored by string match (`run_matrix.py`).
The prompt is prefilled once with ordinary attention (all but the last prompt token). The last prompt token and the answer are then decoded under each configuration on a copy of the cache:

- `full`: ordinary attention.
- `lsa:QUANT:K`: on the 8 full-attention layers, read only 4 sinks, the last 544 keys and K keys chosen by the layer-shared PCA(64) flat index (the recipe tested in experiment 002). The index is built from the unquantized keys.
- `bsp:QUANT:K`: same, but K keys chosen by scoring means of 4-key blocks (block-sparse control).
- QUANT is `none`, `int8`, `int4` or `int4rht` (random-sign Hadamard rotation then 4-bit), all per token and head, **simulated by quantize then dequantize** of the cached K and V (no real packed kernel, no speed claim).
- K = 0 is a control: no long-range keys. It should fail.

## Reading the results (written 2026-11-04 after the 128K hard run)
- At budgets of 128 keys and up, LSA, block-sparse and full attention are indistinguishable on this test, at 32K and 128K, with 20 distractors, and with a simulated 4-bit cache.
- At a budget of 32 keys at 128K, LSA found 5 of 10 facts against 9 of 10 for block-sparse (at 32K, 8 against 9). Suggestive, not significant at 10 prompts. A possible reason, untested: block-sparse reads contiguous 4-key blocks, which keep the planted sentence's neighbouring tokens together, while Ward buckets group tokens by key similarity and split a sentence across buckets, so a tiny budget covers it only partly.
- The one full-attention miss at 128K (depth 0.25, vault 243) is a model error: it answered with a distractor's code. It is not a retrieval failure of the sparse path.
- `bsp:*:0` is not a true zero-budget control: the block-sparse code always takes at least one 4-key block, so it is effectively K = 4 (and got 3 of 10 at 128K). `lsa:*:0` reads no long-range keys (0 of 10 at both lengths).

## What these results do and do not show
- Sparse decoding on the 8 full-attention layers finds a single planted fact as reliably as full attention at the tested lengths, even with very small budgets and a simulated 4-bit cache. One fact is an easy target: the test saturates, which is why a harder variant with 20 similar distractor facts was added.
- The model is a hybrid. The 24 linear-attention layers carry some information from the whole context regardless of what the 8 full-attention layers read, so even K = 0 can sometimes recover part of the answer (seen in a smoke test). The K = 0 rows measure this floor.
- 10 prompts per cell (5 depths x 2 facts). One haystack text. Differences of one prompt are noise.
- GPU numerical faults (non-finite logits, intermittent on this machine) are detected, the prompt is re-run, and the count is reported. They are not scored as misses.
- An earlier runner (`run_needle.py`) fed the first generated token through full attention, so its sparse results only cover later tokens. `run_matrix.py` fixes that and supersedes the `results/005-needle-lsa-*.json` files, which are kept for the record.
- Not tested: aggregation or multi-fact questions, real corpora, 512K and beyond on the R9700, real KV quantization kernels, offloaded gathers, speed.

Reproduce: `HIP_VISIBLE_DEVICES=1 python experiments/005-needle/run_matrix.py --length 131072 --facts 2 --distractors 20 --configs full:none,lsa:none:2048,bsp:none:2048,lsa:none:0 --out results/005-hard-131072.json`, then `python experiments/005-needle/summarize.py`.
