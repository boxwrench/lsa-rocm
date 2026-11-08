# Research directions: LSA, ZHA, HySparse2, and local ultra-long context

**Status: research notes and proposed experiments only (2026-11-08).** No ZHA or HySparse2 implementation, quality result, speedup, or combined-system validation is claimed by this repository. Paper claims, our measured results, and hypotheses are separated below. This is an independent AMD/ROCm lab notebook, not an official implementation of any of these methods.

## Research question

How do we make **reliable** 1M+ token inference usable on local AMD hardware? It has at least four different bottlenecks: (1) prompt prefill compute, (2) which historical KV positions are searched/read during decode, (3) KV-cache capacity and movement between CPU RAM and VRAM, and (4) whether the base model can actually use the entire context. Optimizing any one does not establish success on the others.

## Established local baseline (not paper claims)

Source of truth: [FINDINGS](FINDINGS.md), [fidelity](fidelity.md), [experiment 006](../experiments/006-long-context/README.md), and the machine-written JSON in `results/`.

- On **Ornith 1.5 9B**, native 256K needle testing: exact attention answered 10/10; when the question itself used sparse attention, training-free LSA at **k=512** answered 10/10, but **k=128** answered 3/10 while block-sparse k=128 answered 9/10. One prompt / ten questions; not proof of general task accuracy.
- At **1M with YaRN x4**, exact full attention answered only 3/10 in the distractor needle protocol. Sparse variants' scores cannot establish long-context fidelity when the teacher fails. Select a base model and evaluation where dense attention is reliable before interpreting 1M+ results.
- Training-free LSA's **flat and tree selectors matched to two decimals** in mean attention-mass retrieval across tested runs to 917K. This is primarily a quality observation, **not** a demonstrated tree-routing speedup.
- Sparse prefill on one R9700 took **3,522 s vs 4,316 s dense at 1M** (1.23x faster); at 128K and 256K, the current PyTorch sparse path was slower. These are implementation-specific wall times, not an asymptotic promise.
- On the R9700, the index/gather path has a significant gather floor. Earlier synthetic index benchmarks are not a substitute for measuring real end-to-end prefill, decode, PCIe traffic, or an optimized HIP/Triton kernel.
- The later [experiment 007](../experiments/007-nemotron-kernels/README.md) provides a useful negative lesson: Nemotron-3-Nano did **not** pass the chosen long-context string-copy needle protocol even under unmodified exact attention, so it cannot serve as a fidelity teacher for that protocol.

## Research comparison

| Approach | What it changes | Preserves original KV? | Practical retrofit to existing checkpoint? | What it does **not** solve alone |
|---|---|---|---|---|
| **Latent Sparse Attention (LSA)**, [Prefrontal paper](https://arxiv.org/abs/2610.42147) | Indexes keys via low-dimensional projected buckets and searches them (flat or hierarchical) for sparse attention | Yes, for the global layers | Training-free PCA index is already tested here; distilled indexer is untested here | Full stored KV capacity, model context generalization, full-model prefill outside the sparsified operations |
| **Zyphra Hierarchical Attention (ZHA)**, [paper v2](https://arxiv.org/abs/2610.42112v2) | Uses recursively compressed summaries over **contiguous** key chunks, learned reference queries, top-p coarse-to-fine routing | Yes; original KV remain at leaves | Paper demonstrates freezing an existing base model and KL-distilling a **small router**; not a drop-in untrained swap | Stored KV capacity, CPU↔GPU movement, or guaranteed long-context fidelity of an extrapolated base |
| **HySparse2**, [Xiaomi paper](https://arxiv.org/abs/2609.26368) | Changes the backbone into self-/cross-decoder portions with **KV Bridging** across them, and **KV Reuse** within hybrid full/sparse blocks; cross-decoder prompt prefill can be skipped | Some KV caches are shared/derived rather than independently built | **No straightforward retrofit**: trained architectural change, not a post-hoc sparse index | Reliable 1M model behavior or a turnkey ROCm implementation |

**Important distinctions:**

- The existing LSA Ward-cluster tree is **not** the ZHA tree. LSA clusters by key similarity; ZHA summarizes contiguous chunks with trained reference queries and routes adaptively by estimated subtree attention mass. Code and correctness logic can be shared, but simply renaming the existing tree would not be a ZHA reproduction.
- Flat index scoring can still grow linearly in history length for each query. ZHA's asymptotic claim assumes the selected tree traversal remains sublinear as the tree depth grows; irregular GPU traversal could offset theoretical savings.
- Sparse retrieval only shrinks the set of KV entries *read*. LSA and ZHA do not inherently shrink the stored leaf cache. Quantization/offload may combine with either, subject to actual gather/transfer cost.
- HySparse2 reduces **layer-level cache duplication and prefill work** by changing the architecture. It is complementary in principle to an efficient index, but any ZHA/LSA + HySparse2 combination would be a new, unvalidated architecture rather than a documented paper result.

## What the papers actually establish

**ZHA** (Lufkin et al., 2026): The authors report adapting frozen pretrained models by training lightweight reference-query parameters with KL distillation; their Qwen3.6-35B-A3B study adds **81,920 trainable parameters**. They evaluate reasoning, coding, and retrieval, including 256K RULER. Their extreme-context throughput advantages are primarily from a **performance model**, not measured end-to-end speed on local RDNA/ROCm hardware. Training the router still requires expensive frozen-model dense-attention teacher computation, despite the small parameter count. See [Zyphra technical explanation](https://www.zyphra.com/our-work/zyphra-hierarchical-attention) and [arXiv](https://arxiv.org/abs/2610.42112v2).

**HySparse2** (Wei et al., Xiaomi LLM-Core, 2026): The 80B-A3B model was studied as a **trained model architecture**, not an existing checkpoint converted by swapping kernels. The paper reports, *by FLOP/cache analysis at 1M*, **5.02x less prefill compute** and ~**4.5x less KV storage** than its Hybrid SWA comparator (2.69 vs 12.09 GB), and improved retrieval evaluation at up to 256K. These are not local measured wall-clock speedups or generic savings for Ornith. See [paper](https://arxiv.org/abs/2609.26368); [YOCO](https://arxiv.org/abs/2405.05254) is relevant architectural prior art for early-exit prefill.

## Possible uses in this repository, ranked

### Priority A — Router alternatives on identical captured Q/K/V (proposed experiment 008)

Add a **ZHA-inspired** reference-query tree as a *new selector*, not as a modification of existing LSA results. Start on one Ornith full-attention layer and the same captured 32K/128K/256K keys. Compare:

- LSA flat and LSA tree (current controls);
- block-sparse and recency;
- ZHA-style contiguous-chunk tree with configurable chunk size, tree depth, and top-p routing;
- exact top-k oracle at a **matched retrieved-token count**.

First make a correctness-only prototype. A random or averaged-key router is an **ablation**, **not** trained ZHA. Match sink/window policy, RoPE handling, head grouping, and real selected-key count. Record index-build latency, nodes visited, selected keys, attention-mass capture vs exact oracle, output error, and memory. An adaptive top-p scheme must be compared at both matched **key budget** and matched **latency**; otherwise it may win by reading more tokens. Evaluate at **native 256K** on the existing reliable needle task, then add multi-fact, aggregate, code-navigation and high-entropy facts in a real corpus.

**Gate A:** Do not proceed to engine integration until a ZHA-style router shows a meaningful retrieval/latency advantage on more than one prompt and correct end-task answers compared with the existing LSA/block-sparse baselines. A tree that matches existing LSA quality but has slower traversal is not a win.

### Priority B — Small ZHA router distillation (follow-on to 008, not yet authorized as a large training run)

If the forward-only ablation is promising, train the reference queries using dense-teacher attention mass over contiguous chunks. Freeze the base model, split training and evaluation **by prompt/corpus**, and retain held-out genres. Try a small one-layer / few-head budget before scaling; record teacher FLOPs, optimizer memory, training tokens, attention KL, and per-layer degradation. Validate RoPE/prefill behavior on the exact model rather than assuming the published Qwen results transfer to Ornith.

**Gate B:** The learned selector must outperform the same-cost untrained contiguous tree, LSA PCA and block-sparse across held-out prompts, with no meaningful accuracy regression at 256K. A tiny trainable-parameter count alone is not evidence that training is cheap.

### Priority C — Quantized KV + CPU-resident selected gather (parallel; highest practical deployment relevance)

Build on [long-context-memory.md](long-context-memory.md). Implement and benchmark *real*, not simulated, quantized KV storage and a selected gather/dequantize path. Preserve a full, independently checkable dense reference for the native-context tests. Measure host RAM footprint, pinned-memory allocation, PCIe bytes, latency per layer, index RAM, overlap, throughput and correctness. Test the R9700 discrete-VRAM path and, separately, the Strix Halo unified-memory path. Neither architecture eliminates the cost of initially constructing KV, and offload is only useful when selected reads beat transferring/attending to the full cache.

### Priority D — HySparse2/YOCO-inspired **isolated** prefill research (separate architecture track)

Do **not** promise HySparse2 conversion of Ornith or claim Xiaomi's 1M gains. Prototype a tiny self-/cross-decoder or KV-bridge test model with learnable projections, and compare bridge reconstruction quality, actual cache bytes, forward FLOPs and elapsed prefill at fixed model quality against a properly trained same-scale baseline. Early-exit is only valid if the cross-decoder can access the necessary representations during generation without recomputing all the saved-prefix hidden states.

**Gate D:** Only consider a larger implementation if a *trained* bridge preserves real task performance while measurably reducing total model prefill cost. Paper FLOP ratios alone are insufficient.

## Shared benchmark protocol for new experiments

1. **Pin** exact model/revision, prompt seed, corpus/license, context length, rope scaling, hardware, ROCm/PyTorch/kernel versions, quantization, and git commit; machine-write JSON under `results/`.
2. **Baseline trust:** evaluate dense end-task correctness **first** at every length. Native 256K Ornith is the current strongest reference. Do not use 1M YaRN failures as proof of sparse router failure or success.
3. **Separate** index-build/maintenance, query routing, CPU→GPU gather, dequant, sparse attention, prefill wall time, first-token latency, decode tok/s, peak VRAM, host RAM, and PCIe traffic.
4. **Measure** retrieval probability mass and selected-token overlap alongside answer accuracy; report each full-attention layer and each genre, not only a model-average score.
5. **Compare fairly:** fixed retrieved tokens, fixed latency budget, same sink/local window, same precision, same input and prefix reuse. Report quality vs cost Pareto curves, not just a single top-line number.
6. **Distinguish** measured local results, calculations, published author claims, projections, and proposed experiments in every write-up; record negative findings unchanged.
7. **Port later:** use PyTorch for correctness, then Triton/HIP only after profiling the true bottleneck. Do not infer RDNA performance from H200/B300/MI355X numbers.

## Open questions worth keeping visible

- Does hierarchical routing reduce **total** index work on RDNA, or does irregular search/gather dominate until multi-million-token scale?
- Are learned ZHA summaries superior to LSA PCA on Ornith's partial-RoPE layers, especially for 256K *question prefill*, where very small k degraded answers?
- Can sparse query routing work efficiently over a **CPU-resident int4 KV cache**, including prompt prefill and multiple simultaneous queries? What happens on Strix Halo UMA?
- Which local checkpoint is accurate with **dense attention at 1M native or validated extrapolated context**, not merely able to allocate a 1M KV cache?
- Does any cross-layer KV sharing generalize through an inexpensive student distillation, or does it require costly pretraining?

## References (primary sources)

- Prefrontal AI / Kubicek et al., *Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention* (2026): https://arxiv.org/abs/2610.42147 ; upstream: https://github.com/Prefrontal-AI/latent-sparse-attention
- Lufkin et al., *Subquadratic Attention with a Hierarchical Tree-Structured Indexer*, ZHA (v2, Nov 2026): https://arxiv.org/abs/2610.42112v2 ; authors' explanation: https://www.zyphra.com/our-work/zyphra-hierarchical-attention
- Wei et al., *HySparse2: Hybrid Sparse Attention with Two-Level KV Sharing* (Sep 2026): https://arxiv.org/abs/2609.26368
- *YOCO: You Only Cache Once* (2024): https://arxiv.org/abs/2405.05254
- Local measured results: [FINDINGS](FINDINGS.md), [experiment 006](../experiments/006-long-context/README.md), [experiment 007](../experiments/007-nemotron-kernels/README.md)

**Scope discipline:** This document proposes work. Existing pre-registered gates in [plan.md](plan.md), results, code, and measured findings have not been altered by this research update.
