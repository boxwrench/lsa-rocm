# B0b — Attention design of Nanbeige/Nanbeige4.3-33B

Status: final, 2026-11-03. Read-only; no GPU, no model download over 1 GB (only JSON configs and
the first 200 KB of each safetensors shard were fetched to read weight headers).

## Verdict

Nanbeige4.3-33B is a **dense, pure-softmax, full-attention Llama**: 52/52 layers full attention,
GQA 56 query / 8 KV heads, head_dim 128, RoPE **default** variant with `rope_theta = 120,000,000`
and **no** `rope_scaling`, **no** QK-norm, **no** sliding window, **no** attention sinks, **no** MLA,
**no** sparse/NSA attention. Hub `total_size` = **66,237,954,048 B** (61.69 GiB; 66.24 GB).

This is the clean "pure full-attention" subject the plan wanted, and it is also the easiest possible
case for LSA: every layer is identical, so B2c can pick any layer; there is no hybrid split.

## Evidence base

| Source | File |
|---|---|
| Hub config (live) | `agents/scratch/nb_config_live.json` (fetched 2026-11-03 from `/resolve/main/config.json`) |
| Hub base config (live) | `agents/scratch/nb_base_config.json` |
| Hub repo listing / tree | `agents/scratch/nb_all.json`, `nb_blobs.json`, `tree_Nanbeige_Nanbeige4.3-33B.json` |
| Weight index | `agents/scratch/nb_model.safetensors.index.json` (`metadata.total_size = 66237954048`, 471 tensors) |
| Modeling code | `agents/scratch/modeling_llama_4.51.1.py`, byte-identical to upstream `transformers` v4.51.1 `models/llama/modeling_llama.py` (verified with `diff` → IDENTICAL; upstream copy in `agents/scratch/upstream_llama_v4.51.1.py`) |
| Quant configs | `agents/scratch/nbq_FP8_config.json`, `nbq_INT8_config.json`, `nbq_INT4_config.json`, `nb_*_index.json`, `nb_bar.json` |

No `modeling_*.py` and no `auto_map` are shipped on the Hub (`tokenizer_config.json` has no
`auto_map`; repo file list has no `.py` files), so `trust_remote_code=True` in the model card is
vestigial and stock `LlamaForCausalLM` runs.

I also read the safetensors headers of **all 14 bf16 shards** (HTTP range 0–200 kB each) and
enumerated every tensor. That is independent proof of the attention layout, not just config trust:

```
471 tensors, and the only per-layer names are:
  self_attn.{q,k,v,o}_proj.weight
  input_layernorm.weight, post_attention_layernorm.weight
  mlp.{gate,up,down}_proj.weight
shapes: q_proj [7168,7168], k_proj [1024,7168], v_proj [1024,7168], o_proj [7168,7168]
```

There are **no** `q_norm`/`k_norm`, no `sink`, no `kv_a_proj`/`kv_b_proj`, no `mla`, no
`sliding_window` parameters anywhere in the checkpoint. A hybrid model would have to reuse one of
these names, and none appear.

## Attention type per layer

- `num_hidden_layers = 52`; `LlamaModel.__init__` builds one `LlamaDecoderLayer` per index
  (`modeling_llama_4.51.1.py:476-478`), and each `LlamaDecoderLayer` constructs the same
  `LlamaAttention` (`modeling_llama_4.51.1.py:295`). There is **no** per-layer `layer_types`, no
  `sliding_window`, no `attention_type`, no `hybrid` flag in `config.json` (full key list checked:
  33 keys, none of these).
- Therefore **all 52 layers are full, causal, unmasked-history attention**. 52/52 = 100% full-attention.
- `is_causal = True` (`:225`); mask is a plain lower-triangular causal mask
  (`_prepare_4d_causal_attention_mask_with_cache_position`, `:704-727`). No window, no block mask.

## Pure softmax or not

Pure softmax, single path:

```
attn_weights = Q K^T * scaling          # :201
attn_weights += causal_mask             # :204
attn_weights = softmax(..., dim=-1, dtype=float32).to(query.dtype)   # :206
attn_output   = attn_weights @ V        # :208
```

(`eager_attention_forward`, `modeling_llama_4.51.1.py:188-211`.) No gating, no linear/SSM branch,
no logit soft-cap, no attention-sink logits.

## Sliding window and sinks

- **Sliding window: none.** No `sliding_window` config field and no windowing in the mask code
  (`:704-727`). Note the base→chat config diff does not add one either.
- **Attention sinks: none.** No learned sink parameter in the checkpoint or the module, and the
  softmax denominator covers only real keys (`:206`). gpt-oss-style sinks would need a parameter,
  which is absent.

## QK-norm

**Absent.** `q_proj`/`k_proj` go straight into RoPE, then attention
(`modeling_llama_4.51.1.py:252-257`). The only RMSNorms per layer are pre-attention and
post-attention (`:298-299`), applied to the residual stream, not to Q/K heads.

## RoPE variant and scaling

- `rope_scaling = null`; `rope_theta = rotary_base = 120,000,000`.
- `LlamaRotaryEmbedding.__init__` therefore selects `rope_type = "default"` and
  `attention_scaling = 1.0` (`modeling_llama_4.51.1.py:96-107`). This is **plain RoPE**, not YaRN,
  not NTK/dynamic, not linear, not Llama-3 longrope.
- RoPE is applied to **Q and K after projection**, V untouched (`:256-257`, `apply_rotary_pos_emb`
  at `:133-157`). Rotation pairs the two halves of the head dim (`rotate_half`, `:127-130`), i.e.
  the GPT-NeoX/"rotate half" convention.
- **Context it is trained for vs `max_position_embeddings`:** the chat model declares
  `max_position_embeddings = 524288` and `seq_length = 524288`, and the model card states
  "supports a context length of up to 524,288 tokens (512K)". The **base** checkpoint is different
  and is the informative one:

  | | Base | Chat (33B) |
  |---|---|---|
  | `max_position_embeddings` | **131072** | **524288** |
  | `rope_theta` | **10,000,000** | **120,000,000** |
  | `rope_scaling` | null | null |

  So the underlying base model was trained at **128K with θ=10M**; the instruct/chat release
  **changed θ to 120M and declares 512K**, i.e. long-context extension was done by retuning the
  base frequency (plus post-training), not by a `rope_scaling` recipe. There is no separate
  documented "trained vs max_position_embeddings" statement beyond that, and no rope-scaling
  factor to reproduce: any B2c run at ≥128K is already outside the base model's trained window and
  relies on the θ=120M chat tuning. **Practical note for B2c: 32K is safely in-distribution; 128K+
  should be treated as "claimed", not proven, until we run it.**

## MLA or sparse attention: both absent

Confirmed from the modeling code, not inferred:

- Four independent dense `nn.Linear` projections `q_proj`, `k_proj`, `v_proj`, `o_proj`
  (`modeling_llama_4.51.1.py:227-238`); no `kv_a_proj_with_mqa`/`kv_b_proj`, no compressed latent,
  no `q_lora_rank`/`kv_lora_rank` anywhere in config or checkpoint.
- One softmax over the full key axis (`:206`); no top-k, no block selection, no NSA/quest-style
  branch, no extra indexer module. There is no code path in the file that can skip a key.
- GQA overlap is `num_key_value_groups = 56/8 = 7` (`:222`), expanded by `repeat_kv`
  (`:176-185`), so 7 query heads share each KV head.

For the LSA study this means: **52 identical full-attention layers, 8 KV heads of dim 128.**

## KV cache layout

- Standard `transformers` cache: `DynamicCache` by default, one K and one V tensor per layer,
  updated at `modeling_llama_4.51.1.py:259-262`.
- **The cached K is post-RoPE** (`apply_rotary_pos_emb` on line 257 runs before
  `past_key_value.update` on line 262); cached V is the raw `v_proj` output. This is directly
  relevant to B1c/B2b: "keys as cached" = post-RoPE. B2b's pre- vs post-RoPE ablation needs a hook
  *before* line 257 or a re-derivation from pre-RoPE `k_proj`.
- Shape per layer: `[batch, 8, seq, 128]` for K and for V. No MLA latent, no shared KV, no
  quantized cache in the base model (`_supports_quantized_cache = True` is a capability flag,
  `:375`, not a config).
- Footprint at bf16: `2 tensors × 8 heads × 128 dim × 2 B = 4096 B/layer/token`; ×52 =
  **212,992 B ≈ 208 KiB per token**. That is **6.5 GiB at 32K, 26.1 GiB at 128K, 104.2 GiB at
  512K** (allocator overhead excluded). The 512K claim is therefore only reachable with paged /
  offloaded / quantized KV.
- `use_cache: false` in `config.json` and `generation_config.json` is only a default; the code
  supports `DynamicCache`/`StaticCache` (`:529-530`, `_supports_cache_class`).

## Attention scaling factor

`self.scaling = self.head_dim ** -0.5` (`modeling_llama_4.51.1.py:223`), applied as
`QK^T * scaling` (`:201`):

```
128 ** -0.5 = 0.08838834764831845   (= 1/sqrt(128))
```

Softmax is accumulated in fp32 in the eager path (`:206`). No `attn_logit_softcapping`, no
`query_pre_attn_scalar` override.

## Model size and quantized variants on the Hub

Parameter count from the index: `66237954048 / 2 = 33,118,977,024` (33.12 B), of which 3.07 B
(F8/int8 payloads) are the linear weights and 2.39 B BF16 are embed/lm_head/norms. bf16 on-disk
= **66,237,954,048 B (61.69 GiB)**.

### Official Nanbeige org

| Repo | Format | Files on disk | Notes |
|---|---|---|---|
| `Nanbeige/Nanbeige4.3-33B` | bf16 | **66,237,954,048 B** (66.24 GB / 61.69 GiB), 14 shards | primary; `use_cache:false` in config |
| `Nanbeige/Nanbeige4.3-33B-Base` | bf16 | 66,240,837,915 B (66.24 GB), 14 shards | base, 128K / θ=10M |
| `Nanbeige/Nanbeige4.3-33B-FP8` | FP8 **W8A8-FP8 block**, compressed-tensors `FP8_BLOCK`: weights 128×128 block, dynamic per-group-128 fp8 activations (`recipe.yaml`, `quantization_config`) | 35,516,330,752 B (35.52 GB / 33.08 GiB), 9 shards | `lm_head` ignored |
| `Nanbeige/Nanbeige4.3-33B-INT8` | INT8 **W8A16** pack-quantized, group 128, symmetric, minmax | 35,992,669,888 B (35.99 GB / 33.52 GiB), 9 shards | `lm_head` ignored |
| `Nanbeige/Nanbeige4.3-33B-INT4` | INT4 **pack-quantized**, group 32, asymmetric (`zp_dtype int8`), QAT+OPD+RL per card | 22,550,315,872 B (22.55 GB / 21.0 GiB), **315 shards** | ignores `lm_head` and `embed_tokens`; `FULL_MANIFEST.json` present |

All three quant repos keep the same `config.json` architecture fields (`model_type: llama`, 52/56/8,
head_dim 128, θ=120M, max_pos 524288, rope_scaling null) — **the attention design is unchanged by
quantization**; only Linear weights are quantized, KV cache is not (`kv_cache_scheme: null`).
Caveat for B2c: FP8 uses dynamic activation quant, INT8/INT4 are weight-only (W8A16/W4A16-style),
so K/V are computed from dequantized weights but stored at bf16 activation dtype.

Note the Hub's per-repo `safetensors.parameters` field reports raw payload counts
(`BF16 2,393,603,072` + `F8_E4M3/I32 30,725,373,952` = 33,118,977,024) — the `I32` entry for
INT8/INT4 is the packed-storage accounting, not a 30 B int32 tensor.

### Community GGUF (biggest, `bartowski/Nanbeige_Nanbeige4.3-33B-GGUF`)

26 GGUF files, repo total 533.78 GB; all sizes are file bytes:

| File | Bytes | GB |
|---|---|---|
| IQ2_XXS | 10,071,403,648 | 10.07 |
| IQ2_XS | 10,634,407,040 | 10.63 |
| IQ2_S | 11,150,560,384 | 11.15 |
| IQ2_M | 12,187,167,872 | 12.19 |
| Q2_K | 13,528,816,768 | 13.53 |
| IQ3_XXS | 14,489,959,552 | 14.49 |
| Q3_K_S | 15,211,261,056 | 15.21 |
| IQ3_XS | 15,212,665,984 | 15.21 |
| Q3_K_M | 15,919,746,176 | 15.92 |
| Q3_K_L | 17,270,484,096 | 17.27 |
| IQ3_M | 17,802,005,632 | 17.80 |
| IQ4_XS | 18,635,041,920 | 18.64 |
| Q4_0 | 19,348,057,216 | 19.35 |
| Q4_K_S | 19,740,519,552 | 19.74 |
| Q4_K_M | 20,788,825,216 | 20.79 |
| IQ4_NL | 20,871,343,232 | 20.87 |
| Q4_1 | 21,200,497,792 | 21.20 |
| Q4_K_L | 22,344,395,904 | 22.34 |
| Q5_K_S | 23,664,311,424 | 23.66 |
| Q5_K_M | 25,073,368,192 | 25.07 |
| Q6_K_S | 27,750,272,128 | 27.75 |
| Q6_K | 29,016,857,728 | 29.02 |
| Q6_K_L | 30,404,324,480 | 30.40 |
| Q8_0 | 35,195,128,960 | 35.20 |
| bf16 part 1/2 | 39,693,381,216 | 39.69 |
| bf16 part 2/2 | 26,550,082,976 | 26.55 |
| imatrix | 27,172,928 | 0.03 |

Other notable community variants (not exhaustive; HF search returns ~100 repos for
"Nanbeige4.3-33B"): NVFP4 (`webbrain-one/…-NVFP4` 22.09 GB, `primitive-ai/…-NVFP4` 22.09 GB,
`axteam/…-NVFP4`), MLX 4/8-bit (`Nurymanau/…-MLX-4bit` 18.65 GB, `fcmeyer/…-MLX-8bit` 35.21 GB),
AWQ-4bit (`axteam/…-AWQ-4bit` 22.57 GB), W4A16 AutoRound/GPTQ (~20.7–20.8 GB), EXL3
(`Luni/…-EXL3` 61.79 GB), MXFP8 (`mph/…-MXFP8` 36.49 GB), FP8-KVcal
(`stoyani/…-FP8-KVcal` 35.55 GB, KV-cache calibrated — potentially relevant to B3), ROCmFPX/FP4
(`Harsha001/…-ROCmFPX`, Q4_0 ROCMFP4 17.60/18.57 GB, Q6_K 27.34 GB, Q8_0 34.16 GB — AMD-oriented,
worth a look for the R9700), ONNX-WebGPU, and several uncensored/heretic fine-tunes.

**Recommendation for B2c:** use `Nanbeige/Nanbeige4.3-33B-FP8` (35.5 GB) or `INT8` (36.0 GB) on one
R9700 pair if it fits, else `INT4` (22.6 GB, but 315 shards and QAT+RL so its weights are *not* a
clean reference). Do **not** compare LSA recall across quantization levels without noting that
INT4 changed the model via QAT/RL.

## Commands actually run (representative)

```bash
# configs / metadata
curl -sL https://huggingface.co/Nanbeige/Nanbeige4.3-33B/resolve/main/config.json
curl -sL https://huggingface.co/Nanbeige/Nanbeige4.3-33B-Base/resolve/main/config.json
curl -sL https://huggingface.co/Nanbeige/Nanbeige4.3-33B/resolve/main/model.safetensors.index.json
timeout 40 curl -s "https://huggingface.co/api/models?search=Nanbeige4.3-33B&limit=100"   # repo census
# weight header read (no >1GB download): HTTP Range bytes=0-200000 per shard, parse safetensors header
# modeling code verification
curl -sL https://raw.githubusercontent.com/huggingface/transformers/v4.51.1/src/transformers/models/llama/modeling_llama.py
diff upstream_llama_v4.51.1.py modeling_llama_4.51.1.py   # -> identical
```

Nothing was downloaded over 1 GB. No GPU was touched. All temp files are in
`/ai/github/lsa-rocm/agents/scratch/` (`nb_*`).

## Failures / caveats, stated plainly

- The first `curl` of `Nanbeige/Nanbeige4.3-33B-FP8/resolve/main/config.json` and the INT4 one
  (done with a bare `-s` redirect, no `-L`) returned an HTML blob and failed to parse. Re-fetching
  with `-L` worked; the numbers above come from the successful fetches and the cached
  `nbq_*_config.json`. No data loss, but note that a redirect-unaware fetch of these repos fails
  silently.
- The Hub `usedStorage` for the INT4 repo (22,571,741,573 B) differs from the index
  `total_size` (22,550,315,872 B) by ~21 MB (manifest + tokenizer). Both are quoted above where
  relevant; the index number is the weight payload.
- The exact training context of the *chat* model is not documented beyond the 512K claim and the
  base→chat θ change; do not treat "trained at 512K" as established.
- I did not verify behaviour at load time (no model load — would need GPU/≥22 GB and this task is
  read-only). A later load should assert `config._attn_implementation` and that the cache really is
  `DynamicCache`, since `use_cache:false` in the config means callers must pass `use_cache=True`.

## Answers to the B0b checklist

| Question | Answer |
|---|---|
| Attention type per layer | 52/52 full attention, identical, causal, dense |
| Pure softmax? | Yes |
| Sliding window? | None |
| Sinks? | None |
| QK-norm? | None |
| RoPE variant | Default RoPE, θ=120M, `rope_scaling=null`; base model θ=10M/128K, chat θ=120M/512K |
| Context trained vs max_pos | base 128K; chat declares 524,288 via θ=120M, not via rope scaling. 512K is advertised, not independently measured |
| MLA? | No (4 dense projections, no latent KV; `:227-238`) |
| Sparse attention? | No (full softmax over all keys; `:206`) |
| KV cache layout | per layer `[B,8,S,128]` K (post-RoPE) and V; 4096 B/layer/token; 208 KiB/token total |
| Attention scaling | `1/sqrt(128) = 0.08838834764831845` |
| Quant variants | Official: bf16 66.24 GB, FP8 35.52 GB, INT8 35.99 GB, INT4 22.55 GB; plus GGUF IQ2–Q8/bf16 (10.07–35.20 GB) and NVFP4/MLX/AWQ/W4A16/EXL3/MXFP8/ROCmFPX community builds |
