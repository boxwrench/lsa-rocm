# B0a — granite-switch-4.1-8b attention design (DRAFT, in progress)

Status: draft written early; being refined. All claims from the cached files in
`agents/scratch/B0a/` (model id `ibm-granite/granite-switch-4.1-8b`, revision
`fb4aa7877e13e5fe09ca328713dc1db72a3f93ce`, `api.json`; granite-switch package
0.1.1, git sha `1cc5225b6ddf512da00eb63f0c9b15705ff17e67`, `gs_commit.json`,
`gs_pyproject.toml:7`). Nothing was downloaded and no GPU was used.

## Headline answers

| Question | Answer |
|---|---|
| Attention type per layer | 40 decoder layers, **all `full_attention`**; plus 2 switch-attention cache slots (also full causal softmax). |
| Softmax | Yes, ordinary softmax (fp32 upcast in the eager path). No linear/SSM attention. |
| Sliding window | **None** in the checkpoint. |
| Attention sinks | **None.** |
| QK-norm | **Off** (`qk_norm` absent → False). |
| RoPE | `default` (unscaled) RoPE, theta 1e7, dim 128, applied to Q and K **before** caching. |
| Does "switch"/LoRA change attention? | Yes for the decoder: per-token LoRA deltas are added to Q/K/V (fused `qkv_proj`) and O and the MLP. No extra mask/bias; control tokens never reach attention because their ids are rewritten before embedding. |
| Adapter "hiding" and K/V | The older "KV-hidden control token" comment is **stale**: there is no hiding mask. Token exchange rewrites the control-token id to a substitute real token id before embedding. K/V therefore contain the substitute token's K/V, computed with the active adapter's LoRA delta; K/V are frozen when written to cache. |
| KV cache layout | HF `DynamicCache`, one `DynamicLayer` per `layer_types` entry (42 slots); tensor shape `[batch, num_kv_heads=8, seq, head_dim=128]`, post-RoPE, per-layer. Slot 0/1 are the switch's counting/memory heads; slots 2..41 are decoder layers 0..39. |

## Layer count: 42 config entries = 40 decoder + 2 switch slots

- `config.json:116` `"num_hidden_layers": 42`; `config.json:46-89` contains 42
  strings, all `"full_attention"` (verified by parsing: `len=42`, `set={'full_attention'}`).
- The model class subtracts the switch's cache slots before building decoder
  layers: `layer_offset = self.switch.num_cache_layers` and
  `num_decoder_layers = config.num_hidden_layers - layer_offset`
  (`modeling_granite_switch.py:439-441`), with global layer index
  `local_idx + layer_offset` (`:455-457`).
- `MultiSwitch.num_cache_layers` returns 2 (`switch_multi.py:332-334`), and
  `SWITCH_CACHE_LAYERS = 2` (`gs_config.py:9-15`).
- So there are 40 real transformer blocks, numbered 2..41 in the cache, and
  slots 0/1 belong to the switch. This matches BUILD.md ("Hidden layers: 40",
  `BUILD.md:9`) and the checkpoint index (`index.json`: 40 distinct
  `model.layers.N`, 0..39). The plan's "42 layers" is the config count, not the
  decoder depth.
- `config.json:207` (in the filtered dump) / actual line 107:
  `"max_position_embeddings": 131072`.

## Per-layer attention implementation

All 40 decoder layers instantiate the same class:
`GraniteSwitchAttentionDecoderLayer` (`modeling_granite_switch.py:50-105`),
whose `self.self_attn` is `GraniteLoRAEmbeddedAttention`
(`modeling_granite_switch.py:72-73`; class at `lora.py:351`). There is no
per-layer type switch: `layer_type = "attention"` is hard-coded
(`modeling_granite_switch.py:66`) and `layer_cls` is chosen only by
`dual_stream`, not by layer index (`:446-453`). `dual_stream` is false in this
checkpoint (`config.json:38`), so all 40 use the plain single-stream class.

Per layer:
- 32 Q heads, 8 KV heads, GQA groups 4 (`config.json:114,117`;
  `lora.py:365-372`).
- head_dim = `projection_head_dim` = 128 (`config.json:122`;
  `lora.py:366-370`). `hidden_size/num_heads` = 4096/32 = 128, so they agree.
  No `head_dim` key exists in config.json; RoPE falls back to
  `hidden_size // num_attention_heads` (`modeling_granitemoehybrid.py:788`).
- scaling is `config.attention_multiplier` = 0.0078125 = 1/128, **not**
  1/sqrt(128) (`config.json:33`; `lora.py:375`; passed at `lora.py:535`).
- QKV is one fused `qkv_proj` of width 4096+1024+1024=6144, split into
  Q/K/V (`lora.py:394-404`, `:467-474`), wrapped as
  `MergedSwitchedLoRALinear` because `"qkv_proj"` is in
  `lora_target_modules` (`config.json:91-96`).
- O projection is `SwitchedLoRALinear` over 4096→4096 (`lora.py:413-425`).
- `attention_bias: false` (`config.json:31`), dropout 0 (`config.json:32`).

## Softmax, no sliding window, no sinks

- Dense eager reference: additive mask → `softmax(dim=-1, dtype=float32)` →
  cast back (`modeling_granite.py:102-108`; same function in
  `modeling_granitemoehybrid.py:110-116`). The switch attention also ends in
  softmax (`ALL_ATTENTION_FUNCTIONS["sdpa"]`, `switch_multi.py:438-450`).
- Decoder layers dispatch through the configured backend
  (`lora.py:520-537`), defaulting to `ALL_ATTENTION_FUNCTIONS[...]`; every
  backend is standard causal softmax attention.
- Sliding window: `sliding_window=getattr(self.config, "sliding_window", None)`
  (`lora.py:536`) and no `sliding_window` key exists in config.json, so this is
  `None`. `layer_types` has no `sliding_attention` (`config.json:46-89`).
  The mask builder has no sliding function applied
  (`modeling_granite_switch.py:597-620`, `create_causal_mask`).
- Attention sinks: no sink parameters in the weight map (`index.json` has no
  key containing `sink`; keys are `qkv_proj`/`o_proj`/`lora_*`/norm/MLP/switch
  only), no `sink` attribute anywhere in the modeling code (`grep -rn -i sink`
  over all cached `.py`/`.md` returns nothing). No learned per-head bias is
  added to logits beyond the causal mask.

## QK-norm: off

`self.qk_norm = getattr(config, "qk_norm", False)` (`lora.py:378-383`), and
`qk_norm` is absent from config.json (full key list confirmed). The weight map
has zero keys matching `q_norm`/`k_norm` (`index.json`). Same guard exists on
the vLLM side (`v_decoder.py:128-133`).

## RoPE variant and scaling

- `"rope_parameters": {"rope_theta": 10000000, "rope_type": "default"}`
  (`config.json:125-128`). No `rope_scaling` field anywhere in config.json;
  the base model config explicitly has `"rope_scaling": null`
  (`granite41_config.json`).
- `rope_type == "default"` selects `compute_default_rope_parameters`, so
  `attention_scaling = 1.0` and `cos/sin` are unscaled
  (`modeling_granitemoehybrid.py:764-771`, `:787-793`, `:805-806`).
- inv_freq uses `dim = head_dim or hidden_size//heads` = 128
  (`modeling_granitemoehybrid.py:788,792`), i.e. 64 frequency pairs.
- RoPE is applied to Q and K (**not V**) and to Q/K only, via
  `apply_rotary_pos_emb` on `[B, H, S, D]` (`lora.py:496-505`).
- Order per layer: project → (optional qk-norm, off) → RoPE → transpose into
  cache layout → `past_key_values.update(...)` → attention (`lora.py:467-518`).
  **Cached keys are post-RoPE.** This answers B1c/B2b's "which side of RoPE":
  the HF cache stores rotated keys.

## Does the "switch" mechanism or its LoRA adapters change attention computation?

### The switch itself

The switch is not a router MLP; it is two tiny single-head softmax attention
modules (`switch_multi.py:1-58`; classes `_CountingAttention` at `:81`,
`_MemoryAttention` at `:100`):
- counting head: head_dim 32 (`config.json:111`; `switch_multi.py:231-235`),
  fp32, one query head, one KV head, integral `1/(1+n)` readout;
- memory head: head_dim = `projection_head_dim` = 128 because memory_dim=64 ≤
  128 (`switch_multi.py:240-258`), fp32, Kerdock codeword retrieval.

Both run with SDPA in fp32, plain causal masks, `sliding_window=None`
(`switch_multi.py:487-512`, `:567-580`), and **write their own K/V into the
same HF cache** at slots `layer_idx` and `layer_idx+1` (`switch_multi.py:425-432`,
`:549-556`). Their output is an integer `adapter_indices` per token:
`round` + `clamp(0, num_adapters)` (`switch_multi.py:582-593`).

So the switch adds **two full-attention layers' worth of KV state** but does not
add any mask, sink, or bias to the 40 decoder attention calls. It influences
them only by choosing which adapter delta each token gets.

### The LoRA adapters

`adapter_indices` is passed into every decoder layer
(`modeling_granite_switch.py:705-714`) and used by:
- `qkv_proj` (`lora.py:467-474`): LoRA delta is added per slice to **Q, K and
  V** before reshape/RoPE/cache;
- `o_proj` (`lora.py:539-544`);
- the shared MLP input/output projections
  (`modeling_granite_switch.py:85-86`, `lora.py:662+`).

`BUILD.md:19-20` confirms both adapters target `k_proj`, `q_proj`, `v_proj`,
`o_proj` (decision also gate/up/down). So the adapter selected for a token
changes that token's K and V, and hence what later queries see. It does **not**
change the attention function itself (still softmax, same scaling, same mask).

LoRA math: output = `base_layer(x) + x @ Aᵀ @ Bᵀ`, with B pre-scaled by
alpha/rank at load time (`lora.py:23-90`, `lora.py:200-209`, `:330-347`).
Ranks 16 and 64, max rank 64 (`config.json:6-9,106`).

### The "hiding" question — control tokens and what K/V see

There is a documentation/code mismatch worth flagging:

- Two comments claim control tokens are "KV-hidden (masked from attention)":
  `modeling_granite_switch.py:388-389` and `v_granite_switch_model.py:143-144`.
- `gs_SUPPORTED_MODELS.md:8-12` advertises "KV Cache Hiding: Yes" for Granite
  4.0/4.1 Dense.
- But the implementation is **token exchange, not a hiding mask**. At the end
  of the switch forward each control-token id is rewritten to a substitute real
  token id (`switch_multi.py:595-598`; `token_exchange.py:143-156`), and the
  decoder embeds the rewritten ids (`modeling_granite_switch.py:664-668`).
  The switch's own explainer states this explicitly: "MultiSwitch does not use
  the group-based KV-hiding path … the control token is neutralised by having
  its id replaced before embedding" (`gs_MULTISWITCH_EXPLAINED.md:200-202`),
  and a grep over the installed wheel finds only two incidental comments
  mentioning "hiding", no mechanism.
- Consequences for K/V at inference:
  1. No control-token K/V is ever produced. Position `p` of the control token
     contributes the substitute token's embedding (102352 `<|guardian-core|>`
     → 27; 100353 `<|decision|>` → 14711; `config.json:10-17`) to the residual
     stream, so its K/V is the K/V of a *normal* token with that substitute id.
  2. Those K/V are computed through the **active adapter's** `qkv_proj` LoRA
     delta for the tokens around/after the control token. The substitition does
     not "hide" the adapter delta from K/V.
  3. Once written, cached K/V are immutable, so a token prefilled under adapter
     A is attended to with A's K/V even if later tokens route to adapter B.
- The README's claim "KV cache normalization — each adapter sees only the base
  model's KV cache, never another adapter's internal state" (`README.md:385`)
  is therefore only true in the weak sense that control tokens are removed
  before embedding. The code does not strip adapter deltas from cached K/V.
  This is an open question for the metric design (B1c/B2): the "keys the model
  actually attends to" include adapter-conditioned K/V.

If the aLoRA invocation region matters: the aLoRA "hiding" of earlier
invocation text discussed in `gs_MULTISWITCH_EXPLAINED.md:508` refers to the
*preview* checkpoints built with a hiding matrix; this composed 0.1.1
checkpoint uses the LUT/token-exchange path, so that text is not hidden here.

## KV cache layout

- Cache object: `DynamicCache(config=self.config)` created whenever
  `use_cache` (`modeling_granite_switch.py:573-575`); `use_cache: true`
  (`config.json:139`).
- `DynamicCache.__init__` reads `layer_types` and builds one layer per entry
  via `DYNAMIC_LAYER_TYPE_MAPPING` (`cache_utils.py:1806-1812`,
  `get_layer_types_and_kwargs` `:1720-1753`). All 42 entries are
  `full_attention`, so every slot is a `DynamicLayer`
  (`cache_utils.py:1237-1238`).
- `DynamicLayer` stores `[batch_size, num_heads, seq_len, head_dim]`
  (`cache_utils.py:114-148`); update is `torch.cat(..., dim=-2)` (`:146-147`),
  i.e. append along the sequence axis. For decoder layers `num_heads` is the
  **KV** head count = 8, since the cache is updated with `key_states` of shape
  `[B, num_key_value_heads, S, 128]` (`lora.py:508-518`). GQA expansion to 32
  happens only inside the attention kernel (`modeling_granite.py:99-100`).
- Index mapping: switch writes slots 0 (counting) and 1 (memory)
  (`switch_multi.py:262-272`, `:425-432`, `:549-556`); decoder layers are
  constructed with `global_layer_idx = local_idx + 2`
  (`modeling_granite_switch.py:439-457`), so layer 0 of the checkpoint writes
  cache slot 2, and layer 39 writes slot 41.
- Values are post-RoPE and post-LoRA (see above), bf16 in the checkpoint
  (`config.json:37` `"dtype": "bfloat16"`); the switch's own K/V are forced
  fp32 before being written (`switch_multi.py:394-400`, `:530-547`), so those
  two slots are fp32 while decoder slots are model dtype. (Confirmed in code;
  not measured.)
- `DynamicCache` has no maximum length (`DynamicLayer.get_max_length` returns
  -1, `cache_utils.py:162-164`), so it grows to the full context.

## Open questions / caveats

- The exact final `layer_types` length equals 42 including switch slots, which
  is why the config advertises 42 "layers". Anyone dumping "full-attention
  layers" for B1c should iterate `model.model.layers` (40 blocks) and remember
  cache slots are offset by +2.
- "KV Cache Hiding: Yes" in the docs is not implemented as a mask in 0.1.1.
  Treat K/V as containing the active adapter delta and the substituted token,
  not as base-only K/V.
- No runtime verification was possible: the checkpoint is not on disk
  (`/ai/huggingface/hub` has no `models--ibm-granite--granite-switch-4.1-8b`)
  and downloading was out of scope. All statements are static code/config
  reading.
