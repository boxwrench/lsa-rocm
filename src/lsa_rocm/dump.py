"""Dump post-RoPE Q/K/V of the full-attention layers of a local HF model.

Usage:
    python -m lsa_rocm.dump --model PATH --prompt-file FILE --max-tokens N \
        --n-queries Q --out DIR [--device cpu|cuda] [--attn sdpa|eager] \
        [--prefill-chunk N]

The tool runs a forward pass over the prompt, captures, per full-attention
layer, keys/values as cached (post-RoPE) [Hkv, T, D], keys before RoPE (when the
model's rotary helper can be intercepted), queries of the last Q positions after
RoPE [Hq, Q, D], their positions, and the attention module's own output for the
last query position [Hq, D]. It then recomputes the attention output of the last
query from the dumped tensors and compares it against the module output; a
cosine below 0.999 aborts the run.

Long context: --attn sdpa (the default) makes the forward pass use PyTorch SDPA,
which dispatches to the flash kernel on ROCm and never materialises the full
attention matrix. --attn eager keeps the original full-attention-matrix path for
small models and regression tests.

Chunked prefill: --prefill-chunk N (tokens per chunk, 0 = one pass over the
whole prompt, the default) runs the prompt through the model in chunks of N
tokens with use_cache=True and a cache object. Each chunk's post-RoPE K/V are
appended into per-layer CPU buffers, and only the last Q query positions are
kept. This is what makes prompts beyond one-pass GPU memory (128K and up) dump.
Qwen3.5-family linear-attention layers carry their recurrent state in the same
cache, so they keep working across chunk boundaries. logits_to_keep=1 keeps the
per-chunk logits tiny. A chunk is exact when the chunked K/V agree with a
single-pass dump: the acceptance check compares their cosines per layer.

KV offload: --kv-offload keeps the model's own K/V cache for full-attention
layers in CPU RAM. A forward pre-hook copies the current layer's cache slice to
the GPU just before its update, and the post-hook copies it back afterwards, so
only the layer being computed holds its cache on the device. This is how a 1M
token Ornith dump (about 32 GiB of cache) fits on one 32 GiB GPU. Linear-
attention layers keep their recurrent state resident (it is tiny and does not
grow with context). The pre/post hooks move cache tensors rather than replacing
the Cache type, so transformers' ordinary DynamicCache is used and the model's
own code path is untouched. Our capture buffers always stay on CPU.

YaRN: --rope-yarn FACTOR rewrites the loaded config before the model is built:
config.text_config.rope_parameters gets rope_type "yarn", factor, and
original_max_position_embeddings equal to the native max context, and
max_position_embeddings is set to factor times the native value. The applied
rope settings are read back from the built model and recorded in meta.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch


def _git_commit() -> str | None:
    try:
        import subprocess

        root = Path(__file__).resolve().parents[2]
        return (
            subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _text_config(config):
    return getattr(config, "text_config", config)


def _full_attention_indices(config) -> tuple[list[int], list[str]]:
    text_cfg = _text_config(config)
    layer_types = list(getattr(text_cfg, "layer_types", []) or [])
    n_layers = int(getattr(text_cfg, "num_hidden_layers", len(layer_types)))
    if not layer_types:
        return list(range(n_layers)), ["full_attention"] * n_layers
    if len(layer_types) < n_layers:
        layer_types = layer_types + ["full_attention"] * (n_layers - len(layer_types))
    return [i for i, t in enumerate(layer_types) if t == "full_attention"], layer_types[:n_layers]


def _find_attention_modules(model, full_indices: list[int]):
    found = {}
    for name, module in model.named_modules():
        idx = getattr(module, "layer_idx", None)
        if not isinstance(idx, int) or idx not in full_indices:
            continue
        if not all(hasattr(module, a) for a in ("q_proj", "k_proj", "v_proj", "o_proj")):
            continue
        found[idx] = (name, module)
    return found


def _apply_rope_yarn(config, factor: float) -> dict:
    """Rewrite the loaded config so the model builds with YaRN rope scaling.

    transformers 5.x keeps rope settings in ``config.text_config.rope_parameters``
    and the rope initialisation function reads ``rope_theta``,
    ``partial_rotary_factor``, ``factor`` and
    ``original_max_position_embeddings`` from there. The native context length is
    the value already in the config (262144 for Ornith). The native value is
    recorded before max_position_embeddings is extended, so re-running on an
    already-extended config does not multiply twice.
    """
    text_cfg = _text_config(config)
    native_max = int(getattr(text_cfg, "max_position_embeddings"))
    params = dict(getattr(text_cfg, "rope_parameters", {}) or {})
    if "rope_type" not in params:
        params["rope_type"] = "default"
    if "rope_theta" not in params:
        raise SystemExit("cannot apply YaRN: the model config has no rope_theta")
    if "partial_rotary_factor" not in params:
        prf = getattr(text_cfg, "partial_rotary_factor", None)
        if prf is not None:
            params["partial_rotary_factor"] = float(prf)
    requested = {
        "rope_type": "yarn",
        "factor": float(factor),
        "original_max_position_embeddings": native_max,
    }
    # A previous run may have extended max_position_embeddings; recover the
    # native value from an existing YaRN original_max_position_embeddings.
    if str(params.get("rope_type", "")).lower() == "yarn" and "original_max_position_embeddings" in params:
        native_max = int(params["original_max_position_embeddings"])
        requested["original_max_position_embeddings"] = native_max
    params.update(requested)
    text_cfg.rope_parameters = params
    text_cfg.max_position_embeddings = int(native_max * float(factor))
    return {
        "native_max_position_embeddings": native_max,
        "requested_rope_parameters": dict(params),
        "requested_max_position_embeddings": int(text_cfg.max_position_embeddings),
    }


def _rope_readback(model, config) -> dict:
    """Read the rope settings back from the built model, not the request."""
    text_cfg = _text_config(model.config)
    params = dict(getattr(text_cfg, "rope_parameters", {}) or {})
    scaling = None
    rope_module = None
    inv_freq_head = None
    for name, module in model.named_modules():
        if hasattr(module, "attention_scaling") and hasattr(module, "inv_freq"):
            scaling = float(module.attention_scaling)
            rope_module = name
            inv_freq_head = [round(float(v), 8) for v in module.inv_freq[:4].float().cpu().tolist()]
            break
    out = dict(params)
    out["rope_type"] = params.get("rope_type")
    out["factor"] = float(params.get("factor", 1.0))
    out["original_max_position_embeddings"] = params.get("original_max_position_embeddings")
    out["max_position_embeddings"] = int(getattr(text_cfg, "max_position_embeddings"))
    out["attention_scaling_from_model"] = scaling
    out["rope_module"] = rope_module
    out["inv_freq_head"] = inv_freq_head
    return out


def _make_offload_cache(config, impl: str):
    """Build a cache that keeps full-attention K/V on the CPU between layers.

    ``impl='transformers'`` uses transformers' own DynamicCache offloading
    (prefetch stream + offload after each layer's update).
    ``impl='custom'`` uses the small subclass below, which moves only the
    DynamicLayer K/V tensors and leaves the linear-attention recurrent state
    resident. Both are constructed the same way so the caller can fall back.
    """
    from transformers.cache_utils import DynamicCache

    if impl == "transformers":
        return DynamicCache(config=config, offloading=True)
    if impl == "custom":
        return _OffloadingDynamicCache(config=config)
    raise ValueError(f"unknown kv offload implementation {impl!r}")


class _OffloadingDynamicCache:
    """DynamicCache that keeps full-attention K/V in CPU RAM between updates.

    transformers' DynamicCache grows K/V on the device of the incoming states;
    the layer only needs them on the GPU while its own attention runs. This
    wrapper copies the layer's cache to the computing device just before
    ``update`` and back to CPU just after, keeping at most one layer's cache on
    the GPU. Linear-attention layers use ``update_conv_state`` /
    ``update_recurrent_state`` (small, context-independent) and are not touched.
    """

    def __init__(self, config):
        from transformers.cache_utils import DynamicCache

        self._inner = DynamicCache(config=config)
        self.offloading = True
        self.moved_layers = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        from transformers.cache_utils import DynamicLayer

        layer = self._inner.layers[layer_idx]
        if isinstance(layer, DynamicLayer) and layer.is_initialized:
            if layer.keys.device != key_states.device:
                layer.keys = layer.keys.to(key_states.device, non_blocking=True)
                layer.values = layer.values.to(key_states.device, non_blocking=True)
                self.moved_layers += 1
        keys, values = self._inner.layers[layer_idx].update(key_states, value_states, *args, **kwargs)
        if isinstance(layer, DynamicLayer):
            layer.keys = layer.keys.to("cpu", non_blocking=True)
            layer.values = layer.values.to("cpu", non_blocking=True)
        return keys, values


def _rope_info(text_cfg) -> dict:
    params = dict(getattr(text_cfg, "rope_parameters", {}) or {})
    head_dim = int(getattr(text_cfg, "head_dim", 0) or 0)
    prf = float(params.get("partial_rotary_factor", 1.0))
    return {
        "rope_parameters": params,
        "rope_theta": params.get("rope_theta"),
        "rope_type": params.get("rope_type"),
        "partial_rotary_factor": prf,
        "mrope_section": params.get("mrope_section"),
        "mrope_interleaved": params.get("mrope_interleaved"),
        "rotary_dim_from_config": int(head_dim * prf) if head_dim else None,
        "head_dim": head_dim,
        "note": "Qwen3.5 rotates only the first rotary_dim dims of each head; dims beyond are passed through.",
    }


def run(args) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_text = Path(args.prompt_file).read_text()
    prompt_bytes = prompt_text.encode()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_tokens)
    input_ids = enc["input_ids"].to(device)
    # The prompt is truncated to the first --max-tokens tokens and never padded,
    # so no 2D mask is needed; passing None lets sdpa use is_causal directly.
    attention_mask = None
    T = int(input_ids.shape[1])
    if T <= 0:
        raise SystemExit("empty prompt after tokenization")
    nq = min(int(args.n_queries), T)
    if nq <= 0:
        raise SystemExit("--n-queries must be positive")

    rope_yarn = float(getattr(args, "rope_yarn", 0.0) or 0.0)
    load_config = None
    yarn_request = None
    if rope_yarn > 0:
        from transformers import AutoConfig

        load_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        yarn_request = _apply_rope_yarn(load_config, rope_yarn)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation=args.attn,
        config=load_config,
    ).to(device)
    model.eval()
    # What transformers actually put on the config (never assume the request was honoured).
    actual_impl = str(getattr(model.config, "_attn_implementation", args.attn))
    config = model.config
    text_cfg = _text_config(config)
    rope_readback = _rope_readback(model, config)
    if rope_yarn > 0:
        got = str(rope_readback.get("rope_type", "")).lower()
        got_factor = float(rope_readback.get("factor", 0) or 0)
        if "yarn" not in got or abs(got_factor - rope_yarn) > 1e-6:
            raise RuntimeError(
                f"YaRN was requested (factor {rope_yarn}) but the built model reads back "
                f"rope_type={got!r}, factor={got_factor}; refusing to dump a mis-scaled model"
            )
        if int(rope_readback.get("max_position_embeddings", 0)) != int(
            yarn_request["requested_max_position_embeddings"]
        ):
            raise RuntimeError(
                f"YaRN max_position_embeddings read back as "
                f"{rope_readback.get('max_position_embeddings')}, expected "
                f"{yarn_request['requested_max_position_embeddings']}"
            )
        print(
            f"YaRN: factor={rope_yarn} native_max={yarn_request['native_max_position_embeddings']} "
            f"max_position_embeddings={rope_readback['max_position_embeddings']} "
            f"attention_scaling={rope_readback.get('attention_scaling_from_model')}",
            flush=True,
        )
    full_indices, layer_types = _full_attention_indices(config)
    attn_modules = _find_attention_modules(model, full_indices)
    missing = sorted(set(full_indices) - set(attn_modules))
    if missing:
        raise RuntimeError(f"could not locate attention modules for layers {missing}")

    positions = torch.arange(T, dtype=torch.long, device=device)

    chunk_size = int(getattr(args, "prefill_chunk", 0) or 0)
    chunked = chunk_size > 0
    if chunked and chunk_size < 1:
        raise SystemExit("--prefill-chunk must be positive")

    kv_offload_impl = getattr(args, "kv_offload_impl", "transformers")
    kv_offload = bool(getattr(args, "kv_offload", False))
    if kv_offload and not chunked:
        raise SystemExit("--kv-offload requires --prefill-chunk (it keeps the chunked cache in CPU RAM)")
    if kv_offload and device.type != "cuda":
        raise SystemExit("--kv-offload requires --device cuda (the cache moves between GPU and CPU)")

    # Per-layer capture. The attention interface receives post-RoPE q/k/v and
    # returns the module's own attention output; pre-RoPE k is captured by
    # wrapping the rotary helper used inside the attention forward.
    #
    # In chunked mode the attention interface sees the whole cache up to and
    # including the current chunk, so the current chunk's K/V are its trailing
    # chunk_len positions. Those are appended to per-layer CPU buffers, keeping
    # only the last --n-queries queries, so the capture footprint does not grow
    # with the number of chunks beyond the dump itself.
    state = {
        "pending_pre_rope": {},
        "captured": {},
        "current_layer": None,
        "own_out_last": {},
        "rotary_dim": {},
        "chunked": chunked,
        "chunk_len": 0,
        "chunk_index": 0,
        "num_chunks": 0,
        "buffers": {idx: {"k": [], "v": [], "q_tail": None} for idx in full_indices},
        "layer_meta": {},
    }

    # Wrap the attention implementation's own interface function. It receives
    # post-RoPE q/k/v and the module's own scaling and returns the module's
    # attention output; pre-RoPE k is captured by wrapping the rotary helper
    # used inside the attention forward.
    import importlib

    if args.attn == "eager":
        # transformers falls back to the module's own `eager_attention_forward`
        # default rather than a registry entry, so recover that function.
        orig_interface = None
        for _idx, (_name, module) in attn_modules.items():
            cls_mod = importlib.import_module(type(module).__module__)
            orig_interface = getattr(cls_mod, "eager_attention_forward", None)
            if orig_interface is not None:
                break
        if orig_interface is None:
            raise RuntimeError("could not locate the model's eager_attention_forward")
    else:
        if args.attn not in ALL_ATTENTION_FUNCTIONS:
            raise RuntimeError(
                f"attention implementation {args.attn!r} is not registered in ALL_ATTENTION_FUNCTIONS"
            )
        orig_interface = ALL_ATTENTION_FUNCTIONS[args.attn]

    def wrapped_interface(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        out, weights = orig_interface(
            module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
        )
        idx = getattr(module, "layer_idx", None)
        if idx in full_indices:
            if scaling is None:
                scaling = float(getattr(module, "scaling", 1.0))
            # In chunked mode `out`, `query` and the trailing part of `key` all
            # refer to the current chunk only; the cache prefix is in `key`.
            q_len = int(query.shape[2])
            if not state["chunked"] and q_len != T:
                raise RuntimeError(f"layer {idx}: query length {q_len}, expected {T}")
            # out: [B, q_len, Hq, D] for both eager and sdpa
            o = out.detach()
            if not (o.dim() in (3, 4) and o.shape[1] == q_len):
                raise RuntimeError(f"unexpected attention output shape {tuple(o.shape)}")
            module_out_last = o[0, q_len - 1].float().to("cpu").contiguous()
            meta = {
                "scaling": float(scaling),
                "num_key_value_groups": int(module.num_key_value_groups),
                "num_attention_heads": int(getattr(module, "config").num_attention_heads),
                "num_key_value_heads": int(getattr(module, "config").num_key_value_heads),
            }
            pre = state["pending_pre_rope"].pop(idx, None)
            if not state["chunked"]:
                q = query[0].detach().to("cpu")
                k = key[0].detach().to("cpu")
                v = value[0].detach().to("cpu")
                if k.shape[-2] != T:
                    raise RuntimeError(f"layer {idx}: captured {k.shape[-2]} keys, expected {T}")
                record = {
                    "queries_post_rope": q[:, T - nq :].contiguous().clone(),
                    "keys_post_rope": k.contiguous().clone(),
                    "values": v.contiguous().clone(),
                    "positions": positions[T - nq :].to("cpu").clone(),
                    **meta,
                    "selfcheck_module_out": module_out_last,
                }
                if pre is not None:
                    record["keys_pre_rope"] = pre
                state["own_out_last"][idx] = module_out_last
                state["captured"][idx] = record
                state["layer_meta"][idx] = meta
            else:
                # Append only this chunk's post-RoPE K/V (trailing q_len rows of
                # the cache) and this chunk's queries to small CPU buffers.
                cl = q_len
                if key.shape[-2] < cl:
                    raise RuntimeError(
                        f"layer {idx}: cache has {key.shape[-2]} keys but chunk has {cl} queries"
                    )
                k_new = key[0, :, -cl:, :].detach().to("cpu")
                v_new = value[0, :, -cl:, :].detach().to("cpu")
                q_new = query[0].detach().to("cpu")
                buf = state["buffers"][idx]
                buf["k"].append(k_new.contiguous().clone())
                buf["v"].append(v_new.contiguous().clone())
                tail = q_new if buf["q_tail"] is None else torch.cat([buf["q_tail"], q_new], dim=1)
                buf["q_tail"] = tail[:, -nq:, :].contiguous().clone()
                if pre is not None:
                    buf.setdefault("pre", []).append(pre.contiguous().clone())
                state["own_out_last"][idx] = module_out_last
                state["layer_meta"][idx] = meta
        return out, weights

    had_key = args.attn in ALL_ATTENTION_FUNCTIONS
    saved_interface = ALL_ATTENTION_FUNCTIONS[args.attn] if had_key else None
    ALL_ATTENTION_FUNCTIONS[args.attn] = wrapped_interface

    # Patch each model module that defines apply_rotary_pos_emb.
    patched_modules = []
    rotary_patch_state = {}
    for idx, (_name, module) in attn_modules.items():
        cls_mod = importlib.import_module(type(module).__module__)
        if cls_mod in patched_modules:
            continue
        if not hasattr(cls_mod, "apply_rotary_pos_emb"):
            continue

        orig_rope = cls_mod.apply_rotary_pos_emb

        def make_wrapper(orig, mod):
            def wrapper(q, k, cos, sin, *a, **kw):
                cur = state["current_layer"]
                if cur in full_indices:
                    nd = cos.shape[-1]
                    state["rotary_dim"][cur] = int(nd)
                    # Pre-RoPE keys are only dumped in single-pass mode; in
                    # chunked mode they are not needed and would grow buffers.
                    if not state["chunked"]:
                        kk = k[0].detach().to("cpu")
                        state["pending_pre_rope"][cur] = kk.contiguous().clone()
                return orig(q, k, cos, sin, *a, **kw)

            return wrapper

        cls_mod.apply_rotary_pos_emb = make_wrapper(orig_rope, cls_mod)
        patched_modules.append(cls_mod)
        rotary_patch_state[str(cls_mod.__name__)] = orig_rope

    hooks = []
    for idx, (_name, module) in attn_modules.items():
        def pre_hook(module, args, kwargs, _idx=idx):
            state["current_layer"] = _idx

        def post_hook(module, args, kwargs, output, _idx=idx):
            state["current_layer"] = None

        hooks.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        hooks.append(module.register_forward_hook(post_hook, with_kwargs=True))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_forward = time.time()
    try:
        with torch.no_grad():
            if not chunked:
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=positions.unsqueeze(0),
                    use_cache=False,
                    # Only the attention tensors are captured, so skip the full-sequence logits
                    # (T x vocab is about 16 GB at T=32768 with a 248k vocabulary).
                    logits_to_keep=1,
                )
            else:
                # Chunked prefill: feed the prompt N tokens at a time through a
                # persistent cache. Linear-attention layers keep their conv and
                # recurrent state in that cache; full-attention layers keep K/V.
                from transformers.cache_utils import DynamicCache

                if kv_offload:
                    past = _make_offload_cache(config, kv_offload_impl)
                    print(
                        f"KV OFFLOAD: {kv_offload_impl} cache, full-attention K/V stay in CPU RAM "
                        f"between layers",
                        flush=True,
                    )
                else:
                    past = DynamicCache(config=config)
                state["past_cache"] = past
                num_chunks = (T + chunk_size - 1) // chunk_size
                state["num_chunks"] = num_chunks
                for ci, start in enumerate(range(0, T, chunk_size)):
                    end = min(start + chunk_size, T)
                    cl = end - start
                    state["chunk_index"] = ci
                    state["chunk_len"] = cl
                    chunk_ids = input_ids[:, start:end]
                    chunk_pos = positions[start:end].unsqueeze(0)
                    out = model(
                        input_ids=chunk_ids,
                        attention_mask=None,
                        position_ids=chunk_pos,
                        past_key_values=past,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    # The model mutates the cache in place, but take the returned
                    # object in case a model rebuilds it.
                    if out.past_key_values is not None:
                        past = out.past_key_values
                    print(
                        f"chunk {ci + 1}/{num_chunks}: tokens [{start}, {end}) "
                        f"({cl} tokens), cache_seq_len={past.get_seq_length()}",
                        flush=True,
                    )
                    del out
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    finally:
        elapsed_s = time.time() - t_forward
        if had_key:
            ALL_ATTENTION_FUNCTIONS[args.attn] = saved_interface
        else:
            ALL_ATTENTION_FUNCTIONS.pop(args.attn, None)
        for mod, orig in zip(patched_modules, rotary_patch_state.values()):
            mod.apply_rotary_pos_emb = orig
        for h in hooks:
            h.remove()
    peak_vram_gib = (
        round(torch.cuda.max_memory_allocated(device) / (2**30), 4) if device.type == "cuda" else None
    )

    if chunked:
        # Materialise per-layer captures from the chunk buffers.
        for idx in full_indices:
            buf = state["buffers"].pop(idx)
            if not buf["k"]:
                raise RuntimeError(f"layer {idx}: no chunks captured")
            rec = {
                "queries_post_rope": buf["q_tail"].contiguous().clone(),
                "keys_post_rope": torch.cat(buf["k"], dim=1).contiguous(),
                "values": torch.cat(buf["v"], dim=1).contiguous(),
                "positions": positions[T - nq :].to("cpu").clone(),
                **state["layer_meta"][idx],
                "selfcheck_module_out": state["own_out_last"][idx],
            }
            if buf.get("pre"):
                rec["keys_pre_rope"] = torch.cat(buf["pre"], dim=1).contiguous()
            state["captured"][idx] = rec

    if sorted(state["captured"]) != sorted(full_indices):
        raise RuntimeError(
            f"captured layers {sorted(state['captured'])} != full-attention layers {full_indices}"
        )
    for idx in full_indices:
        rec = state["captured"][idx]
        if rec["keys_post_rope"].shape[-2] != T:
            raise RuntimeError(
                f"layer {idx}: captured {rec['keys_post_rope'].shape[-2]} keys, expected {T}"
            )

    # Self-check: recompute the last query's attention output from the dump.
    check_layer = full_indices[0]
    rec = state["captured"][check_layer]
    q_last = rec["queries_post_rope"][:, -1, :].float()
    k_all = rec["keys_post_rope"].float()
    v_all = rec["values"].float()
    groups = rec["num_key_value_groups"]
    k_rep = k_all.repeat_interleave(groups, dim=0)
    v_rep = v_all.repeat_interleave(groups, dim=0)
    logits = (q_last.unsqueeze(1) @ k_rep.transpose(1, 2)).squeeze(1) * rec["scaling"]
    probs = torch.softmax(logits, dim=-1)
    out_recomputed = (probs.unsqueeze(1) @ v_rep).squeeze(1)
    own = state["own_out_last"][check_layer].float()
    if own.shape != out_recomputed.shape:
        raise RuntimeError(f"self-check shape mismatch: own {tuple(own.shape)} vs {tuple(out_recomputed.shape)}")
    cos = torch.nn.functional.cosine_similarity(
        out_recomputed.flatten(), own.flatten(), dim=0
    ).item()
    print(
        f"SELF-CHECK layer {check_layer}: cosine(last-query attention output) = {cos:.8f} "
        f"(threshold 0.999)"
    )
    if not cos > 0.999:
        raise RuntimeError(f"self-check failed: cosine {cos} <= 0.999")
    del k_rep, v_rep, logits, probs, out_recomputed, own

    # Persist each layer.
    files = []
    total_bytes = 0
    layer_index_map = {}
    scaling_per_layer = {}
    for idx in full_indices:
        rec = state["captured"].pop(idx)
        layer_index_map[str(idx)] = {
            "module_name": attn_modules[idx][0],
            "layer_type": layer_types[idx],
            "num_attention_heads": rec["num_attention_heads"],
            "num_key_value_heads": rec["num_key_value_heads"],
            "num_key_value_groups": rec["num_key_value_groups"],
            "head_dim": int(rec["keys_post_rope"].shape[-1]),
            "has_pre_rope_keys": "keys_pre_rope" in rec,
        }
        scaling_per_layer[str(idx)] = rec["scaling"]
        path = out_dir / f"layer_{idx:02d}.pt"
        torch.save(rec, path)
        files.append(path.name)
        total_bytes += path.stat().st_size

    meta = {
        "model_path": os.path.abspath(args.model),
        "model_path_real": os.path.realpath(args.model),
        "model_revision": getattr(config, "_commit_hash", None),
        "architectures": getattr(config, "architectures", None),
        "model_type": getattr(config, "model_type", None),
        "max_tokens": int(args.max_tokens),
        "prompt_file": os.path.basename(args.prompt_file),
        "prompt_file_path": os.path.abspath(args.prompt_file),
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "peak_vram_gib": peak_vram_gib,
        "elapsed_s": round(elapsed_s, 3),
        "attention_impl": actual_impl,
        "prefill_chunk": chunk_size,
        "chunked_prefill": chunked,
        "num_chunks": int(state["num_chunks"]),
        "self_check_layer": check_layer,
        "self_check_cosine": cos,
        "tokens": T,
        "n_queries": nq,
        "query_positions": state.get("query_positions", list(range(T - nq, T))),
        "layer_types": layer_types,
        "full_attention_layer_indices": full_indices,
        "num_full_attention_layers": len(full_indices),
        "layer_index_map": layer_index_map,
        "attention_scaling_per_layer": scaling_per_layer,
        "gqa_group_size": {
            "num_attention_heads": int(text_cfg.num_attention_heads),
            "num_key_value_heads": int(text_cfg.num_key_value_heads),
            "groups": int(text_cfg.num_attention_heads // text_cfg.num_key_value_heads),
        },
        "rope": _rope_info(text_cfg),
        "rope_scaling": {
            "rope_type": rope_readback.get("rope_type"),
            "factor": rope_readback.get("factor"),
            "original_max_position_embeddings": rope_readback.get("original_max_position_embeddings"),
            "max_position_embeddings": rope_readback.get("max_position_embeddings"),
            "attention_scaling": rope_readback.get("attention_scaling_from_model"),
            "requested": yarn_request,
            "readback": rope_readback,
        },
        "kv_offload": kv_offload,
        "kv_offload_impl": kv_offload_impl if kv_offload else None,
        "kv_offload_moved_layers": (
            int(getattr(state.get("past_cache"), "moved_layers", 0)) if kv_offload else 0
        ),
        "rotary_dim_per_layer": {str(k): v for k, v in sorted(state["rotary_dim"].items())},
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or platform.machine(),
        "torch_version": torch.__version__,
        "rocm_version": torch.version.hip,
        "transformers_version": __import__("transformers").__version__,
        "seed": args.seed,
        "attention_implementation": actual_impl,
        "self_check": {
            "layer": check_layer,
            "cosine": cos,
            "threshold": 0.999,
            "passed": bool(cos > 0.999),
        },
        "dump_files": files,
        "dump_bytes": total_bytes,
        "dump_mib": round(total_bytes / (1024 * 1024), 3),
        "git_commit": _git_commit(),
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"WROTE {meta_path} ({total_bytes / (1024*1024):.2f} MiB of tensors across {len(files)} files)")
    print(json.dumps({"full_attention_layer_indices": full_indices,
                      "self_check_cosine": cos,
                      "dump_mib": meta["dump_mib"]}))
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--n-queries", type=int, default=8)
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument(
        "--attn",
        choices=["sdpa", "eager"],
        default="sdpa",
        help="attention implementation for the forward pass; sdpa never materialises the full matrix",
    )
    p.add_argument(
        "--prefill-chunk",
        type=int,
        default=0,
        help="tokens per prefill chunk; 0 (default) is a single pass over the whole prompt",
    )
    p.add_argument(
        "--kv-offload",
        action="store_true",
        help="keep the model's own full-attention K/V cache in CPU RAM between layers "
        "(requires --prefill-chunk; the layer being computed is copied to the GPU)",
    )
    p.add_argument(
        "--kv-offload-impl",
        choices=["transformers", "custom"],
        default="transformers",
        help="offload implementation: transformers' DynamicCache(offloading=True) (default) "
        "or the dumper's own cache wrapper",
    )
    p.add_argument(
        "--rope-yarn",
        type=float,
        default=0.0,
        help="apply YaRN rope scaling with this factor (0 = leave the model's rope unchanged)",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
