"""Dump post-RoPE Q/K/V of the full-attention layers of a local HF model.

Usage:
    python -m lsa_rocm.dump --model PATH --prompt-file FILE --max-tokens N \
        --n-queries Q --out DIR [--device cpu|cuda]

The tool runs a single forward pass over the prompt, captures, per full-attention
layer, keys/values as cached (post-RoPE) [Hkv, T, D], keys before RoPE (when the
model's rotary helper can be intercepted), queries of the last Q positions after
RoPE [Hq, Q, D] and their positions. It then recomputes the attention output of
the last query from the dumped tensors and compares it against the attention
module's own output; a cosine below 0.999 aborts the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
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
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_tokens)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    T = int(input_ids.shape[1])
    if T <= 0:
        raise SystemExit("empty prompt after tokenization")
    nq = min(int(args.n_queries), T)
    if nq <= 0:
        raise SystemExit("--n-queries must be positive")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    config = model.config
    text_cfg = _text_config(config)
    full_indices, layer_types = _full_attention_indices(config)
    attn_modules = _find_attention_modules(model, full_indices)
    missing = sorted(set(full_indices) - set(attn_modules))
    if missing:
        raise RuntimeError(f"could not locate attention modules for layers {missing}")

    positions = torch.arange(T, dtype=torch.long, device=device)

    # Per-layer capture. The attention interface receives post-RoPE q/k/v and
    # returns the module's own attention output; pre-RoPE k is captured by
    # wrapping the rotary helper used inside the attention forward.
    state = {
        "pending_pre_rope": {},
        "captured": {},
        "current_layer": None,
        "own_out_last": {},
        "rotary_dim": {},
    }

    # Under the "eager" implementation transformers falls back to the module's
    # own `eager_attention_forward` default rather than a registry entry, so we
    # recover that function and register our wrapper under the "eager" key,
    # which get_interface then finds first.
    import importlib

    eager_fn = None
    for _idx, (_name, module) in attn_modules.items():
        cls_mod = importlib.import_module(type(module).__module__)
        eager_fn = getattr(cls_mod, "eager_attention_forward", None)
        if eager_fn is not None:
            break
    if eager_fn is None:
        raise RuntimeError("could not locate the model's eager_attention_forward")
    orig_interface = eager_fn

    def wrapped_interface(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        out, weights = orig_interface(
            module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
        )
        idx = getattr(module, "layer_idx", None)
        if idx in full_indices:
            q = query[0].detach().to("cpu")
            k = key[0].detach().to("cpu")
            v = value[0].detach().to("cpu")
            if k.shape[-2] != T:
                raise RuntimeError(f"layer {idx}: captured {k.shape[-2]} keys, expected {T}")
            pre = state["pending_pre_rope"].pop(idx, None)
            record = {
                "queries_post_rope": q[:, T - nq :].contiguous().clone(),
                "keys_post_rope": k.contiguous().clone(),
                "values": v.contiguous().clone(),
                "positions": positions[T - nq :].to("cpu").clone(),
                "scaling": float(scaling),
                "num_key_value_groups": int(module.num_key_value_groups),
                "num_attention_heads": int(getattr(module, "config").num_attention_heads),
                "num_key_value_heads": int(getattr(module, "config").num_key_value_heads),
            }
            if pre is not None:
                record["keys_pre_rope"] = pre
            state["captured"][idx] = record
            # out: [B, T, Hq, D] under eager
            o = out.detach()
            if o.dim() == 4 and o.shape[1] == T:
                state["own_out_last"][idx] = o[0, T - 1].float().to("cpu")
            elif o.dim() == 3:
                state["own_out_last"][idx] = o[0, T - 1].float().to("cpu")
            else:
                raise RuntimeError(f"unexpected attention output shape {tuple(o.shape)}")
        return out, weights

    ALL_ATTENTION_FUNCTIONS["eager"] = wrapped_interface

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

    try:
        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=positions.unsqueeze(0),
                use_cache=False,
            )
    finally:
        ALL_ATTENTION_FUNCTIONS.pop("eager", None)
        for mod, orig in zip(patched_modules, rotary_patch_state.values()):
            mod.apply_rotary_pos_emb = orig
        for h in hooks:
            h.remove()

    if sorted(state["captured"]) != sorted(full_indices):
        raise RuntimeError(
            f"captured layers {sorted(state['captured'])} != full-attention layers {full_indices}"
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
        "prompt_file": os.path.abspath(args.prompt_file),
        "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
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
        "rotary_dim_per_layer": {str(k): v for k, v in sorted(state["rotary_dim"].items())},
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or platform.machine(),
        "torch_version": torch.__version__,
        "rocm_version": torch.version.hip,
        "transformers_version": __import__("transformers").__version__,
        "seed": args.seed,
        "attention_implementation": "eager",
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
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
