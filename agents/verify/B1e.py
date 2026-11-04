#!/usr/bin/env python3
"""Acceptance for B1e (dumper scaling). Written by the supervisor, not the implementer."""
import json, sys, os, glob
import torch
R = "/ai/github/lsa-rocm"
man = json.load(open(f"{R}/experiments/data/manifest.compact.json"))
ids = {p["id"] for p in man["prompts"]}
fails = []
def need(c, msg):
    if not c: fails.append(msg)
for L in (8192, 16384, 32768):
    d = f"{R}/results/dumps/ornith-1.5-9b-{L}"
    mp = f"{d}/meta.json"
    if not os.path.exists(mp): fails.append(f"{L}: missing {mp}"); continue
    m = json.load(open(mp))
    need(m.get("max_tokens") == L, f"{L}: meta.max_tokens != {L}: {m.get('max_tokens')}")
    need(m.get("prompt_file", "").split("/")[-1].replace(".txt", "") in ids, f"{L}: prompt_file not a manifest prompt: {m.get('prompt_file')}")
    for k in ("peak_vram_gib", "elapsed_s", "attention_impl", "self_check_layer", "self_check_cosine"):
        need(k in m, f"{L}: meta missing {k}")
    need(m.get("self_check_cosine", 0) > 0.999, f"{L}: self_check_cosine {m.get('self_check_cosine')}")
    need(len(m.get("full_attention_layer_indices", [])) == 8, f"{L}: expected 8 full-attention layers")
    need("eager" != str(m.get("attention_impl")), f"{L}: attention_impl is eager, will not scale")
    sl = m.get("self_check_layer")
    if sl is None: continue
    lp = f"{d}/layer_{int(sl):02d}.pt"
    if not os.path.exists(lp): fails.append(f"{L}: missing {lp}"); continue
    t = torch.load(lp, map_location="cpu")
    K, V, Q = t["keys_post_rope"].float(), t["values"].float(), t["queries_post_rope"].float()
    need(K.shape[1] == L, f"{L}: keys length {K.shape[1]}")
    out_key = next((k for k in t if "module_out" in k), None)
    need(out_key is not None, f"{L}: layer file lacks a saved module output (key containing 'module_out')")
    if out_key is None: continue
    ref = t[out_key].float().reshape(Q.shape[0], -1)          # [Hq, D] module output of last query
    scale = float(t["scaling"]); g = Q.shape[0] // K.shape[0]
    q = Q[:, -1, :]                                            # last query, all heads
    Kr = K.repeat_interleave(g, 0); Vr = V.repeat_interleave(g, 0)
    w = torch.softmax((q.unsqueeze(1) @ Kr.transpose(1, 2)).squeeze(1) * scale, -1)
    mine = (w.unsqueeze(1) @ Vr).squeeze(1)
    cos = torch.nn.functional.cosine_similarity(mine.flatten(), ref.flatten(), dim=0).item()
    print(f"{L}: independent recompute cosine={cos:.6f} peak_vram_gib={m.get('peak_vram_gib')} elapsed_s={m.get('elapsed_s')} impl={m.get('attention_impl')}")
    need(cos > 0.999, f"{L}: independent recompute cosine {cos}")
print("FAILS:", fails if fails else "none")
sys.exit(1 if fails else 0)
