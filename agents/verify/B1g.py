#!/usr/bin/env python3
"""Acceptance for B1g (chunked prefill dumper). Supervisor-written."""
import json, sys, os
import torch
R = "/ai/github/lsa-rocm"
fails = []
def need(c, msg):
    if not c: fails.append(msg)
def load(d, layer):
    return torch.load(f"{d}/layer_{layer:02d}.pt", map_location="cpu")
# 1) chunked vs non-chunked agree on 32K (same prompt prose-32768)
ref, chk = f"{R}/results/dumps/ornith-1.5-9b-32768", f"{R}/results/dumps/ornith-chunked-32768"
for d in (ref, chk):
    if not os.path.exists(f"{d}/meta.json"): fails.append(f"missing {d}/meta.json")
if not fails:
    mr, mc = json.load(open(f"{ref}/meta.json")), json.load(open(f"{chk}/meta.json"))
    need(mc.get("prefill_chunk"), "chunked meta lacks prefill_chunk")
    need(mc.get("max_tokens") == 32768, "chunked 32K meta.max_tokens")
    for L in mr["full_attention_layer_indices"][:3]:
        a, b = load(ref, L)["keys_post_rope"].float(), load(chk, L)["keys_post_rope"].float()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        print(f"32K layer {L}: chunked-vs-full keys cosine={cos:.6f} shape={tuple(b.shape)}")
        need(a.shape == b.shape and cos > 0.999, f"layer {L}: chunked keys differ (cos {cos})")
# 2) the 128K dump, with independent recompute
d = f"{R}/results/dumps/ornith-chunked-131072"
mp = f"{d}/meta.json"
if not os.path.exists(mp): fails.append(f"missing {mp}")
else:
    m = json.load(open(mp))
    for k in ("peak_vram_gib", "elapsed_s", "attention_impl", "prefill_chunk", "self_check_layer", "self_check_cosine", "prompt_file"):
        need(k in m, f"128K meta missing {k}")
    need(m.get("max_tokens") == 131072, f"128K max_tokens {m.get('max_tokens')}")
    need(m.get("self_check_cosine", 0) > 0.999, "128K self_check_cosine")
    need("prose-131072" in m.get("prompt_file", "") or "coh-131072" in m.get("prompt_file", ""), f"128K prompt_file {m.get('prompt_file')}")
    sl = m.get("self_check_layer")
    t = load(d, int(sl)) if sl is not None else None
    if t is not None:
        K, V, Q = t["keys_post_rope"].float(), t["values"].float(), t["queries_post_rope"].float()
        need(K.shape[1] == 131072, f"128K key length {K.shape[1]}")
        out_key = next((k for k in t if "module_out" in k), None)
        need(out_key is not None, "128K layer lacks module_out")
        if out_key:
            g = Q.shape[0] // K.shape[0]; q = Q[:, -1, :]
            w = torch.softmax((q.unsqueeze(1) @ K.repeat_interleave(g, 0).transpose(1, 2)).squeeze(1) * float(t["scaling"]), -1)
            mine = (w.unsqueeze(1) @ V.repeat_interleave(g, 0)).squeeze(1)
            cos = torch.nn.functional.cosine_similarity(mine.flatten(), t[out_key].float().reshape(Q.shape[0], -1).flatten(), dim=0).item()
            print(f"128K independent recompute cosine={cos:.6f} peak_vram_gib={m.get('peak_vram_gib')} elapsed_s={m.get('elapsed_s')} chunk={m.get('prefill_chunk')}")
            need(cos > 0.999, f"128K recompute cosine {cos}")
print("FAILS:", fails if fails else "none")
sys.exit(1 if fails else 0)
