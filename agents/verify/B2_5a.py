#!/usr/bin/env python3
"""Acceptance for B2.5a (long-context dumper features). Supervisor-written."""
import json, sys, os
import torch
R = "/ai/github/lsa-rocm"
fails = []
def need(c, msg):
    if not c: fails.append(msg)
def load(d, layer): return torch.load(f"{d}/layer_{layer:02d}.pt", map_location="cpu")
def meta(d):
    p = f"{d}/meta.json"
    if not os.path.exists(p): fails.append(f"missing {p}"); return None
    return json.load(open(p))
# 1) KV offload gives the same keys as the accepted chunked dump (32K)
ref, off = f"{R}/results/dumps/ornith-chunked-32768", f"{R}/results/dumps/ornith-offload-32768"
mr, mo = meta(ref), meta(off)
if mr and mo:
    need(mo.get("kv_offload") is True, "offload meta.kv_offload must be true")
    need(mo.get("max_tokens") == 32768, "offload max_tokens")
    need(mo.get("self_check_cosine", 0) > 0.999, "offload self_check_cosine")
    for L in mr["full_attention_layer_indices"][:4]:
        a, b = load(ref, L)["keys_post_rope"].float(), load(off, L)["keys_post_rope"].float()
        c = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        print(f"offload vs chunked 32K layer {L}: keys cosine={c:.6f} peak_vram_gib={mo.get('peak_vram_gib')} (ref {mr.get('peak_vram_gib')})")
        need(a.shape == b.shape and c > 0.999, f"layer {L}: offload keys differ (cos {c})")
# 2) YaRN: RoPE-free dims unchanged, rotated dims rescaled by the YaRN attention factor
ya, base = f"{R}/results/dumps/ornith-yarn4-8192", f"{R}/results/dumps/ornith-1.5-9b-8192"
my, mb = meta(ya), meta(base)
if my and mb:
    rs = my.get("rope_scaling") or {}
    need(float(rs.get("factor", 0)) == 4.0, f"yarn meta.rope_scaling.factor {rs.get('factor')}")
    need("yarn" in str(rs.get("rope_type", rs.get("type", ""))).lower(), "yarn rope_type missing")
    need(my.get("max_tokens") == 8192, "yarn max_tokens")
    rd = int(mb.get("rotary_dim", 64))
    L = mb["full_attention_layer_indices"][0]   # first full-attention layer: nothing upstream is YaRN-scaled (later layers drift: 0.9985 at the 2nd, 0.9967 at the 3rd)
    a, b = load(base, L)["keys_post_rope"].float(), load(ya, L)["keys_post_rope"].float()
    free = torch.nn.functional.cosine_similarity(a[..., rd:].flatten(), b[..., rd:].flatten(), dim=0).item()
    ratio = (b[..., 1:, :rd].norm() / a[..., 1:, :rd].norm()).item()
    print(f"yarn vs plain 8K layer {L}: rope-free dims cosine={free:.6f}, rotated-dims norm ratio={ratio:.4f} (YaRN mscale for factor 4 is about 1.1386)")
    need(free > 0.999, f"rope-free dims changed under YaRN (cos {free})")
    need(1.05 < ratio < 1.30, f"rotated dims not rescaled as YaRN would (ratio {ratio})")
print("FAILS:", fails if fails else "none")
sys.exit(1 if fails else 0)
