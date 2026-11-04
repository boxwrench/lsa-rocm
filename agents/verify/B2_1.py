#!/usr/bin/env python3
"""Acceptance for B2.1 (paper mode). Supervisor-written."""
import json, sys, os, subprocess
import torch
R = "/ai/github/lsa-rocm"
PY = f"{R}/agents/scratch/venv-lsa-min/bin/python"
dump = f"{R}/results/dumps/ornith-1.5-9b-8192"
out = f"{R}/results/b2_1-verify.json"
if os.path.exists(out): os.remove(out)
cmd = [PY, "-I", f"{R}/experiments/002-real-kv-recall/run.py", "--mode", "paper", "--dump", dump, "--out", out,
       "--layers", "7", "--n-queries", "4", "--quick"]
env = dict(os.environ, PYTHONPATH=f"{R}/src:{R}/third_party/lsa/src")
r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1200)
print((r.stdout + r.stderr)[-500:])
fails = []
def need(c, msg):
    if not c: fails.append(msg)
need(r.returncode == 0, f"run.py exit {r.returncode}")
if not os.path.exists(out): print("FAILS: no output json"); sys.exit(1)
j = json.load(open(out)); meta = j.get("meta", {}); recs = j.get("records", [])
for k, v in (("d_idx", 64), ("n_pool", 4), ("sink_size", 4)):
    need(meta.get(k) == v, f"meta.{k} == {v}, got {meta.get(k)}")
need(meta.get("window_size") == meta.get("n_cluster"), "window_size must equal n_cluster")
need(meta.get("index_dims_source") == "rope_free_dims_all_kv_heads", f"index_dims_source {meta.get('index_dims_source')}")
need("rotary_dim" in meta and "k" in meta and "p" in meta, "meta needs rotary_dim, k, p")
need(meta.get("p") == meta.get("k", 0) // 4, "p must equal k // 4")
meths = {x.get("method") for x in recs}
for m in ("flat", "tree", "random", "recency", "exhaustive"): need(m in meths, f"missing method {m}")
for x in recs:
    need(x.get("index_mode") == "paper-pca", "index_mode must be paper-pca")
    need(x.get("kv_head") == "all", "paper mode index is layer-shared: kv_head must be 'all'")
    for f in ("mass_outside_headavg", "top_recall", "efficiency", "mass_captured", "budget_fraction", "matched_to"):
        need(f in x, f"record missing {f}")
    if fails: break
import collections
agg = collections.defaultdict(list)
for x in recs:
    g = x["method"] if x["method"] in ("flat", "tree") else x.get("matched_to")
    agg[(g, x["method"])].append(x["mass_outside_headavg"])
mean = {k: sum(v) / len(v) for k, v in agg.items() if v}
print("mean outside-window head-avg mass:", {f"{a}/{b}": round(v, 4) for (a, b), v in mean.items()})
for g in ("flat", "tree"):
    try:
        need(mean[(g, "exhaustive")] >= mean[(g, g)] - 1e-6, f"{g}: exhaustive < method")
        need(mean[(g, g)] > mean[(g, "random")], f"{g}: not above random")
    except KeyError as e: fails.append(f"missing group {e}")
# independent recompute of the paper-target metric from the dump
s = j["samples"][0]
t = torch.load(f"{dump}/layer_{int(s['layer']):02d}.pt", map_location="cpu")
K, Q = t["keys_post_rope"].float(), t["queries_post_rope"].float()
Hkv, T, D = K.shape; Hq = Q.shape[0]; g = Hq // Hkv
qi = int(s["query_index"]); pos = int(t["positions"][qi]); sink, win = int(meta["sink_size"]), int(meta["window_size"])
lo, hi = sink, pos - win + 1                                   # R_i = [sink, pos-window]
need(hi > lo, "region R_i empty")
A = torch.zeros(hi - lo)
for h in range(Hq):
    A += torch.softmax((K[h // g, lo:hi] @ Q[h, qi]) * float(t["scaling"]), -1)
A /= Hq
sel = torch.tensor(s["selected"], dtype=torch.long); sel = sel[(sel >= lo) & (sel < hi)] - lo
mine = A[sel].sum().item()
print(f"independent mass_outside_headavg={mine:.5f} reported={s['mass_outside_headavg']:.5f}")
need(abs(mine - s["mass_outside_headavg"]) < 2e-3, "sample metric does not reproduce")
print("FAILS:", fails if fails else "none")
sys.exit(1 if fails else 0)
