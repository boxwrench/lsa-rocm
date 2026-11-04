#!/usr/bin/env python3
"""Acceptance for B2.0 (analysis harness, CPU). Supervisor-written."""
import json, sys, os, subprocess
import torch
R = "/ai/github/lsa-rocm"
PY = f"{R}/agents/scratch/venv-lsa-min/bin/python"
dump = f"{R}/results/dumps/qwen3.5-0.8b-2048"
out = f"{R}/results/b2_0-verify.json"
if os.path.exists(out): os.remove(out)
cmd = [PY, "-I", f"{R}/experiments/002-real-kv-recall/run.py", "--dump", dump, "--out", out,
       "--layers", "3", "--max-kv-heads", "1", "--n-queries", "4", "--quick"]
env = dict(os.environ, PYTHONPATH=f"{R}/src:{R}/third_party/lsa/src")
r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
print((r.stdout + r.stderr)[-600:])
fails = []
def need(c, msg):
    if not c: fails.append(msg)
need(r.returncode == 0, f"run.py exit {r.returncode}")
if not os.path.exists(out): print("FAILS: no output json"); sys.exit(1)
j = json.load(open(out))
need(all(k in j for k in ("meta", "records", "samples")), "json needs meta, records, samples")
recs = j.get("records", [])
need(len(recs) >= 4, f"only {len(recs)} records")
methods = {r_["method"] for r_ in recs}
for mth in ("flat", "tree", "random", "recency", "exhaustive"):
    need(mth in methods, f"missing method {mth}")
for r_ in recs:
    for k in ("layer", "kv_head", "method", "index_mode", "mass_captured", "recall_at_k", "output_error", "budget_fraction", "mass_excluding_sinks"):
        need(k in r_, f"record missing {k}"); 
    if fails: break
# control ordering within matched-size groups
import collections
agg = collections.defaultdict(list)
for r_ in recs:
    grp = r_["method"] if r_["method"] in ("flat", "tree") else r_.get("matched_to")
    need(grp in ("flat", "tree"), f"record {r_['method']} lacks matched_to in (flat, tree)")
    agg[(grp, r_["method"])].append(r_["mass_captured"])
mean = {k: sum(v) / len(v) for k, v in agg.items()}
print("mean mass:", {f"{a}/{b}": round(v, 4) for (a, b), v in mean.items()})
for grp in ("flat", "tree"):
    try:
        need(mean[(grp, "exhaustive")] >= mean[(grp, grp)] - 1e-6, f"{grp}: exhaustive upper bound < method")
        need(mean[(grp, grp)] > mean[(grp, "random")], f"{grp}: not above random control")
    except KeyError as e:
        fails.append(f"missing group/method {e}")
# independent recompute of a sample
s = j["samples"][0]
t = torch.load(f"{dump}/layer_{int(s['layer']):02d}.pt", map_location="cpu")
K, Q = t["keys_post_rope"].float(), t["queries_post_rope"].float()
g = Q.shape[0] // K.shape[0]; h = int(s["query_head"]); kv = h // g
q = Q[h, int(s["query_index"])]; pos = int(t["positions"][int(s["query_index"])])
logit = (K[kv, : pos + 1] @ q) * float(t["scaling"])
w = torch.softmax(logit, -1)
sel = torch.tensor(s["selected"], dtype=torch.long); sel = sel[sel <= pos]
mass = w[sel].sum().item()
print(f"sample recompute mass={mass:.6f} reported={s['mass_captured']:.6f}")
need(abs(mass - s["mass_captured"]) < 1e-3, "sample mass does not reproduce from dump + selected indices")
print("FAILS:", fails if fails else "none")
sys.exit(1 if fails else 0)
