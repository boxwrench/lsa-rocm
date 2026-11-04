#!/usr/bin/env python3
"""Run paper-mode analysis on one dump, write per-k result JSON, a summary JSON, and a README section.
usage: analyze_dump.py <dump_dir_name> <label> [--ks 512,2048,4096] [--tree scaled|upstream] [--n-queries 16]
Example: analyze_dump.py ornith-code-coh-32768 ornith-code32768
Uses the CPU venv. Idempotent: replaces the README section for the same label."""
import sys, os, json, subprocess, collections, re
R = "/ai/github/lsa-rocm"
PY = f"{R}/agents/scratch/venv-lsa-min/bin/python"
dump, label = sys.argv[1], sys.argv[2]
opt = dict(zip(sys.argv[3::2], sys.argv[4::2]))
ks = [int(x) for x in opt.get("--ks", "512,2048,4096").split(",")]
tree = opt.get("--tree", "scaled"); nq = opt.get("--n-queries", "16")
meta = json.load(open(f"{R}/results/dumps/{dump}/meta.json")); T = int(meta["max_tokens"])
env = dict(os.environ, PYTHONPATH=f"{R}/src:{R}/third_party/lsa/src")
for K in ks:
    out = f"{R}/results/b2_1-{label}-k{K}.json"
    if os.path.exists(out): continue
    r = subprocess.run([PY, f"{R}/experiments/002-real-kv-recall/run.py", "--mode", "paper", "--dump", f"{R}/results/dumps/{dump}",
                        "--out", out, "--layers", "all", "--n-queries", nq, "--n-cluster", "512", "--k", str(K), "--tree", tree],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0: print("FAILED", K, r.stderr[-400:]); sys.exit(1)
per = {}; rows = []
for K in ks:
    j = json.load(open(f"{R}/results/b2_1-{label}-k{K}.json")); agg = collections.defaultdict(list)
    for x in j["records"]:
        g = x["method"] if x["method"] in ("flat", "tree", "block_sparse") else x.get("matched_to")
        agg[(x["layer"], g, x["method"])].append(x["efficiency"])
    layers = sorted({l for l, _, _ in agg}); m = lambda l, g, n: sum(agg[(l, g, n)]) / len(agg[(l, g, n)])
    s = {n: [m(l, g, n) for l in layers] for n, g in (("flat", "flat"), ("tree", "tree"), ("block_sparse", "block_sparse"), ("recency", "flat"))}
    per[K] = {"layers": layers, **s, "budget_fraction": round(K / T, 5)}
    n = len(layers)
    rows.append(f"| {K} ({K/T:.2%}) | {sum(s['flat'])/n:.2f} | {sum(s['tree'])/n:.2f} | {sum(s['block_sparse'])/n:.2f} | {sum(s['recency'])/n:.2f} | "
                f"{sum(x>=0.7 for x in s['flat'])}/{n} | {sum(a>b for a,b in zip(s['flat'],s['recency']))}/{n} | {sum(a>b for a,b in zip(s['flat'],s['block_sparse']))}/{n} |")
json.dump({"dump": dump, "tokens": T, "tree_preset": tree, "per_k": per}, open(f"{R}/results/b2_1-summary-{label}.json", "w"), indent=1)
tbl = ("| k (share of keys) | flat | tree | block-sparse | recency | layers flat >= 0.70 | flat > recency | flat > block-sparse |\n|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
sec = (f"\n### {label}: {T} tokens\n\nDump `{dump}` (self-check cosine {meta.get('self_check_cosine')}), tree preset `{tree}`, window 512, "
       f"mean retrieval efficiency over layers. Source: `results/b2_1-summary-{label}.json`.\n\n{tbl}")
p = f"{R}/experiments/002-real-kv-recall/README.md"; s = open(p).read()
s = re.sub(rf"\n### {re.escape(label)}: .*?(?=\n### |\n## )", "", s, flags=re.S)
s = s.replace("\n## Conclusion", sec + "\n## Conclusion", 1); open(p, "w").write(s)
print(tbl)
