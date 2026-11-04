#!/usr/bin/env python3
"""Collate results/b2_1-summary-*.json into markdown tables (never hand-copy numbers). Prints to stdout.
Rows: genre x context length. Columns: mean retrieval efficiency of flat / tree / block-sparse / recency at a fixed k."""
import json, glob, re, sys, os
R = "/ai/github/lsa-rocm/results"
rows = []
for f in sorted(glob.glob(f"{R}/b2_1-summary-*.json")):
    j = json.load(open(f)); label = os.path.basename(f)[len("b2_1-summary-"):-5]
    m = re.match(r"(.+?)-?(prose|code|tech)(\d+)$", label)
    if not m: continue
    model, genre, n = m.group(1) or "ornith", m.group(2), int(m.group(3))
    rows.append((model, genre, int(j.get("tokens", n)), j.get("tree_preset", "scaled"), j["per_k"]))
rows.sort(key=lambda r: (r[0], r[1], r[2]))
def mean(x): return sum(x) / len(x)
for K in (2048, 4096):
    print(f"\n### Mean retrieval efficiency at k = {K}\n")
    print("| model | genre | tokens | k share of keys | flat | tree | block-sparse | recency | layers flat>=0.70 | flat>recency | flat>block |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for model, genre, T, tree, pk in rows:
        d = pk.get(str(K))
        if not d: continue
        f, t, b, rc = (d[x] for x in ("flat", "tree", "block_sparse", "recency")); n = len(f)
        print(f"| {model} | {genre} | {T:,} | {K/T:.2%} | {mean(f):.2f} | {mean(t):.2f} | {mean(b):.2f} | {mean(rc):.2f} | "
              f"{sum(x>=0.7 for x in f)}/{n} | {sum(a>b_ for a,b_ in zip(f,rc))}/{n} | {sum(a>b_ for a,b_ in zip(f,b))}/{n} |")
