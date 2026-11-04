#!/usr/bin/env python3
"""Acceptance for B1f (coherent prompts). Supervisor-written."""
import json, sys, os, hashlib, re
from transformers import AutoTokenizer
R = "/ai/github/lsa-rocm"
tok = AutoTokenizer.from_pretrained("/ai/models/llm/Ornith-1.5-9B")
fails = []
def need(c, msg):
    if not c: fails.append(msg)
mp = f"{R}/experiments/data/manifest.coherent.compact.json"
if not os.path.exists(mp): print("FAILS: missing", mp); sys.exit(1)
m = json.load(open(mp)); ps = {p["id"]: p for p in m["prompts"]}
bad_src = re.compile(r"humaneval|mmlu|gsm8k|ifeval|paul_graham", re.I)
for kind in ("code-coh", "tech-coh"):
    for L in (4096, 32768, 131072):
        pid = f"{kind}-{L}"
        if pid not in ps: fails.append(f"missing prompt {pid}"); continue
        p = ps[pid]; f = f"{R}/experiments/data/texts/{pid}.txt"
        if not os.path.exists(f): fails.append(f"missing text {f}"); continue
        t = open(f, encoding="utf-8").read()
        ids = tok(t, add_special_tokens=False)["input_ids"]
        need(len(ids) == L, f"{pid}: {len(ids)} tokens != {L}")
        need(hashlib.sha256(t.encode()).hexdigest() == p.get("sha256"), f"{pid}: sha mismatch")
        ns = p.get("num_sources", 99)
        need(ns <= 3, f"{pid}: {ns} sources (>3, stitched)")
        need(not any(bad_src.search(s) for s in p.get("sources", [])), f"{pid}: banned source {p.get('sources')}")
        need(bool(p.get("licences")) and "unknown" not in [x.lower() for x in p["licences"]], f"{pid}: licence missing or unknown")
        grams = [tuple(ids[i:i+8]) for i in range(0, len(ids) - 8)]
        rep = 1 - len(set(grams)) / max(1, len(grams))
        need(rep < 0.15, f"{pid}: 8-gram repetition {rep:.3f} >= 0.15")
        print(f"{pid}: tokens={len(ids)} sources={ns} 8gram_rep={rep:.3f} licences={p.get('licences')}")
print("FAILS:", fails if fails else "none")
sys.exit(1 if fails else 0)
