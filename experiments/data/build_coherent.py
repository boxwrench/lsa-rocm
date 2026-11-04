#!/usr/bin/env python3
"""Coherent long prompts: ONE repository per prompt, files in path order. Seeded, no GPU.
code-coh: Python sources of exllamav3 (MIT). tech-coh: documentation of composable_kernel (MIT)."""
import json, hashlib, random, os, glob, sys
from transformers import AutoTokenizer
R = "/ai/github/lsa-rocm/experiments/data"
tok = AutoTokenizer.from_pretrained("/ai/models/llm/Ornith-1.5-9B")
SRC = {
 "code-coh": dict(genre="code", root="/ai/github/exllamav3-pr", name="exllamav3 (turboderp, MIT)", licence="MIT",
                  pats=["exllamav3/**/*.py"], hdr="# ==== file: {} ====\n"),
 "tech-coh": dict(genre="technical", root="/ai/github/composable_kernel", name="composable_kernel docs (AMD, MIT)", licence="MIT",
                  pats=["docs/**/*.rst", "docs/**/*.md", "**/*.md"], hdr=".. ==== file: {} ====\n"),
}
def _materialise_essays(dst="/ai/github/lsa-rocm/agents/scratch/essays"):
    """One text file per essay of the sgoel9/paul_graham_essays dataset (gitignored scratch, never committed)."""
    import csv
    os.makedirs(dst, exist_ok=True)
    if glob.glob(dst + "/*.txt"): return dst
    csv.field_size_limit(10**9)
    f = glob.glob("/ai/huggingface/hub/datasets--sgoel9--paul_graham_essays/snapshots/*/*.csv")[0]
    for r in list(csv.DictReader(open(f, encoding="utf-8", errors="ignore"))):
        open(f"{dst}/{int(r['id']):04d}.txt", "w", encoding="utf-8").write(f"{r['title']}\n\n{r['text']}\n")
    return dst
SRC["prose-coh"] = dict(genre="prose", root=_materialise_essays(), name="paul_graham_essays dataset (sgoel9; dataset card MIT, essays are the author's)",
                        licence="MIT (dataset card)", pats=["*.txt"], hdr="== essay: {} ==\n")
def files(cfg):
    fs = sorted({p for pat in cfg["pats"] for p in glob.glob(os.path.join(cfg["root"], pat), recursive=True)
                 if os.path.isfile(p) and os.path.getsize(p) > 1500 and "/third_party/" not in p})
    return fs
def exact(text, L):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) < L: return None
    cut = L
    for _ in range(40):
        t = tok.decode(ids[:cut]); n = len(tok(t, add_special_tokens=False)["input_ids"])
        if n == L: return t
        cut += (L - n)
    return None
def rep8(t):
    ids = tok(t, add_special_tokens=False)["input_ids"]
    g = [tuple(ids[i:i+8]) for i in range(0, len(ids) - 8)]
    return 1 - len(set(g)) / max(1, len(g))
def build(fs, cfg, start, L):
    parts, used, n = [], [], 0
    for p in fs[start:] + fs[:start]:
        s = open(p, encoding="utf-8", errors="ignore").read()
        piece = cfg["hdr"].format(os.path.relpath(p, cfg["root"])) + s
        parts.append(piece); used.append(p)
        n += len(tok(piece, add_special_tokens=False)["input_ids"])
        if n >= L + 64: break
    return exact("\n\n".join(parts), L), used
# usage: build_coherent.py [kind ...] [--lengths 4096,32768,131072]   (partial runs merge into the manifest)
args = sys.argv[1:]; LENGTHS = (4096, 32768, 131072); only = []
while args:
    a = args.pop(0)
    if a == "--lengths": LENGTHS = tuple(int(x) for x in args.pop(0).split(","))
    else: only.append(a)
mp = f"{R}/manifest.coherent.compact.json"
manifest = json.load(open(mp)) if os.path.exists(mp) else {"tokenizer": "Ornith-1.5-9B", "add_special_tokens": False, "seed": 0, "prompts": []}
for kind, cfg in SRC.items():
    if only and kind not in only: continue
    fs = files(cfg)
    for L in LENGTHS:
        best = None
        for attempt in range(14):
            start = (attempt * 37 + 11) % len(fs)
            text, used = build(fs, cfg, start, L)
            if text is None: continue
            r = rep8(text)
            if best is None or r < best[0]: best = (r, text, used, start)
            if r < 0.12: break
        pid = f"{kind}-{L}"
        if best is None: print("FAILED to fill", pid, "files", len(fs)); continue
        r, text, used, start = best
        open(f"{R}/texts/{pid}.txt", "w", encoding="utf-8").write(text)
        manifest["prompts"] = [x for x in manifest["prompts"] if x["id"] != pid]
        manifest["prompts"].append(dict(id=pid, genre=cfg["genre"], length=L, token_count=L,
            sha256=hashlib.sha256(text.encode()).hexdigest(), licences=[cfg["licence"]], sources=[cfg["name"]],
            num_sources=1, num_files=len(used), start_file_index=start, eight_gram_repetition=round(r, 3)))
        print(pid, "files", len(used), "start", start, "rep8", round(r, 3), flush=True)
json.dump(manifest, open(f"{R}/manifest.coherent.compact.json", "w"), indent=1)
