#!/usr/bin/env python3
"""Needle-in-haystack, exact attention, Nemotron-3-Nano-30B-A3B on Nimo. Haystack/scoring protocol copied from 006-long-context/run_big.py.
Chunked prefill (patched fused kernels unless --unpatched), cache snapshot after haystack, one question at a time (Mamba state restored,
attention KV cropped back). usage: run_needle_nemotron.py --length 32768 --facts 8 --out results/007-nemotron-32768.json"""
import os, sys, json, time, argparse, random, copy
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
sys.path.insert(0, "/home/keith/lsa/src")
import torch
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/home/keith/lsa/models/Nemotron-3-Nano-30B-A3B-BF16")
ap.add_argument("--texts", default="/home/keith/lsa/texts/prose-coh-524288.txt,/home/keith/lsa/texts/code-coh-917504.txt")
ap.add_argument("--length", type=int, required=True); ap.add_argument("--facts", type=int, default=8); ap.add_argument("--distractors", type=int, default=20)
ap.add_argument("--chunk", type=int, default=8192); ap.add_argument("--max-new", type=int, default=24); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--prompt", default="raw", choices=["raw","plain","chat"]); ap.add_argument("--unpatched", action="store_true"); ap.add_argument("--out", required=True)
a = ap.parse_args()
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
rng = random.Random(a.seed)
def code_str(r): return f"{r.randint(1000,9999)}-{''.join(r.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(3))}"
def fact_ids(vn, vc): return tok(f"\n\nNote for the record: the access code for vault {vn} is {vc}.\n\n", add_special_tokens=False)["input_ids"]
# ---- haystack (same builder as run_big) ----
base = []
for tp in a.texts.split(","):
    base += tok(open(tp).read(), add_special_tokens=False)["input_ids"]
    if len(base) >= a.length: break
vaults = rng.sample(range(100, 1000), a.facts + a.distractors)
targets = [(vaults[i], code_str(rng)) for i in range(a.facts)]
dists = [(vaults[a.facts + i], code_str(rng)) for i in range(a.distractors)]
marks = [(0.03 + 0.94 * (i + 0.5) / a.facts, v, c, "T") for i, (v, c) in enumerate(targets)]
marks += [(0.03 + 0.94 * (i + 0.25) / a.distractors, v, c, "D") for i, (v, c) in enumerate(dists)]
marks.sort()
fids = {v: fact_ids(v, c) for _, v, c, _ in marks}
extra = sum(len(x) for x in fids.values()); body = base[: a.length - extra]
assert len(body) == a.length - extra, f"only {len(body)+extra} tokens of text for {a.length}"
ids, prev, depth_of = [], 0, {}
for f, v, c, kind in marks:
    cut = int(len(body) * f); ids += body[prev:cut]; depth_of[v] = len(ids); ids += fids[v]; prev = cut
ids += body[prev:]
if a.prompt == "chat":
    pre = tok("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    ids = pre + ids; depth_of = {k: v + len(pre) for k, v in depth_of.items()}
T0 = len(ids)
print(f"context {T0} tokens, {a.facts} targets, {a.distractors} distractors", flush=True)
# ---- resume ----
res = json.load(open(a.out)) if os.path.exists(a.out) else None
if res and res.get("complete"): print("already complete"); sys.exit(0)
GiB = 2**30
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
if not a.unpatched:
    from lsa_rocm.kernels import nemotron_patch; nemotron_patch.apply()
from transformers.cache_utils import DynamicCache
cache = DynamicCache(config=model.config)
weights = torch.cuda.memory_allocated() / GiB
def fin(x): return bool(torch.isfinite(x).all())
def lin_state_finite():
    for l in cache.layers:
        for k, v in vars(l).items():
            for t in (v if isinstance(v, (list, tuple)) else [v]):
                if torch.is_tensor(t) and t.is_floating_point() and k not in ("keys", "values") and not fin(t): return False
    return True
# warmup (autotune) with throwaway cache
if not a.unpatched:
    with torch.no_grad():
        for n in (2048, 4096, a.chunk):
            model(input_ids=torch.tensor([ids[:n]], device="cuda"), past_key_values=DynamicCache(config=model.config), use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
inp = torch.tensor([ids], device="cuda"); nonfinite = []
torch.cuda.reset_peak_memory_stats(); t0 = time.time()
with torch.no_grad():
    for s in range(0, T0, a.chunk):
        e = min(s + a.chunk, T0)
        out = model(input_ids=inp[:, s:e], position_ids=torch.arange(s, e, device="cuda")[None], past_key_values=cache, use_cache=True, logits_to_keep=1)
        if not fin(out.logits): nonfinite.append([s, e]); print(f"NONFINITE logits chunk [{s},{e})", flush=True)
        if (s // a.chunk) % 8 == 0:
            print(f"prefill {e}/{T0} {time.time()-t0:.0f}s ({e/(time.time()-t0):.0f} tok/s)", flush=True)
            if not lin_state_finite(): nonfinite.append([s, e, "state"]); print(f"NONFINITE mamba state by {e}", flush=True)
torch.cuda.synchronize(); prefill_s = time.time() - t0
if not lin_state_finite(): nonfinite.append(["final-state"])
peak = torch.cuda.max_memory_allocated() / GiB
print(f"prefill done {prefill_s:.1f}s {T0/prefill_s:.0f} tok/s peak {peak:.1f} GiB nonfinite={nonfinite}", flush=True)
del out
# ---- snapshot ----
def clone_tree(x):
    if torch.is_tensor(x): return x.clone()
    if isinstance(x, (list, tuple)): return type(x)(clone_tree(v) for v in x)
    if isinstance(x, dict): return {k: clone_tree(v) for k, v in x.items()}
    return x
snap = []
for l in cache.layers:
    snap.append({k: clone_tree(v) for k, v in vars(l).items() if k not in ("keys", "values")})
def restore():
    for l, st in zip(cache.layers, snap):
        for k, v in st.items(): setattr(l, k, clone_tree(v))
        if getattr(l, "keys", None) is not None and torch.is_tensor(l.keys) and l.keys.dim() == 4:
            l.keys = l.keys[:, :, :T0]; l.values = l.values[:, :, :T0]
# ---- questions ----
rows = (res or {}).get("results", []); done = {r["vault"] for r in rows}
tset = {c for _, c in targets}; dmap = {c: "D" for _, c in dists}
def save(complete):
    n = len(rows); k = sum(r["correct"] for r in rows if not r["nonfinite"]); nn = sum(not r["nonfinite"] for r in rows)
    br = {"distractor": 0, "other_target": 0, "neither": 0}
    for r in rows:
        if not r["correct"] and not r["nonfinite"]: br[r["wrong_kind"]] += 1
    json.dump({"length": T0, "model": a.model, "patched": not a.unpatched, "chunk": a.chunk, "prefill_s": prefill_s, "tok_s": T0 / prefill_s, "peak_gib": peak,
               "weights_gib": weights, "nonfinite_chunks": nonfinite, "facts": a.facts, "distractors": a.distractors, "accuracy": k / max(nn, 1), "correct": k, "scored": nn,
               "wrong_breakdown": br, "results": rows, "complete": complete, "torch": torch.__version__}, open(a.out, "w"), indent=1)
for vn, code in targets:
    if vn in done: continue
    restore(); t1 = time.time(); gen = []
    qtxt = {"raw": f"\n\nQuestion: What is the access code for vault {vn}?\nAnswer: The access code for vault {vn} is",
            "plain": f"\n\nThe access code for vault {vn} is",
            "chat": f"\n\nQuestion: What is the access code for vault {vn}? Answer with the code only.<|im_end|>\n<|im_start|>assistant\n<think></think>"}[a.prompt]
    qids = tok(qtxt, add_special_tokens=False)["input_ids"]
    qfin = True
    with torch.no_grad():
        pos = T0
        o = model(input_ids=torch.tensor([qids], device="cuda"), position_ids=torch.arange(pos, pos + len(qids), device="cuda")[None], past_key_values=cache, use_cache=True, logits_to_keep=1); pos += len(qids)
        for _ in range(a.max_new):
            qfin &= fin(o.logits)
            nxt = o.logits[:, -1].argmax(-1, keepdim=True); gen.append(nxt.item())
            if nxt.item() == tok.eos_token_id or "\n" in tok.decode(gen): break
            o = model(input_ids=nxt, position_ids=torch.tensor([[pos]], device="cuda"), past_key_values=cache, use_cache=True, logits_to_keep=1); pos += 1
    ans = tok.decode(gen); ok = code in ans
    wk = None
    if not ok:
        wk = "distractor" if any(c in ans for c in dmap) else "other_target" if any(c in ans for c in tset if c != code) else "neither"
    rows.append({"vault": vn, "code": code, "depth_tokens": depth_of[vn], "depth_frac": depth_of[vn] / T0, "answer": ans, "correct": ok, "wrong_kind": wk,
                 "nonfinite": (not qfin) or bool(nonfinite), "seconds": time.time() - t1})
    print(f"vault {vn} depth {depth_of[vn]/T0:.2f} {'OK ' if ok else 'MISS'} {time.time()-t1:.1f}s -> {ans[:40]!r} {wk or ''}", flush=True)
    save(False)
save(True)
print("SUMMARY", T0, f"{sum(r['correct'] for r in rows)}/{len(rows)}", flush=True)
