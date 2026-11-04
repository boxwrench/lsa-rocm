#!/usr/bin/env python3
"""Fact-at-depth (needle) test on Ornith 1.5 9B: chunked prefill, then greedy decode, scored by string match.
Haystack = an existing prose text (experiments/data/texts) cut to --length tokens; one synthetic fact is inserted at each
--depths fraction; the question is appended. Prompts are rebuilt deterministically (seed), nothing large is stored.
--mode full  : ordinary attention at decode.
(--mode lsa  : added in a later commit.)
usage: run_needle.py --length 32768 --depths 0.05,0.25,0.5,0.75,0.95 --facts 2 --out results/005-needle-full-32768.json
"""
import sys, json, time, argparse, random
sys.path[:0] = ["/ai/github/lsa-rocm/src", "/ai/github/lsa-rocm/third_party/lsa/src"]
import torch
from lsa_rocm.dump import _make_gqa_sdpa

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/ai/models/llm/Ornith-1.5-9B"); ap.add_argument("--length", type=int, default=32768)
ap.add_argument("--text", default=None); ap.add_argument("--depths", default="0.05,0.25,0.5,0.75,0.95")
ap.add_argument("--facts", type=int, default=2); ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--max-new", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mode", default="full"); ap.add_argument("--out", required=True)
a = ap.parse_args()
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import DynamicCache
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
text_path = a.text or f"/ai/github/lsa-rocm/experiments/data/texts/prose-{131072 if a.length > 32768 else 32768}.txt"
base_ids = tok(open(text_path).read(), add_special_tokens=False)["input_ids"]
rng = random.Random(a.seed)
def make_fact():
    n = rng.randint(100, 999); code = f"{rng.randint(1000,9999)}-{''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(3))}"
    return n, code
jobs = []
for d in [float(x) for x in a.depths.split(",")]:
    for f in range(a.facts):
        n, code = make_fact(); jobs.append((d, n, code))

model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa").to("cuda").eval()
gqa = _make_gqa_sdpa(128)
def attn(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kw):
    return gqa(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kw)  # decode (q_len 1) uses the same exact path
ALL_ATTENTION_FUNCTIONS["sdpa"] = attn
if a.mode != "full": raise SystemExit("only --mode full implemented so far")

results = []
for d, n, code in jobs:
    body = base_ids[: a.length - 80]
    cut = int(len(body) * d)
    pre, suf = tok.decode(body[:cut]), tok.decode(body[cut:])
    needle = f"\n\nNote for the record: the access code for vault {n} is {code}.\n\n"
    question = f"\n\nQuestion: What is the access code for vault {n}?\nAnswer: The access code for vault {n} is"
    ids = tok(pre + needle + suf + question, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
    T = ids.shape[1]; t0 = time.time()
    with torch.no_grad():
        past = DynamicCache(config=model.config)
        for s in range(0, T, a.chunk):
            out = model(input_ids=ids[:, s:s + a.chunk], position_ids=torch.arange(s, min(s + a.chunk, T), device="cuda")[None],
                        past_key_values=past, use_cache=True, logits_to_keep=1)
        pre_s = time.time() - t0; nan_after_prefill = bool(torch.isnan(out.logits).any()); gen = []; t1 = time.time()
        for i in range(a.max_new):
            nxt = out.logits[:, -1].argmax(-1, keepdim=True); gen.append(nxt.item())
            out = model(input_ids=nxt, position_ids=torch.tensor([[T + i]], device="cuda"), past_key_values=past, use_cache=True, logits_to_keep=1)
        dec_s = time.time() - t1
    ans = tok.decode(gen); ok = code in ans; degenerate = len(set(gen)) == 1 or nan_after_prefill
    results.append({"depth": d, "vault": n, "code": code, "tokens": T, "answer": ans, "correct": ok, "nan_after_prefill": nan_after_prefill, "degenerate": degenerate, "prefill_s": pre_s, "decode_s": dec_s})
    print(f"depth {d:.2f} T={T} {'OK ' if ok else ('DEGENERATE' if degenerate else 'MISS')} prefill {pre_s:.0f}s decode {dec_s:.1f}s -> {ans!r}", flush=True)
    del past; torch.cuda.empty_cache()
acc = sum(r["correct"] for r in results) / len(results)
print(f"accuracy {acc:.2f} ({sum(r['correct'] for r in results)}/{len(results)})")
json.dump({"mode": a.mode, "length": a.length, "model": a.model, "text": text_path, "accuracy": acc, "results": results, "torch": torch.__version__}, open(a.out, "w"), indent=1)
