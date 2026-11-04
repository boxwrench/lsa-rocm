#!/usr/bin/env python3
"""Fact-at-depth (needle) test on Ornith 1.5 9B: chunked prefill, then greedy decode, scored by string match.
Haystack = an existing prose text (experiments/data/texts) cut to --length tokens; one synthetic fact is inserted at each
--depths fraction; the question is appended. Prompts are rebuilt deterministically (seed), nothing large is stored.
--mode full  : ordinary attention at decode.
--mode lsa   : decode steps on the 8 full-attention layers read only sinks, a window and k keys picked by the layer-shared PCA index (flat).
usage: run_needle.py --length 32768 --depths 0.05,0.25,0.5,0.75,0.95 --facts 2 --out results/005-needle-full-32768.json
"""
import sys, json, time, argparse, random
sys.path[:0] = ["/ai/github/lsa-rocm/src", "/ai/github/lsa-rocm/third_party/lsa/src"]
import torch
from lsa_rocm.dump import _make_gqa_sdpa, _full_attention_indices
from lsa_rocm.oracle import OracleConfig, build_flat_index
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/ai/models/llm/Ornith-1.5-9B"); ap.add_argument("--length", type=int, default=32768)
ap.add_argument("--text", default=None); ap.add_argument("--depths", default="0.05,0.25,0.5,0.75,0.95")
ap.add_argument("--facts", type=int, default=2); ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--max-new", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--k", type=int, default=2048); ap.add_argument("--only-index", type=int, default=None); ap.add_argument("--mode", default="full"); ap.add_argument("--out", required=True)
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

if a.only_index is not None: jobs = [jobs[a.only_index]]
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa").to("cuda").eval()
gqa = _make_gqa_sdpa(128)
FULL = set(_full_attention_indices(model.config)[0])
tcfg = getattr(model.config, "text_config", model.config)
ROT = int(tcfg.head_dim * tcfg.partial_rotary_factor)
SINK, WIN, CAP = 4, 512 + 32, 16
sparse_state = {}   # layer -> index built lazily from the prefilled cache; cleared per prompt
def build_layer_index(key, Tp):
    """key [1,Hkv,Tp+1,D] post-RoPE cache. Layer-shared PCA(64) index over RoPE-free dims, clustered in blocks of 512."""
    Hkv, D = key.shape[1], key.shape[3]
    raw = key[0, :, :Tp, ROT:].float().permute(1, 0, 2).reshape(Tp, -1)
    mean = raw.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(raw - mean, q=64, center=False); basis = V[:, :64].T.contiguous()
    idxk = F.normalize((raw - mean) @ basis.T, dim=-1)
    ncl = (Tp // 512) * 512
    cfg = OracleConfig(pool_size=4, bucket_size=8, bucket_cap=CAP, cluster_block_size=512)
    b = build_flat_index(idxk[:ncl].contiguous().cpu(), cfg).buckets
    return {"mean": mean, "basis": basis, "cent": b.bucket_centroids.cuda(), "mem": b.bucket_members.long().cuda()}
def attn(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kw):
    layer = getattr(module, "layer_idx", None)
    if a.mode == "full" or query.shape[2] != 1 or layer not in FULL:
        return gqa(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kw)
    B, Hq, _, D = query.shape; Hkv = key.shape[1]; G = Hq // Hkv; Tk = key.shape[2]; p = Tk - 1
    if layer not in sparse_state: sparse_state[layer] = dict(build_layer_index(key, Tk - 1), Tp=Tk - 1)
    st = sparse_state[layer]
    q = query[0, :, 0].view(Hkv, G, D)
    qi = F.normalize(((q.float().sum(1)[:, ROT:].reshape(1, -1)) - st["mean"]) @ st["basis"].T, dim=-1)[0]
    sc = (st["cent"] @ qi).masked_fill(st["mem"][:, 0] > p - WIN, -1e4)
    top = sc.topk(a.k // 4).indices
    m = st["mem"][top].flatten(); ok = m >= 0
    keep = ok & (ok.cumsum(0) <= a.k) & (m <= p - WIN)
    ar = torch.arange(WIN, device=key.device)
    idx = torch.cat([m.clamp(min=0), torch.arange(SINK, device=key.device), (p + 1 - WIN + ar).clamp(min=0)])
    valid = torch.cat([keep, torch.ones(SINK + WIN, dtype=torch.bool, device=key.device)]) & (idx <= p)
    kk = key[0].index_select(1, idx); vv = value[0].index_select(1, idx)
    o = F.scaled_dot_product_attention(q.unsqueeze(0), kk.unsqueeze(0), vv.unsqueeze(0), attn_mask=valid.view(1, 1, 1, -1), scale=scaling)
    return o.reshape(1, Hq, 1, -1).transpose(1, 2).contiguous(), None
ALL_ATTENTION_FUNCTIONS["sdpa"] = attn
if a.mode not in ("full", "lsa"): raise SystemExit("--mode must be full or lsa")

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
    del past; sparse_state.clear(); torch.cuda.empty_cache()
acc = sum(r["correct"] for r in results) / len(results)
print(f"accuracy {acc:.2f} ({sum(r['correct'] for r in results)}/{len(results)})")
json.dump({"mode": a.mode, "length": a.length, "model": a.model, "text": text_path, "accuracy": acc, "results": results, "torch": torch.__version__}, open(a.out, "w"), indent=1)
