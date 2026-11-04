#!/usr/bin/env python3
"""Needle test, one prefill per prompt, many decode configurations on a copy of the cache.
A config is MODE:QUANT[:K], MODE full|lsa, QUANT none|int8|int4|int4rht (simulated: quantize then dequantize the cached
K/V of the 8 full-attention layers, per token and head, asymmetric; int4rht rotates with a random-sign Hadamard first),
K the long-range key budget for lsa. The LSA index is built from the unquantized keys before the cache is quantized,
because a real system would index keys as they are produced and store them quantized.
usage: run_matrix.py --length 131072 --configs full:none,lsa:none:2048,lsa:int4:2048 --facts 2 --out results/005-matrix-131072.json
"""
import sys, json, time, argparse, random, copy
sys.path[:0] = ["/ai/github/lsa-rocm/src", "/ai/github/lsa-rocm/third_party/lsa/src"]
import torch, torch.nn.functional as F
from lsa_rocm.dump import _make_gqa_sdpa, _full_attention_indices, _apply_rope_yarn
from lsa_rocm.oracle import OracleConfig, build_flat_index

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/ai/models/llm/Ornith-1.5-9B"); ap.add_argument("--length", type=int, default=131072)
ap.add_argument("--depths", default="0.05,0.25,0.5,0.75,0.95"); ap.add_argument("--facts", type=int, default=2)
ap.add_argument("--chunk", type=int, default=2048); ap.add_argument("--max-new", type=int, default=16)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--configs", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--only-index", type=int, default=None); ap.add_argument("--text", default=None); ap.add_argument("--distractors", type=int, default=0); ap.add_argument("--multi", type=int, default=1); ap.add_argument("--rope-yarn", type=float, default=0.0)
a = ap.parse_args()
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import DynamicCache
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
text_path = a.text or f"/ai/github/lsa-rocm/experiments/data/texts/prose-{131072 if a.length > 32768 else 32768}.txt"
base_ids = tok(open(text_path).read(), add_special_tokens=False)["input_ids"]
rng = random.Random(a.seed); jobs = []
for d in [float(x) for x in a.depths.split(",")]:
    for _ in range(a.facts):
        jobs.append((d, rng.randint(100, 999), f"{rng.randint(1000,9999)}-{''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(3))}"))
if a.only_index is not None: jobs = [jobs[a.only_index]]
configs = []
for c in a.configs.split(","):
    p = c.split(":"); configs.append((p[0], p[1], int(p[2]) if len(p) > 2 else 2048))

load_cfg = None
if a.rope_yarn > 0:
    from transformers import AutoConfig
    load_cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True); print("YARN", _apply_rope_yarn(load_cfg, a.rope_yarn), flush=True)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa", config=load_cfg).to("cuda").eval()
gqa = _make_gqa_sdpa(128)
FULL = set(_full_attention_indices(model.config)[0]); tcfg = getattr(model.config, "text_config", model.config)
ROT = int(tcfg.head_dim * tcfg.partial_rotary_factor); SINK, WIN, CAP = 4, 512 + 32, 16
cur = {"mode": "full", "k": 2048}; sparse_state = {}

def build_layer_index(key, Tp):
    raw = key[0, :, :Tp, ROT:].float().permute(1, 0, 2).reshape(Tp, -1)
    if not torch.isfinite(raw).all(): raise RuntimeError("non-finite keys")
    mean = raw.mean(0, keepdim=True); x = (raw - mean).double(); cov = x.T @ x / x.shape[0]
    _, evec = torch.linalg.eigh(cov); basis = evec[:, -64:].flip(1).T.float().contiguous()
    idxk = F.normalize((raw - mean) @ basis.T, dim=-1); ncl = (Tp // 512) * 512
    cfg = OracleConfig(pool_size=4, bucket_size=8, bucket_cap=CAP, cluster_block_size=512)
    b = build_flat_index(idxk[:ncl].contiguous().cpu(), cfg).buckets
    return {"mean": mean, "basis": basis, "cent": b.bucket_centroids.cuda(), "mem": b.bucket_members.long().cuda()}

def build_bsp(key, Tp):
    raw = key[0, :, :Tp, ROT:].float().permute(1, 0, 2).reshape(Tp, -1); nb = Tp // 4
    return {"blk": F.normalize(raw[: nb * 4].reshape(nb, 4, -1).mean(1), dim=-1).to(torch.bfloat16)}

def attn(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kw):
    layer = getattr(module, "layer_idx", None)
    if cur["mode"] == "full" or query.shape[2] != 1 or layer not in FULL:
        return gqa(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kw)
    B, Hq, _, D = query.shape; Hkv = key.shape[1]; G = Hq // Hkv; p = key.shape[2] - 1; st = sparse_state[layer]; k = cur["k"]
    q = query[0, :, 0].view(Hkv, G, D)
    if cur["mode"] == "bsp":   # block-sparse control: score mean raw keys of 4-key blocks, take the top blocks
        rq = F.normalize(q.float().sum(1)[:, ROT:].reshape(-1), dim=-1).to(torch.bfloat16); nb = st["blk"].shape[0]
        sc = (st["blk"] @ rq).float().masked_fill(torch.arange(nb, device=key.device) * 4 > p - WIN, -1e4)
        top = sc.topk(max(k // 4, 1)).indices
        idx = torch.cat([(top[:, None] * 4 + torch.arange(4, device=key.device)).flatten(), torch.arange(SINK, device=key.device),
                         (p + 1 - WIN + torch.arange(WIN, device=key.device)).clamp(min=0)])
        valid = torch.cat([(top[:, None] * 4 + torch.arange(4, device=key.device)).flatten() <= p - WIN,
                           torch.ones(SINK + WIN, dtype=torch.bool, device=key.device)]) & (idx <= p)
        o = F.scaled_dot_product_attention(q.unsqueeze(0), key[0].index_select(1, idx).unsqueeze(0), value[0].index_select(1, idx).unsqueeze(0),
                                           attn_mask=valid.view(1, 1, 1, -1), scale=scaling)
        return o.reshape(1, Hq, 1, -1).transpose(1, 2).contiguous(), None
    qi = F.normalize(((q.float().sum(1)[:, ROT:].reshape(1, -1)) - st["mean"]) @ st["basis"].T, dim=-1)[0]
    sc = (st["cent"] @ qi).masked_fill(st["mem"][:, 0] > p - WIN, -1e4)
    top = sc.topk(max(k // 4, 1)).indices; m = st["mem"][top].flatten(); ok = m >= 0
    keep = ok & (ok.cumsum(0) <= k) & (m <= p - WIN)
    idx = torch.cat([m.clamp(min=0), torch.arange(SINK, device=key.device), (p + 1 - WIN + torch.arange(WIN, device=key.device)).clamp(min=0)])
    valid = torch.cat([keep, torch.ones(SINK + WIN, dtype=torch.bool, device=key.device)]) & (idx <= p)
    o = F.scaled_dot_product_attention(q.unsqueeze(0), key[0].index_select(1, idx).unsqueeze(0), value[0].index_select(1, idx).unsqueeze(0),
                                       attn_mask=valid.view(1, 1, 1, -1), scale=scaling)
    return o.reshape(1, Hq, 1, -1).transpose(1, 2).contiguous(), None
ALL_ATTENTION_FUNCTIONS["sdpa"] = attn

def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / n ** 0.5
Hm = hadamard(tcfg.head_dim).cuda(); signs = (torch.randint(0, 2, (tcfg.head_dim,), generator=torch.Generator().manual_seed(7)) * 2 - 1).float().cuda()
def qdq(x, quant):
    """x [1,Hkv,T,D] bf16 -> quantize-dequantize per token and head."""
    if quant == "none": return x
    bits = 8 if quant == "int8" else 4; out = torch.empty_like(x)
    for s in range(0, x.shape[2], 16384):
        y = x[:, :, s:s + 16384].float()
        if quant == "int4rht": y = (y * signs) @ Hm
        lo = y.amin(-1, keepdim=True); hi = y.amax(-1, keepdim=True); sc = ((hi - lo) / (2 ** bits - 1)).clamp(min=1e-8)
        y = ((y - lo) / sc).round() * sc + lo
        if quant == "int4rht": y = (y @ Hm.T) * signs
        out[:, :, s:s + 16384] = y.to(x.dtype)
    return out

results = []
for d, n, code in jobs:
    body = base_ids[: a.length - 80]; cut = int(len(body) * d)
    marks = [(cut, n, code)]; targets = [(n, code)]
    mrng = random.Random(a.seed * 77 + n)
    for j in range(1, a.multi):   # extra target facts, other depths; the question asks for all of them
        tn = mrng.choice([x for x in range(100, 1000) if x != n]); tc = f"{mrng.randint(1000,9999)}-{''.join(mrng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(3))}"
        marks.append((int(len(body) * min(max(d + (j - 0.5 * (a.multi - 1)) * 0.15, 0.03), 0.97)), tn, tc)); targets.append((tn, tc))
    drng = random.Random(a.seed * 1000 + n)
    for j in range(a.distractors):   # similar facts about other vaults, spread over the whole text
        vn = drng.choice([x for x in range(100, 1000) if x != n]); marks.append((int(len(body) * (j + 0.5) / a.distractors), vn, f"{drng.randint(1000,9999)}-{''.join(drng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(3))}"))
    marks.sort(); pieces = []; prev = 0
    for c_, vn, vc in marks:
        pieces.append(tok.decode(body[prev:c_])); pieces.append(f"\n\nNote for the record: the access code for vault {vn} is {vc}.\n\n"); prev = c_
    pieces.append(tok.decode(body[prev:]))
    text = "".join(pieces) + \
        (f"\n\nQuestion: What is the access code for vault {n}?\nAnswer: The access code for vault {n} is" if a.multi == 1 else
         f"\n\nQuestion: What are the access codes for vaults {', '.join(str(t[0]) for t in targets)}?\nAnswer: The access codes are")
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda(); T = ids.shape[1]; faults = 0
    for attempt in range(1, 4):
        try:
            with torch.no_grad():
                t0 = time.time(); past = DynamicCache(config=model.config); cur["mode"] = "full"
                for s in range(0, T - 1, a.chunk):   # all but the last prompt token: that token goes through each config's own decode path
                    e = min(s + a.chunk, T - 1)
                    out = model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device="cuda")[None],
                                past_key_values=past, use_cache=True, logits_to_keep=1)
                if not bool(torch.isfinite(out.logits).all()): raise RuntimeError("non-finite logits")
            break
        except RuntimeError as e:
            if "non-finite" not in str(e): raise
            faults += 1; print(f"  FAULT attempt {attempt}: {e}", flush=True); torch.cuda.empty_cache()
    pre_s = time.time() - t0
    row = {"depth": d, "vault": n, "code": code, "tokens": T, "prefill_s": pre_s, "faults_retried": faults, "configs": {}}
    for mode, quant, k in configs:
        name = f"{mode}:{quant}" + (f":{k}" if mode != "full" else "")
        pc = copy.deepcopy(past); sparse_state.clear(); cur.update(mode=mode, k=k)
        with torch.no_grad():
            if mode == "lsa":
                for l in FULL: sparse_state[l] = build_layer_index(pc.layers[l].keys, T - 1)   # index from unquantized keys
            if mode == "bsp":
                for l in FULL: sparse_state[l] = build_bsp(pc.layers[l].keys, T - 1)
            for l in FULL:
                pc.layers[l].keys = qdq(pc.layers[l].keys, quant); pc.layers[l].values = qdq(pc.layers[l].values, quant)
            gen = []; t1 = time.time()
            o = model(input_ids=ids[:, -1:], position_ids=torch.tensor([[T - 1]], device="cuda"), past_key_values=pc, use_cache=True, logits_to_keep=1)
            for i in range(a.max_new):
                nxt = o.logits[:, -1].argmax(-1, keepdim=True); gen.append(nxt.item())
                o = model(input_ids=nxt, position_ids=torch.tensor([[T + i]], device="cuda"), past_key_values=pc, use_cache=True, logits_to_keep=1)
        ans = tok.decode(gen); found = sum(c in ans for _, c in targets); ok = found == len(targets)
        row["configs"][name] = {"answer": ans, "correct": ok, "found": found, "n_targets": len(targets), "decode_s": time.time() - t1}
        print(f"depth {d:.2f} {name:18s} {'OK ' if ok else 'MISS'} -> {ans[:40]!r}", flush=True)
        del pc; sparse_state.clear(); torch.cuda.empty_cache()
    results.append(row); del past; torch.cuda.empty_cache()
    json.dump({"length": a.length, "configs": [list(c) for c in configs], "results": results}, open(a.out, "w"), indent=1)
summ = {}
for r in results:
    for name, v in r["configs"].items(): summ.setdefault(name, []).append(v["correct"])
for name, v in summ.items(): print(f"SUMMARY {name:18s} {sum(v)}/{len(v)}")
