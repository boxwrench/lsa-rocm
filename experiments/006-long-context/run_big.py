#!/usr/bin/env python3
"""Needle test at very long context with the KV cache of the full-attention layers resident in CPU RAM.

One dense chunked prefill of a haystack with planted facts and distractors (K/V of the 8 full-attention layers moved to CPU RAM
after every layer update), saved to disk so a GPU fault does not cost the prefill. Then many questions, each decoded from the
same cache under different attention paths on the 8 full-attention layers:
  full:none:0    exact attention, streaming the whole CPU cache through the GPU in chunks (online softmax). The reference.
  lsa:none:K     sinks + last 544 prefix keys + new tokens + K keys picked by the layer-shared PCA(64) flat index; only those keys are
                 gathered from CPU RAM to the GPU.
  bsp:none:K     same with block-sparse selection (means of 4-key blocks of raw RoPE-free keys).
The cache is shared and read-only: a question's own tokens are held in a small GPU tail; the small linear-attention states are
snapshotted and restored around each question. Question tokens are fed one at a time through the decode path.

usage: run_big.py --length 131072 --text A.txt[,B.txt] --facts 8 --distractors 20 --configs full:none:0,lsa:none:2048 --out results/006-x.json
"""
import sys, os, json, time, argparse, random, copy
sys.path[:0] = ["/ai/github/lsa-rocm/src", "/ai/github/lsa-rocm/third_party/lsa/src"]
import torch, torch.nn.functional as F
from lsa_rocm.dump import _make_gqa_sdpa, _full_attention_indices, _apply_rope_yarn, _OffloadingDynamicCache
from lsa_rocm.oracle import OracleConfig, build_flat_index

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/ai/models/llm/Ornith-1.5-9B"); ap.add_argument("--length", type=int, default=131072)
ap.add_argument("--text", required=True, help="comma-separated text files, concatenated in order to fill the haystack")
ap.add_argument("--facts", type=int, default=8); ap.add_argument("--distractors", type=int, default=20)
ap.add_argument("--chunk", type=int, default=1024); ap.add_argument("--max-new", type=int, default=16)
ap.add_argument("--rope-yarn", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--configs", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cache-dir", default=None, help="save/load the prefilled cache here (default results/dumps/big-cache-<length>)")
ap.add_argument("--split-gpus", action="store_true", help="prefill with the model split over cuda:0 and cuda:1, KV kept in VRAM (needs both visible)")
ap.add_argument("--question-full", action="store_true", help="feed the question tokens with exact (streamed) attention; only the answer tokens use the config"); ap.add_argument("--slow-prefill-attn", action="store_true", help="use the old masked folded-GQA attention instead of the flash prefix + diagonal path"); ap.add_argument("--prefill-only", action="store_true"); ap.add_argument("--stream-chunk", type=int, default=65536); ap.add_argument("--max-questions", type=int, default=0)
a = ap.parse_args()
cache_dir = a.cache_dir or f"/ai/github/lsa-rocm/results/dumps/big-cache-{a.length}"
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import DynamicLayer

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
rng = random.Random(a.seed)
def code_str(r): return f"{r.randint(1000,9999)}-{''.join(r.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(3))}"
def fact_ids(vn, vc): return tok(f"\n\nNote for the record: the access code for vault {vn} is {vc}.\n\n", add_special_tokens=False)["input_ids"]

# ---- haystack with planted facts (token level, exact length) --------------------------------------------------------------
base = []
for tp in a.text.split(","):
    base += tok(open(tp).read(), add_special_tokens=False)["input_ids"]
vaults = rng.sample(range(100, 1000), a.facts + a.distractors)
targets = [(vaults[i], code_str(rng)) for i in range(a.facts)]
dists = [(vaults[a.facts + i], code_str(rng)) for i in range(a.distractors)]
marks = [(0.03 + 0.94 * (i + 0.5) / a.facts, v, c, "T") for i, (v, c) in enumerate(targets)]
marks += [(0.03 + 0.94 * (i + 0.25) / a.distractors, v, c, "D") for i, (v, c) in enumerate(dists)]
marks.sort()
fids = {(v): fact_ids(v, c) for _, v, c, _ in marks}
extra = sum(len(x) for x in fids.values())
body = base[: a.length - extra]
if len(body) < a.length - extra: print(f"WARNING: only {len(body) + extra} tokens available for requested {a.length}", flush=True)
ids, prev, depth_of = [], 0, {}
for f, v, c, kind in marks:
    cut = int(len(body) * f); ids += body[prev:cut]; depth_of[v] = len(ids); ids += fids[v]; prev = cut
ids += body[prev:]; T0 = len(ids)
print(f"context {T0} tokens, {a.facts} targets, {a.distractors} distractors", flush=True)

# ---- model ------------------------------------------------------------------------------------------------------------
load_cfg = None
if a.rope_yarn > 0:
    load_cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True); print("YARN", _apply_rope_yarn(load_cfg, a.rope_yarn)["requested_rope_parameters"], flush=True)
have_cache = os.path.exists(f"{cache_dir}/meta.json") and json.load(open(f"{cache_dir}/meta.json")).get("tokens") == T0 and json.load(open(f"{cache_dir}/meta.json")).get("seed") == a.seed
if a.split_gpus and not have_cache:
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa", config=load_cfg,
                                                 device_map="auto", max_memory={0: "9.5GiB", 1: "12GiB"}).eval()
    print("device map:", sorted(set(map(str, model.hf_device_map.values()))), flush=True)
else:
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa", config=load_cfg).to("cuda").eval()
FULL = sorted(_full_attention_indices(model.config)[0]); FULLS = set(FULL)
tcfg = getattr(model.config, "text_config", model.config)
ROT = int(tcfg.head_dim * tcfg.partial_rotary_factor); SINK, WINP, CAP = 4, 544, 16
from lsa_rocm.fastattn import make_fast_prefill_attn
gqa = _make_gqa_sdpa(128) if a.slow_prefill_attn else make_fast_prefill_attn()

class BigCache(_OffloadingDynamicCache):
    """Prefill: parent behaviour (K/V of full layers bounce through the GPU, rest on CPU). Decode: full layers keep only a small GPU tail."""
    def __init__(self, config):
        super().__init__(config); self.decode = False; self.tail = {}
    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if not self.decode: return super().update(key_states, value_states, layer_idx, *args, **kwargs)
        k0, v0 = self.tail.get(layer_idx, (None, None))
        k = key_states if k0 is None else torch.cat([k0, key_states], 2); v = value_states if v0 is None else torch.cat([v0, value_states], 2)
        self.tail[layer_idx] = (k, v); return k, v

def cpu_tree(x):
    if torch.is_tensor(x): return x.detach().cpu()
    if isinstance(x, dict): return {k: cpu_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(cpu_tree(v) for v in x)
    return x
def layer_state(layer):
    """Everything a cache layer holds (tensors, dicts of tensors, flags) so a linear-attention layer is restored exactly."""
    return {k: cpu_tree(v) for k, v in vars(layer).items() if k != "device"}
def to_dev(x, dev):
    if torch.is_tensor(x): return x.to(dev)
    if isinstance(x, dict): return {k: to_dev(v, dev) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(to_dev(v, dev) for v in x)
    return x
def set_layer_state(layer, st, dev=None):
    for k, v in copy.deepcopy(st).items(): setattr(layer, k, to_dev(v, dev) if dev else v)

# ---- prefill (or load) ------------------------------------------------------------------------------------------------
cur = {"mode": "full", "k": 2048, "quant": "none"}; STATE = {}
def attn_prefill(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kw):
    return gqa(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kw)
ALL_ATTENTION_FUNCTIONS["sdpa"] = attn_prefill
if a.split_gpus and not have_cache:
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache(config=model.config); inner = cache      # K/V stay on each layer's GPU (no CPU bounce)
else:
    cache = BigCache(model.config); inner = cache._inner
meta_path = f"{cache_dir}/meta.json"; t_pre = 0.0
if have_cache:
    t0 = time.time()
    for i, layer in enumerate(inner.layers):
        set_layer_state(layer, torch.load(f"{cache_dir}/layer_{i:02d}.pt", map_location="cpu", weights_only=False), None if i in FULLS else "cuda")
    print(f"loaded prefilled cache from {cache_dir} in {time.time()-t0:.0f}s", flush=True)
else:
    t0 = time.time(); d0 = "cuda:0" if a.split_gpus else "cuda"; inp = torch.tensor([ids], device=d0)
    with torch.no_grad():
        for s in range(0, T0, a.chunk):
            e = min(s + a.chunk, T0)
            out = model(input_ids=inp[:, s:e], position_ids=torch.arange(s, e, device=d0)[None], past_key_values=cache, use_cache=True, logits_to_keep=1)
            if (s // a.chunk) % 50 == 0: print(f"prefill {e}/{T0} {time.time()-t0:.0f}s", flush=True)
    if not bool(torch.isfinite(out.logits).all()): raise SystemExit("non-finite logits after prefill: rerun")
    t_pre = time.time() - t0; print(f"prefill done {t_pre:.0f}s", flush=True)
    os.makedirs(cache_dir, exist_ok=True)
    for i, layer in enumerate(inner.layers): torch.save(layer_state(layer), f"{cache_dir}/layer_{i:02d}.pt")
    json.dump({"tokens": T0, "seed": a.seed, "prefill_s": t_pre}, open(meta_path, "w"))
    del out
    if a.prefill_only: raise SystemExit(0)

# ---- per-layer decode-time structures ---------------------------------------------------------------------------------
# CPU-resident prefix K/V as pinned, head-contiguous [Hkv,T0,D] tensors: streaming then runs near PCIe speed (pageable, strided copies were ~2 GB/s)
Kc, Vc = {}, {}
for l in FULL:
    Kc[l] = inner.layers[l].keys[0].contiguous().pin_memory(); Vc[l] = inner.layers[l].values[0].contiguous().pin_memory()
    stub = torch.zeros(1, dtype=torch.bfloat16).expand(1, *Kc[l].shape)   # shape-only stand-in (no memory) so get_seq_length() still reports T0
    inner.layers[l].keys = stub; inner.layers[l].values = stub
assert all(Kc[l].shape[1] == T0 for l in FULL), "cache length mismatch"
Hkv, D = Kc[FULL[0]].shape[0], Kc[FULL[0]].shape[2]
WK = {l: Kc[l][:, T0 - WINP:].cuda() for l in FULL}; WV = {l: Vc[l][:, T0 - WINP:].cuda() for l in FULL}
SK = {l: Kc[l][:, :SINK].cuda() for l in FULL}; SV = {l: Vc[l][:, :SINK].cuda() for l in FULL}

def raw_chunks(l, step=65536):
    for s in range(0, T0, step):
        yield s, Kc[l][:, s:s + step, ROT:].cuda().float().permute(1, 0, 2).reshape(-1, Hkv * (D - ROT))

def build_lsa(l):
    n = 0; ssum = torch.zeros(Hkv * (D - ROT), device="cuda", dtype=torch.float64)
    for _, r in raw_chunks(l): ssum += r.double().sum(0); n += r.shape[0]
    mean = (ssum / n).float()[None]; cov = torch.zeros(ssum.shape[0], ssum.shape[0], device="cuda", dtype=torch.float64)
    for _, r in raw_chunks(l): x = (r - mean).double(); cov += x.T @ x
    _, evec = torch.linalg.eigh(cov / n); basis = evec[:, -64:].flip(1).T.float().contiguous()
    idxk = torch.cat([F.normalize((r - mean) @ basis.T, dim=-1).cpu() for _, r in raw_chunks(l)])
    ncl = (T0 // 512) * 512
    b = build_flat_index(idxk[:ncl].contiguous(), OracleConfig(pool_size=4, bucket_size=8, bucket_cap=CAP, cluster_block_size=512)).buckets
    return {"mean": mean, "basis": basis, "cent": b.bucket_centroids.cuda(), "mem": b.bucket_members.long().cuda()}

def build_bsp(l):
    blks = []
    for s, r in raw_chunks(l, 65536):
        m = (r.shape[0] // 4) * 4; blks.append(F.normalize(r[:m].reshape(-1, 4, r.shape[1]).mean(1), dim=-1).to(torch.bfloat16))
    return {"blk": torch.cat(blks)}

def ensure_index(mode):
    for l in FULL:
        key = (mode, l)
        if key not in STATE:
            t = time.time(); STATE[key] = build_lsa(l) if mode == "lsa" else build_bsp(l)
            print(f"  built {mode} index layer {l} in {time.time()-t:.1f}s", flush=True)

def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / n ** 0.5
Hm = hadamard(D).cuda(); signs = (torch.randint(0, 2, (D,), generator=torch.Generator().manual_seed(7)) * 2 - 1).float().cuda()
def qdq(x, quant):
    """x [Hkv,S,D] -> quantize-dequantize per token and head (simulates a quantized cache read)."""
    if quant == "none": return x
    bits = 8 if quant == "int8" else 4; y = x.float()
    if quant == "int4rht": y = (y * signs) @ Hm
    lo = y.amin(-1, keepdim=True); hi = y.amax(-1, keepdim=True); sc = ((hi - lo) / (2 ** bits - 1)).clamp(min=1e-8)
    y = ((y - lo) / sc).round() * sc + lo
    if quant == "int4rht": y = (y @ Hm.T) * signs
    return y.to(x.dtype)

def gather(l, idx):
    """idx: 1-D long tensor of prefix positions (GPU) -> K,V [Hkv,S,D] on GPU, read from the CPU cache."""
    ic = idx.cpu(); return Kc[l].index_select(1, ic).cuda(non_blocking=True), Vc[l].index_select(1, ic).cuda(non_blocking=True)

def attend(q, k, v, valid, scaling):
    G = q.shape[1]
    o = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attn_mask=valid.view(1, 1, 1, -1), scale=scaling)
    return o

def stream_full(l, q, tk, tv, scaling):
    """Exact attention over the whole prefix (streamed from CPU in chunks) plus the GPU tail, online softmax."""
    m = l_ = o = None
    for s in range(0, T0, a.stream_chunk):
        e = min(s + a.stream_chunk, T0)
        k = torch.empty(Hkv, e - s, D, dtype=torch.bfloat16, device="cuda"); v = torch.empty_like(k)
        for h in range(Hkv):   # one contiguous pinned slice per head -> a plain DMA copy
            k[h].copy_(Kc[l][h, s:e], non_blocking=True); v[h].copy_(Vc[l][h, s:e], non_blocking=True)
        k = qdq(k, cur["quant"]); v = qdq(v, cur["quant"])
        m, l_, o = fold(q, k, v, scaling, m, l_, o)
    m, l_, o = fold(q, tk[0], tv[0], scaling, m, l_, o)
    return (o / l_).to(q.dtype)

def fold(q, k, v, scaling, m, l_, o):
    sc = torch.matmul(q.float(), k.float().transpose(1, 2)) * scaling          # [Hkv,G,c]
    mc = sc.amax(-1, keepdim=True); pc = torch.exp(sc - mc); lc = pc.sum(-1, keepdim=True); oc = torch.matmul(pc, v.float())
    if m is None: return mc, lc, oc
    mn = torch.maximum(m, mc); a1 = torch.exp(m - mn); a2 = torch.exp(mc - mn)
    return mn, l_ * a1 + lc * a2, o * a1 + oc * a2

def attn_decode(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kw):
    layer = getattr(module, "layer_idx", None)
    if not cache.decode or layer not in FULLS:
        return gqa(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kw)
    B, Hq, _, Dh = query.shape; G = Hq // Hkv; q = query[0, :, 0].view(Hkv, G, Dh)
    tk, tv = key, value                       # the GPU tail: question and answer tokens so far, including this one
    mode, k = cur["mode"], cur["k"]; tl = tk.shape[2]
    if mode == "full":
        o = stream_full(layer, q, tk, tv, scaling)
        return o.reshape(1, Hq, 1, -1).transpose(1, 2).contiguous().to(query.dtype), None
    st = STATE[(mode, layer)]; dev = q.device; limit = T0 - WINP   # long-range candidates are prefix positions < limit
    if mode == "lsa":
        qi = F.normalize(((q.float().sum(1)[:, ROT:].reshape(1, -1)) - st["mean"]) @ st["basis"].T, dim=-1)[0]
        sc = (st["cent"] @ qi).masked_fill(st["mem"][:, 0] >= limit, -1e4)
        top = sc.topk(max(k // 4, 1)).indices; mm = st["mem"][top].flatten(); ok = mm >= 0
        keep = ok & (ok.cumsum(0) <= k) & (mm < limit); idx = mm.clamp(min=0)
    else:
        rq = F.normalize(q.float().sum(1)[:, ROT:].reshape(-1), dim=-1).to(torch.bfloat16); nb = st["blk"].shape[0]
        sc = (st["blk"] @ rq).float().masked_fill(torch.arange(nb, device=dev) * 4 >= limit, -1e4)
        top = sc.topk(max(k // 4, 1)).indices; idx = (top[:, None] * 4 + torch.arange(4, device=dev)).flatten(); keep = idx < limit
    gk, gv = gather(layer, idx); qn = cur["quant"]
    K_all = torch.cat([qdq(SK[layer], qn), qdq(gk, qn), qdq(WK[layer], qn), tk[0]], 1); V_all = torch.cat([qdq(SV[layer], qn), qdq(gv, qn), qdq(WV[layer], qn), tv[0]], 1)
    valid = torch.cat([torch.ones(SINK, dtype=torch.bool, device=dev), keep, torch.ones(WINP + tl, dtype=torch.bool, device=dev)])
    o = attend(q, K_all, V_all, valid, scaling)
    return o.reshape(1, Hq, 1, -1).transpose(1, 2).contiguous().to(query.dtype), None
ALL_ATTENTION_FUNCTIONS["sdpa"] = attn_decode

# ---- questions ---------------------------------------------------------------------------------------------------------
configs = []
for c in a.configs.split(","):
    p = c.split(":"); configs.append((p[0], int(p[2]) if len(p) > 2 else 0, p[1] if len(p) > 1 else "none"))
for mode in {m for m, _, _ in configs if m != "full"}: ensure_index(mode)
lin_ids = [i for i in range(len(inner.layers)) if i not in FULLS]
snap = {i: layer_state(inner.layers[i]) for i in lin_ids}   # complete linear-attention state after the context (tensors, dicts, flags)
results = []; qs = targets[: a.max_questions] if a.max_questions else targets
for vn, code in qs:
    qids = tok(f"\n\nQuestion: What is the access code for vault {vn}?\nAnswer: The access code for vault {vn} is", add_special_tokens=False)["input_ids"]
    row = {"vault": vn, "code": code, "depth_tokens": depth_of[vn], "depth_frac": depth_of[vn] / T0, "configs": {}}
    for mode, k, qn in configs:
        name = (f"{mode}:{qn}:{k}" if mode != "full" else f"full:{qn}")
        for i in lin_ids: set_layer_state(inner.layers[i], snap[i], "cuda")
        cache.decode = True; cache.tail = {}; cur.update(mode=mode, k=k, quant=qn); t1 = time.time(); gen = []
        with torch.no_grad():
            pos = T0; nxt = None
            for t in qids:
                if a.question_full: cur.update(mode="full", quant=qn)
                o = model(input_ids=torch.tensor([[t]], device="cuda"), position_ids=torch.tensor([[pos]], device="cuda"), past_key_values=cache, use_cache=True, logits_to_keep=1); pos += 1
            cur.update(mode=mode, k=k, quant=qn)
            for _ in range(a.max_new):
                nxt = o.logits[:, -1].argmax(-1, keepdim=True); gen.append(nxt.item())
                o = model(input_ids=nxt, position_ids=torch.tensor([[pos]], device="cuda"), past_key_values=cache, use_cache=True, logits_to_keep=1); pos += 1
        cache.decode = False
        ans = tok.decode(gen); ok = code in ans
        row["configs"][name] = {"answer": ans, "correct": ok, "seconds": time.time() - t1}
        print(f"vault {vn} depth {depth_of[vn]/T0:.2f} {name:10s} {'OK ' if ok else 'MISS'} {time.time()-t1:5.1f}s -> {ans[:34]!r}", flush=True)
    results.append(row)
    json.dump({"length": T0, "text": a.text, "rope_yarn": a.rope_yarn, "question_full": a.question_full, "prefill_s": t_pre, "facts": a.facts, "distractors": a.distractors,
               "configs": [list(c) for c in configs], "results": results, "torch": torch.__version__}, open(a.out, "w"), indent=1)
summ = {}
for r in results:
    for n, v in r["configs"].items(): summ.setdefault(n, []).append(v["correct"])
for n, v in summ.items(): print(f"SUMMARY {n:10s} {sum(v)}/{len(v)}")
