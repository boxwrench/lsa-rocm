#!/usr/bin/env python3
"""End-to-end decode-step attention on real captured KV, one layer, one GPU.

Three paths, each a full decode step for one query (select -> gather K/V -> attention), timed with GPU sync:
  full         SDPA over every key (GQA folded into rows).
  blocksparse  score mean raw keys of 4-key blocks, take the top blocks, gather, attend.
  lsa_flat     PCA(64) index, score bucket centroids, expand the top buckets to k keys, gather, attend.
All three also read the 4 sink keys and the last 512 keys. Output error is relative L2 vs full attention over the
same query's heads, averaged over the last --n-err queries (each causal at its own position).
Index building (clustering) is done once on CPU and excluded; per-step maintenance is not modelled.
usage: bench.py --dump results/dumps/<name> --layer 11 --out results/004-<name>-L11.json [--k 2048]
"""
import sys, json, time, argparse, statistics
sys.path[:0] = ["/ai/github/lsa-rocm/src", "/ai/github/lsa-rocm/third_party/lsa/src"]
import torch, torch.nn.functional as F
from lsa_rocm.oracle import OracleConfig, build_flat_index

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True); ap.add_argument("--layer", type=int, default=11)
ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=2048)
ap.add_argument("--n-err", type=int, default=8); ap.add_argument("--reps", type=int, default=30)
a = ap.parse_args()
dev = torch.device("cuda")
meta = json.load(open(f"{a.dump}/meta.json"))
rec = torch.load(f"{a.dump}/layer_{a.layer:02d}.pt", map_location="cpu")
keys, vals, qs = rec["keys_post_rope"], rec["values"], rec["queries_post_rope"]
pos_all = rec["positions"].long(); scale = float(rec["scaling"])
Hkv, T, D = keys.shape; Hq = qs.shape[0]; G = Hq // Hkv
rot = meta["rotary_dim_per_layer"][str(a.layer)]
SINK, WIN, BLK, CAP = 4, 512, 4, 16
nq = min(a.n_err, qs.shape[1]); off = qs.shape[1] - nq

def unit(x): return F.normalize(x.float(), dim=-1)
# index keys: RoPE-free dims, all KV heads concatenated; queries summed over each group's heads
raw_k = keys.float()[:, :, rot:].permute(1, 0, 2).reshape(T, -1)
raw_q = qs[:, off:].float().reshape(Hkv, G, nq, D).sum(1)[:, :, rot:].permute(1, 0, 2).reshape(nq, -1)
mean = raw_k.mean(0, keepdim=True)
_, _, V = torch.pca_lowrank(raw_k - mean, q=64, center=False); basis = V[:, :64].T.contiguous()
idx_k = unit((raw_k - mean) @ basis.T); idx_q = unit((raw_q - mean) @ basis.T)
nclu = (T // 512) * 512
cfg = OracleConfig(pool_size=4, bucket_size=8, bucket_cap=CAP, cluster_block_size=512)
t0 = time.time(); buckets = build_flat_index(idx_k[:nclu].contiguous(), cfg).buckets
print(f"cluster {time.time()-t0:.1f}s buckets {buckets.bucket_centroids.shape[0]}", flush=True)
cent = buckets.bucket_centroids.to(dev); mem = buckets.bucket_members.long().to(dev)
P = a.k // 4  # buckets expanded, as in the harness (p = k/4)

K = keys.to(dev, torch.bfloat16); Vv = vals.to(dev, torch.bfloat16)
nb = T // BLK
blk_mean = unit(raw_k[: nb * BLK].reshape(nb, BLK, -1).mean(1)).to(dev, torch.bfloat16)
iq = idx_q.to(dev); rq = unit(raw_q).to(dev, torch.bfloat16)
Q = qs[:, off:].to(dev, torch.bfloat16)  # [Hq, nq, D]
posq = pos_all[off:].to(dev)
sinks = torch.arange(SINK, device=dev)
ar_win = torch.arange(WIN, device=dev)

def attend(q, idx, valid):
    """q [Hq,D]; idx [S] key ids; valid [S] bool. Folded GQA SDPA over the gathered keys."""
    k = K.index_select(1, idx); v = Vv.index_select(1, idx)
    out = F.scaled_dot_product_attention(q.view(1, Hkv, G, D), k.unsqueeze(0), v.unsqueeze(0),
                                         attn_mask=valid.view(1, 1, 1, -1), scale=scale)
    return out.reshape(Hq, D)

def fixed(pos):
    win = (pos + 1 - WIN + ar_win).clamp(min=0); return torch.cat([sinks, win])

def f_full(i):
    p = posq[i]; ar = torch.arange(T, device=dev)
    out = F.scaled_dot_product_attention(Q[:, i].view(1, Hkv, G, D), K.unsqueeze(0), Vv.unsqueeze(0),
                                         attn_mask=(ar <= p).view(1, 1, 1, -1), scale=scale)
    return out.reshape(Hq, D)

def f_full_nomask(i):
    # last query sees every key, so no mask is needed: lets SDPA pick its fast (flash/efficient) kernel
    out = F.scaled_dot_product_attention(Q[:, i].view(1, Hkv, G, D), K.unsqueeze(0), Vv.unsqueeze(0), scale=scale)
    return out.reshape(Hq, D)

def f_full_bmm(i):
    # decode-style: scores = q K^T, softmax, @ V as batched matmuls (bandwidth-bound, no mask: last query only)
    q = Q[:, i].view(Hkv, G, D)
    sc = torch.matmul(q, K.transpose(1, 2)).float() * scale
    pr = torch.softmax(sc, dim=-1).to(torch.bfloat16)
    return torch.matmul(pr, Vv).reshape(Hq, D)

def f_block(i):
    p = posq[i]; sc = blk_mean @ rq[i]
    sc = sc.masked_fill(torch.arange(nb, device=dev) * BLK > p - WIN, -1e4)
    top = sc.topk(a.k // BLK).indices
    idx = torch.cat([(top[:, None] * BLK + torch.arange(BLK, device=dev)).flatten(), fixed(p)])
    return attend(Q[:, i], idx, idx <= p)

def f_lsa(i):
    p = posq[i]; sc = cent @ iq[i]
    sc = sc.masked_fill(mem[:, 0] > p - WIN, -1e4)  # bucket is long-range only if it starts before the window
    top = sc.topk(P).indices
    m = mem[top].flatten(); ok = m >= 0
    keep = ok & (ok.cumsum(0) <= a.k) & (m <= p - WIN)
    idx = torch.cat([m.clamp(min=0), fixed(p)]); valid = torch.cat([keep, torch.ones(SINK + WIN, dtype=torch.bool, device=dev)]) & (idx <= p)
    return attend(Q[:, i], idx, valid)

def timeit(fn, i):
    for _ in range(3): fn(i)
    torch.cuda.synchronize(); ts = []
    for _ in range(a.reps):
        t = time.perf_counter(); fn(i); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3

res = {"dump": a.dump, "layer": a.layer, "T": T, "k": a.k, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "kv_bytes_layer_gib": 2 * K.numel() * 2 / 2**30, "paths": {}}
last = nq - 1
ref = [f_full(i).float() for i in range(nq)]
for name, fn in (("full_masked", f_full), ("full_sdpa_nomask", f_full_nomask), ("full_bmm", f_full_bmm), ("blocksparse", f_block), ("lsa_flat", f_lsa)):
    rng = [last] if fn in (f_full_nomask, f_full_bmm) else range(nq)  # unmasked paths are valid for the last query only
    err = statistics.mean(((fn(i).float() - ref[i]).norm() / ref[i].norm()).item() for i in rng)
    ms = timeit(fn, last)
    res["paths"][name] = {"ms_per_step": ms, "rel_l2_err_vs_full": err}
    print(f"T={T} {name:12s} {ms:7.3f} ms  err {err:.4f}", flush=True)
res["selected_keys_read_per_step"] = a.k + SINK + WIN
res["kv_read_fraction"] = (a.k + SINK + WIN) / T
json.dump(res, open(a.out, "w"), indent=1)
