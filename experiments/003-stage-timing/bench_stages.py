#!/usr/bin/env python3
"""Per-stage cost of the LSA index at 32K ... 10M keys, on CPU or one GPU. Synthetic keys (cost only, not quality).
usage: bench_stages.py --device cpu|cuda --out results/003-timing-<dev>.json [--sizes 32768,131072,1048576,10485760]
Stages: cluster (Ward, lsa.core reference, timed on a few cluster blocks and scaled to all blocks),
score_centroids (q . every bucket centroid + top-p), score_exhaustive (q . every pool key + top-k),
gather (index_select of k rows from a [T, 2*Hkv*D] bf16 KV table, measured on a 1M-row table).
Index space: d_idx=64, n_pool=4, bucket 8 pool keys (32 keys per bucket), cluster block 512, k=4096, p=1024."""
import sys, time, json, argparse, subprocess, statistics
sys.path[:0] = ["/ai/github/lsa-rocm/src", "/ai/github/lsa-rocm/third_party/lsa/src"]
import torch
ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu"); ap.add_argument("--out", required=True)
ap.add_argument("--sizes", default="32768,131072,1048576,10485760"); ap.add_argument("--nq", type=int, default=32)
a = ap.parse_args(); dev = torch.device(a.device)
D, POOL, BUCKET, BLOCK, K, P = 64, 4, 8, 512, 4096, 1024
def sync():
    if dev.type == "cuda": torch.cuda.synchronize()
def timeit(fn, reps=5, warm=1):
    for _ in range(warm): fn()
    sync(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)
def mixture(n, d, seed=0):
    g = torch.Generator().manual_seed(seed); nc = max(8, n // 256)
    centers = torch.nn.functional.normalize(torch.randn(nc, d, generator=g), dim=-1)
    idx = torch.randint(0, nc, (n,), generator=g)
    x = centers[idx] + 0.25 * torch.randn(n, d, generator=g) / d ** 0.5
    return torch.nn.functional.normalize(x, dim=-1)
res = {"device": a.device, "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "torch": torch.__version__,
       "params": dict(d_idx=D, n_pool=POOL, bucket=BUCKET, cluster_block=BLOCK, k=K, p=P, nq=a.nq), "rows": []}
# cluster stage: reference clustering on CPU tensors (lsa.core), a few blocks, then scaled
try:
    from lsa.core import build_buckets
    nb_meas = 16; keys = mixture(BLOCK * nb_meas, D)
    _nt = torch.get_num_threads(); torch.set_num_threads(1)  # many tiny ops: multi-thread oversubscription is ~60x slower
    t = timeit(lambda: build_buckets(keys, POOL, BUCKET, 2 * BUCKET, BLOCK, "ward"), reps=3)
    torch.set_num_threads(_nt); per_block = t / nb_meas
except Exception as e:
    per_block = None; res["cluster_error"] = repr(e)[:200]
res["cluster_s_per_512_key_block_cpu_reference"] = per_block
# gather table (1M rows of 2*4*256 bf16) once
rows_tab = 1 << 20
table = torch.randn(rows_tab, 2048, device=dev, dtype=torch.bfloat16) if dev.type == "cpu" or torch.cuda.mem_get_info()[0] > 6e9 else None
for T in [int(x) for x in a.sizes.split(",")]:
    n_pool = T // POOL; n_buck = n_pool // BUCKET
    pool = mixture(n_pool, D, 1).to(dev); cent = mixture(n_buck, D, 2).to(dev); q = mixture(a.nq, D, 3).to(dev)
    row = {"T": T, "pool_keys": n_pool, "buckets": n_buck}
    row["score_centroids_s"] = timeit(lambda: torch.topk(q @ cent.T, P, dim=-1))
    row["score_exhaustive_s"] = timeit(lambda: torch.topk(q @ pool.T, K, dim=-1)) if n_pool * a.nq * 4 < 6e9 else None
    if table is not None:
        sel = torch.randint(0, rows_tab, (a.nq, K), device=dev)
        row["gather_s"] = timeit(lambda: table[sel.flatten()])
    if per_block: row["cluster_total_s_cpu_reference"] = per_block * T / BLOCK
    row["bytes_index_vectors_pool_GB"] = round(n_pool * D * 2 / 1e9, 3)
    res["rows"].append(row); print(row, flush=True)
    del pool, cent
res["git_commit"] = subprocess.run(["git", "-C", "/ai/github/lsa-rocm", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
json.dump(res, open(a.out, "w"), indent=1)
