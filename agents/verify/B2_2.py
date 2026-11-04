#!/usr/bin/env python3
"""Acceptance for B2.2 (block-sparse baseline). Supervisor-written."""
import sys, subprocess, os
import torch
R = "/ai/github/lsa-rocm"
sys.path[:0] = [f"{R}/src", f"{R}/third_party/lsa/src"]
fails = []
def need(c, msg):
    if not c: fails.append(msg)
try:
    from lsa_rocm import baselines
    fn = baselines.block_sparse_select
except Exception as e:
    print("FAILS: cannot import baselines.block_sparse_select:", e); sys.exit(1)
torch.manual_seed(0)
N, D, B, nb, sink, win = 1024, 32, 8, 6, 4, 32
keys = torch.nn.functional.normalize(torch.randn(N, D), dim=-1)
q = torch.nn.functional.normalize(torch.randn(3, D), dim=-1)
keys[200:208] = q[0].repeat(8, 1) * 1.0                      # plant a block exactly aligned with query 0
pos = torch.full((3,), N - 1, dtype=torch.int64)
sel, ln = fn(q, keys, pos, block_size=B, n_blocks=nb, sink_size=sink, window_size=win)
need(sel.shape[0] == 3 and ln.shape == (3,), f"bad output shapes {tuple(sel.shape)} {tuple(ln.shape)}")
s0 = set(sel[0, : int(ln[0])].tolist())
need(set(range(200, 208)) <= s0, "planted block 200-207 not selected for query 0")
need(set(range(sink)) <= s0 and set(range(N - win, N)) <= s0, "sinks or window missing")
# brute-force reference: mean-pooled closed blocks outside sink/window, top n_blocks
for i in range(3):
    last_closed = (int(pos[i]) + 1 - win) // B                    # blocks fully before the window
    cand = [b for b in range(last_closed) if b * B >= sink]
    scores = torch.stack([keys[b * B:(b + 1) * B].mean(0) @ q[i] for b in cand])
    top = [cand[j] for j in scores.topk(nb).indices.tolist()]
    exp = set(range(sink)) | set(range(int(pos[i]) + 1 - win, int(pos[i]) + 1))
    for b in top: exp |= set(range(b * B, (b + 1) * B))
    got = set(sel[i, : int(ln[i])].tolist())
    need(got == exp, f"query {i}: selection differs from brute-force reference (got {len(got)}, expected {len(exp)})")
need(bool((sel[:, :1] >= 0).all()), "negative index")
for i in range(3): need(int(sel[i, : int(ln[i])].max()) <= int(pos[i]), "non-causal index")
r = subprocess.run([f"{R}/agents/scratch/venv-lsa-min/bin/python", "-I", "-m", "unittest", "discover", "-s", f"{R}/tests", "-p", "test_baselines.py"],
                   capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=f"{R}/src:{R}/third_party/lsa/src"))
print((r.stdout + r.stderr)[-300:])
need(r.returncode == 0, "tests/test_baselines.py failed or missing")
print("FAILS:", fails if fails else "none")
sys.exit(1 if fails else 0)
