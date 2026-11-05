import sys, json, torch, torch.nn.functional as F
sys.path[:0] = ["/ai/github/lsa-rocm/src", "/ai/github/lsa-rocm/third_party/lsa/src"]
from lsa_rocm.oracle import OracleConfig, build_flat_index, flat_select
dump = sys.argv[1]; layer = int(sys.argv[2]); K = int(sys.argv[3])
meta = json.load(open(f"{dump}/meta.json")); rec = torch.load(f"{dump}/layer_{layer:02d}.pt", map_location="cpu")
keys, qs, pos = rec["keys_post_rope"].float(), rec["queries_post_rope"].float(), rec["positions"].long()
Hkv, T, D = keys.shape; Hq = qs.shape[0]; G = Hq // Hkv; ROT = meta["rotary_dim_per_layer"][str(layer)]; SINK, WIN, CAP = 4, 544, 16
raw_k = keys[:, :, ROT:].permute(1, 0, 2).reshape(T, -1)
mean = raw_k.mean(0, keepdim=True); x = (raw_k - mean).double(); _, ev = torch.linalg.eigh(x.T @ x / T); basis = ev[:, -64:].flip(1).T.float()
idxk = F.normalize((raw_k - mean) @ basis.T, dim=-1); ncl = (T // 512) * 512
cfg = OracleConfig(pool_size=4, bucket_size=8, bucket_cap=CAP, cluster_block_size=512, top_p=K // 4, top_k=K, sink_size=SINK, window_size=WIN)
index = build_flat_index(idxk[:ncl].contiguous(), cfg); cent, mem = index.buckets.bucket_centroids, index.buckets.bucket_members.long()
nq = 12; off = qs.shape[1] - nq
raw_q = qs[:, off:].reshape(Hkv, G, nq, D).sum(1)[:, :, ROT:].permute(1, 0, 2).reshape(nq, -1)
iq = F.normalize((raw_q - mean) @ basis.T, dim=-1); p_all = pos[off:]
sel, lens = flat_select(iq, p_all, index, cfg)
tot = {"oracle": 0, "ours": 0, "inter": 0, "sink_dup": 0, "in_window_zone": 0}
for i in range(nq):
    p = int(p_all[i]); L = int(lens[i]); o = sel[i, :L].tolist()
    # oracle row = expansion, then window (<= WIN keys), then sinks: split the long-range part by position
    wstart = max(0, p + 1 - WIN); oracle_lr = {j for j in o if SINK <= j < wstart}
    # our selection logic, exactly as in run_matrix.py / run_big.py
    sc = (cent @ iq[i]).masked_fill(mem[:, 0] > p - WIN, -1e4); top = sc.topk(K // 4).indices
    m = mem[top].flatten(); ok = m >= 0; keep = ok & (ok.cumsum(0) <= K) & (m <= p - WIN)
    ours = set(m[keep].tolist()); ours_lr = {j for j in ours if SINK <= j < wstart}
    tot["oracle"] += len(oracle_lr); tot["ours"] += len(ours_lr); tot["inter"] += len(oracle_lr & ours_lr)
    tot["sink_dup"] += sum(1 for j in ours if j < SINK); tot["in_window_zone"] += sum(1 for j in ours if j >= wstart)
print(dump.split("/")[-1], "layer", layer, "k", K, "queries", nq, "keys", T)
print(f"  oracle long-range keys {tot['oracle']}, ours {tot['ours']}, overlap {tot['inter']} ({100*tot['inter']/max(tot['oracle'],1):.1f}% of oracle)")
print(f"  ours also contains sink keys (duplicated with the always-read sinks): {tot['sink_dup']}; keys inside the window zone: {tot['in_window_zone']}")
