import sys, json, glob, time, torch, torch.nn.functional as F
sys.path[:0] = ["/ai/github/lsa-rocm/src", "/ai/github/lsa-rocm/third_party/lsa/src"]
from lsa_rocm import oracle, metrics
dump = sys.argv[1]; meta = json.load(open(f"{dump}/meta.json"))
rd = int(meta.get("rotary_dim", 64)); cfg = oracle.OracleConfig()
rows = []
for f in sorted(glob.glob(f"{dump}/layer_*.pt")):
    t = torch.load(f, map_location="cpu")
    K, V, Q = t["keys_post_rope"].float(), t["values"].float(), t["queries_post_rope"].float()
    pos = t["positions"].long(); scale = float(t["scaling"]); Hq, Nq, D = Q.shape; Hkv, T, _ = K.shape; g = Hq // Hkv
    layer = int(f.split("_")[-1].split(".")[0])
    for mode in ("full", "posfree"):
        sl = slice(0, D) if mode == "full" else slice(rd, D)
        acc = {m: [] for m in ("flat", "tree", "flat_random", "flat_recency", "flat_exh", "tree_random", "tree_recency", "tree_exh")}
        for kv in range(Hkv):
            keys = F.normalize(K[kv][:, sl], dim=-1)
            qs = F.normalize(Q[kv*g:(kv+1)*g, :, sl].sum(0), dim=-1)          # group-summed queries [Nq, d]
            fi, ti = oracle.build_flat_index(keys, cfg), oracle.build_tree_index(keys, cfg)
            for name, (sel, ln) in (("flat", oracle.flat_select(qs, pos, fi, cfg)), ("tree", oracle.tree_select(qs, pos, ti, cfg))):
                for h in range(kv*g, (kv+1)*g):
                    w = metrics.attention_weights(Q[h], K[kv], scale, pos)       # true [Nq, T]
                    acc[name].append(metrics.mass_captured(w, sel, ln, 0)["total_mass"].mean().item())
                    cnt = ln.long()
                    g_r = torch.Generator().manual_seed(0)
                    rnd = [torch.randperm(int(pos[i])+1, generator=g_r)[:cnt[i]] for i in range(Nq)]
                    rec = [torch.arange(max(0, int(pos[i])+1-int(cnt[i])), int(pos[i])+1) for i in range(Nq)]
                    exh = [w[i].topk(int(cnt[i])).indices for i in range(Nq)]
                    for lab, sets in (("random", rnd), ("recency", rec), ("exh", exh)):
                        acc[f"{name}_{lab}"].append(sum(w[i, sets[i]].sum().item() for i in range(Nq)) / Nq)
        row = {"layer": layer, "mode": mode, **{k: round(sum(v)/len(v), 4) for k, v in acc.items()}, "sel_frac": round(float(ln.float().mean())/T, 3)}
        rows.append(row); print(row, flush=True)
json.dump(rows, open(sys.argv[2], "w"), indent=1)
