"""Sparse prefill attention: training-free flat LSA used while the prompt is processed in chunks (batch 1, GQA, partial-RoPE models).

For every full-attention layer and every chunk of S queries, each query attends to
  * the first ``sink`` keys,
  * a local band: the previous ``win`` keys, plus every key of the current chunk up to the query (so nothing between the window and the
    chunk start is missing for queries late in the chunk),
  * ``k`` long-range keys chosen per query by the layer-shared PCA(64) flat index over the closed 512-key blocks before the chunk.
The PCA basis is fitted once per layer on the first ``fit_tokens`` keys (the paper does not say what the basis is fitted on) and frozen; the
first ``start`` tokens use dense attention. The index grows by whole 512-key blocks after each chunk. Index and selection follow
``lsa.core`` (Ward buckets of at most 16 keys, score = unit centroid . unit query, top-p buckets expanded best first until k keys).
Differences from the paper: plain PyTorch ops, query sub-blocks instead of fused kernels, band wider than 512 late in a chunk, basis fitted
on a prefix of the same prompt.
"""
import torch
import torch.nn.functional as F

from lsa_rocm.oracle import OracleConfig, build_flat_index

CAP = 16
BLOCK = 512


class SparsePrefill:
    def __init__(self, full_layers, rot, k=2048, sink=4, win=544, start=8192, fit_tokens=8192, qblock=128):
        self.full, self.rot, self.k, self.sink, self.win = set(full_layers), rot, k, sink, win
        self.start, self.fit_tokens, self.qblock = start, fit_tokens, qblock
        self.p = max(k // 4, 1)
        self.cfg = OracleConfig(pool_size=4, bucket_size=8, bucket_cap=CAP, cluster_block_size=BLOCK)
        self.state = {}
        self.sparse_chunks = 0
        self.dense_chunks = 0

    # ---- index maintenance -------------------------------------------------------------------------------------------
    def _raw(self, key, a, b):
        """RoPE-free dims of keys [a,b) of every KV head, concatenated per token: [b-a, Hkv*(D-rot)] fp32."""
        return key[0, :, a:b, self.rot:].float().permute(1, 0, 2).reshape(b - a, -1)

    def _project(self, raw, st):
        return F.normalize((raw - st["mean"]) @ st["basis"].T, dim=-1)

    def observe(self, layer, key, s, S):
        """Feed the keys of the chunk [s, s+S) to the layer's index (whole 512-blocks only)."""
        end = s + S
        st = self.state.get(layer)
        if st is None:
            if end < self.start:
                return
            raw = self._raw(key, 0, min(end, self.fit_tokens))
            mean = raw.mean(0, keepdim=True)
            x = (raw - mean).double()
            _, ev = torch.linalg.eigh(x.T @ x / x.shape[0])
            st = {"mean": mean, "basis": ev[:, -64:].flip(1).T.float().contiguous(), "covered": 0,
                  "cent": None, "mem": None, "first": None}
            self.state[layer] = st
        ncl = (end // BLOCK) * BLOCK
        a = st["covered"]
        if ncl <= a:
            return
        idxk = self._project(self._raw(key, a, ncl), st)
        b = build_flat_index(idxk.cpu().contiguous(), self.cfg).buckets
        cent = b.bucket_centroids.to(key.device)
        mem = b.bucket_members.long().to(key.device)
        mem = torch.where(mem >= 0, mem + a, mem)                       # members are positions inside this block range
        st["cent"] = cent if st["cent"] is None else torch.cat([st["cent"], cent])
        st["mem"] = mem if st["mem"] is None else torch.cat([st["mem"], mem])
        st["first"] = st["mem"][:, 0]
        st["covered"] = ncl

    # ---- attention ---------------------------------------------------------------------------------------------------
    def usable(self, layer, s):
        st = self.state.get(layer)
        return layer in self.full and st is not None and st["covered"] >= s and s >= self.start and st["covered"] > 0

    def attend(self, layer, query, key, value, scaling, s, S):
        st = self.state[layer]
        _, Hq, _, D = query.shape
        Hkv = key.shape[1]
        G = Hq // Hkv
        dev = query.device
        K0, V0 = key[0], value[0]                                        # [Hkv,Tk,D]
        bs = max(0, s - self.win + 1)
        kb, vb = K0[:, bs:s + S], V0[:, bs:s + S]                        # local band keys: prefix window + the whole chunk
        ks, vs = K0[:, :self.sink], V0[:, :self.sink]
        jpos = bs + torch.arange(kb.shape[1], device=dev)
        outs = []
        for qs in range(0, S, self.qblock):
            qe = min(qs + self.qblock, S)
            QB = qe - qs
            qpos = s + torch.arange(qs, qe, device=dev)
            qb = query[0, :, qs:qe].reshape(Hkv, G, QB, D)
            # --- per-query selection from the closed blocks before the chunk
            raw_q = qb.float().sum(1)[:, :, self.rot:].permute(1, 0, 2).reshape(QB, -1)
            qi = self._project(raw_q, st)
            sc = qi @ st["cent"].T                                         # [QB, nb]
            sc = sc.masked_fill(st["first"][None, :] > (qpos[:, None] - self.win), -1e4)
            top = sc.topk(self.p, dim=1).indices
            m = st["mem"][top].reshape(QB, -1)                             # members of the picked buckets, best bucket first
            ok = (m >= self.sink) & (m <= qpos[:, None] - self.win)
            keep = ok & (ok.cumsum(1) <= self.k)
            order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)[:, :self.k]
            sel = m.gather(1, order).clamp(min=0)
            valid = keep.gather(1, order)                                  # [QB,k]
            flat = sel.reshape(-1)   # index_select copies whole rows (advanced indexing used a slow generic kernel: 65% of the time)
            Ks = K0.index_select(1, flat).reshape(Hkv, QB, -1, D)
            Vs = V0.index_select(1, flat).reshape(Hkv, QB, -1, D)
            # --- scores of the three parts, one joint softmax
            s_band = torch.einsum("hgqd,hjd->hgqj", qb, kb).float()
            allow = (jpos[None, :] <= qpos[:, None]) & ((jpos[None, :] > qpos[:, None] - self.win) | (jpos[None, :] >= s))
            s_band = s_band.masked_fill(~allow[None, None], float("-inf"))
            s_sink = torch.einsum("hgqd,hjd->hgqj", qb, ks).float()
            s_sel = torch.einsum("hgqd,hqkd->hgqk", qb, Ks).float().masked_fill(~valid[None, None], float("-inf"))
            pr = torch.softmax(torch.cat([s_band, s_sink, s_sel], -1) * scaling, dim=-1).to(query.dtype)
            nb_, ns_ = s_band.shape[-1], s_sink.shape[-1]
            o = torch.einsum("hgqj,hjd->hgqd", pr[..., :nb_], vb) + torch.einsum("hgqj,hjd->hgqd", pr[..., nb_:nb_ + ns_], vs) \
                + torch.einsum("hgqk,hqkd->hgqd", pr[..., nb_ + ns_:], Vs)
            outs.append(o.reshape(Hq, QB, D))
        out = torch.cat(outs, 1)                                           # [Hq,S,D]
        return out.transpose(0, 1).unsqueeze(0).contiguous()               # [1,S,Hq,D]


def make_sparse_prefill_attn(sp, dense_fn):
    """HF attention-interface function: sparse for full-attention layers once the index exists, dense otherwise."""
    def fn(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        layer = getattr(module, "layer_idx", None)
        S = query.shape[2]
        s = key.shape[2] - S
        if scaling is None:
            scaling = query.shape[-1] ** -0.5
        if sp.usable(layer, s) and S > 1:
            out = sp.attend(layer, query, key, value, scaling, s, S)
            sp.sparse_chunks += 1
        else:
            out, _ = dense_fn(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
            if layer in sp.full:
                sp.dense_chunks += 1
        if layer in sp.full:
            sp.observe(layer, key, s, S)
        return out, None
    return fn
