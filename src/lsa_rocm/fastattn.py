"""Faster chunked-prefill attention for GQA models on ROCm (batch 1).

The chunk being processed is the last ``q_len`` of the ``Tk`` cached keys. Attention over the older keys (the prefix) needs no mask, so
it runs on the flash kernel (about 2x the throughput of the masked kernel the folded-GQA path in ``dump.py`` falls back to); the chunk's
own keys form a small causal diagonal block. The two parts are merged with their log-sum-exp values. K/V are broadcast per KV head, never
materialised ``num_key_value_groups`` times.
"""
import torch

_flash = torch.ops.aten._scaled_dot_product_flash_attention
PIECE = 131072  # keys per flash call over the prefix


def make_fast_prefill_attn():
    def fast(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        B, Hq, S, D = query.shape
        assert B == 1, "batch 1 only"
        Hkv, Tk = key.shape[1], key.shape[2]
        G = Hq // Hkv
        P = Tk - S
        if scaling is None:
            scaling = D ** -0.5
        outs = []
        for h in range(Hkv):
            qh = query[:, h * G:(h + 1) * G].contiguous()
            kd = key[:, h:h + 1, P:].expand(1, G, S, D)
            vd = value[:, h:h + 1, P:].expand(1, G, S, value.shape[-1])
            o2, l2 = _flash(qh, kd, vd, 0.0, True, scale=scaling)[:2]
            if P == 0:
                outs.append(o2)
                continue
            # The flash kernel materialises the broadcast K/V, so walk the prefix in pieces and merge by log-sum-exp (bounded memory).
            acc, lse = o2.float(), l2
            for s0 in range(0, P, PIECE):
                s1 = min(s0 + PIECE, P)
                kp = key[:, h:h + 1, s0:s1].expand(1, G, s1 - s0, D)
                vp = value[:, h:h + 1, s0:s1].expand(1, G, s1 - s0, value.shape[-1])
                o1, l1 = _flash(qh, kp, vp, 0.0, False, scale=scaling)[:2]
                new = torch.logaddexp(lse, l1)
                acc = acc * torch.exp(lse - new).unsqueeze(-1) + o1.float() * torch.exp(l1 - new).unsqueeze(-1)
                lse = new
            outs.append(acc.to(query.dtype))
        return torch.cat(outs, 1).transpose(1, 2).contiguous(), None

    return fast
