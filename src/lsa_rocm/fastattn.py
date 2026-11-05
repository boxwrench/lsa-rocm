"""Faster chunked-prefill attention for GQA models on ROCm (batch 1).

The chunk being processed is the last ``q_len`` of the ``Tk`` cached keys. Attention over the older keys (the prefix) needs no mask, so
it runs on the flash kernel (about 2x the throughput of the masked kernel the folded-GQA path in ``dump.py`` falls back to); the chunk's
own keys form a small causal diagonal block. The two parts are merged with their log-sum-exp values. K/V are broadcast per KV head, never
materialised ``num_key_value_groups`` times.
"""
import torch

_flash = torch.ops.aten._scaled_dot_product_flash_attention


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
            kp = key[:, h:h + 1, :P].expand(1, G, P, D)
            vp = value[:, h:h + 1, :P].expand(1, G, P, value.shape[-1])
            o1, l1 = _flash(qh, kp, vp, 0.0, False, scale=scaling)[:2]
            m = torch.maximum(l1, l2)
            w1 = torch.exp(l1 - m).unsqueeze(-1)
            w2 = torch.exp(l2 - m).unsqueeze(-1)
            outs.append(((o1.float() * w1 + o2.float() * w2) / (w1 + w2)).to(query.dtype))
        return torch.cat(outs, 1).transpose(1, 2).contiguous(), None

    return fast
