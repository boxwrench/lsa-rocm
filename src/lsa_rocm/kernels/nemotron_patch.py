# SPDX-License-Identifier: Apache-2.0
"""Monkeypatch transformers' Nemotron-H (5.16.1) to use fused Triton kernels for prefill.
Call apply() before running the model. Does not modify site-packages."""
import torch
import transformers.models.nemotron_h.modeling_nemotron_h as mh

from .conv1d import causal_conv1d_fn as _conv_fn
from .moe import fused_experts_relu2
from .ssd_combined import _mamba_chunk_scan_combined_fwd

_orig = {}


def chunk_scan_triton(hidden_states, dt, A, B, C, chunk_size, D=None, dt_bias=None, initial_states=None,
                      dt_softplus=False, dt_limit=(0.0, float("inf")), return_final_states=False, **kw):
    Bt, T, H, Pd = hidden_states.shape
    dev = hidden_states.device
    cu_chunk = torch.arange(0, T, chunk_size, device=dev, dtype=torch.int32)
    cu_chunk = torch.cat([cu_chunk, torch.tensor([T], device=dev, dtype=torch.int32)])
    nch = cu_chunk.numel() - 1
    outs, finals = [], []
    for b in range(Bt):
        out = torch.empty((T, H, Pd), device=dev, dtype=hidden_states.dtype)
        fs = _mamba_chunk_scan_combined_fwd(
            hidden_states[b].contiguous(), dt[b].contiguous(), A.float(), B[b].contiguous(), C[b].contiguous(),
            chunk_size, out, D=D.float() if D is not None else None, z=None,
            dt_bias=dt_bias.float() if dt_bias is not None else None,
            initial_states=initial_states[b:b + 1].float() if initial_states is not None else None,
            seq_idx=torch.zeros(nch, device=dev, dtype=torch.int32),
            cu_seqlens=torch.tensor([0, T], device=dev, dtype=torch.int32),
            cu_chunk_seqlens=cu_chunk,
            last_chunk_indices=torch.tensor([nch - 1], device=dev, dtype=torch.int64),
            dt_softplus=dt_softplus, dt_limit=dt_limit, state_dtype=torch.float32)
        outs.append(out)
        finals.append(fs)
    out = torch.stack(outs)
    if return_final_states:
        return out, torch.cat(finals, 0)
    return out


def experts_forward(self, hidden_states, top_k_index, top_k_weights):
    return fused_experts_relu2(hidden_states, self.up_proj, self.down_proj, top_k_index, top_k_weights)


def apply(conv=True, scan=True, moe=True):
    if scan:
        mh.mamba2_chunk_scan = chunk_scan_triton
    if conv:
        mh.causal_conv1d_fn = _conv_fn
    if moe and "experts" not in _orig:
        _orig["experts"] = mh.NemotronHExperts.forward
        mh.NemotronHExperts.forward = experts_forward
