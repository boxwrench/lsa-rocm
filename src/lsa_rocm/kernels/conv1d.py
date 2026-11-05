# SPDX-License-Identifier: Apache-2.0
# Original (not vendored) fused depthwise causal conv1d + SiLU Triton kernel for
# prefill, written for lsa-rocm. vLLM's causal_conv1d.py (1100+ lines, varlen /
# paged-cache machinery) was read and judged too heavy to vendor for this use.
import torch
import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_kernel(x_ptr, w_ptr, b_ptr, o_ptr, T, C,
                          sxb, sxc, sxt, sob, sot, soc,
                          WIDTH: tl.constexpr, HAS_BIAS: tl.constexpr, SILU: tl.constexpr,
                          BT: tl.constexpr, BC: tl.constexpr):
    pb = tl.program_id(0)
    pt = tl.program_id(1)
    pc = tl.program_id(2)
    t = pt * BT + tl.arange(0, BT)
    c = pc * BC + tl.arange(0, BC)
    cm = c < C
    acc = tl.zeros((BT, BC), dtype=tl.float32)
    for k in tl.static_range(WIDTH):
        ts = t - (WIDTH - 1) + k
        m = (ts >= 0)[:, None] & (ts < T)[:, None] & cm[None, :]
        xv = tl.load(x_ptr + pb * sxb + ts[:, None] * sxt + c[None, :] * sxc, mask=m, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + c * WIDTH + k, mask=cm, other=0.0).to(tl.float32)
        acc += xv * wv[None, :]
    if HAS_BIAS:
        acc += tl.load(b_ptr + c, mask=cm, other=0.0).to(tl.float32)[None, :]
    if SILU:
        acc = acc * tl.sigmoid(acc)
    om = (t < T)[:, None] & cm[None, :]
    tl.store(o_ptr + pb * sob + t[:, None] * sot + c[None, :] * soc, acc.to(o_ptr.dtype.element_ty), mask=om)


def causal_conv1d_fn(x, weight, bias=None, activation=None, **kwargs):
    """x: (B, C, T) (any strides, e.g. transposed view of (B,T,C)); weight: (C, W).
    Returns (B, C, T) view of a (B, T, C)-contiguous buffer."""
    B, C, T = x.shape
    W = weight.shape[-1]
    out = torch.empty((B, T, C), device=x.device, dtype=x.dtype)
    BT, BC = 32, 128
    grid = (B, triton.cdiv(T, BT), triton.cdiv(C, BC))
    _causal_conv1d_kernel[grid](
        x, weight.contiguous(), bias if bias is not None else weight, out, T, C,
        x.stride(0), x.stride(1), x.stride(2), out.stride(0), out.stride(1), out.stride(2),
        WIDTH=W, HAS_BIAS=bias is not None, SILU=activation in ("silu", "swish"),
        BT=BT, BC=BC, num_warps=4)
    return out.transpose(1, 2)
