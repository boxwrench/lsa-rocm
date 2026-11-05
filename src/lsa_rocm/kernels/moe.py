# SPDX-License-Identifier: Apache-2.0
# Original grouped-GEMM fused-MoE Triton kernels for non-gated (up -> relu^2 -> down)
# experts, written for lsa-rocm. Token/expert block alignment follows the idea of
# vLLM's fused_moe (moe_align_block_size) but is implemented in plain torch ops
# (vLLM's version is a compiled CUDA op, which is skipped). No vLLM code is copied.
import torch
import triton
import triton.language as tl

_CFGS = [
    triton.Config({"BN": 128, "BK": 64}, num_warps=4, num_stages=2),
    triton.Config({"BN": 64, "BK": 64}, num_warps=4, num_stages=2),
    triton.Config({"BN": 128, "BK": 32}, num_warps=8, num_stages=2),
    triton.Config({"BN": 256, "BK": 32}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_CFGS, key=["N", "K", "BM", "A_DIV", "RELU2", "MUL_W"])
@triton.jit
def _moe_gemm(a_ptr, w_ptr, o_ptr, sorted_ids_ptr, block_expert_ptr, topw_ptr,
              P, N, K, sa, swe, swn, so,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
              A_DIV: tl.constexpr, RELU2: tl.constexpr, MUL_W: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(block_expert_ptr + pm)
    if e < 0:
        return
    pair = tl.load(sorted_ids_ptr + pm * BM + tl.arange(0, BM))
    valid = pair < P
    arow = pair // A_DIV
    n = pn * BN + tl.arange(0, BN)
    nm = n < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    wbase = w_ptr + e.to(tl.int64) * swe
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        km = k < K
        a = tl.load(a_ptr + arow.to(tl.int64)[:, None] * sa + k[None, :], mask=valid[:, None] & km[None, :], other=0.0)
        w = tl.load(wbase + n[None, :] * swn + k[:, None], mask=nm[None, :] & km[:, None], other=0.0)
        acc += tl.dot(a, w)
    if RELU2:
        acc = tl.maximum(acc, 0.0)
        acc = acc * acc
    if MUL_W:
        wt = tl.load(topw_ptr + pair, mask=valid, other=0.0)
        acc = acc * wt[:, None]
    tl.store(o_ptr + pair.to(tl.int64)[:, None] * so + n[None, :], acc.to(o_ptr.dtype.element_ty),
             mask=valid[:, None] & nm[None, :])


def _align(topk_idx, E, BM):
    P = topk_idx.numel()
    flat = topk_idx.reshape(-1).to(torch.int64)
    order = torch.argsort(flat, stable=True)
    counts = torch.zeros(E, dtype=torch.int64, device=flat.device).scatter_add_(0, flat, torch.ones_like(flat))
    padded = (counts + BM - 1) // BM * BM
    pad_cum = torch.cumsum(padded, 0)
    pad_start = pad_cum - padded
    cum_start = torch.cumsum(counts, 0) - counts
    se = flat[order]
    dest = pad_start[se] + (torch.arange(P, device=flat.device) - cum_start[se])
    max_p = (P + BM - 1) // BM * BM + E * BM  # static upper bound, no host sync
    sorted_ids = torch.full((max_p,), P, dtype=torch.int32, device=flat.device)
    sorted_ids[dest] = order.to(torch.int32)
    nblk = max_p // BM
    starts = torch.arange(nblk, device=flat.device) * BM
    be = torch.searchsorted(pad_cum, starts, right=True)
    be = torch.where(be >= E, torch.full_like(be, -1), be).to(torch.int32)
    return sorted_ids, be, nblk


def fused_experts_relu2(x, up_w, down_w, topk_idx, topk_w):
    """x (M,K); up_w (E,I,K); down_w (E,K,I); topk_idx/topk_w (M,top). Returns (M,K) (weighted sum)."""
    M, K = x.shape
    E, I, _ = up_w.shape
    top = topk_idx.shape[1]
    P = M * top
    BM = 64 if P // E < 128 else 128
    sorted_ids, be, nblk = _align(topk_idx, E, BM)
    x = x.contiguous()
    h = torch.empty((P, I), device=x.device, dtype=x.dtype)
    w32 = topk_w.reshape(-1).float().contiguous()
    _moe_gemm[lambda m: (nblk, triton.cdiv(I, m["BN"]))](
        x, up_w, h, sorted_ids, be, w32, P, I, K, x.stride(0), up_w.stride(0), up_w.stride(1), h.stride(0),
        BM=BM, A_DIV=top, RELU2=True, MUL_W=False)
    o = torch.empty((P, K), device=x.device, dtype=x.dtype)
    _moe_gemm[lambda m: (nblk, triton.cdiv(K, m["BN"]))](
        h, down_w, o, sorted_ids, be, w32, P, K, I, h.stride(0), down_w.stride(0), down_w.stride(1), o.stride(0),
        BM=BM, A_DIV=1, RELU2=False, MUL_W=True)
    return o.view(M, top, K).float().sum(1).to(x.dtype)
