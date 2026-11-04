"""Retrieval quality metrics for LSA on cached keys.

All functions are pure torch, device agnostic, and work on a single
(layer, KV head) at a time. Key/query tensors use the repo-wide convention
of an explicit leading query dimension and no batch dimension:

    q        [Nq, D]
    k        [N, D]
    weights  [Nq, N]   post-softmax attention weights
    selected [Nq, S]   padded int tensor, valid entries are the first
                       ``lengths[i]`` entries of row ``i``
    lengths  [Nq]      int tensor

``selected`` is the same format returned by ``lsa.core.read``.

Ties in the true top-k are broken by ascending key index: the ranking is a
stable sort of ``-weights`` (equivalently a stable descending sort of the
weights), so equal weights keep their original key order.
"""

from __future__ import annotations

import torch

__all__ = [
    "attention_weights",
    "mass_captured",
    "recall_at_k",
    "output_error",
    "budget_fraction",
]


def _as_2d(name: str, x: torch.Tensor, cols: int | None = None) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError(f"{name} must be 2-D, got shape {tuple(x.shape)}")
    if cols is not None and x.shape[1] != cols:
        raise ValueError(
            f"{name} must have {cols} columns, got shape {tuple(x.shape)}"
        )
    return x


def _lengths(lengths: torch.Tensor, nq: int, device: torch.device) -> torch.Tensor:
    lengths = torch.as_tensor(lengths, device=device).to(torch.long).reshape(-1)
    if lengths.numel() != nq:
        raise ValueError(f"lengths must have {nq} entries, got {lengths.numel()}")
    return lengths


def _padding_mask(
    selected: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    if selected.dim() != 2:
        raise ValueError(f"selected must be 2-D, got shape {tuple(selected.shape)}")
    if selected.shape[0] != lengths.numel():
        raise ValueError(
            "selected and lengths disagree on the number of queries: "
            f"{selected.shape[0]} vs {lengths.numel()}"
        )
    positions = torch.arange(selected.shape[1], device=selected.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def _gather_weights(weights: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """Per-query gather of weights at the selected key indices -> ``[Nq, S]``."""
    n = weights.shape[1]
    idx = selected.to(torch.long)
    if n == 0:
        return weights.new_zeros(idx.shape)
    return weights.gather(1, idx.clamp_(0, n - 1))


def _gather_x(x: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """Gather rows of ``x`` along dim 0 for every selected index, clamped.

    Advanced indexing supports an arbitrary index shape, so this returns
    ``x[selected]`` with shape ``selected.shape + x.shape[1:]``.
    """
    n = x.shape[0]
    idx = selected.to(torch.long)
    if n == 0:
        return x.new_zeros(idx.shape + x.shape[1:])
    return x[idx.clamp_(0, n - 1)]


def attention_weights(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    q_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    """Softmax attention weights ``softmax(q K^T * scale)`` in float32.

    When ``q_pos`` is given, key ``j`` is allowed for query ``i`` iff
    ``j <= q_pos[i]`` (inclusive causal mask, matching a query at position
    ``q_pos[i]`` attending to keys at positions ``0..q_pos[i]``). Future keys
    receive zero weight. A query whose mask allows no keys yields an all-zero
    row.
    """
    q = _as_2d("q", q)
    k = _as_2d("k", k, cols=q.shape[1])
    logits = (q.to(torch.float32) @ k.to(torch.float32).transpose(0, 1)) * float(scale)
    if q_pos is not None:
        q_pos = torch.as_tensor(q_pos, device=logits.device).to(torch.long).reshape(-1)
        if q_pos.numel() != logits.shape[0]:
            raise ValueError(
                f"q_pos must have {logits.shape[0]} entries, got {q_pos.numel()}"
            )
        keys = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
        allowed = keys <= q_pos.unsqueeze(1)
        logits = logits.masked_fill(~allowed, float("-inf"))
    weights = torch.softmax(logits, dim=-1)
    if q_pos is not None:
        empty = ~allowed.any(dim=1)
        if empty.any():
            weights = weights.masked_fill(empty.unsqueeze(1), 0.0)
    return weights.to(torch.float32)


def mass_captured(
    weights: torch.Tensor,
    selected: torch.Tensor,
    lengths: torch.Tensor,
    sink_size: int = 0,
) -> dict[str, torch.Tensor]:
    """Attention mass captured by the selected keys, per query.

    Valid selected entries are the first ``lengths[i]`` entries of row ``i``;
    padding is ignored. Duplicate valid indices would be counted more than
    once, so selections are expected to be distinct.

    Returns a dict of per-query float32 tensors ``[Nq]``:

    * ``total_mass`` -- sum of the weights of all valid selected keys,
    * ``mass_excluding_sinks`` -- the same sum restricted to key indices
      ``>= sink_size``,
    * ``sink_mass`` -- the part of ``total_mass`` coming from keys
      ``< sink_size``.

    ``sink_size=0`` makes the last two equal to ``total_mass`` and ``0``.
    """
    weights = _as_2d("weights", weights)
    lengths = _lengths(lengths, weights.shape[0], weights.device)
    mask = _padding_mask(selected, lengths)
    if sink_size < 0:
        raise ValueError("sink_size must be >= 0")
    w_sel = _gather_weights(weights, selected) * mask
    total = w_sel.sum(dim=1)
    is_sink = (selected.to(torch.long) < int(sink_size)) & mask
    sink = (w_sel * is_sink).sum(dim=1)
    return {
        "total_mass": total.to(torch.float32),
        "mass_excluding_sinks": (total - sink).to(torch.float32),
        "sink_mass": sink.to(torch.float32),
    }


def recall_at_k(
    weights: torch.Tensor,
    selected: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Recall of the selected set against the true top-k, per query.

    Returns ``|S ∩ topk(a, k)| / min(k, N)`` as a float32 tensor ``[Nq]``.
    The true top-k is computed with ties broken by ascending key index.
    """
    weights = _as_2d("weights", weights)
    nq, n = weights.shape
    lengths = _lengths(lengths, nq, weights.device)
    if k <= 0:
        raise ValueError("k must be positive")
    kk = min(int(k), n)
    ranking = torch.argsort(-weights.to(torch.float32), dim=1, stable=True)
    top = ranking[:, :kk]
    top_mask = torch.zeros_like(weights, dtype=torch.bool)
    top_mask.scatter_(1, top, True)

    mask = _padding_mask(selected, lengths)
    sel = selected.to(torch.long).clamp(0, max(n - 1, 0))
    hit = top_mask.gather(1, sel) & mask
    return hit.sum(dim=1).to(torch.float32) / float(kk)


def output_error(
    weights: torch.Tensor,
    v: torch.Tensor,
    selected: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Relative error of attention restricted to the selected keys, per query.

    ``o_full = weights @ v``. ``o_S`` is computed from only the valid
    selected keys with their weights renormalised to sum to one. Returns
    ``||o_full - o_S|| / ||o_full||`` as a float32 tensor ``[Nq]``.

    A query with zero captured mass gets ``o_S = 0`` and therefore an error
    of 1 (unless ``o_full`` is also zero, in which case the error is 0).
    """
    weights = _as_2d("weights", weights)
    v = _as_2d("v", v)
    if v.shape[0] != weights.shape[1]:
        raise ValueError(
            "v must have one row per key: "
            f"{v.shape[0]} vs {weights.shape[1]}"
        )
    lengths = _lengths(lengths, weights.shape[0], weights.device)
    v = v.to(torch.float32)
    mask = _padding_mask(selected, lengths)
    w_sel = _gather_weights(weights, selected).to(torch.float32) * mask
    v_sel = _gather_x(v, selected)
    mass = w_sel.sum(dim=1)
    o_full = weights.to(torch.float32) @ v
    o_sel = (w_sel.unsqueeze(-1) * v_sel).sum(dim=1)
    has_mass = mass > 0
    o_sel = torch.where(
        has_mass.unsqueeze(1),
        o_sel / mass.clamp_min(1e-30).unsqueeze(1),
        torch.zeros_like(o_sel),
    )
    num = torch.linalg.vector_norm(o_full - o_sel, dim=1)
    den = torch.linalg.vector_norm(o_full, dim=1)
    ratio = num / den.clamp_min(1e-30)
    return torch.where(den > 0, ratio, torch.zeros_like(den)).to(torch.float32)


def budget_fraction(lengths: torch.Tensor, n_keys: int) -> torch.Tensor:
    """Fraction of the cached keys scored, ``lengths / n_keys``, per query."""
    if n_keys <= 0:
        raise ValueError("n_keys must be positive")
    lengths = torch.as_tensor(lengths).to(torch.long).reshape(-1)
    return lengths.to(torch.float32) / float(n_keys)
