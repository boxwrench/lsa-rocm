"""Block-sparse baselines: the paper's QSA/HiP-style control (Section 4.1).

These are indexers that score *blocks* of consecutive keys instead of individual keys:

* ``block_sparse_select`` -- the flat family. Every closed block of ``block_size``
  consecutive keys is scored by the dot product of the query with the block's mean
  key. Only blocks that lie fully before the recent window and start at or after the
  sinks are candidates; the best ``n_blocks`` are expanded to their keys, and the
  sinks plus the window are always added.
* ``block_sparse_tree_select`` -- a two-level variant: coarse blocks of
  ``block_size * coarse_factor`` keys are scored first, then the surviving coarse
  blocks are refined at the fine ``block_size`` granularity.

Both return the repo-wide selection format: padded int32 ids ``[Nq, S]`` with ``-1``
padding and a length ``[Nq]`` int32. Rows hold distinct keys only, all ``<= pos``.

Keys are ``[N, D]`` raw (unprojected) keys and queries ``[Nq, D]``; both are cast to
float32 for scoring. A query at position ``p`` always reads the sinks
``[0, min(sink_size, window_start, p + 1))`` and the window
``[window_start, p + 1)`` with ``window_start = max(0, p + 1 - window_size)``, exactly
as ``lsa_rocm.oracle.exhaustive_select`` does, so selections are directly comparable.
"""

from __future__ import annotations

import torch

__all__ = ["block_sparse_select", "block_sparse_tree_select"]


def _check(q: torch.Tensor, keys: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    if q.dim() != 2:
        raise ValueError(f"q must be 2-D [Nq, D], got shape {tuple(q.shape)}")
    if keys.dim() != 2:
        raise ValueError(f"keys must be 2-D [N, D], got shape {tuple(keys.shape)}")
    if q.shape[-1] != keys.shape[-1]:
        raise ValueError(
            f"q and keys must share the key dim - {q.shape[-1]} != {keys.shape[-1]}"
        )
    positions = torch.as_tensor(pos, dtype=torch.int64).reshape(-1)
    if positions.numel() != q.shape[0]:
        raise ValueError(
            f"pos must have {q.shape[0]} entries, got {positions.numel()}"
        )
    if positions.numel() and int(positions.min()) < 0:
        raise ValueError("positions must be non-negative")
    return positions


def _block_means(keys: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mean key of every full closed block -> ``[N // block_size, D]``."""
    n_blocks = keys.shape[0] // block_size
    if n_blocks == 0:
        return keys.new_zeros((0, keys.shape[1]))
    return keys[: n_blocks * block_size].reshape(n_blocks, block_size, -1).mean(1)


def _row_bounds(position: int, n_keys: int, sink_size: int, window_size: int):
    """The sink, window and long-range ranges for one query, as in the oracle."""
    window_start = max(0, position + 1 - window_size)
    window_high = min(position + 1, n_keys)
    window = torch.arange(
        window_start, window_start + max(0, window_high - window_start), dtype=torch.int64
    )
    sinks = torch.arange(min(sink_size, window_start, position + 1), dtype=torch.int64)
    return sinks, window, window_start


def _assemble(
    nq: int,
    n_keys: int,
    chosen: list[torch.Tensor],
    sinks: list[torch.Tensor],
    windows: list[torch.Tensor],
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = torch.full((nq, width), -1, dtype=torch.int32)
    lengths = torch.zeros(nq, dtype=torch.int32)
    for row in range(nq):
        mask = torch.zeros(n_keys, dtype=torch.bool)
        parts = []
        for piece in (chosen[row], windows[row], sinks[row]):
            if piece.numel():
                keep = piece[~mask[piece]]
                mask[keep] = True
                parts.append(keep)
        held = torch.cat(parts) if parts else torch.empty(0, dtype=torch.int64)
        selected[row, : held.numel()] = held.to(torch.int32)
        lengths[row] = held.numel()
    return selected, lengths


def _top_blocks(scores: torch.Tensor, blocks: torch.Tensor, count: int) -> torch.Tensor:
    """The best ``count`` blocks, ties broken to the lower block id."""
    if blocks.numel() == 0 or count <= 0:
        return blocks[:0]
    count = min(count, blocks.numel())
    order = torch.argsort(-scores[blocks], stable=True)[:count]
    return blocks[order]


def block_sparse_select(
    q: torch.Tensor,
    keys: torch.Tensor,
    pos: torch.Tensor,
    block_size: int,
    n_blocks: int,
    sink_size: int,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat block-sparse selection (QSA/HiP style): score every closed block, pick the best.

    Args:
        q: Queries ``[Nq, D]``.
        keys: Raw keys ``[N, D]`` (float32).
        pos: Each query's position ``[Nq]``, integer.
        block_size: Keys per scored block, ``>= 1``.
        n_blocks: Blocks selected, ``>= 0``.
        sink_size: First positions always read, ``>= 0``.
        window_size: Recent positions always read, the query's own included.

    Returns:
        Padded selected key ids ``[Nq, S]`` int32 with ``-1`` padding, where
        ``S = min(n_blocks * block_size + window_size + sink_size, N)``, and the
        number of valid ids per row ``[Nq]`` int32.

    Raises:
        ValueError: On a bad shape, a negative position, or a non-positive block size.
    """
    positions = _check(q, keys, pos)
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if n_blocks < 0 or sink_size < 0 or window_size < 0:
        raise ValueError("n_blocks, sink_size and window_size must be non-negative")
    n_keys = keys.shape[0]
    keyf = keys.to(torch.float32)
    means = _block_means(keyf, block_size)  # [n_closed, D]
    scores = q.to(torch.float32) @ means.T  # [Nq, n_closed]
    n_closed = means.shape[0]
    first_block = (sink_size + block_size - 1) // block_size

    width = min(n_blocks * block_size + window_size + sink_size, n_keys)
    chosen, sinks, windows = [], [], []
    for row, position in enumerate(positions.tolist()):
        sinks_row, window_row, window_start = _row_bounds(
            position, n_keys, sink_size, window_size
        )
        last_closed = min(window_start // block_size, n_closed)
        candidates = torch.arange(
            first_block, first_block + max(0, last_closed - first_block)
        )
        best = _top_blocks(scores[row], candidates, n_blocks)
        blocks = torch.arange(n_keys, dtype=torch.int64)
        keys_row = torch.cat(
            [blocks[b * block_size : (b + 1) * block_size] for b in best.tolist()]
        ) if best.numel() else torch.empty(0, dtype=torch.int64)
        chosen.append(keys_row)
        sinks.append(sinks_row)
        windows.append(window_row)
    return _assemble(q.shape[0], n_keys, chosen, sinks, windows, width)


def block_sparse_tree_select(
    q: torch.Tensor,
    keys: torch.Tensor,
    pos: torch.Tensor,
    block_size: int,
    n_blocks: int,
    sink_size: int,
    window_size: int,
    coarse_factor: int = 8,
    n_coarse: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-level block-sparse selection: coarse blocks first, then refine at ``block_size``.

    Coarse blocks of ``block_size * coarse_factor`` keys are scored by their mean key
    and the best ``n_coarse`` (default: enough to hold ``n_blocks`` fine blocks) are
    kept. The fine blocks inside them are then scored by their own mean and the best
    ``n_blocks`` are expanded, with the same sinks and window as the flat version.

    Args:
        q: Queries ``[Nq, D]``.
        keys: Raw keys ``[N, D]`` (float32).
        pos: Each query's position ``[Nq]``, integer.
        block_size: Fine keys per scored block, ``>= 1``.
        n_blocks: Fine blocks selected, ``>= 0``.
        sink_size: First positions always read, ``>= 0``.
        window_size: Recent positions always read, the query's own included.
        coarse_factor: Coarse block size in fine blocks, ``>= 1``.
        n_coarse: Coarse blocks kept; defaults to ``ceil(n_blocks / coarse_factor)``.

    Returns:
        Padded selected key ids ``[Nq, S]`` int32 with ``-1`` padding, and lengths
        ``[Nq]`` int32, in the same format as :func:`block_sparse_select`.

    Raises:
        ValueError: On a bad shape, a negative position, or a non-positive block size.
    """
    positions = _check(q, keys, pos)
    if block_size < 1 or coarse_factor < 1:
        raise ValueError("block_size and coarse_factor must be >= 1")
    if n_blocks < 0 or sink_size < 0 or window_size < 0:
        raise ValueError("n_blocks, sink_size and window_size must be non-negative")
    if n_coarse is None:
        n_coarse = (n_blocks + coarse_factor - 1) // coarse_factor
    if n_coarse < 0:
        raise ValueError("n_coarse must be non-negative")

    n_keys = keys.shape[0]
    keyf = keys.to(torch.float32)
    fine = _block_means(keyf, block_size)  # [n_fine, D]
    coarse_span = block_size * coarse_factor
    coarse = _block_means(keyf, coarse_span)  # [n_coarse_total, D]
    fine_scores = q.to(torch.float32) @ fine.T
    coarse_scores = q.to(torch.float32) @ coarse.T
    n_fine = fine.shape[0]
    n_coarse_total = coarse.shape[0]
    first_fine = (sink_size + block_size - 1) // block_size

    width = min(n_blocks * block_size + window_size + sink_size, n_keys)
    chosen, sinks, windows = [], [], []
    for row, position in enumerate(positions.tolist()):
        sinks_row, window_row, window_start = _row_bounds(
            position, n_keys, sink_size, window_size
        )
        # Coarse blocks fully before the window that could hold a fine block >= sink.
        last_coarse = min(window_start // coarse_span, n_coarse_total)
        coarse_candidates = torch.arange(
            last_coarse + 1
        )
        coarse_candidates = coarse_candidates[
            (coarse_candidates + 1) * coarse_span > sink_size
        ]
        coarse_candidates = coarse_candidates[coarse_candidates < last_coarse]
        kept_coarse = _top_blocks(coarse_scores[row], coarse_candidates, n_coarse)
        # Fine blocks inside the surviving coarse blocks.
        if kept_coarse.numel():
            offsets = torch.arange(coarse_factor, dtype=torch.int64)
            fine_ids = (kept_coarse.unsqueeze(1) * coarse_factor + offsets).reshape(-1)
            last_fine = min(window_start // block_size, n_fine)
            fine_ids = fine_ids[(fine_ids >= first_fine) & (fine_ids < last_fine)]
            fine_ids = torch.unique(fine_ids, sorted=True)
        else:
            fine_ids = torch.empty(0, dtype=torch.int64)
        best = _top_blocks(fine_scores[row], fine_ids, n_blocks)
        blocks = torch.arange(n_keys, dtype=torch.int64)
        keys_row = torch.cat(
            [blocks[b * block_size : (b + 1) * block_size] for b in best.tolist()]
        ) if best.numel() else torch.empty(0, dtype=torch.int64)
        chosen.append(keys_row)
        sinks.append(sinks_row)
        windows.append(window_row)
    return _assemble(q.shape[0], n_keys, chosen, sinks, windows, width)
