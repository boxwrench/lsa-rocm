"""CPU correctness oracle: a thin wrapper over the pinned ``lsa.core`` reference.

``lsa.core`` (``third_party/lsa``) is the torch specification of the flat and tree designs. This module
exposes the repo's selection format on top of it:

    selected [Nq, S] int32, -1 padded, followed by lengths [Nq] int32

Keys and queries are ``[N, D]`` / ``[Nq, D]`` float32 unit rows, positions are ``[Nq]`` integers, and a
query at position ``p`` may read keys ``0..p`` only. The long-range part of a selection is the keys in
``[sink_size, max(0, p + 1 - window_size))``; the window is ``[max(0, p + 1 - window_size), p]`` and the
sinks are the first ``min(sink_size, window_start, p + 1)`` positions, exactly as ``expand_buckets`` builds
them. ``exhaustive_select`` is the true dot-product top-k over the same long-range candidates, with the same
window and sinks, and is the baseline the flat and tree selections are recalled against.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch

from lsa.core import Buckets, Hierarchy, build_buckets, build_hierarchy, read, read_hierarchy


@dataclass(frozen=True)
class OracleConfig:
    """The knobs of the flat and tree indexes and of a read.

    The defaults nest (``pool_size | bucket_size | cluster_block_size``, ``bucket_cap >= 2 * bucket_size``,
    ``page_size % cluster_block_size == 0``, ``node_bucket_size`` a power of two dividing
    ``node_cluster_block_size``); ``lsa.core`` refuses a config that does not.
    """

    # Flat index: pool, Ward, bucket.
    pool_size: int = 4
    bucket_size: int = 8
    bucket_cap: int = 16
    cluster_block_size: int = 64
    cluster_algorithm: str = "ward"
    # A read: buckets picked, long-range keys, always-read sinks and window.
    top_p: int = 8
    top_k: int = 64
    sink_size: int = 4
    window_size: int = 128
    # Tree index: node bucket levels over the buckets, and the beam down them.
    top_level: int = 4
    node_bucket_size: int = 4
    node_cluster_block_size: int = 16
    node_cluster_algorithm: str = "ward"
    beam_width: int = 8
    start_nodes: int | None = None
    page_size: int = 512
    growth_buckets: int = 0


class FlatIndex(NamedTuple):
    """A flat index: the buckets of the cache's closed cluster blocks."""

    buckets: Buckets


class TreeIndex(NamedTuple):
    """A tree index: the buckets and the node bucket levels over them."""

    buckets: Buckets
    hierarchy: Hierarchy


def build_flat_index(keys: torch.Tensor, cfg: OracleConfig) -> FlatIndex:
    """Cluster ``keys`` into buckets with ``lsa.core.build_buckets``.

    Args:
        keys: The cache's index keys ``[N, D]`` float32 or bfloat16, unit rows.
        cfg: The index and read knobs.

    Returns:
        The flat index, whose ``buckets`` are the closed cluster blocks' buckets in block order.

    Raises:
        ValueError: As ``lsa.core.build_buckets`` does, on a wrong shape, dtype, norm or size nesting.
    """
    return FlatIndex(
        build_buckets(
            keys,
            cfg.pool_size,
            cfg.bucket_size,
            cfg.bucket_cap,
            cfg.cluster_block_size,
            cfg.cluster_algorithm,
        )
    )


def flat_select(
    q: torch.Tensor, pos: torch.Tensor, index: FlatIndex, cfg: OracleConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select with the flat design: score every visible bucket, pick ``top_p``, expand to ``top_k``.

    Args:
        q: Index queries ``[Nq, D]`` float32 or bfloat16, unit rows.
        pos: Each query's position ``[Nq]``, int32 or int64.
        index: The flat index from ``build_flat_index``.
        cfg: The read knobs.

    Returns:
        Selected key ids ``[Nq, top_k + window_size + sink_size]`` int32 - the expansion, then the window,
        then the sinks, -1 padded - and keys each row holds ``[Nq]`` int32.

    Raises:
        ValueError: As ``lsa.core.read`` does, on a wrong shape, dtype, norm or position.
    """
    return read(
        q,
        torch.as_tensor(pos, dtype=torch.int64),
        index.buckets,
        cfg.top_p,
        cfg.top_k,
        cfg.sink_size,
        cfg.window_size,
    )


def build_tree_index(keys: torch.Tensor, cfg: OracleConfig) -> TreeIndex:
    """Build the buckets and the node bucket levels over them with ``lsa.core.build_hierarchy``.

    Args:
        keys: The cache's index keys ``[N, D]`` float32 or bfloat16, unit rows.
        cfg: The index knobs.

    Returns:
        The tree index: the buckets and the levels that fill at least one node cluster block, level 1 first.

    Raises:
        ValueError: As ``lsa.core.build_hierarchy`` does, on a wrong shape, dtype, norm or level nesting.
    """
    buckets = build_buckets(
        keys,
        cfg.pool_size,
        cfg.bucket_size,
        cfg.bucket_cap,
        cfg.cluster_block_size,
        cfg.cluster_algorithm,
    )
    hierarchy = build_hierarchy(
        keys,
        buckets,
        cfg.top_level,
        cfg.node_bucket_size,
        cfg.node_cluster_block_size,
        cfg.node_cluster_algorithm,
    )
    return TreeIndex(buckets, hierarchy)


def tree_select(
    q: torch.Tensor, pos: torch.Tensor, index: TreeIndex, cfg: OracleConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select with the tree design: a beam down the levels a query's page saw, then a flat pick.

    Args:
        q: Index queries ``[Nq, D]`` float32 or bfloat16, unit rows.
        pos: Each query's position ``[Nq]``, int32 or int64.
        index: The tree index from ``build_tree_index``.
        cfg: The read and beam knobs.

    Returns:
        Selected key ids ``[Nq, top_k + window_size + sink_size]`` int32 - the expansion, then the window,
        then the sinks, -1 padded - and keys each row holds ``[Nq]`` int32.

    Raises:
        ValueError: As ``lsa.core.read_hierarchy`` does, on a wrong shape, dtype, norm, position or page.
    """
    return read_hierarchy(
        q,
        torch.as_tensor(pos, dtype=torch.int64),
        index.buckets,
        index.hierarchy,
        cfg.beam_width,
        cfg.start_nodes,
        cfg.top_p,
        cfg.top_k,
        cfg.sink_size,
        cfg.window_size,
        cfg.page_size,
        cfg.growth_buckets,
    )


def exhaustive_select(
    q: torch.Tensor,
    k: torch.Tensor,
    pos: torch.Tensor,
    top_k: int,
    sink_size: int,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The baseline: true dot-product top-k over the same candidates, with the same window and sinks.

    A query at position ``p`` ranks the long-range candidates ``[sink_size, max(0, p + 1 - window_size))``
    by ``q . key`` and keeps the ``top_k`` best, a tie to the lower key. The window
    ``[max(0, p + 1 - window_size), p]`` and the sinks ``[0, min(sink_size, window_start, p + 1))`` are
    appended exactly as ``expand_buckets`` appends them, so a selected row is a permutation of the same
    candidate universe a flat or tree read draws from.

    Args:
        q: Queries ``[Nq, D]``, floating.
        k: Every key ``[N, D]``, floating.
        pos: Each query's position ``[Nq]``, integer.
        top_k: Long-range keys a query reads at most.
        sink_size: First positions always read.
        window_size: Recent positions always read, the query's own included.

    Returns:
        Selected key ids ``[Nq, top_k + window_size + sink_size]`` int32, -1 padded, and keys each row
        holds ``[Nq]`` int32.

    Raises:
        ValueError: If ``q`` and ``k`` do not share their key dim, or a position is negative.
    """
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q and k must share the key dim - {q.shape[-1]} != {k.shape[-1]}")
    positions = torch.as_tensor(pos, dtype=torch.int64)
    if positions.numel() and int(positions.min()) < 0:
        raise ValueError("positions must be non-negative")
    if min(top_k, window_size) < 1 or sink_size < 0:
        raise ValueError("top_k and window_size must be positive and sink_size non-negative")

    scores = q.float() @ k.float().T
    width = top_k + window_size + sink_size
    selected = torch.full((q.shape[0], width), -1, dtype=torch.int32)
    lengths = torch.zeros(q.shape[0], dtype=torch.int32)
    for row, position in enumerate(positions.tolist()):
        window_start = max(0, position + 1 - window_size)
        # A range of zero length is empty, not an error, at the start of a document.
        long_high = min(window_start, k.shape[0])
        candidates = torch.arange(
            sink_size, sink_size + max(0, long_high - sink_size), dtype=torch.int64
        )
        if candidates.numel():
            best = candidates[
                torch.argsort(scores[row, candidates], descending=True, stable=True)[:top_k]
            ]
        else:
            best = candidates
        window_high = min(position + 1, k.shape[0])
        window = torch.arange(
            window_start, window_start + max(0, window_high - window_start), dtype=torch.int64
        )
        sinks = torch.arange(min(sink_size, window_start, position + 1), dtype=torch.int64)
        held = torch.cat([best, window, sinks]).to(torch.int32)
        selected[row, : held.numel()] = held
        lengths[row] = held.numel()
    return selected, lengths
