#!/usr/bin/env python3
"""Real-KV retrieval recall harness (CPU only) for B2.0.

For every selected full-attention layer of a dump and every KV head, this builds an
index over the dumped post-RoPE keys, runs the flat and tree reads of
``lsa_rocm.oracle`` for the last ``--n-queries`` queries, and scores each selection
against the *true* attention of every query head in that KV group.  The selection is
shared across the group: the index query is the sum of the group's query heads, unit
normalised.  Three controls are matched to the flat and to the tree selection length
per query: random causal keys, the most recent keys, and the true top-count keys of
the query head being scored.

Index modes
-----------
``full``     every cached dim of the post-RoPE keys (the whole head, 256 dims here).
``posfree``  dims ``[rotary_dim:]`` only, i.e. the position-free part of the head
             (Qwen3.5-family rotates only the first ``rotary_dim`` dims, 64 here).

Both modes unit-normalise the index keys and the summed group query before the
oracle sees them, because ``lsa.core`` requires unit rows.

Config grids
------------
``--quick`` runs exactly one flat and one tree config (``bucket_size=8``,
``top_p=8``, ``top_k=64``) over one KV head per selected layer.

Full mode sweeps the small grid::

    bucket_size in {8, 16}
    top_p       in {4, 8, 16}
    top_k       in {32, 64, 128}

with the oracle defaults for everything else (``pool_size=4``,
``bucket_cap=max(16, 2*bucket_size)``, ``cluster_block_size=64``, Ward,
``sink_size=4``, ``window_size=128``, ``top_level=4``, ``node_bucket_size=4``,
``node_cluster_block_size=16``, ``beam_width=8``, ``page_size=512``,
``growth_buckets=0``).  ``sink_size`` and ``window_size`` are reported in meta.

Usage::

    PYTHONPATH=src:third_party/lsa/src agents/scratch/venv-lsa-min/bin/python \
        experiments/002-real-kv-recall/run.py --dump results/dumps/qwen3.5-0.8b-2048 \
        --out results/b2_0.json --layers all --n-queries 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
# The verifier runs this file with ``-I``, which ignores PYTHONPATH, so the repo's
# own packages and the pinned lsa reference are put on the path here.
for _entry in (ROOT / "src", ROOT / "third_party" / "lsa" / "src"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from lsa_rocm import metrics  # noqa: E402
from lsa_rocm.oracle import (
    OracleConfig,
    build_flat_index,
    build_tree_index,
    flat_select,
    tree_select,
)

# The design defaults of OracleConfig; kept explicit so meta can report them.
SINK_SIZE = 4
WINDOW_SIZE = 128
POOL_SIZE = 4
CLUSTER_BLOCK_SIZE = 64
TOP_LEVEL = 4
NODE_BUCKET_SIZE = 4
NODE_CLUSTER_BLOCK_SIZE = 16
BEAM_WIDTH = 8
PAGE_SIZE = 512
GROWTH_BUCKETS = 0

BUCKET_SIZES = (8, 16)
TOP_PS = (4, 8, 16)
TOP_KS = (32, 64, 128)
QUICK = dict(bucket_size=8, top_p=8, top_k=64)

SAMPLE_CAP = 5000
START = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - START:7.1f}s] {message}", flush=True)


def timed(label: str, fn):
    t0 = time.time()
    value = fn()
    elapsed = time.time() - t0
    if elapsed > 30:
        log(f"SLOW {label}: {elapsed:.1f}s")
    return value, elapsed


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def make_config(bucket_size: int, top_p: int, top_k: int) -> OracleConfig:
    return OracleConfig(
        pool_size=POOL_SIZE,
        bucket_size=bucket_size,
        bucket_cap=max(16, 2 * bucket_size),
        cluster_block_size=CLUSTER_BLOCK_SIZE,
        cluster_algorithm="ward",
        top_p=top_p,
        top_k=top_k,
        sink_size=SINK_SIZE,
        window_size=WINDOW_SIZE,
        top_level=TOP_LEVEL,
        node_bucket_size=NODE_BUCKET_SIZE,
        node_cluster_block_size=NODE_CLUSTER_BLOCK_SIZE,
        node_cluster_algorithm="ward",
        beam_width=BEAM_WIDTH,
        start_nodes=None,
        page_size=PAGE_SIZE,
        growth_buckets=GROWTH_BUCKETS,
    )


def config_dict(cfg: OracleConfig) -> dict:
    return {
        "pool_size": cfg.pool_size,
        "bucket_size": cfg.bucket_size,
        "bucket_cap": cfg.bucket_cap,
        "cluster_block_size": cfg.cluster_block_size,
        "cluster_algorithm": cfg.cluster_algorithm,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "sink_size": cfg.sink_size,
        "window_size": cfg.window_size,
        "top_level": cfg.top_level,
        "node_bucket_size": cfg.node_bucket_size,
        "node_cluster_block_size": cfg.node_cluster_block_size,
        "node_cluster_algorithm": cfg.node_cluster_algorithm,
        "beam_width": cfg.beam_width,
        "start_nodes": cfg.start_nodes,
        "page_size": cfg.page_size,
        "growth_buckets": cfg.growth_buckets,
    }


def unit_rows(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-30)


def make_index_keys(keys_post_rope: torch.Tensor, mode: str, rotary_dim: int) -> torch.Tensor:
    """The [N, D] unit index keys for one KV head."""
    k = keys_post_rope.float()
    if mode == "posfree":
        k = k[:, rotary_dim:]
    return unit_rows(k)


def make_index_queries(
    queries_post_rope: torch.Tensor,
    kv_head: int,
    group: int,
    offset: int,
    n_q: int,
    mode: str,
    rotary_dim: int,
) -> torch.Tensor:
    """The [n_q, D] unit group query: summed over the KV group's heads."""
    start = kv_head * group
    q = queries_post_rope[start : start + group, offset : offset + n_q].float().sum(dim=0)
    if mode == "posfree":
        q = q[:, rotary_dim:]
    return unit_rows(q)


def pad_rows(rows: list[torch.Tensor], width: int) -> tuple[torch.Tensor, torch.Tensor]:
    selected = torch.full((len(rows), width), -1, dtype=torch.int32)
    lengths = torch.zeros(len(rows), dtype=torch.int32)
    for i, row in enumerate(rows):
        held = torch.as_tensor(row, dtype=torch.int64).reshape(-1)
        selected[i, : held.numel()] = held.to(torch.int32)
        lengths[i] = held.numel()
    return selected, lengths


def random_control(lengths: torch.Tensor, positions: torch.Tensor, seed: int):
    """Uniform distinct causal keys, ``lengths[i]`` of them."""
    width = int(lengths.max().item())
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for i, position in enumerate(positions.tolist()):
        count = min(int(lengths[i]), int(position) + 1)
        rows.append(torch.randperm(int(position) + 1, generator=generator)[:count])
    return pad_rows(rows, width)


def recency_control(lengths: torch.Tensor, positions: torch.Tensor):
    """The most recent ``lengths[i]`` keys, ending at the query."""
    width = int(lengths.max().item())
    rows = []
    for i, position in enumerate(positions.tolist()):
        count = min(int(lengths[i]), int(position) + 1)
        start = int(position) + 1 - count
        rows.append(torch.arange(start, int(position) + 1))
    return pad_rows(rows, width)


def exhaustive_control(weights: torch.Tensor, lengths: torch.Tensor):
    """The head's true top-``lengths[i]`` keys among its causal keys."""
    width = int(lengths.max().item())
    rows = []
    n_keys = weights.shape[1]
    for i in range(weights.shape[0]):
        count = int(lengths[i])
        order = torch.argsort(-weights[i].float(), stable=True)[: min(count, n_keys)]
        rows.append(order)
    return pad_rows(rows, width)


def head_scores(weights, values, selected, lengths, sink_size):
    """Per-query scores of one head's weights [Nq,N] against a shared selection."""
    mass = metrics.mass_captured(weights, selected, lengths, sink_size=sink_size)
    error = metrics.output_error(weights, values, selected, lengths)
    recall = torch.zeros(weights.shape[0], dtype=torch.float32)
    for i in range(weights.shape[0]):
        k = int(lengths[i])
        if k > 0:
            recall[i] = metrics.recall_at_k(
                weights[i : i + 1], selected[i : i + 1], lengths[i : i + 1], k=k
            )[0]
    return mass["total_mass"], mass["mass_excluding_sinks"], recall, error


def mean_of(values: torch.Tensor) -> float:
    return float(values.mean().item()) if values.numel() else 0.0


def build_record(
    layer, kv, method, matched_to, mode, cfg, selected, lengths, build_s,
    positions, n_keys, head_weights, values,
):
    per_head = []
    for weights in head_weights:
        total, excl, recall, error = head_scores(
            weights, values, selected, lengths, cfg.sink_size
        )
        per_head.append(
            {
                "mass_captured": mean_of(total),
                "mass_excluding_sinks": mean_of(excl),
                "recall_at_k": mean_of(recall),
                "output_error": mean_of(error),
            }
        )
    keys = ("mass_captured", "mass_excluding_sinks", "recall_at_k", "output_error")
    averaged = {key: sum(h[key] for h in per_head) / len(per_head) for key in keys}
    return {
        "layer": layer,
        "kv_head": kv,
        "method": method,
        "index_mode": mode,
        "matched_to": matched_to,
        "config": config_dict(cfg),
        "mass_captured": averaged["mass_captured"],
        "mass_excluding_sinks": averaged["mass_excluding_sinks"],
        "recall_at_k": averaged["recall_at_k"],
        "output_error": averaged["output_error"],
        "budget_fraction": mean_of(lengths.float() / float(n_keys)),
        "index_build_s": float(build_s),
        "mean_over_heads": per_head,
    }


def add_samples(
    samples, layer, kv, group, method, matched_to, mode, cfg, head_offset,
    weights, selected, lengths, offset,
):
    """Append one sample for the first query, for the scored head's first query."""
    if len(samples) >= SAMPLE_CAP:
        return
    h = kv * group + head_offset
    if not weights:
        return
    w = weights[0]
    count = int(lengths[0])
    mass = metrics.mass_captured(
        w[0:1], selected[0:1], lengths[0:1], sink_size=cfg.sink_size
    )["total_mass"][0]
    samples.append(
        {
            "layer": layer,
            "query_head": h,
            "query_index": offset,
            "selected": [int(x) for x in selected[0, :count].tolist()],
            "mass_captured": float(mass.item()),
            "method": method,
            "matched_to": matched_to,
            "index_mode": mode,
        }
    )


def run(args) -> dict:
    torch.manual_seed(args.seed)
    dump = Path(args.dump).resolve()
    meta_path = dump / "meta.json"
    dump_meta = json.loads(meta_path.read_text())
    full_layers = list(dump_meta["full_attention_layer_indices"])
    if args.layers.strip().lower() == "all":
        layers = full_layers
    else:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
        missing = [x for x in layers if x not in full_layers]
        if missing:
            raise SystemExit(f"layers {missing} are not full-attention layers of {dump}")

    rotary_per_layer = {
        int(k): int(v) for k, v in (dump_meta.get("rotary_dim_per_layer") or {}).items()
    }
    default_rotary = (dump_meta.get("rope") or {}).get("rotary_dim_from_config")

    if args.quick:
        configs = [make_config(**QUICK)]
    else:
        configs = [
            make_config(bs, tp, tk) for bs in BUCKET_SIZES for tp in TOP_PS for tk in TOP_KS
        ]

    records: list[dict] = []
    samples: list[dict] = []
    index_cache: dict[tuple, tuple] = {}
    layer_meta_out: list[dict] = []

    for layer in layers:
        layer_path = dump / f"layer_{layer:02d}.pt"
        if not layer_path.exists():
            raise SystemExit(f"missing {layer_path}")
        t0 = time.time()
        rec = torch.load(layer_path, map_location="cpu")
        keys = rec["keys_post_rope"].float()
        values = rec["values"].float()
        queries = rec["queries_post_rope"].float()
        positions_all = rec["positions"].long()
        scaling = float(rec["scaling"])
        n_kv, n_keys, head_dim = keys.shape
        n_q_heads = queries.shape[0]
        group = n_q_heads // n_kv
        rotary = rotary_per_layer.get(layer, default_rotary)
        if rotary is None:
            raise SystemExit(f"no rotary_dim for layer {layer}")
        q_total = queries.shape[1]
        n_q = min(int(args.n_queries), q_total)
        offset = q_total - n_q
        positions = positions_all[offset:]
        kv_heads = (
            list(range(min(int(args.max_kv_heads), n_kv)))
            if args.max_kv_heads
            else list(range(n_kv))
        )

        log(
            f"layer {layer}: keys {tuple(keys.shape)} queries {tuple(queries.shape)} "
            f"group {group} rotary_dim {rotary} n_queries {n_q} "
            f"(load {time.time() - t0:.1f}s)"
        )

        true_weights: dict[int, list[torch.Tensor]] = {}
        for kv in kv_heads:
            per_head = []
            for h in range(kv * group, kv * group + group):
                w, dt = timed(
                    f"layer {layer} kv {kv} head {h} true attention",
                    lambda h=h, kv=kv: metrics.attention_weights(
                        queries[h, offset : offset + n_q], keys[kv], scaling, positions
                    ),
                )
                if dt > 30:
                    log(f"  true attention layer {layer} head {h} took {dt:.1f}s")
                per_head.append(w)
            true_weights[kv] = per_head

        layer_meta_out.append(
            {
                "layer": layer,
                "query_heads": n_q_heads,
                "kv_heads": n_kv,
                "group_size": group,
                "head_dim": head_dim,
                "rotary_dim": rotary,
                "scaling": scaling,
                "n_keys": n_keys,
                "query_indices": [offset + i for i in range(n_q)],
            }
        )

        for kv in kv_heads:
            for mode in ("full", "posfree"):
                ik = make_index_keys(keys[kv], mode, rotary)
                iq = make_index_queries(queries, kv, group, offset, n_q, mode, rotary)

                for cfg in configs:
                    cache_key = (layer, kv, mode, cfg.bucket_size)
                    if cache_key not in index_cache:
                        flat, flat_s = timed(
                            f"index layer {layer} kv {kv} {mode} bs {cfg.bucket_size} flat",
                            lambda: build_flat_index(ik, cfg),
                        )
                        tree, tree_s = timed(
                            f"index layer {layer} kv {kv} {mode} bs {cfg.bucket_size} tree",
                            lambda: build_tree_index(ik, cfg),
                        )
                        index_cache[cache_key] = (flat, flat_s, tree, tree_s)
                        log(
                            f"  index layer {layer} kv {kv} {mode} bs {cfg.bucket_size}: "
                            f"flat {flat_s:.2f}s tree {tree_s:.2f}s"
                        )
                    flat_index, flat_s, tree_index, tree_s = index_cache[cache_key]

                    selections = {}
                    for which, index, build_s in (
                        ("flat", flat_index, flat_s),
                        ("tree", tree_index, tree_s),
                    ):
                        selected, lengths = select(index, which, iq, positions, cfg)
                        selections[which] = (selected, lengths, build_s)
                        records.append(
                            build_record(
                                layer, kv, which, which, mode, cfg, selected, lengths,
                                build_s, positions, n_keys, true_weights[kv], values[kv],
                            )
                        )
                        add_samples(
                            samples, layer, kv, group, which, which, mode, cfg, 0,
                            true_weights[kv], selected, lengths, offset,
                        )
                        record = records[-1]
                        log(
                            f"  layer {layer} kv {kv} {mode} {which} "
                            f"tp {cfg.top_p} tk {cfg.top_k} bs {cfg.bucket_size}: "
                            f"mass {record['mass_captured']:.4f} "
                            f"len/N {record['budget_fraction']:.4f} "
                            f"build {build_s:.2f}s"
                        )

                    for matched_to in ("flat", "tree"):
                        base_selected, base_lengths, build_s = selections[matched_to]
                        rand, rand_lengths = random_control(
                            base_lengths, positions, args.seed + 17 * layer + kv
                        )
                        recency, recency_lengths = recency_control(base_lengths, positions)
                        for name, (sel, lens) in (
                            ("random", (rand, rand_lengths)),
                            ("recency", (recency, recency_lengths)),
                        ):
                            records.append(
                                build_record(
                                    layer, kv, name, matched_to, mode, cfg, sel, lens,
                                    build_s, positions, n_keys, true_weights[kv], values[kv],
                                )
                            )
                            add_samples(
                                samples, layer, kv, group, name, matched_to, mode, cfg, 0,
                                true_weights[kv], sel, lens, offset,
                            )
                        for h in range(group):
                            sel_h, lens_h = exhaustive_control(
                                true_weights[kv][h], base_lengths
                            )
                            records.append(
                                build_record(
                                    layer, kv, "exhaustive", matched_to, mode, cfg,
                                    sel_h, lens_h, build_s, positions, n_keys,
                                    [true_weights[kv][h]], values[kv],
                                )
                            )
                            add_samples(
                                samples, layer, kv, group, "exhaustive", matched_to, mode,
                                cfg, h, [true_weights[kv][h]], sel_h, lens_h, offset,
                            )

        log(f"layer {layer} done in {time.time() - t0:.1f}s ({len(records)} records)")

    return {
        "meta": {
            "git_commit": git_commit(),
            "dump": str(dump),
            "dump_meta_path": str(meta_path),
            "dump_git_commit": dump_meta.get("git_commit"),
            "torch_version": torch.__version__,
            "device": "cpu",
            "seed": args.seed,
            "args": vars(args),
            "sink_size": SINK_SIZE,
            "window_size": WINDOW_SIZE,
            "index_modes": ["full", "posfree"],
            "quick": bool(args.quick),
            "configs": [config_dict(cfg) for cfg in configs],
            "grids": {
                "quick": QUICK if args.quick else None,
                "bucket_sizes": list(BUCKET_SIZES),
                "top_ps": list(TOP_PS),
                "top_ks": list(TOP_KS),
            },
            "layers": layer_meta_out,
            "note": (
                "Index keys and group query are unit-normalised rows. posfree uses "
                "dims [rotary_dim:] of the post-RoPE tensors. Controls are matched to "
                "the flat/tree selection length per query; exhaustive is the scored "
                "head's true top-count causal keys."
            ),
        },
        "records": records,
        "samples": samples,
    }


def select(index, which, q, positions, cfg):
    if which == "flat":
        return flat_select(q, positions, index, cfg)
    return tree_select(q, positions, index, cfg)


# ---------------------------------------------------------------------------
# Paper mode (B2.1): one layer-shared PCA index over the RoPE-free dims of every
# KV head, scored against the paper's head-averaged long-range attention target.
# ---------------------------------------------------------------------------

PAPER_D_IDX = 64
PAPER_N_POOL = 4
PAPER_SINK_SIZE = 4
PAPER_N_CLUSTER = 512
PAPER_BUCKET_SIZE = 8
PAPER_BUCKET_CAP = 16
PAPER_K = 2048


def make_paper_config(n_cluster: int, k: int, tree: str = "scaled") -> OracleConfig:
    """The paper's flat/tree knobs (Table 2), scaled to our context by ``n_cluster`` and ``k``.

    ``tree="upstream"`` uses the upstream production tree settings for Qwen3.8-27B
    (``configs/evaluation/production/qwen38_27b/lsa_tree_tp1.yaml``: top_level 16, node bucket 4,
    node cluster block 4096, beam 2048, start nodes 32768). Those only prune once a context holds
    more candidates than the beam, which is around 1M tokens.
    """
    up = tree == "upstream"
    return OracleConfig(
        pool_size=PAPER_N_POOL,
        bucket_size=PAPER_BUCKET_SIZE,
        bucket_cap=PAPER_BUCKET_CAP,
        cluster_block_size=n_cluster,
        cluster_algorithm="ward",
        top_p=max(1, k // 4),
        top_k=k,
        sink_size=PAPER_SINK_SIZE,
        window_size=n_cluster,
        # Tree: paper Table 2 uses node buckets of 4, 4096-bucket blocks, beam 2048.
        top_level=16 if up else 4,
        node_bucket_size=4,
        node_cluster_block_size=4096 if up else max(4, n_cluster // PAPER_BUCKET_SIZE),
        node_cluster_algorithm="ward",
        beam_width=2048 if up else min(2048, max(8, k)),
        start_nodes=32768 if up else None,
        page_size=max(n_cluster, ((n_cluster + 511) // 512) * 512),
        growth_buckets=0,
    )


def paper_index_keys(keys_post_rope: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """``[T, Hkv*(D-rotary_dim)]`` RoPE-free dims of every KV head, concatenated per token."""
    k = keys_post_rope.float()[:, :, rotary_dim:]
    return k.permute(1, 0, 2).reshape(k.shape[1], -1)


def paper_raw_queries(
    queries_post_rope: torch.Tensor, group: int, offset: int, n_q: int, rotary_dim: int
) -> torch.Tensor:
    """``[n_q, Hkv*(D-rotary_dim)]`` RoPE-free dims of each group's summed query heads."""
    q = queries_post_rope[:, offset : offset + n_q].float()
    n_kv = q.shape[0] // group
    summed = q.reshape(n_kv, group, n_q, q.shape[-1]).sum(dim=1)
    return summed[:, :, rotary_dim:].permute(1, 0, 2).reshape(n_q, -1)


def fit_pca(keys: torch.Tensor, d_idx: int) -> torch.Tensor:
    """Top ``d_idx`` principal directions of the centred keys, as a ``[d_idx, E]`` basis."""
    x = keys.float()
    centred = x - x.mean(dim=0, keepdim=True)
    q = min(d_idx, min(centred.shape))
    _, _, basis = torch.pca_lowrank(centred, q=q, center=False)
    return basis[:, :q].T.contiguous()


def project_pca(x: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return unit_rows((x.float() - mean) @ basis.T)


def build_paper_record(
    layer, method, matched_to, cfg, selected, lengths, positions, n_keys,
    weights_list, values, sink_size, window_size, d_idx, n_cluster, k, p,
    rotary_dim, pca_source, build_s,
):
    """Score one selection on the paper's head-averaged long-range target."""
    n_q = weights_list[0].shape[0]
    targets = []
    regions = []
    for i in range(n_q):
        pos = int(positions[i])
        lo = min(sink_size, pos + 1)
        hi = min(max(lo, pos - window_size + 1), n_keys)
        region = torch.arange(lo, hi)
        regions.append(region)
        if region.numel() == 0:
            targets.append(torch.zeros(0))
            continue
        acc = torch.zeros(region.numel())
        for w in weights_list:
            mass = w[i, region].float()
            acc += mass / mass.sum().clamp_min(1e-30)
        targets.append(acc / len(weights_list))

    masses = []
    recalls = []
    for i in range(n_q):
        region = regions[i]
        count = int(lengths[i])
        sel = selected[i, :count].to(torch.long)
        target = targets[i]
        if region.numel() and target.numel():
            in_region = (sel >= region[0]) & (sel < region[-1] + 1)
            idx = sel[in_region] - region[0]
            masses.append(float(target[idx].sum().item()) if idx.numel() else 0.0)
            top_n = min(2048, max(1, target.numel() // 2))
            order = torch.argsort(-target, stable=True)[:top_n]
            hit = torch.zeros(target.numel(), dtype=torch.bool)
            hit[idx] = True
            recalls.append(float(hit[order].sum().item()) / float(top_n))
        else:
            masses.append(0.0)
            recalls.append(0.0)

    # Mass captured per head (the old metric) for continuity.
    per_head = []
    for w in weights_list:
        m = metrics.mass_captured(w, selected, lengths, sink_size=sink_size)
        per_head.append(mean_of(m["total_mass"]))
    mass_captured = sum(per_head) / len(per_head)

    return {
        "layer": layer,
        "kv_head": "all",
        "method": method,
        "index_mode": "paper-pca",
        "matched_to": matched_to,
        "config": config_dict(cfg),
        "mass_captured": mass_captured,
        "mass_excluding_sinks": 0.0,
        "mass_outside_headavg": sum(masses) / len(masses) if masses else 0.0,
        "top_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "efficiency": 0.0,
        "recall_at_k": 0.0,
        "output_error": 0.0,
        "budget_fraction": mean_of(lengths.float() / float(n_keys)),
        "index_build_s": float(build_s),
        "d_idx": d_idx,
        "n_cluster": n_cluster,
        "window_size": window_size,
        "sink_size": sink_size,
        "k": k,
        "p": p,
        "rotary_dim": rotary_dim,
        "pca_source": pca_source,
        "mass_outside_by_query": masses,
        "top_recall_by_query": recalls,
    }


def paper_run(args) -> dict:
    torch.manual_seed(args.seed)
    dump = Path(args.dump).resolve()
    meta_path = dump / "meta.json"
    dump_meta = json.loads(meta_path.read_text())
    full_layers = list(dump_meta["full_attention_layer_indices"])
    if args.layers.strip().lower() == "all":
        layers = full_layers
    else:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
        missing = [x for x in layers if x not in full_layers]
        if missing:
            raise SystemExit(f"layers {missing} are not full-attention layers of {dump}")

    rotary_per_layer = {
        int(a): int(b) for a, b in (dump_meta.get("rotary_dim_per_layer") or {}).items()
    }
    default_rotary = (dump_meta.get("rope") or {}).get("rotary_dim_from_config")

    n_cluster = int(args.n_cluster)
    k = int(args.k)
    p = max(1, k // 4)
    if n_cluster % PAPER_BUCKET_SIZE:
        raise SystemExit(f"--n-cluster must be a multiple of {PAPER_BUCKET_SIZE}")
    cfg = make_paper_config(n_cluster, k, getattr(args, 'tree', 'scaled'))
    if args.quick:
        cfg = make_paper_config(n_cluster, min(k, 64), getattr(args, 'tree', 'scaled'))
        k, p = cfg.top_k, cfg.top_p

    pca_from = None
    pca_from_meta = None
    if args.pca_from:
        pca_path = Path(args.pca_from).resolve()
        pca_from = torch.load(pca_path, map_location="cpu")
        pca_from_meta = str(pca_path)

    records: list[dict] = []
    samples: list[dict] = []
    layer_meta_out: list[dict] = []
    pca_bases: dict[int, tuple] = {}

    for layer in layers:
        layer_path = dump / f"layer_{layer:02d}.pt"
        if not layer_path.exists():
            raise SystemExit(f"missing {layer_path}")
        t0 = time.time()
        rec = torch.load(layer_path, map_location="cpu")
        keys = rec["keys_post_rope"].float()
        values = rec["values"].float()
        queries = rec["queries_post_rope"].float()
        positions_all = rec["positions"].long()
        scaling = float(rec["scaling"])
        n_kv, n_keys, head_dim = keys.shape
        n_q_heads = queries.shape[0]
        group = n_q_heads // n_kv
        rotary = rotary_per_layer.get(layer, default_rotary)
        if rotary is None:
            raise SystemExit(f"no rotary_dim for layer {layer}")
        q_total = queries.shape[1]
        n_q = min(int(args.n_queries), q_total)
        offset = q_total - n_q
        positions = positions_all[offset:]

        raw_keys = paper_index_keys(keys, rotary)
        raw_queries = paper_raw_queries(queries, group, offset, n_q, rotary)

        if pca_from is not None:
            entry = pca_from.get(str(layer)) if isinstance(pca_from, dict) else None
            if entry is None and isinstance(pca_from, dict):
                entry = pca_from.get(layer)
            if entry is None:
                raise SystemExit(f"--pca-from has no basis for layer {layer}")
            mean, basis = entry["mean"].float(), entry["basis"].float()
            pca_source = f"pca-from:{pca_from_meta}"
        else:
            mean = raw_keys.mean(dim=0, keepdim=True)
            basis = fit_pca(raw_keys, PAPER_D_IDX)
            pca_source = "fit-on-this-prompt"
            pca_bases[layer] = {"mean": mean, "basis": basis}

        index_keys = project_pca(raw_keys, mean, basis)
        index_queries = project_pca(raw_queries, mean, basis)

        true_weights = []
        for h in range(n_q_heads):
            w, _ = timed(
                f"paper layer {layer} head {h} true attention",
                lambda h=h: metrics.attention_weights(
                    queries[h, offset : offset + n_q], keys[h // group], scaling, positions
                ),
            )
            true_weights.append(w)

        layer_meta_out.append(
            {
                "layer": layer,
                "query_heads": n_q_heads,
                "kv_heads": n_kv,
                "group_size": group,
                "head_dim": head_dim,
                "rotary_dim": rotary,
                "scaling": scaling,
                "n_keys": n_keys,
                "query_indices": [offset + i for i in range(n_q)],
                "index_dim": int(index_keys.shape[1]),
                "pca_d_idx": int(basis.shape[0]),
                "pca_source": pca_source,
            }
        )

        flat, flat_s = timed(
            f"paper layer {layer} flat index", lambda: build_flat_index(index_keys, cfg)
        )
        tree, tree_s = timed(
            f"paper layer {layer} tree index", lambda: build_tree_index(index_keys, cfg)
        )
        log(
            f"paper layer {layer}: index dim {index_keys.shape[1]} "
            f"flat {flat_s:.2f}s tree {tree_s:.2f}s"
        )

        selections = {}
        for which, index, build_s in (("flat", flat, flat_s), ("tree", tree, tree_s)):
            selected, lengths = select(index, which, index_queries, positions, cfg)
            selections[which] = (selected, lengths, build_s)
            rec_out = build_paper_record(
                layer, which, which, cfg, selected, lengths, positions, n_keys,
                true_weights, values, cfg.sink_size, cfg.window_size, PAPER_D_IDX,
                n_cluster, k, p, rotary, pca_source, build_s,
            )
            records.append(rec_out)
            add_paper_sample(
                samples, layer, offset, which, which, selected, lengths, rec_out, cfg
            )
            log(
                f"  paper layer {layer} {which}: outside {rec_out['mass_outside_headavg']:.4f} "
                f"recall {rec_out['top_recall']:.4f} mass {rec_out['mass_captured']:.4f}"
            )

        for matched_to in ("flat", "tree"):
            base_selected, base_lengths, build_s = selections[matched_to]
            rand, rand_lengths = random_control(
                base_lengths, positions, args.seed + 17 * layer
            )
            recency, recency_lengths = recency_control(base_lengths, positions)
            for name, (sel, lens) in (
                ("random", (rand, rand_lengths)),
                ("recency", (recency, recency_lengths)),
            ):
                rec_out = build_paper_record(
                    layer, name, matched_to, cfg, sel, lens, positions, n_keys,
                    true_weights, values, cfg.sink_size, cfg.window_size, PAPER_D_IDX,
                    n_cluster, k, p, rotary, pca_source, build_s,
                )
                records.append(rec_out)
                add_paper_sample(
                    samples, layer, offset, name, matched_to, sel, lens, rec_out, cfg
                )
            # Exhaustive control: the true top-|S| keys of the head-averaged target, per query.
            ex_sel, ex_lens = exhaustive_paper_control(
                true_weights, positions, base_lengths, cfg.sink_size, cfg.window_size
            )
            rec_out = build_paper_record(
                layer, "exhaustive", matched_to, cfg, ex_sel, ex_lens, positions, n_keys,
                true_weights, values, cfg.sink_size, cfg.window_size, PAPER_D_IDX,
                n_cluster, k, p, rotary, pca_source, build_s,
            )
            records.append(rec_out)
            add_paper_sample(
                samples, layer, offset, "exhaustive", matched_to, ex_sel, ex_lens,
                rec_out, cfg,
            )

        # Optional block-sparse baseline (QSA/HiP family), on the raw RoPE-free dims.
        try:
            from lsa_rocm.baselines import block_sparse_select
        except Exception:
            block_sparse_select = None
        if block_sparse_select is not None:
            for block_size in (4, 8):
                n_blocks = max(1, k // block_size)
                bs_sel, bs_len = block_sparse_select(
                    raw_queries, raw_keys, positions, block_size=block_size,
                    n_blocks=n_blocks, sink_size=cfg.sink_size,
                    window_size=cfg.window_size,
                )
                rec_out = build_paper_record(
                    layer, "block_sparse", "block_sparse", cfg, bs_sel, bs_len,
                    positions, n_keys, true_weights, values, cfg.sink_size,
                    cfg.window_size, PAPER_D_IDX, n_cluster, k, p, rotary,
                    pca_source, 0.0,
                )
                rec_out["block_size"] = block_size
                records.append(rec_out)
                add_paper_sample(
                    samples, layer, offset, "block_sparse", "block_sparse", bs_sel,
                    bs_len, rec_out, cfg,
                )
                # Its own matched controls.
                rand, rand_lengths = random_control(
                    bs_len, positions, args.seed + 31 * layer + block_size
                )
                recency, recency_lengths = recency_control(bs_len, positions)
                for name, (sel, lens) in (
                    ("random", (rand, rand_lengths)),
                    ("recency", (recency, recency_lengths)),
                ):
                    ctrl = build_paper_record(
                        layer, name, "block_sparse", cfg, sel, lens, positions,
                        n_keys, true_weights, values, cfg.sink_size,
                        cfg.window_size, PAPER_D_IDX, n_cluster, k, p, rotary,
                        pca_source, 0.0,
                    )
                    ctrl["block_size"] = block_size
                    records.append(ctrl)
                ex_sel, ex_lens = exhaustive_paper_control(
                    true_weights, positions, bs_len, cfg.sink_size, cfg.window_size
                )
                ex_rec = build_paper_record(
                    layer, "exhaustive", "block_sparse", cfg, ex_sel, ex_lens,
                    positions, n_keys, true_weights, values, cfg.sink_size,
                    cfg.window_size, PAPER_D_IDX, n_cluster, k, p, rotary,
                    pca_source, 0.0,
                )
                ex_rec["block_size"] = block_size
                records.append(ex_rec)

        log(f"paper layer {layer} done in {time.time() - t0:.1f}s")

    # Efficiency: mass_outside_headavg relative to the exhaustive control at the same count.
    by_group = {}
    for r in records:
        grp = r["method"] if r["method"] in ("flat", "tree") else r["matched_to"]
        key = (r["layer"], grp, r.get("block_size"))
        by_group.setdefault(key, {})[r["method"]] = r
    for (layer, grp, _block), methods in by_group.items():
        base = methods.get("exhaustive")
        if not base:
            continue
        denom = base["mass_outside_headavg"]
        for r in methods.values():
            r["efficiency"] = (
                r["mass_outside_headavg"] / denom if denom > 0 else 0.0
            )

    meta = {
        "git_commit": git_commit(),
        "dump": str(dump),
        "dump_meta_path": str(meta_path),
        "dump_git_commit": dump_meta.get("git_commit"),
        "torch_version": torch.__version__,
        "device": "cpu",
        "seed": args.seed,
        "mode": "paper",
        "args": vars(args),
        "d_idx": PAPER_D_IDX,
        "n_pool": PAPER_N_POOL,
        "n_cluster": n_cluster,
        "window_size": n_cluster,
        "sink_size": PAPER_SINK_SIZE,
        "k": k,
        "p": p,
        "rotary_dim": int(default_rotary) if default_rotary is not None else None,
        "rotary_dim_per_layer": rotary_per_layer,
        "index_dims_source": "rope_free_dims_all_kv_heads",
        "pca_fit_source": (
            f"pca-from:{pca_from_meta}" if pca_from is not None else "fit-on-this-prompt"
        ),
        "bucket_size": PAPER_BUCKET_SIZE,
        "bucket_cap": PAPER_BUCKET_CAP,
        "top_level": cfg.top_level,
        "node_bucket_size": cfg.node_bucket_size,
        "node_cluster_block_size": cfg.node_cluster_block_size,
        "beam_width": cfg.beam_width,
        "quick": bool(args.quick),
        "config": config_dict(cfg),
        "layers": layer_meta_out,
        "note": (
            "Paper mode: one layer-shared index per layer over the concatenated "
            "RoPE-free dims of every KV head, projected onto the layer's top-d_idx "
            "principal directions (centred), then L2 normalised. mass_outside_headavg "
            "is the paper target: A_i = mean over query heads of softmax over "
            "R_i = [sink_size, pos - window_size], summed over selected keys in R_i. "
            "top_recall is the share of the top min(2048, |R_i|//2) A_i keys selected. "
            "efficiency = method / exhaustive control at the same selection count."
        ),
    }
    return {"meta": meta, "records": records, "samples": samples}


def exhaustive_paper_control(
    weights_list, positions, lengths, sink_size, window_size
):
    """The true top-``lengths[i]`` keys of ``A_i`` inside ``R_i``, plus the window and sinks."""
    n_q = weights_list[0].shape[0]
    n_keys = weights_list[0].shape[1]
    width = int(lengths.max().item()) if lengths.numel() else 0
    rows = []
    for i in range(n_q):
        pos = int(positions[i])
        lo = min(sink_size, pos + 1)
        hi = min(max(lo, pos - window_size + 1), n_keys)
        region = torch.arange(lo, hi)
        count = min(int(lengths[i]), max(0, region.numel()))
        if region.numel():
            acc = torch.zeros(region.numel())
            for w in weights_list:
                mass = w[i, region].float()
                acc += mass / mass.sum().clamp_min(1e-30)
            acc /= len(weights_list)
            order = region[torch.argsort(-acc, stable=True)[:count]]
        else:
            order = region[:0]
        rows.append(order)
    return pad_rows(rows, width)


def add_paper_sample(
    samples, layer, offset, method, matched_to, selected, lengths, record, cfg
):
    if len(samples) >= SAMPLE_CAP:
        return
    count = int(lengths[0])
    samples.append(
        {
            "layer": layer,
            "query_head": "all",
            "query_index": offset,
            "selected": [int(x) for x in selected[0, :count].tolist()],
            "mass_outside_headavg": record["mass_outside_by_query"][0],
            "mass_captured": record["mass_captured"],
            "method": method,
            "matched_to": matched_to,
            "index_mode": "paper-pca",
        }
    )


def paper_pca_from(args) -> None:
    """Fit per-layer PCA bases on one dump and write ``{layer: {mean, basis}}`` for ``--pca-from``."""
    dump = Path(args.dump).resolve()
    meta = json.loads((dump / "meta.json").read_text())
    full_layers = list(meta["full_attention_layer_indices"])
    rotary_per_layer = {
        int(a): int(b) for a, b in (meta.get("rotary_dim_per_layer") or {}).items()
    }
    default_rotary = (meta.get("rope") or {}).get("rotary_dim_from_config")
    bases = {}
    for layer in full_layers:
        rec = torch.load(dump / f"layer_{layer:02d}.pt", map_location="cpu")
        rotary = rotary_per_layer.get(layer, default_rotary)
        raw = paper_index_keys(rec["keys_post_rope"].float(), rotary)
        mean = raw.mean(dim=0, keepdim=True)
        basis = fit_pca(raw, PAPER_D_IDX)
        bases[str(layer)] = {"mean": mean, "basis": basis}
        log(f"pca layer {layer}: basis {tuple(basis.shape)}")
    out = Path(args.pca_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bases, out)
    log(f"WROTE {out}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", default="all")
    p.add_argument("--max-kv-heads", type=int, default=None)
    p.add_argument("--n-queries", type=int, default=8)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=("legacy", "paper"), default="legacy")
    p.add_argument("--n-cluster", type=int, default=PAPER_N_CLUSTER)
    p.add_argument("--tree", choices=("scaled", "upstream"), default="scaled", help="tree knobs: scaled to short contexts (default) or the upstream production settings (long context)")
    p.add_argument("--k", type=int, default=PAPER_K)
    p.add_argument("--pca-from", default=None)
    p.add_argument("--pca-out", default=None,
                   help="with --mode paper --pca-out and no evaluation inputs, fit and save PCA bases")
    args = p.parse_args(argv)

    if args.mode == "paper":
        if args.pca_out and args.out.strip().lower() in ("", "none", "-"):
            paper_pca_from(args)
            return
        result = paper_run(args)
    else:
        result = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    log(
        f"WROTE {out} ({len(result['records'])} records, "
        f"{len(result['samples'])} samples) in {time.time() - START:.1f}s"
    )


if __name__ == "__main__":
    main()
