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


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", default="all")
    p.add_argument("--max-kv-heads", type=int, default=None)
    p.add_argument("--n-queries", type=int, default=8)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

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
