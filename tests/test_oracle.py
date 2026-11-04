"""Tests for the ``lsa.core`` CPU oracle wrapper (``src/lsa_rocm/oracle.py``).

Run with the CPU venv, either directly (which also writes ``results/b1b-oracle-tests.json``):

    PYTHONPATH=src agents/scratch/venv-lsa-min/bin/python -I tests/test_oracle.py

or under pytest:

    PYTHONPATH=src agents/scratch/venv-lsa/bin/python -m pytest tests/test_oracle.py -v

The synthetic keys are 32 well separated Gaussian clusters in 32 dims (256 keys a cluster), and the
queries are noisy copies of a cluster centroid placed in the last page, so the causal rule leaves every
long-range candidate in a closed block. The baseline is the true dot-product top-k over the same
long-range candidates, with the same window and sinks.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lsa_rocm.oracle import (  # noqa: E402
    FlatIndex,
    OracleConfig,
    TreeIndex,
    build_flat_index,
    build_tree_index,
    exhaustive_select,
    flat_select,
    tree_select,
)

D, N, C = 32, 8192, 32
NQ = 64
SINK, WINDOW = 4, 128
RECALL_TOP_P, RECALL_TOP_K = 32, 256
SMALL_TOP_P, SMALL_TOP_K = 8, 64
SIGMA = 0.15

RECALL_CFG = OracleConfig(
    top_p=RECALL_TOP_P, top_k=RECALL_TOP_K, sink_size=SINK, window_size=WINDOW
)
SMALL_CFG = OracleConfig(
    top_p=SMALL_TOP_P, top_k=SMALL_TOP_K, sink_size=SINK, window_size=WINDOW
)

_STORE: dict[str, tuple] = {}
_RESULTS: dict[str, dict] = {}


def clustered_keys(seed: int = 0, sigma: float = SIGMA) -> tuple[torch.Tensor, torch.Tensor]:
    """32 well separated Gaussian clusters, 256 keys a cluster, as unit rows."""
    generator = torch.Generator().manual_seed(seed)
    centroids = F.normalize(torch.randn(C, D, generator=generator), dim=-1)
    assign = torch.arange(N) // (N // C)
    keys = F.normalize(
        centroids[assign] + sigma * torch.randn(N, D, generator=generator), dim=-1
    )
    return keys, centroids


def random_keys(seed: int = 2) -> torch.Tensor:
    """Random unit rows, no cluster structure."""
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(N, D, generator=generator), dim=-1)


def make_queries(
    centroids: torch.Tensor, nq: int = NQ, seed: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Noisy copies of a random cluster centroid, at positions in the last page."""
    generator = torch.Generator().manual_seed(seed)
    positions = torch.arange(N - 1, N - 1 - 3 * nq, -3)[:nq].to(torch.int64)
    clusters = torch.randint(0, C, (nq,), generator=generator)
    queries = F.normalize(
        centroids[clusters] + 0.05 * torch.randn(nq, D, generator=generator), dim=-1
    )
    return queries, positions, clusters


def _build_store(name: str, keys: torch.Tensor, queries: torch.Tensor, positions: torch.Tensor,
                 cfg: OracleConfig) -> tuple:
    if name not in _STORE:
        _STORE[name] = (
            keys,
            queries,
            positions,
            cfg,
            build_flat_index(keys, cfg),
            build_tree_index(keys, cfg),
        )
    return _STORE[name]


def clustered_store() -> tuple:
    keys, centroids = clustered_keys()
    queries, positions, _ = make_queries(centroids)
    return _build_store("clustered", keys, queries, positions, RECALL_CFG)


def random_store() -> tuple:
    keys = random_keys()
    generator = torch.Generator().manual_seed(3)
    positions = torch.arange(N - 1, N - 1 - 3 * NQ, -3)[:NQ].to(torch.int64)
    queries = F.normalize(torch.randn(NQ, D, generator=generator), dim=-1)
    return _build_store("random", keys, queries, positions, RECALL_CFG)


def long_range(
    selected: torch.Tensor, lengths: torch.Tensor, pos: torch.Tensor, sink_size: int, window_size: int
) -> list[torch.Tensor]:
    """Each row's long-range keys: the part before the window and at or past the sinks."""
    rows = []
    for row, position in enumerate(pos.tolist()):
        held = selected[row, : int(lengths[row])].long()
        window_start = max(0, position + 1 - window_size)
        rows.append(held[(held >= sink_size) & (held < window_start)])
    return rows


def recall_against(
    selected: torch.Tensor,
    lengths: torch.Tensor,
    pos: torch.Tensor,
    cfg: OracleConfig,
    exhaustive: list[torch.Tensor],
) -> float:
    """Fraction of the exhaustive top-k long-range keys a selection holds."""
    got = long_range(selected, lengths, pos, cfg.sink_size, cfg.window_size)
    hits = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(got, exhaustive))
    return hits / (cfg.top_k * len(got))


def baseline(
    keys: torch.Tensor, queries: torch.Tensor, pos: torch.Tensor, cfg: OracleConfig
) -> list[torch.Tensor]:
    selected, lengths = exhaustive_select(
        queries, keys, pos, cfg.top_k, cfg.sink_size, cfg.window_size
    )
    return long_range(selected, lengths, pos, cfg.sink_size, cfg.window_size)


def test_determinism_same_seed_identical_selections() -> None:
    """The same seed gives the same keys and identical flat, tree and exhaustive selections."""
    keys, centroids = clustered_keys(seed=7)
    again, _ = clustered_keys(seed=7)
    assert torch.equal(keys, again)
    queries, positions, _ = make_queries(centroids, nq=8, seed=3)

    first = build_flat_index(keys, SMALL_CFG)
    second = build_flat_index(again, SMALL_CFG)
    assert torch.equal(first.buckets.bucket_members, second.buckets.bucket_members)
    assert torch.equal(first.buckets.bucket_centroids, second.buckets.bucket_centroids)
    flat_a = flat_select(queries, positions, first, SMALL_CFG)
    flat_b = flat_select(queries, positions, second, SMALL_CFG)
    assert torch.equal(flat_a[0], flat_b[0]) and torch.equal(flat_a[1], flat_b[1])

    tree_a = tree_select(queries, positions, build_tree_index(keys, SMALL_CFG), SMALL_CFG)
    tree_b = tree_select(queries, positions, build_tree_index(again, SMALL_CFG), SMALL_CFG)
    assert torch.equal(tree_a[0], tree_b[0]) and torch.equal(tree_a[1], tree_b[1])

    exhaustive_a = exhaustive_select(queries, keys, positions, SMALL_TOP_K, SINK, WINDOW)
    exhaustive_b = exhaustive_select(queries, again, positions, SMALL_TOP_K, SINK, WINDOW)
    assert torch.equal(exhaustive_a[0], exhaustive_b[0])
    assert torch.equal(exhaustive_a[1], exhaustive_b[1])


def test_selected_indices_are_valid_and_causal() -> None:
    """Every selected id is a real key at or below the query, read once, with -1 padding after."""
    keys, queries, positions, cfg, flat, tree = clustered_store()
    early = torch.tensor([0, 1, 5, 63, 64, 100, 200, 1000, N - 1], dtype=torch.int64)
    generator = torch.Generator().manual_seed(11)
    early_queries = F.normalize(torch.randn(len(early), D, generator=generator), dim=-1)
    width = cfg.top_k + cfg.window_size + cfg.sink_size

    selections = {
        "flat": flat_select(early_queries, early, flat, cfg),
        "tree": tree_select(early_queries, early, tree, cfg),
        "exhaustive": exhaustive_select(
            early_queries, keys, early, cfg.top_k, cfg.sink_size, cfg.window_size
        ),
    }
    for name, (selected, lengths) in selections.items():
        assert selected.dtype == torch.int32, name
        assert lengths.dtype == torch.int32, name
        assert selected.shape == (len(early), width), name
        assert lengths.shape == (len(early),), name
        for row, position in enumerate(early.tolist()):
            held = selected[row, : int(lengths[row])]
            assert int(lengths[row]) <= width, name
            assert bool((held >= 0).all()) and bool((held <= position).all()), name
            assert held.unique().numel() == held.numel(), name
            assert bool((selected[row, int(lengths[row]) :] == -1).all()), name


def test_sinks_and_window_are_always_included() -> None:
    """The window and the sinks are in every selection, at every position."""
    keys, _, _, cfg, flat, tree = clustered_store()
    positions = torch.tensor([0, 1, 5, 63, 64, 100, 200, 1000, N - 1], dtype=torch.int64)
    generator = torch.Generator().manual_seed(12)
    queries = F.normalize(torch.randn(len(positions), D, generator=generator), dim=-1)

    selections = {
        "flat": flat_select(queries, positions, flat, cfg),
        "tree": tree_select(queries, positions, tree, cfg),
        "exhaustive": exhaustive_select(
            queries, keys, positions, cfg.top_k, cfg.sink_size, cfg.window_size
        ),
    }
    for name, (selected, lengths) in selections.items():
        for row, position in enumerate(positions.tolist()):
            held = set(selected[row, : int(lengths[row])].tolist())
            window_start = max(0, position + 1 - cfg.window_size)
            assert set(range(window_start, position + 1)) <= held, name
            assert set(range(min(cfg.sink_size, window_start, position + 1))) <= held, name


def test_clustered_recall_is_clearly_above_chance() -> None:
    """On 32 separated clusters, flat and tree recall against the exhaustive top-k is far above chance."""
    keys, queries, positions, cfg, flat, tree = clustered_store()
    exhaustive = baseline(keys, queries, positions, cfg)
    flat_selected, flat_lengths = flat_select(queries, positions, flat, cfg)
    tree_selected, tree_lengths = tree_select(queries, positions, tree, cfg)
    flat_recall = recall_against(flat_selected, flat_lengths, positions, cfg, exhaustive)
    tree_recall = recall_against(tree_selected, tree_lengths, positions, cfg, exhaustive)
    candidates = min(positions.tolist()) + 1 - cfg.window_size - cfg.sink_size
    chance = cfg.top_k / candidates
    _RESULTS["clustered"] = {
        "flat_recall": flat_recall,
        "tree_recall": tree_recall,
        "chance": chance,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "n_queries": int(len(positions)),
        "clusters": C,
        "keys": N,
        "index_dim": D,
    }
    print(
        f"clustered: flat recall {flat_recall:.3f}, tree recall {tree_recall:.3f}, "
        f"chance {chance:.4f} (top_p={cfg.top_p}, top_k={cfg.top_k}, {len(positions)} queries)"
    )
    assert flat_recall > 0.5, flat_recall
    assert tree_recall > 0.5, tree_recall
    assert flat_recall > 10 * chance, flat_recall
    assert tree_recall > 10 * chance, tree_recall


def test_random_keys_recall_far_below_clustered() -> None:
    """On random keys the same budget is near chance, which is the contrast the table reports."""
    keys, queries, positions, cfg, flat, tree = random_store()
    exhaustive = baseline(keys, queries, positions, cfg)
    flat_selected, flat_lengths = flat_select(queries, positions, flat, cfg)
    tree_selected, tree_lengths = tree_select(queries, positions, tree, cfg)
    flat_recall = recall_against(flat_selected, flat_lengths, positions, cfg, exhaustive)
    tree_recall = recall_against(tree_selected, tree_lengths, positions, cfg, exhaustive)
    candidates = min(positions.tolist()) + 1 - cfg.window_size - cfg.sink_size
    chance = cfg.top_k / candidates
    _RESULTS["random"] = {
        "flat_recall": flat_recall,
        "tree_recall": tree_recall,
        "chance": chance,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "n_queries": int(len(positions)),
        "clusters": 0,
        "keys": N,
        "index_dim": D,
    }
    print(
        f"random:     flat recall {flat_recall:.3f}, tree recall {tree_recall:.3f}, "
        f"chance {chance:.4f} (top_p={cfg.top_p}, top_k={cfg.top_k}, {len(positions)} queries)"
    )
    if "clustered" not in _RESULTS:
        test_clustered_recall_is_clearly_above_chance()
    clustered = _RESULTS["clustered"]
    assert flat_recall < 0.3, flat_recall
    assert tree_recall < 0.3, tree_recall
    assert flat_recall < clustered["flat_recall"] / 2
    assert tree_recall < clustered["tree_recall"] / 2


def test_full_top_p_and_full_top_k_covers_every_causal_key() -> None:
    """With every bucket picked and a full long-range budget, the flat read covers every causal key once."""
    _, queries, positions, cfg, flat, _ = clustered_store()
    buckets = flat.buckets.bucket_centroids.shape[0]
    full = replace(cfg, top_p=buckets, top_k=N)
    selected, lengths = flat_select(queries, positions, flat, full)
    for row, position in enumerate(positions.tolist()):
        held = selected[row, : int(lengths[row])].long()
        assert torch.equal(held.sort().values, torch.arange(position + 1, dtype=torch.int32))


def test_full_top_p_with_a_limited_top_k_does_not_contain_the_exhaustive_top_k() -> None:
    """The actual behaviour: a full bucket budget with a limited top-k misses true top keys.

    ``expand_buckets`` takes the first ``top_k`` members in bucket-centroid order (position order within
    a bucket), not the ``top_k`` highest-scoring keys. With every bucket picked, the first ``top_k``
    members are the first ``top_k / bucket_size`` buckets' keys, so a key in a lower-centroid bucket with
    a higher dot product is missed. The containment is therefore below 1; a full long-range budget is
    what makes the selection cover every causal key (the test above).
    """
    keys, queries, positions, cfg, flat, _ = clustered_store()
    buckets = flat.buckets.bucket_centroids.shape[0]
    full = replace(cfg, top_p=buckets)
    exhaustive = baseline(keys, queries, positions, cfg)
    selected, lengths = flat_select(queries, positions, flat, full)
    got = long_range(selected, lengths, positions, cfg.sink_size, cfg.window_size)
    per_row = [
        len(set(a.tolist()) & set(b.tolist())) / cfg.top_k for a, b in zip(got, exhaustive)
    ]
    containment = sum(per_row) / len(per_row)
    _RESULTS["containment_full_top_p_limited_top_k"] = {
        "mean": containment,
        "min": min(per_row),
        "max": max(per_row),
        "top_p": buckets,
        "top_k": cfg.top_k,
        "n_buckets": buckets,
    }
    print(
        f"full top_p ({buckets} buckets), top_k={cfg.top_k}: exhaustive top-k containment "
        f"mean {containment:.3f}, min {min(per_row):.3f}, max {max(per_row):.3f}"
    )
    assert containment < 1.0, containment
    assert min(per_row) < 1.0, min(per_row)


TESTS = [
    test_determinism_same_seed_identical_selections,
    test_selected_indices_are_valid_and_causal,
    test_sinks_and_window_are_always_included,
    test_clustered_recall_is_clearly_above_chance,
    test_random_keys_recall_far_below_clustered,
    test_full_top_p_and_full_top_k_covers_every_causal_key,
    test_full_top_p_with_a_limited_top_k_does_not_contain_the_exhaustive_top_k,
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    results = []
    failures = 0
    for test in TESTS:
        started = time.perf_counter()
        try:
            test()
            status = "pass"
            detail = ""
        except Exception as error:  # noqa: BLE001 - a test runner reports every failure
            status = "fail"
            detail = f"{type(error).__name__}: {error}"
            failures += 1
        elapsed = time.perf_counter() - started
        results.append(
            {"test": f"test_oracle.{test.__name__}", "status": status, "seconds": elapsed,
             "detail": detail}
        )
        print(f"{status.upper():4} {test.__name__} ({elapsed:.3f}s) {detail}")

    clustered = _RESULTS.get("clustered", {})
    random = _RESULTS.get("random", {})
    payload = {
        "task": "B1b",
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "device": "cpu",
        "model": None,
        "seed": 0,
        "config": {
            "keys": N,
            "index_dim": D,
            "clusters": C,
            "sigma": SIGMA,
            "n_queries": NQ,
            "recall_top_p": RECALL_TOP_P,
            "recall_top_k": RECALL_TOP_K,
            "sink_size": SINK,
            "window_size": WINDOW,
            "pool_size": RECALL_CFG.pool_size,
            "bucket_size": RECALL_CFG.bucket_size,
            "bucket_cap": RECALL_CFG.bucket_cap,
            "cluster_block_size": RECALL_CFG.cluster_block_size,
            "top_level": RECALL_CFG.top_level,
            "node_bucket_size": RECALL_CFG.node_bucket_size,
            "node_cluster_block_size": RECALL_CFG.node_cluster_block_size,
            "beam_width": RECALL_CFG.beam_width,
            "page_size": RECALL_CFG.page_size,
            "growth_buckets": RECALL_CFG.growth_buckets,
        },
        "tests_run": len(TESTS),
        "failures": failures,
        "successful": failures == 0,
        "recall": {"clustered": clustered, "random": random},
        "containment": _RESULTS.get("containment_full_top_p_limited_top_k", {}),
        "tests": results,
    }
    out = ROOT / "results" / "b1b-oracle-tests.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nrecall table (long-range top-{RECALL_TOP_K} vs exhaustive):")
    print(f"  {'keys':>10} {'flat':>8} {'tree':>8} {'chance':>8}")
    print(
        f"  {'clustered':>10} {clustered.get('flat_recall', float('nan')):>8.3f} "
        f"{clustered.get('tree_recall', float('nan')):>8.3f} {clustered.get('chance', float('nan')):>8.4f}"
    )
    print(
        f"  {'random':>10} {random.get('flat_recall', float('nan')):>8.3f} "
        f"{random.get('tree_recall', float('nan')):>8.3f} {random.get('chance', float('nan')):>8.4f}"
    )
    print(f"\nwrote {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
