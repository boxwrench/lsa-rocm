"""Tests for ``lsa_rocm.baselines`` (the block-sparse QSA/HiP-style controls).

Run with the CPU venv:

    PYTHONPATH=src agents/scratch/venv-lsa-min/bin/python -I -m unittest discover -s tests -p test_baselines.py

The reference is an explicit brute-force implementation: score every closed block
outside the sinks and the recent window by its mean key, take the top ``n_blocks``,
expand, then union the sinks and window.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from lsa_rocm.baselines import block_sparse_select, block_sparse_tree_select  # noqa: E402


def _brute_force(q, keys, pos, block_size, n_blocks, sink_size, window_size):
    """Explicit reference, mirroring the verify script's definition."""
    nq = q.shape[0]
    rows = []
    for i in range(nq):
        p = int(pos[i])
        window_start = max(0, p + 1 - window_size)
        last_closed = window_start // block_size
        cand = [b for b in range(last_closed) if b * block_size >= sink_size]
        if cand:
            scores = torch.stack(
                [keys[b * block_size : (b + 1) * block_size].mean(0) @ q[i] for b in cand]
            )
            keep = min(n_blocks, len(cand))
            order = torch.argsort(-scores, stable=True)[:keep]
            top = [cand[int(j)] for j in order]
        else:
            top = []
        chosen = set()
        for b in top:
            chosen |= set(range(b * block_size, (b + 1) * block_size))
        window = set(range(window_start, p + 1))
        sinks = set(range(min(sink_size, window_start, p + 1)))
        rows.append(chosen | window | sinks)
    return rows


def _as_sets(selected, lengths):
    return [set(selected[i, : int(lengths[i])].tolist()) for i in range(selected.shape[0])]


class BlockSparseSelectTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.n, self.d = 1024, 32
        self.keys = torch.nn.functional.normalize(torch.randn(self.n, self.d), dim=-1)
        self.q = torch.nn.functional.normalize(torch.randn(5, self.d), dim=-1)
        self.pos = torch.full((5,), self.n - 1, dtype=torch.int64)

    def test_planted_block_recovered(self) -> None:
        keys = self.keys.clone()
        keys[200:208] = self.q[0].repeat(8, 1) * 1.5
        sel, ln = block_sparse_select(
            self.q, keys, self.pos, block_size=8, n_blocks=3, sink_size=4, window_size=32
        )
        got = set(sel[0, : int(ln[0])].tolist())
        self.assertTrue(set(range(200, 208)) <= got)

    def test_sinks_and_window_always_included(self) -> None:
        sink, win = 4, 32
        sel, ln = block_sparse_select(
            self.q, self.keys, self.pos, block_size=8, n_blocks=2, sink_size=sink, window_size=win
        )
        for i in range(self.q.shape[0]):
            got = set(sel[i, : int(ln[i])].tolist())
            self.assertTrue(set(range(sink)) <= got)
            self.assertTrue(set(range(self.n - win, self.n)) <= got)

    def test_causal_and_distinct(self) -> None:
        pos = torch.tensor([0, 1, 33, 500, 1023], dtype=torch.int64)
        sel, ln = block_sparse_select(
            self.q, self.keys, pos, block_size=8, n_blocks=6,
            sink_size=4, window_size=32,
        )
        for i, p in enumerate(pos.tolist()):
            row = sel[i, : int(ln[i])].tolist()
            self.assertEqual(len(row), len(set(row)), "duplicate keys in a row")
            self.assertTrue(all(0 <= j <= p for j in row), "key beyond the query position")
            self.assertTrue(bool((sel[i, int(ln[i]) :] == -1).all()))

    def test_matches_brute_force(self) -> None:
        for block_size, n_blocks, sink, win in [
            (8, 6, 4, 32),
            (4, 10, 0, 64),
            (16, 3, 8, 128),
            (8, 0, 4, 32),
        ]:
            with self.subTest(block_size=block_size, n_blocks=n_blocks):
                pos = torch.tensor([0, 17, 300, 512, 1023], dtype=torch.int64)
                sel, ln = block_sparse_select(
                    self.q, self.keys, pos, block_size=block_size,
                    n_blocks=n_blocks, sink_size=sink, window_size=win,
                )
                expected = _brute_force(
                    self.q, self.keys, pos, block_size, n_blocks, sink, win
                )
                self.assertEqual(_as_sets(sel, ln), expected)

    def test_output_format(self) -> None:
        sel, ln = block_sparse_select(
            self.q, self.keys, self.pos, block_size=8, n_blocks=4, sink_size=4, window_size=32
        )
        self.assertEqual(sel.dtype, torch.int32)
        self.assertEqual(ln.dtype, torch.int32)
        self.assertEqual(sel.shape[0], self.q.shape[0])
        self.assertEqual(ln.shape, (self.q.shape[0],))
        self.assertTrue(bool((ln == sel.shape[1] - (sel == -1).sum(1)).all()))

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            block_sparse_select(self.q, self.keys, self.pos, 0, 2, 4, 32)
        with self.assertRaises(ValueError):
            block_sparse_select(self.q, self.keys, torch.tensor([-1]), 8, 2, 4, 32)
        with self.assertRaises(ValueError):
            block_sparse_select(self.q[:, :4], self.keys, self.pos, 8, 2, 4, 32)


class BlockSparseTreeSelectTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1)
        self.n, self.d = 1024, 32
        self.keys = torch.nn.functional.normalize(torch.randn(self.n, self.d), dim=-1)
        self.q = torch.nn.functional.normalize(torch.randn(4, self.d), dim=-1)
        self.pos = torch.full((4,), self.n - 1, dtype=torch.int64)

    def test_planted_block_recovered(self) -> None:
        keys = self.keys.clone()
        keys[600:608] = self.q[1].repeat(8, 1) * 1.5
        sel, ln = block_sparse_tree_select(
            self.q, keys, self.pos, block_size=8, n_blocks=4,
            sink_size=4, window_size=32, coarse_factor=8,
        )
        got = set(sel[1, : int(ln[1])].tolist())
        self.assertTrue(set(range(600, 608)) <= got)

    def test_sinks_window_causal_distinct(self) -> None:
        pos = torch.tensor([0, 40, 511, 1023], dtype=torch.int64)
        sel, ln = block_sparse_tree_select(
            self.q, self.keys, pos, block_size=8, n_blocks=4,
            sink_size=4, window_size=32,
        )
        for i, p in enumerate(pos.tolist()):
            row = sel[i, : int(ln[i])].tolist()
            self.assertEqual(len(row), len(set(row)))
            self.assertTrue(all(0 <= j <= p for j in row))
            self.assertTrue(set(range(min(4, p + 1))) <= set(row))
            self.assertTrue(set(range(max(0, p + 1 - 32), p + 1)) <= set(row))

    def test_single_coarse_block_matches_flat(self) -> None:
        # With enough coarse blocks to expose every fine block, the tree refines to
        # the same fine blocks the flat selector would consider.
        # window_size 64 aligns the window boundary to a coarse block (992 -> 960),
        # so no candidate fine block is hidden inside a partial coarse block.
        sel, ln = block_sparse_tree_select(
            self.q, self.keys, self.pos, block_size=8, n_blocks=4,
            sink_size=4, window_size=64, coarse_factor=8, n_coarse=16,
        )
        expected = _brute_force(
            self.q, self.keys, self.pos, 8, 4, 4, 64
        )
        self.assertEqual(_as_sets(sel, ln), expected)


if __name__ == "__main__":
    unittest.main()
