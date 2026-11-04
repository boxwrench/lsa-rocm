"""Unit tests for lsa_rocm.metrics on hand-made tensors with known answers.

Run with the CPU venv:
    PYTHONPATH=src <cpu-venv>/bin/python -I -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from lsa_rocm.metrics import (
    attention_weights,
    budget_fraction,
    mass_captured,
    output_error,
    recall_at_k,
)


def _sel(*rows: list[int]) -> torch.Tensor:
    """Padded int32 selection; rows shorter than the widest are padded with 0."""
    width = max((len(r) for r in rows), default=0)
    padded = [r + [0] * (width - len(r)) for r in rows]
    return torch.tensor(padded, dtype=torch.int32).reshape(len(rows), width)


def _len(*values: int) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32)


class AttentionWeightsTest(unittest.TestCase):
    def test_matches_manual_softmax(self) -> None:
        q = torch.tensor([[1.0, 0.0]])
        k = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        w = attention_weights(q, k, scale=1.0)
        expected = torch.softmax(torch.tensor([[0.0, 1.0]]), dim=-1)
        self.assertTrue(torch.allclose(w, expected, atol=0, rtol=0))
        self.assertEqual(w.dtype, torch.float32)
        self.assertEqual(w.sum(dim=-1).item(), 1.0)

    def test_scale_is_applied(self) -> None:
        q = torch.tensor([[1.0]])
        k = torch.tensor([[0.0], [1.0]])
        w = attention_weights(q, k, scale=0.5)
        expected = torch.softmax(torch.tensor([[0.0, 0.5]]), dim=-1)
        self.assertTrue(torch.allclose(w, expected, atol=0, rtol=0))

    def test_causal_mask_excludes_future_keys(self) -> None:
        q = torch.tensor([[1.0], [1.0], [1.0]])
        k = torch.tensor([[0.0], [1.0], [2.0]])
        q_pos = torch.tensor([2, 1, 0], dtype=torch.int32)
        w = attention_weights(q, k, scale=1.0, q_pos=q_pos)
        # query 0 sees all three keys
        self.assertTrue(
            torch.allclose(w[0], torch.softmax(torch.tensor([0.0, 1.0, 2.0]), dim=-1))
        )
        # query 1 sees keys 0..1 only, key 2 exactly zero
        self.assertTrue(
            torch.allclose(
                w[1], torch.softmax(torch.tensor([0.0, 1.0, float("-inf")]), dim=-1)
            )
        )
        self.assertEqual(w[1, 2].item(), 0.0)
        # query 2 sees key 0 only
        self.assertEqual(w[2, 0].item(), 1.0)
        self.assertEqual(w[2, 1].item(), 0.0)
        self.assertEqual(w[2, 2].item(), 0.0)
        # each non-empty row sums to one
        self.assertTrue(torch.allclose(w.sum(dim=-1), torch.ones(3)))

    def test_causal_mask_inclusive_boundary(self) -> None:
        # q_pos == j must be allowed; q_pos == j-1 must not.
        q = torch.zeros(2, 4)
        k = torch.eye(4)
        w = attention_weights(q, k, scale=1.0, q_pos=torch.tensor([1, 3]))
        self.assertGreater(w[0, 1].item(), 0.0)
        self.assertEqual(w[0, 2].item(), 0.0)
        self.assertEqual(w[0, 3].item(), 0.0)
        self.assertGreater(w[1, 3].item(), 0.0)

    def test_fully_masked_query_gives_zero_row(self) -> None:
        # q_pos = -1 allows no keys; the row must be all zeros, not NaN,
        # and the other row must be unaffected.
        q = torch.ones(2, 2)
        k = torch.eye(3)[:, :2]
        w = attention_weights(q, k, scale=1.0, q_pos=torch.tensor([-1, 2]))
        self.assertFalse(torch.isnan(w).any())
        self.assertTrue(torch.allclose(w[0], torch.zeros(3)))
        self.assertAlmostEqual(w[1].sum().item(), 1.0, places=6)

    def test_input_validation(self) -> None:
        with self.assertRaises(ValueError):
            attention_weights(torch.zeros(3), torch.zeros(3, 4), 1.0)
        with self.assertRaises(ValueError):
            attention_weights(torch.zeros(2, 4), torch.zeros(3, 5), 1.0)
        with self.assertRaises(ValueError):
            attention_weights(
                torch.zeros(2, 4), torch.zeros(3, 4), 1.0, q_pos=torch.tensor([1])
            )


class MassCapturedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.weights = torch.tensor(
            [
                [0.5, 0.2, 0.2, 0.1],
                [0.1, 0.1, 0.1, 0.7],
            ]
        )

    def test_all_keys_selected_gives_mass_one(self) -> None:
        selected = _sel([0, 1, 2, 3], [0, 1, 2, 3])
        out = mass_captured(self.weights, selected, _len(4, 4))
        self.assertTrue(torch.allclose(out["total_mass"], torch.ones(2)))
        self.assertTrue(torch.allclose(out["mass_excluding_sinks"], torch.ones(2)))
        self.assertTrue(torch.allclose(out["sink_mass"], torch.zeros(2)))

    def test_missing_heavy_keys_gives_computed_mass(self) -> None:
        # query 0 selects keys 1 and 2 -> 0.2 + 0.2 = 0.4
        # query 1 selects key 0      -> 0.1
        selected = _sel([1, 2], [0])
        out = mass_captured(self.weights, selected, _len(2, 1))
        self.assertAlmostEqual(out["total_mass"][0].item(), 0.4, places=6)
        self.assertAlmostEqual(out["total_mass"][1].item(), 0.1, places=6)

    def test_padding_beyond_length_is_ignored(self) -> None:
        # row 0 has length 1, but padding holds key 0 (weight 0.5);
        # only key 1 (0.2) should count.
        selected = _sel([1, 0], [3, 3])
        out = mass_captured(self.weights, selected, _len(1, 1))
        self.assertAlmostEqual(out["total_mass"][0].item(), 0.2, places=6)
        self.assertAlmostEqual(out["total_mass"][1].item(), 0.7, places=6)

    def test_sink_mass_reported_separately(self) -> None:
        # sink_size=1: keys 0 are sinks. query 0 selects 0,1 -> total 0.7,
        # excluding sinks 0.2, sink 0.5. query 1 selects 3 -> all non-sink.
        selected = _sel([0, 1], [3])
        out = mass_captured(self.weights, selected, _len(2, 1), sink_size=1)
        self.assertAlmostEqual(out["total_mass"][0].item(), 0.7, places=6)
        self.assertAlmostEqual(out["mass_excluding_sinks"][0].item(), 0.2, places=6)
        self.assertAlmostEqual(out["sink_mass"][0].item(), 0.5, places=6)
        self.assertAlmostEqual(out["total_mass"][1].item(), 0.7, places=6)
        self.assertAlmostEqual(out["mass_excluding_sinks"][1].item(), 0.7, places=6)
        self.assertAlmostEqual(out["sink_mass"][1].item(), 0.0, places=6)

    def test_minus_one_padding_from_lsa_core_is_ignored(self) -> None:
        # lsa.core.pad rows with -1 beyond the length.
        selected = torch.tensor([[1, -1], [3, -1]], dtype=torch.int32)
        out = mass_captured(self.weights, selected, _len(1, 1))
        self.assertAlmostEqual(out["total_mass"][0].item(), 0.2, places=6)
        self.assertAlmostEqual(out["total_mass"][1].item(), 0.7, places=6)

    def test_empty_selection_gives_zero(self) -> None:
        selected = _sel([], [])
        out = mass_captured(self.weights, selected, _len(0, 0))
        self.assertTrue(torch.allclose(out["total_mass"], torch.zeros(2)))


class RecallAtKTest(unittest.TestCase):
    def test_all_keys_selected_gives_one(self) -> None:
        weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        selected = _sel([0, 1, 2, 3])
        r = recall_at_k(weights, selected, _len(4), 2)
        self.assertEqual(r.item(), 1.0)

    def test_top_k_miss(self) -> None:
        # true top-2 is keys 3 (0.4) and 2 (0.3); selecting 2,3 -> 1.0,
        # selecting 0,1 -> 0.0
        weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        r_hit = recall_at_k(weights, _sel([2, 3]), _len(2), 2)
        r_miss = recall_at_k(weights, _sel([0, 1]), _len(2), 2)
        r_half = recall_at_k(weights, _sel([3, 0]), _len(2), 2)
        self.assertEqual(r_hit.item(), 1.0)
        self.assertEqual(r_miss.item(), 0.0)
        self.assertEqual(r_half.item(), 0.5)

    def test_ties_broken_by_ascending_index(self) -> None:
        # all equal: true top-2 is keys 0 and 1.
        weights = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        self.assertEqual(recall_at_k(weights, _sel([0, 1]), _len(2), 2).item(), 1.0)
        self.assertEqual(recall_at_k(weights, _sel([2, 3]), _len(2), 2).item(), 0.0)
        self.assertEqual(recall_at_k(weights, _sel([0, 3]), _len(2), 2).item(), 0.5)

    def test_padding_is_ignored(self) -> None:
        weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        # length 1: only key 3 counts; padding key 2 must not add a hit.
        r = recall_at_k(weights, _sel([3, 2]), _len(1), 2)
        self.assertEqual(r.item(), 0.5)

    def test_k_larger_than_n_is_clamped(self) -> None:
        weights = torch.tensor([[0.5, 0.5]])
        r = recall_at_k(weights, _sel([0]), _len(1), 10)
        self.assertEqual(r.item(), 0.5)


class OutputErrorTest(unittest.TestCase):
    def test_all_keys_selected_gives_zero_error(self) -> None:
        weights = torch.tensor([[0.2, 0.3, 0.5]])
        v = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
        err = output_error(weights, v, _sel([0, 1, 2]), _len(3))
        self.assertAlmostEqual(err.item(), 0.0, places=6)

    def test_error_formula_on_a_known_case(self) -> None:
        # weights [0.5, 0.5] over v rows [1,0] and [0,1];
        # select only key 0 -> o_full = [0.5, 0.5], o_S = [1, 0];
        # diff = [-0.5, 0.5], ratio = sqrt(0.5)/sqrt(0.5) = 1.0
        weights = torch.tensor([[0.5, 0.5]])
        v = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        err = output_error(weights, v, _sel([0]), _len(1))
        self.assertAlmostEqual(err.item(), 1.0, places=6)

    def test_renormalisation_is_used(self) -> None:
        # select key 0 only; o_S must be v[0] itself (renormalised), not 0.5*v[0]
        weights = torch.tensor([[0.9, 0.1]])
        v = torch.tensor([[2.0, 4.0], [100.0, 100.0]])
        err = output_error(weights, v, _sel([0]), _len(1))
        o_full = torch.tensor([0.9 * 2.0 + 0.1 * 100.0, 0.9 * 4.0 + 0.1 * 100.0])
        o_sel = torch.tensor([2.0, 4.0])
        expected = (o_full - o_sel).norm() / o_full.norm()
        self.assertAlmostEqual(err.item(), expected.item(), places=6)

    def test_zero_selection_gives_error_one(self) -> None:
        weights = torch.tensor([[0.5, 0.5]])
        v = torch.tensor([[1.0], [3.0]])
        err = output_error(weights, v, _sel([]), _len(0))
        self.assertAlmostEqual(err.item(), 1.0, places=6)

    def test_padding_is_ignored(self) -> None:
        weights = torch.tensor([[0.5, 0.5]])
        v = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        # length 1 selects key 0; padded key 1 must not contribute.
        err = output_error(weights, v, _sel([0, 1]), _len(1))
        self.assertAlmostEqual(err.item(), 1.0, places=6)


class BudgetFractionTest(unittest.TestCase):
    def test_basic(self) -> None:
        out = budget_fraction(torch.tensor([0, 50, 100], dtype=torch.int32), 100)
        self.assertTrue(torch.allclose(out, torch.tensor([0.0, 0.5, 1.0])))
        self.assertEqual(out.dtype, torch.float32)

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            budget_fraction(torch.tensor([1]), 0)


if __name__ == "__main__":
    unittest.main()
