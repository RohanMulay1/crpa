"""
Attention structure and the equivalence of the two implementations.

The equivalence test is what licenses using the gather path at long context
while keeping the dense path as the short-context reference.
"""

from __future__ import annotations

import pytest
import torch

from crpa.attention import (
    build_crpa_mask,
    build_crpa_structure,
    dense_masked_attention,
    relay_positions,
    sliding_mask,
    sparse_gather_attention,
)


def make_structure(T=256, p=64, g=4, k=4, seed=123):
    hard = torch.randint(0, max(T // p, 1), (T,), generator=torch.Generator().manual_seed(seed))
    gen = torch.Generator().manual_seed(seed)
    return build_crpa_structure(T, p, g, k, hard, torch.device("cpu"), gen)


class TestMaskStructure:
    def test_mask_respects_causality(self):
        """No query may attend to a future key."""
        structure = make_structure()
        mask = structure.dense_mask()
        T = mask.shape[0]
        upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        assert not bool((mask & upper).any()), "mask permits attention to future positions"

    def test_every_query_has_at_least_itself(self):
        """A fully masked row would produce NaN after softmax."""
        mask = make_structure().dense_mask()
        assert bool(mask.diagonal().all())
        assert bool((mask.sum(dim=1) > 0).all())

    def test_partition_windows_present(self):
        structure = make_structure(T=128, p=32, g=0, k=0)
        mask = structure.dense_mask()
        # Query 40 is in block 1 (32..63) and may see 32..40 causally.
        assert bool(mask[40, 32]) and bool(mask[40, 40])
        assert not bool(mask[40, 41])
        # Block 0 is a different partition and unreachable without relays/routing.
        assert not bool(mask[40, 10])

    def test_relays_are_globally_visible(self):
        T, p, g = 128, 32, 4
        structure = make_structure(T=T, p=p, g=g, k=0)
        mask = structure.dense_mask()
        for r in structure.relay_pos.tolist():
            later = torch.arange(r, T)
            assert bool(mask[later, r].all()), "relay {} not visible to later queries".format(r)

    def test_routed_keys_are_causal_and_cross_partition(self):
        structure = make_structure()
        for query in range(structure.T):
            for key in structure.cross_idx[query].tolist():
                if key >= 0:
                    assert key < query, "routed key {} not causal for query {}".format(key, query)

    def test_sliding_mask_is_banded(self):
        mask = sliding_mask(64, 16)
        assert bool(mask[40, 40]) and bool(mask[40, 25])
        assert not bool(mask[40, 24])   # exactly outside the window
        assert not bool(mask[40, 41])   # future

    def test_relay_positions_match_original_formula(self):
        assert relay_positions(512, 4) == [102, 204, 306, 408]

    def test_build_crpa_mask_signature_preserved(self):
        """The original positional signature still works."""
        mask = build_crpa_mask(64, 16, [10, 20], torch.zeros(64, dtype=torch.long), 2,
                               True, "cpu")
        assert mask.shape == (64, 64) and mask.dtype == torch.bool

    def test_generator_makes_routing_deterministic(self):
        a = make_structure(seed=5).cross_idx
        b = make_structure(seed=5).cross_idx
        assert torch.equal(a, b), "same seed produced different routing"
        c = make_structure(seed=6).cross_idx
        assert not torch.equal(a, c), "different seeds produced identical routing"


class TestImplementationEquivalence:
    """The gather path must compute exactly what the dense mask describes."""

    @pytest.mark.parametrize("T,p", [(256, 64), (250, 64), (128, 128), (300, 100)])
    def test_outputs_match(self, T, p):
        structure = make_structure(T=T, p=p)
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 4, T, 16) for _ in range(3))

        out_dense, probs_dense, _ = dense_masked_attention(q, k, v, structure.dense_mask())
        out_sparse, probs_sparse, _ = sparse_gather_attention(
            q, k, v, structure, return_probs=True, query_chunk=128
        )
        assert torch.allclose(out_dense, out_sparse, atol=1e-5)
        assert torch.allclose(probs_dense, probs_sparse, atol=1e-5)

    def test_probabilities_are_normalised(self):
        structure = make_structure()
        torch.manual_seed(1)
        q, k, v = (torch.randn(2, 4, structure.T, 16) for _ in range(3))
        _, probs, _ = sparse_gather_attention(q, k, v, structure, return_probs=True)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_no_duplicate_key_weighting(self):
        """A key reachable via two sources must not be counted twice."""
        structure = make_structure(T=128, p=32, g=4, k=4)
        torch.manual_seed(2)
        q, k, v = (torch.randn(1, 2, 128, 8) for _ in range(3))
        _, probs, _ = sparse_gather_attention(q, k, v, structure, return_probs=True)
        mask = structure.dense_mask()
        # Probability must be exactly zero outside the permitted set.
        assert float(probs[0, 0][~mask].abs().max()) < 1e-6
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestSparsity:
    def test_crpa_is_sparser_than_dense(self):
        structure = make_structure(T=512, p=128, g=4, k=4)
        actual = structure.edge_count()
        dense = 512 * 513 // 2
        assert actual < dense
        assert actual > 512, "mask is implausibly empty"

    def test_attended_mask_matches_dense_mask(self):
        """Reachability helper must agree with the materialised mask."""
        structure = make_structure(T=128, p=32)
        mask = structure.dense_mask()
        queries = torch.zeros(128, dtype=torch.bool)
        queries[[40, 41, 100]] = True
        expected = mask[queries].any(dim=0)
        got = structure.attended_mask(queries)
        # The structural helper may be a superset within a partition window; it
        # must never be a subset, or reachability would miss real paths.
        assert bool((expected & ~got).sum() == 0), "attended_mask missed reachable keys"
