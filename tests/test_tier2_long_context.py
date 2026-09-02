"""
Tier 2 correctness at realistic context lengths.

The gather path was only ever equivalence-tested at T <= 300, then run at
16384. "Did not crash" is not "is correct": a chunking or indexing mistake that
only appears once the sequence spans many query chunks would have gone
unnoticed, and every Tier 2 number would have been wrong in a way no test
covered.

These tests run the same equivalence and intervention checks at 4096 and 8192,
across chunk boundaries, with head counts and head dimensions small enough that
the dense reference still fits on CPU. They are marked slow because a dense
(T, T) reference at 8192 takes a few seconds and a few hundred megabytes.
"""

from __future__ import annotations

import pytest
import torch

from crpa.attention import (
    build_crpa_structure,
    dense_masked_attention,
    relay_positions,
    sparse_gather_attention,
)

pytestmark = pytest.mark.slow

#: Small head count and head dim keep the dense reference affordable at 8192:
#: (1, 2, 8192, 8192) float32 is about 537 MB.
HEADS = 2
DIM = 8


def _structure(T, p, g=8, k=4, seed=11):
    hard = torch.randint(0, max(T // p, 1), (T,),
                         generator=torch.Generator().manual_seed(seed))
    return build_crpa_structure(T, p, g, k, hard, torch.device("cpu"),
                                torch.Generator().manual_seed(seed))


def _qkv(T, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(1, HEADS, T, DIM, generator=gen) for _ in range(3))


class TestEquivalenceAtLength:
    """The two implementations must agree at the lengths Tier 2 actually runs."""

    @pytest.mark.parametrize("T,p,chunk", [
        (4096, 512, 4096),      # one chunk
        (4096, 512, 1024),      # several chunks
        (4096, 512, 512),       # one chunk per partition
        (8192, 512, 2048),      # many chunks, Tier 2's middle length
        (4100, 512, 1024),      # ragged final partition
    ])
    def test_outputs_match_the_dense_reference(self, T, p, chunk):
        structure = _structure(T, p)
        q, k, v = _qkv(T)

        dense_out, _, _ = dense_masked_attention(q, k, v, structure.dense_mask())
        sparse_out, _, _ = sparse_gather_attention(
            q, k, v, structure, query_chunk=chunk
        )
        assert torch.allclose(dense_out, sparse_out, atol=1e-5), (
            "paths diverge at T={} with chunk {}".format(T, chunk)
        )

    def test_chunking_does_not_change_the_answer(self):
        """The chunk size is a memory knob and must not affect the result."""
        T, p = 4096, 512
        structure = _structure(T, p)
        q, k, v = _qkv(T)

        reference, _, _ = sparse_gather_attention(q, k, v, structure, query_chunk=4096)
        for chunk in (2048, 1024, 512):
            other, _, _ = sparse_gather_attention(q, k, v, structure, query_chunk=chunk)
            assert torch.allclose(reference, other, atol=1e-6), (
                "chunk size {} changed the output".format(chunk)
            )

    def test_probabilities_stay_normalised_at_length(self):
        T, p = 8192, 512
        structure = _structure(T, p)
        q, k, v = _qkv(T)
        _, probs, _ = sparse_gather_attention(
            q, k, v, structure, return_probs=True, query_chunk=2048
        )
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestInterventionAtLength:
    """Edge removal must behave the same at 4k as it does at 256."""

    @pytest.mark.parametrize("T,chunk", [(4096, 1024), (8192, 2048)])
    def test_an_edge_is_removed_on_both_paths(self, T, chunk):
        structure = _structure(T, 512)
        q, k, v = _qkv(T)

        query = T - 100
        keys = structure.allowed_keys(query)
        assert keys.numel() >= 2
        key = int(keys[len(keys) // 2])
        edges = [(1, query, key)]

        _, dense_probs, n_dense = dense_masked_attention(
            q, k, v, structure.dense_mask(), edges=edges)
        _, sparse_probs, n_sparse = sparse_gather_attention(
            q, k, v, structure, edges=edges, return_probs=True, query_chunk=chunk)

        assert n_dense == n_sparse == 1
        assert float(sparse_probs[0, 1, query, key]) == 0.0
        assert torch.allclose(dense_probs, sparse_probs, atol=1e-5)
        row = sparse_probs[0, 1, query].sum()
        assert float(row) == pytest.approx(1.0, abs=1e-5)

    def test_an_edge_in_a_non_first_chunk_is_reached(self):
        """Query-offset arithmetic must be right for chunks after the first."""
        T, chunk = 4096, 512
        structure = _structure(T, 512)
        q, k, v = _qkv(T)

        # Sits in the seventh chunk, so query_offset is non-zero.
        query = 3500
        keys = structure.allowed_keys(query)
        key = int(keys[len(keys) // 2])
        _, probs, touched = sparse_gather_attention(
            q, k, v, structure, edges=[(0, query, key)],
            return_probs=True, query_chunk=chunk)
        assert touched == 1
        assert float(probs[0, 0, query, key]) == 0.0

    def test_other_heads_and_rows_are_untouched(self):
        T = 4096
        structure = _structure(T, 512)
        q, k, v = _qkv(T)
        query = 2000
        key = int(structure.allowed_keys(query)[3])

        _, base, _ = sparse_gather_attention(
            q, k, v, structure, return_probs=True, query_chunk=1024)
        _, cut, _ = sparse_gather_attention(
            q, k, v, structure, edges=[(0, query, key)],
            return_probs=True, query_chunk=1024)

        assert torch.allclose(base[:, 1], cut[:, 1], atol=1e-7)
        others = [r for r in (query - 1, query + 1, 0, T - 1) if r != query]
        for row in others:
            assert torch.allclose(base[0, 0, row], cut[0, 0, row], atol=1e-7)


class TestWindowedCaptureAtLength:
    """The gathered capture is what makes long-context diagnostics possible."""

    def test_window_matches_the_dense_capture_at_4k(self):
        from crpa.metrics import support_key_sets, top_p_support_mask

        T, p = 4096, 512
        structure = _structure(T, p)
        q, k, v = _qkv(T)
        window = (T - 1024, T)

        _, dense_probs, _ = sparse_gather_attention(
            q, k, v, structure, return_probs=True, query_chunk=1024)
        collected: list = []
        sparse_gather_attention(q, k, v, structure, query_chunk=1024,
                                probs_window=window, sparse_probs_out=collected)
        assert collected, "no windowed capture was produced"

        from crpa.attention import SparseProbs

        sp = collected[0] if len(collected) == 1 else SparseProbs(
            probs=torch.cat([c.probs for c in collected], dim=2),
            key_idx=torch.cat([c.key_idx for c in collected], dim=0),
            query_lo=collected[0].query_lo,
            is_relay=torch.cat([c.is_relay for c in collected], dim=0),
        )
        assert sp.query_lo == window[0]
        assert sp.n_queries == window[1] - window[0]

        dense_support = top_p_support_mask(dense_probs[:, 0].mean(dim=0), 0.6)
        sparse_support = support_key_sets(sp.head(0), sp.key_idx, 0.6)
        checked = 0
        for local in range(0, sp.n_queries, 37):
            if bool(sp.is_relay[local]):
                continue
            absolute = sp.query_lo + local
            expected = set(dense_support[absolute].nonzero(as_tuple=True)[0].tolist())
            assert sparse_support[local] == expected, (
                "support differs at query {}".format(absolute))
            checked += 1
        assert checked > 10

    def test_capture_cost_does_not_grow_with_context(self):
        """The point of the gathered form: size is independent of T."""
        sizes = {}
        for T in (4096, 8192):
            structure = _structure(T, 512)
            q, k, v = _qkv(T)
            collected: list = []
            sparse_gather_attention(q, k, v, structure, query_chunk=2048,
                                    probs_window=(T - 1024, T),
                                    sparse_probs_out=collected)
            sizes[T] = sum(c.probs.numel() for c in collected)
        assert sizes[4096] == sizes[8192], (
            "windowed capture grew with context length: {}".format(sizes))


class TestStructureAtLength:
    def test_relays_and_routing_stay_causal_at_8k(self):
        T = 8192
        structure = _structure(T, 512)
        for query in (100, 4000, T - 1):
            keys = structure.allowed_keys(query)
            assert int(keys.max()) <= query
        routed = structure.cross_idx
        rows = torch.arange(T).unsqueeze(1).expand_as(routed)
        live = routed >= 0
        assert bool((routed[live] < rows[live]).all())

    def test_relay_positions_are_where_the_mask_says(self):
        T, g = 8192, 8
        structure = _structure(T, 512, g=g)
        assert structure.relay_pos.tolist() == relay_positions(T, g)

    def test_sparsity_is_far_below_dense(self):
        T = 4096
        structure = _structure(T, 512)
        actual = structure.edge_count()
        dense = T * (T + 1) // 2
        assert actual < dense * 0.25
        assert actual > T
