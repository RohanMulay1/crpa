"""
Interventions: the properties the whole scientific claim depends on.

Three things must hold, or a reported delta means nothing:
  1. the targeted edge actually loses its probability mass,
  2. the row renormalises, so the delta reflects redistribution not deletion,
  3. an intervention that removes nothing is refused, not reported as zero.
"""

from __future__ import annotations

import pytest
import torch

import numpy as np

from crpa.attention import build_crpa_structure, dense_masked_attention, sparse_gather_attention
from crpa.intervention import (
    Candidate,
    Edge,
    InterventionError,
    InterventionPlan,
    make_needle_loss_fn,
    measure_delta,
    reachable_queries,
    score_candidates,
    score_candidates_chunked,
    sample_candidate_edges,
    select_contribution_gated,
    select_naive,
    split_high_overlap_groups,
)


def _structure(T=128, p=32, g=2, k=2, seed=11):
    hard = torch.randint(0, max(T // p, 1), (T,),
                         generator=torch.Generator().manual_seed(seed))
    return build_crpa_structure(T, p, g, k, hard, torch.device("cpu"),
                                torch.Generator().manual_seed(seed))


class TestEdgeRemoval:
    def test_intervention_removes_exactly_the_requested_edge(self):
        structure = _structure()
        mask = structure.dense_mask()
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 4, structure.T, 8) for _ in range(3))

        _, base_probs, _ = dense_masked_attention(q, k, v, mask)
        query = 100
        key = int(structure.allowed_keys(query)[3])
        assert base_probs[0, 1, query, key] > 0, "test targets an edge with no mass"

        _, probs, touched = dense_masked_attention(
            q, k, v, mask, edges=[(1, query, key)]
        )
        assert touched == 1
        assert float(probs[0, 1, query, key]) == 0.0

    def test_other_heads_are_untouched(self):
        structure = _structure()
        mask = structure.dense_mask()
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 4, structure.T, 8) for _ in range(3))
        _, base, _ = dense_masked_attention(q, k, v, mask)
        _, probs, _ = dense_masked_attention(q, k, v, mask, edges=[(1, 100, 96)])

        for head in (0, 2, 3):
            assert torch.allclose(base[:, head], probs[:, head], atol=1e-7), (
                "intervention on head 1 changed head {}".format(head)
            )

    def test_attention_renormalises_after_intervention(self):
        structure = _structure()
        mask = structure.dense_mask()
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 4, structure.T, 8) for _ in range(3))
        _, probs, _ = dense_masked_attention(q, k, v, mask, edges=[(1, 100, 96)])
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), (
            "rows do not sum to 1 after intervention; mass was deleted rather "
            "than redistributed"
        )

    def test_removed_mass_is_redistributed_not_lost(self):
        structure = _structure()
        mask = structure.dense_mask()
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 2, structure.T, 8) for _ in range(3))
        query, head = 100, 0
        key = int(structure.allowed_keys(query)[2])

        _, base, _ = dense_masked_attention(q, k, v, mask)
        _, after, _ = dense_masked_attention(q, k, v, mask, edges=[(head, query, key)])

        removed = float(base[0, head, query, key])
        others = [j for j in structure.allowed_keys(query).tolist() if j != key]
        gained = sum(float(after[0, head, query, j] - base[0, head, query, j])
                     for j in others)
        assert gained == pytest.approx(removed, abs=1e-5)

    def test_intervention_works_on_gather_path_too(self):
        structure = _structure()
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 4, structure.T, 8) for _ in range(3))
        query = 100
        key = int(structure.allowed_keys(query)[3])
        edges = [(1, query, key)]

        _, dense_probs, n_dense = dense_masked_attention(
            q, k, v, structure.dense_mask(), edges=edges)
        _, sparse_probs, n_sparse = sparse_gather_attention(
            q, k, v, structure, edges=edges, return_probs=True)

        assert n_dense == n_sparse == 1
        assert torch.allclose(dense_probs, sparse_probs, atol=1e-5)


class TestNoOpRefusal:
    def test_measure_delta_refuses_a_no_op(self, tiny_model, needle_batch):
        """An edge that is not in the mask must raise, not return delta 0.

        This is the failure the original implementation had: because it sampled
        query-row pairs (i, j) without ordering and then masked the edge i->j,
        every sample with i < j hit an already-False entry, produced delta 0,
        and was classified redundant.
        """
        x, y = needle_batch
        loss_fn = make_needle_loss_fn(x, y)
        future_edge = Edge(layer=0, head=0, query=5, key=60)  # key > query
        with pytest.raises(InterventionError):
            measure_delta(tiny_model, [future_edge], loss_fn, strict=True)

    def test_score_candidates_drops_no_ops(self, tiny_model, needle_batch):
        x, y = needle_batch
        bogus = [Candidate(layer=0, head=0, query=5, key=60, overlap=0.9)]
        scored = score_candidates(tiny_model, bogus, make_needle_loss_fn(x, y),
                                  eps=0.03, skip_no_ops=True)
        assert scored == [], "a no-op intervention was recorded as an observation"

    def test_chunked_scoring_matches_materialized_scoring(self, tiny_model,
                                                           needle_batch):
        x, y = needle_batch
        candidates = [
            Candidate(layer=0, head=0, query=x.shape[1] - 1, key=k,
                      overlap=1.0 - k / 100.0)
            for k in (0, 1, 2)
        ]
        loss_fn = make_needle_loss_fn(x, y)
        direct = score_candidates(tiny_model, candidates, loss_fn, eps=0.03)
        fresh = [Candidate(layer=c.layer, head=c.head, query=c.query,
                           key=c.key, overlap=c.overlap) for c in candidates]
        chunked = score_candidates_chunked(
            tiny_model, iter(fresh), loss_fn, eps=0.03, chunk_size=1)
        assert [c.to_row() for c in chunked] == [c.to_row() for c in direct]

    def test_chunk_size_must_be_positive(self, tiny_model, needle_batch):
        x, y = needle_batch
        with pytest.raises(ValueError, match="positive"):
            score_candidates_chunked(
                tiny_model, [], make_needle_loss_fn(x, y), eps=0.03,
                chunk_size=0)


class TestScopedProbabilityCapture:
    def test_only_requested_layer_retains_probabilities(self, tiny_model,
                                                        needle_batch):
        x, _ = needle_batch
        with torch.no_grad(), tiny_model.capture_probabilities(
                True, window=(0, x.shape[1]), layers=[0]):
            tiny_model(x, last_only=True)
        captured = tiny_model.attention_probabilities()
        assert captured[0] is not None
        assert all(value is None for value in captured[1:])

    def test_unknown_layer_is_rejected(self, tiny_model):
        with pytest.raises(ValueError, match="out of range"):
            with tiny_model.capture_probabilities(True, layers=[999]):
                pass


class TestReachability:
    def test_unreachable_queries_are_identified(self, tiny_model, needle_batch):
        """Under a last-token loss, most early edges cannot matter at all."""
        x, _ = needle_batch
        with tiny_model.frozen_structure():
            tiny_model(x)
            reach = reachable_queries(tiny_model, x.shape[1])
        assert len(reach) == len(tiny_model.blocks)
        # The final layer can only matter at the queried position.
        assert int(reach[-1].sum()) == 1
        assert bool(reach[-1][x.shape[1] - 1])
        # Lower layers reach at least as much as higher ones.
        assert int(reach[0].sum()) >= int(reach[-1].sum())

    def test_unreachable_edge_has_exactly_zero_delta(self, tiny_model, needle_batch):
        """Confirms the reachability filter is necessary, not merely tidy."""
        x, y = needle_batch
        loss_fn = make_needle_loss_fn(x, y)
        with tiny_model.frozen_structure():
            tiny_model(x)
            reach = reachable_queries(tiny_model, x.shape[1])
            unreachable = (~reach[-1]).nonzero(as_tuple=True)[0]
            assert unreachable.numel() > 0
            query = int(unreachable[-1])
            structure = tiny_model.blocks[-1].attn._structure
            key = int(structure.allowed_keys(query)[0])
            base, after, removed = measure_delta(
                tiny_model,
                [Edge(len(tiny_model.blocks) - 1, 0, query, key)],
                loss_fn, strict=False,
            )
        assert removed >= 1, "edge should exist in the mask"
        assert after == pytest.approx(base, abs=1e-9), (
            "an unreachable edge produced a nonzero delta"
        )


class TestSelection:
    def _pool(self):
        return [
            Candidate(layer=0, head=0, query=10, key=5, overlap=0.9, delta_loss=0.50),
            Candidate(layer=0, head=0, query=11, key=6, overlap=0.85, delta_loss=0.01),
            Candidate(layer=0, head=0, query=12, key=7, overlap=0.20, delta_loss=0.00),
            Candidate(layer=0, head=0, query=13, key=8, overlap=0.80, delta_loss=-0.05),
        ]

    def test_naive_ranks_by_overlap_only(self):
        picked = select_naive(self._pool(), 2)
        assert [c.query for c in picked] == [10, 11]

    def test_contribution_ranks_by_measured_delta(self):
        picked = select_contribution_gated(self._pool(), 2)
        assert [c.query for c in picked] == [13, 12]

    def test_both_criteria_respect_the_same_budget(self):
        """The matched-budget property that makes the comparison controlled."""
        pool = self._pool()
        for budget in (1, 2, 3, 4):
            assert len(select_naive(pool, budget)) == budget
            assert len(select_contribution_gated(pool, budget)) == budget

    def test_criteria_disagree_at_similar_overlap(self):
        """Edges 10 and 13 have comparable overlap but opposite contribution."""
        pool = self._pool()
        naive = {c.query for c in select_naive(pool, 1)}
        contribution = {c.query for c in select_contribution_gated(pool, 1)}
        assert naive != contribution

    def test_high_overlap_groups_split_on_behaviour(self):
        groups = split_high_overlap_groups(self._pool(), 0.5, 0.5, 0.5)
        a = groups["high_overlap_low_contribution"]
        b = groups["high_overlap_high_contribution"]
        assert a and b
        assert max(c.delta_loss for c in a) <= min(c.delta_loss for c in b)


class TestPlan:
    def test_plan_routes_edges_to_their_layer(self):
        plan = InterventionPlan.of([
            Edge(0, 1, 10, 5), Edge(2, None, 11, 6), Edge(0, 3, 12, 7),
        ])
        assert len(plan.for_layer(0)) == 2
        assert plan.for_layer(1) == []
        assert plan.for_layer(2) == [(None, 11, 6)]


class TestRowEmptying:
    """An intervention must remove an interaction, never a whole query.

    Position 0 has exactly one permitted key: itself. Removing it leaves the
    row with nothing to attend to, so softmax produces NaN. This surfaced only
    on the full-scale GPU run, where the candidate sampler eventually proposed
    that edge.
    """

    def test_removing_a_rows_last_key_is_refused(self):
        structure = _structure(T=64, p=16, g=2, k=2)
        mask = structure.dense_mask()
        assert int(mask[0].sum()) == 1, "query 0 should have exactly one key"

        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 2, 64, 8) for _ in range(3))
        out, probs, touched = dense_masked_attention(
            q, k, v, mask, edges=[(0, 0, 0)]
        )
        assert touched == 0, "the last remaining key was removed"
        assert not torch.isnan(probs).any()
        assert float(probs[0, 0, 0].sum()) == pytest.approx(1.0, abs=1e-6)

    def test_sampler_never_proposes_such_an_edge(self, tiny_model, needle_batch):
        x, _ = needle_batch
        rng = np.random.default_rng(0)
        with tiny_model.frozen_structure():
            with tiny_model.capture_probabilities(True):
                tiny_model(x)
            probs = tiny_model.attention_probabilities()
            structures = tiny_model.structures()
            for depth, layer_probs in enumerate(probs):
                candidates = sample_candidate_edges(
                    layer_probs, depth, 16, 0.6, 20, rng
                )
                for cand in candidates:
                    allowed = structures[depth].allowed_keys(cand.query)
                    assert allowed.numel() >= 2, (
                        "sampled query {} has only one permitted key".format(cand.query)
                    )
