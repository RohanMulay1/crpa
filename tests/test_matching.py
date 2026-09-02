"""
Matched-overlap pairing.

The whole point of the sweep is that pairing happens on *realized* overlap
measured after training, never on the configured regularization strength. These
tests pin that.
"""

from __future__ import annotations

import pytest

from experiments.matched_overlap import find_matched_overlap_pairs


def run(variant, seed, overlap, retrieval, lam, rid=None):
    return {
        "variant": variant, "seed": seed, "realized_overlap": overlap,
        "retrieval_accuracy": retrieval, "lambda_red": lam,
        "removal_budget": 8, "run_id": rid or "{}_{}_{}".format(variant, seed, lam),
    }


class TestMatching:
    def test_finds_the_nearest_counterpart(self):
        runs = [
            run("crpa_naive", 42, 0.250, 10.0, 0.05),
            run("crpa_contribution", 42, 0.400, 40.0, 0.01),
            run("crpa_contribution", 42, 0.252, 35.0, 0.10),   # nearest
            run("crpa_contribution", 42, 0.300, 30.0, 0.05),
        ]
        pairs = find_matched_overlap_pairs(runs, tolerance=0.01)
        assert len(pairs) == 1
        pair = pairs[0]
        assert pair["crpa_contribution_overlap"] == pytest.approx(0.252)
        assert pair["overlap_abs_diff"] == pytest.approx(0.002)
        assert pair["retrieval_delta"] == pytest.approx(25.0)

    def test_does_not_match_on_lambda(self):
        """Same lambda, far apart in realized overlap: must not pair."""
        runs = [
            run("crpa_naive", 42, 0.200, 10.0, 0.05),
            run("crpa_contribution", 42, 0.500, 40.0, 0.05),   # identical lambda
        ]
        assert find_matched_overlap_pairs(runs, tolerance=0.01) == []

    def test_tolerance_is_respected(self):
        runs = [
            run("crpa_naive", 42, 0.250, 10.0, 0.05),
            run("crpa_contribution", 42, 0.270, 40.0, 0.05),
        ]
        # 0.02 exactly is avoided: 0.270 - 0.250 is 0.020000000000000018 in
        # binary floating point, so the boundary itself is not a meaningful
        # thing to assert on.
        assert find_matched_overlap_pairs(runs, tolerance=0.01) == []
        assert len(find_matched_overlap_pairs(runs, tolerance=0.025)) == 1

    def test_pairs_only_within_a_seed_by_default(self):
        runs = [
            run("crpa_naive", 42, 0.250, 10.0, 0.05),
            run("crpa_contribution", 1337, 0.250, 40.0, 0.05),
        ]
        assert find_matched_overlap_pairs(runs, tolerance=0.01) == []
        assert len(find_matched_overlap_pairs(runs, tolerance=0.01,
                                              within_seed=False)) == 1

    def test_results_are_sorted_by_closeness(self):
        runs = [
            run("crpa_naive", 1, 0.300, 10.0, 0.05),
            run("crpa_contribution", 1, 0.305, 20.0, 0.05),
            run("crpa_naive", 2, 0.400, 10.0, 0.05),
            run("crpa_contribution", 2, 0.401, 20.0, 0.05),
        ]
        pairs = find_matched_overlap_pairs(runs, tolerance=0.01)
        assert len(pairs) == 2
        diffs = [p["overlap_abs_diff"] for p in pairs]
        assert diffs == sorted(diffs)

    def test_non_finite_overlaps_are_ignored(self):
        runs = [
            run("crpa_naive", 42, float("nan"), 10.0, 0.05),
            run("crpa_contribution", 42, 0.250, 40.0, 0.05),
        ]
        assert find_matched_overlap_pairs(runs, tolerance=0.5) == []

    def test_empty_and_single_method_inputs(self):
        assert find_matched_overlap_pairs([], tolerance=0.01) == []
        only_one = [run("crpa_naive", 42, 0.25, 10.0, 0.05)]
        assert find_matched_overlap_pairs(only_one, tolerance=0.01) == []

    def test_negative_tolerance_is_rejected(self):
        with pytest.raises(ValueError):
            find_matched_overlap_pairs([], tolerance=-0.1)

    def test_pair_records_both_lambdas(self):
        runs = [
            run("crpa_naive", 42, 0.250, 10.0, 0.20),
            run("crpa_contribution", 42, 0.251, 40.0, 0.01),
        ]
        pair = find_matched_overlap_pairs(runs, tolerance=0.01)[0]
        # Matched overlap, very different lambdas: exactly the situation the
        # experiment exists to surface.
        assert pair["crpa_naive_lambda_red"] == 0.20
        assert pair["crpa_contribution_lambda_red"] == 0.01


class TestLambdaControl:
    """The matched-overlap experiment rests on lambda actually steering overlap.

    If it does not, the "matched" pairs are matched on run-to-run variation
    rather than on a controlled structural budget. That is a different and much
    weaker claim, so the experiment has to detect and say it.
    """

    def _rows(self, overlap_fn):
        return [
            {"variant": "crpa_naive", "seed": seed, "lambda_red": lam,
             "realized_overlap": overlap_fn(seed, lam)}
            for seed in (42, 1337, 2024)
            for lam in (0.0, 0.01, 0.05, 0.10, 0.20)
        ]

    def test_detects_that_lambda_steers_overlap(self):
        from experiments.matched_overlap import lambda_controls_overlap

        rows = self._rows(lambda seed, lam: 0.30 - 0.5 * lam)
        out = lambda_controls_overlap(rows)["crpa_naive"]
        assert out["lambda_dominates_seed"] is True
        assert out["spearman_lambda_vs_overlap"] == pytest.approx(-1.0)

    def test_detects_that_lambda_does_nothing(self):
        from experiments.matched_overlap import lambda_controls_overlap

        # Overlap depends only on the seed, not at all on lambda.
        rows = self._rows(lambda seed, lam: 0.24 + (seed % 7) * 0.005)
        out = lambda_controls_overlap(rows)["crpa_naive"]
        assert out["lambda_dominates_seed"] is False
        assert out["max_overlap_span_across_lambda_within_a_seed"] == pytest.approx(0.0)
        assert out["max_overlap_span_across_seeds_at_fixed_lambda"] > 0

    def test_too_few_runs_is_inconclusive(self):
        from experiments.matched_overlap import lambda_controls_overlap

        out = lambda_controls_overlap([
            {"variant": "crpa_naive", "seed": 42, "lambda_red": 0.0,
             "realized_overlap": 0.24}
        ])
        assert out["conclusive"] is False
