"""Check 0 applied to this repository's own headline claim.

The decision rule is ported from xsa-controls unchanged. These tests pin it in
both places so the two projects cannot silently drift to different standards,
and pin the arithmetic that decides whether a correlation may be reported at
all.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from crpa.resolvability import (
    RESOLVABILITY_RULE,
    ResolvabilityResult,
    assess,
    disattenuate,
    max_observable_correlation,
    pooled_verdict,
    ulp,
    verdict_for,
)

RESULTS = Path(__file__).resolve().parents[1] / "results" / "resolvability"


class TestDecisionRule:
    """Ported verbatim. If these change, xsa-controls must change too."""

    def test_thresholds_are_the_pre_registered_ones(self):
        assert [t for t, _, _ in RESOLVABILITY_RULE] == [0.6, 0.3, 0.0]

    def test_names_are_the_pre_registered_ones(self):
        assert [n for _, n, _ in RESOLVABILITY_RULE] == [
            "reliable", "attenuated", "unresolvable"]

    @pytest.mark.parametrize("r,expected", [
        (0.9, "reliable"), (0.6, "reliable"),
        (0.45, "attenuated"), (0.3, "attenuated"),
        (0.1, "unresolvable"), (0.0, "unresolvable"), (-0.5, "unresolvable"),
    ])
    def test_verdict_boundaries(self, r, expected):
        assert verdict_for(r)[0] == expected

    def test_nan_reliability_is_unresolvable_not_optimistic(self):
        assert verdict_for(float("nan"))[0] == "unresolvable"

    def test_the_unresolvable_action_forbids_the_correlation_claim(self):
        action = verdict_for(0.1)[1]
        assert "No decoupling may be claimed" in action
        assert "drop the correlation claim" in action


class TestCeilingAndDisattenuation:
    def test_ceiling_is_the_geometric_mean_of_reliabilities(self):
        assert max_observable_correlation(0.25, 0.64) == pytest.approx(0.4)

    def test_a_zero_reliability_gives_a_zero_ceiling(self):
        assert max_observable_correlation(0.0, 0.9) == 0.0

    def test_a_negative_reliability_gives_a_zero_ceiling(self):
        assert max_observable_correlation(-0.1, 0.9) == 0.0

    def test_disattenuation_corrects_upward(self):
        assert disattenuate(0.3, 0.5, 0.5) == pytest.approx(0.6)

    def test_disattenuation_is_undefined_at_non_positive_reliability(self):
        """Dividing by a near-zero reliability is an artifact, not a finding."""
        assert math.isnan(disattenuate(0.3, 0.0, 0.5))
        assert math.isnan(disattenuate(0.3, -0.02, 0.5))

    def test_observed_correlation_inside_the_ceiling_is_uninformative(self):
        """The exact situation this repository is in."""
        ceiling = max_observable_correlation(0.088, 0.119)
        assert abs(0.018) < ceiling, (
            "an observed r below the ceiling cannot distinguish a real null "
            "from measurement noise")


class TestULP:
    def test_ulp_at_a_typical_loss_is_the_float32_step(self):
        # A loss near 6.8 sits in the [4, 8) binade: spacing is 2^-21.
        assert ulp(6.8) == pytest.approx(2.0 ** -21, rel=1e-6)

    def test_ulp_grows_with_magnitude(self):
        assert ulp(100.0) > ulp(1.0)

    def test_ulp_of_zero_is_positive(self):
        assert ulp(0.0) > 0


class TestAssess:
    def _paired(self, n=40, noise=0.0, seed=0):
        import random
        rng = random.Random(seed)
        truth = [rng.random() for _ in range(n)]
        a = [t + rng.gauss(0, noise) for t in truth]
        b = [t + rng.gauss(0, noise) for t in truth]
        return truth, a, b

    def test_a_reliable_measurement_passes(self):
        truth, a, b = self._paired(noise=0.001)
        res = assess(a, b, truth, truth, seed=1, baseline_loss=6.8)
        assert res.r_delta > 0.6
        assert res.verdict == "reliable"
        assert res.passed is True

    def test_pure_noise_is_unresolvable(self):
        import random
        rng = random.Random(3)
        a = [rng.random() for _ in range(60)]
        b = [rng.random() for _ in range(60)]
        stat = [rng.random() for _ in range(60)]
        res = assess(a, b, stat, stat, seed=1, baseline_loss=6.8)
        assert res.verdict == "unresolvable"
        assert res.passed is False
        assert res.ceiling < 0.4

    def test_delta_scale_is_reported_in_ulps(self):
        res = assess([1e-6] * 20, [1e-6] * 20, list(range(20)),
                     list(range(20)), seed=1, baseline_loss=6.8)
        assert res.ulp_at_loss == pytest.approx(2.0 ** -21, rel=1e-6)
        assert res.deltas_per_ulp == pytest.approx(1e-6 / (2.0 ** -21), rel=0.1)

    def test_a_flat_budget_curve_is_reported_as_non_convergent(self):
        truth, a, b = self._paired(noise=1.0)
        res = assess(a, b, truth, truth, seed=1, baseline_loss=6.8,
                     budget_curve={2: -0.1, 4: 0.05, 8: 0.02, 16: 0.04})
        assert res.converges is False

    def test_a_rising_budget_curve_is_reported_as_convergent(self):
        truth, a, b = self._paired(noise=0.001)
        res = assess(a, b, truth, truth, seed=1, baseline_loss=6.8,
                     budget_curve={2: 0.1, 4: 0.4, 8: 0.7, 16: 0.85})
        assert res.converges is True

    def test_summary_names_the_verdict_and_the_ceiling(self):
        truth, a, b = self._paired(noise=1.0)
        text = assess(a, b, truth, truth, seed=42, baseline_loss=6.8).summary()
        assert "Check 0" in text and "ceiling" in text

    def test_result_is_json_serialisable(self):
        truth, a, b = self._paired(noise=0.01)
        json.dumps(assess(a, b, truth, truth, seed=1,
                          baseline_loss=6.8).to_dict())


class TestPooledVerdict:
    def _res(self, seed, r_delta, r_stat=0.2):
        name, action = verdict_for(r_delta)
        return ResolvabilityResult(
            seed=seed, r_delta=r_delta, r_stat=r_stat, verdict=name,
            action=action, n_candidates=100,
            ceiling=max_observable_correlation(r_delta, r_stat))

    def test_pooling_takes_the_best_seed_not_the_mean(self):
        """If the measurement fails at its most favourable seed, averaging
        does not rescue it."""
        out = pooled_verdict([self._res(1, 0.05), self._res(2, 0.7)])
        assert out["r_delta_best"] == pytest.approx(0.7)
        assert out["verdict"] == "reliable"

    def test_all_seeds_failing_gives_unresolvable(self):
        out = pooled_verdict([self._res(1, 0.05), self._res(2, -0.02),
                              self._res(3, 0.01)])
        assert out["verdict"] == "unresolvable"
        assert out["any_seed_passed"] is False
        assert "not reliable enough" in out["claim_permitted"]

    def test_empty_input_is_unresolvable_not_an_error(self):
        assert pooled_verdict([])["verdict"] == "unresolvable"

    def test_the_permitted_claim_is_stated_explicitly(self):
        out = pooled_verdict([self._res(1, 0.8, 0.8)])
        assert "overlap does not predict contribution" in out["claim_permitted"]


@pytest.mark.skipif(not (RESULTS / "resolvability.json").exists(),
                    reason="Check 0 has not been run in this checkout")
class TestCommittedResultIsConsistent:
    """The committed artifact must still say what the README says it says."""

    @pytest.fixture
    def payload(self):
        return json.loads(
            (RESULTS / "resolvability.json").read_text(encoding="utf-8"))

    def test_verdict_is_unresolvable(self, payload):
        assert payload["pooled"]["verdict"] == "unresolvable"

    def test_best_reliability_is_below_the_threshold(self, payload):
        assert payload["pooled"]["r_delta_best"] < 0.3

    def test_no_observed_correlation_escapes_the_best_ceiling(self, payload):
        """The argument the README makes, stated exactly.

        A per-seed ceiling is not a hard bound: where reliability is negative
        the geometric mean degenerates to zero, and a zero ceiling means "this
        seed carries no information", not "the observed r must be zero". The
        defensible statement is that every observed correlation is smaller
        than the LARGEST ceiling any seed achieved, so none of them can be
        distinguished from measurement noise.
        """
        best_ceiling = max(s["max_observable_correlation"]
                           for s in payload["per_seed"])
        for seed in payload["per_seed"]:
            assert abs(seed["rho_observed"]) < best_ceiling, (
                "seed {} reports |rho| = {:.4f} which is not below the best "
                "achievable ceiling {:.4f}".format(
                    seed["seed"], abs(seed["rho_observed"]), best_ceiling))

    def test_a_zero_ceiling_seed_carries_no_information(self, payload):
        """Where reliability went negative, disattenuation must be undefined
        rather than produce a large 'corrected' correlation."""
        for seed in payload["per_seed"]:
            if seed["max_observable_correlation"] == 0.0:
                assert seed["rho_corrected"] != seed["rho_corrected"], (
                    "seed {} has zero reliability but reports a finite "
                    "disattenuated correlation".format(seed["seed"]))

    def test_no_seed_converged(self, payload):
        assert all(s["converges"] is False for s in payload["per_seed"])

    def test_deltas_are_a_handful_of_ulps(self, payload):
        for seed in payload["per_seed"]:
            assert 1 <= seed["deltas_per_ulp"] <= 50

    def test_provenance_of_the_rule_is_recorded(self, payload):
        assert "xsa-controls" in payload["provenance"]
