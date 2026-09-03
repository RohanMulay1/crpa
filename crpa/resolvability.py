"""
crpa.resolvability - Check 0, applied to this repository's own headline claim.

Ported from the ``xsa-controls`` project's ``xsac/checks.py``. The decision
rule below is reproduced unchanged so that both projects apply the same
pre-registered standard; do not re-tune it here.

Why this module exists
----------------------
Tier 1 reports that structural overlap is a weak predictor of behavioural
contribution: Pearson r of +0.080, -0.054 and -0.149 across three seeds, with
the sign unstable. That is only interpretable if the contribution measurement
is itself reliable.

It may well not be. This repository's own results record

    delta_abs_p95            = 3.81e-06
    eps_over_delta_scale     = 7864
    frac_classified_suppressible = 1.0
    vacuous                  = true

and a float32 loss near 6.8 has a ULP of about 4.8e-07, so a typical delta is
only a handful of representable steps. **A near-zero correlation between a
reliable statistic and an unreliable one is guaranteed by construction.**
Reporting the correlation as a decoupling without first showing the delta is
resolvable would be measuring our own noise floor and calling it a finding.

The diagnostic is a split-half: measure each candidate edge's delta twice, on
two disjoint halves of the evaluation data, and correlate the two estimates.
The same treatment applied to the overlap statistic gives its reliability, and
the two together bound how large the true correlation could be.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from crpa.metrics import correlations

#: The pre-registered rule, copied verbatim from xsa-controls so that the two
#: projects cannot drift apart. Thresholds are on the split-half reliability of
#: the measured effect.
RESOLVABILITY_RULE = (
    (0.6, "reliable",
     "Delta is reliable. Report rho and rho_corrected. A near-zero rho is a "
     "real decoupling."),
    (0.3, "attenuated",
     "Report both, lead with rho_corrected, and state the attenuation "
     "explicitly."),
    (0.0, "unresolvable",
     "Delta is not resolvable at this budget. No decoupling may be claimed. "
     "Report the reliability failure itself as the Check 0 result, and drop "
     "the correlation claim from the abstract."),
)


def verdict_for(r_delta: float) -> tuple:
    if not math.isfinite(r_delta):
        return RESOLVABILITY_RULE[-1][1], RESOLVABILITY_RULE[-1][2]
    for threshold, name, action in RESOLVABILITY_RULE:
        if r_delta >= threshold:
            return name, action
    return RESOLVABILITY_RULE[-1][1], RESOLVABILITY_RULE[-1][2]


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation, NaN when either side has no variance."""
    xs, ys = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if xs.size < 3:
        return float("nan")
    stats = correlations(xs.tolist(), ys.tolist())
    r = stats.get("spearman_r", float("nan"))
    return float(r) if r is not None else float("nan")


def disattenuate(rho_observed: float, r_delta: float, r_stat: float) -> float:
    """Correct a correlation for the unreliability of both measures.

    Returns NaN when either reliability is non-positive. Dividing by a
    near-zero reliability produces an arbitrarily large "corrected"
    correlation, which is an artifact of the formula rather than a finding.
    """
    if not all(math.isfinite(x) for x in (rho_observed, r_delta, r_stat)):
        return float("nan")
    if r_delta <= 0 or r_stat <= 0:
        return float("nan")
    return float(rho_observed / math.sqrt(r_delta * r_stat))


def max_observable_correlation(r_delta: float, r_stat: float) -> float:
    """The ceiling that unreliability puts on any observable correlation.

    Even a perfect underlying relationship cannot show up above
    ``sqrt(r_delta * r_stat)``. When that ceiling is near zero, an observed
    r of zero carries no information about the underlying relationship, and
    reporting it as evidence of independence is a category error.
    """
    if not all(math.isfinite(x) for x in (r_delta, r_stat)):
        return float("nan")
    if r_delta <= 0 or r_stat <= 0:
        return 0.0
    return float(math.sqrt(r_delta * r_stat))


def ulp(value: float) -> float:
    """Float32 spacing at ``value``. The quantisation floor of a measurement."""
    v = np.float32(abs(value))
    if v == 0:
        return float(np.spacing(np.float32(0)))
    return float(np.spacing(v))


@dataclass
class ResolvabilityResult:
    """Check 0 on one seed. ``passed`` means the effect can be correlated."""

    seed: int
    r_delta: float
    r_stat: float
    verdict: str
    action: str
    n_candidates: int
    rho_observed: float = float("nan")
    rho_corrected: float = float("nan")
    ceiling: float = float("nan")
    delta_scale: float = float("nan")
    ulp_at_loss: float = float("nan")
    deltas_per_ulp: float = float("nan")
    budget_curve: Dict[int, float] = field(default_factory=dict)
    converges: Optional[bool] = None
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == "reliable"

    def to_dict(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "r_delta": self.r_delta,
            "r_stat": self.r_stat,
            "verdict": self.verdict,
            "action": self.action,
            "passed": self.passed,
            "n_candidates": self.n_candidates,
            "rho_observed": self.rho_observed,
            "rho_corrected": self.rho_corrected,
            "max_observable_correlation": self.ceiling,
            "delta_scale_p95": self.delta_scale,
            "ulp_at_loss": self.ulp_at_loss,
            "deltas_per_ulp": self.deltas_per_ulp,
            "budget_curve": {str(k): v for k, v in self.budget_curve.items()},
            "converges": self.converges,
            "note": self.note,
        }

    def summary(self) -> str:
        lines = [
            "Check 0 (resolvability), seed {}: {}".format(
                self.seed, self.verdict.upper()),
            "  split-half reliability of delta      r_delta = {:+.3f}".format(
                self.r_delta),
            "  split-half reliability of overlap    r_stat  = {:+.3f}".format(
                self.r_stat),
            "  n candidate edges                            = {}".format(
                self.n_candidates),
            "  delta scale (95th pct |delta|)               = {:.3e}".format(
                self.delta_scale),
            "  float32 ULP at the baseline loss             = {:.3e}".format(
                self.ulp_at_loss),
            "  typical delta in ULPs                        = {:.1f}".format(
                self.deltas_per_ulp),
            "  observed rho(overlap, delta)                 = {:+.3f}".format(
                self.rho_observed),
            "  ceiling sqrt(r_delta * r_stat)               = {:.3f}".format(
                self.ceiling),
            "  disattenuated rho                            = {}".format(
                "{:+.3f}".format(self.rho_corrected)
                if math.isfinite(self.rho_corrected) else "undefined"),
        ]
        if self.budget_curve:
            pts = ", ".join("{}:{:+.3f}".format(b, v)
                            for b, v in sorted(self.budget_curve.items()))
            lines.append("  budget sweep (replicate agreement)  {}".format(pts))
            if self.converges is False:
                lines.append("  the curve is FLAT: more evaluation data does "
                             "not make the estimate converge")
        lines.append("  -> {}".format(self.action))
        return "\n".join(lines)


def assess(delta_half_a: Sequence[float], delta_half_b: Sequence[float],
           stat_half_a: Sequence[float], stat_half_b: Sequence[float],
           seed: int, baseline_loss: float = float("nan"),
           budget_curve: Optional[Dict[int, float]] = None
           ) -> ResolvabilityResult:
    """Run Check 0 on paired split-half measurements.

    ``delta_half_a[i]`` and ``delta_half_b[i]`` must be two independent
    estimates of the SAME edge's contribution, measured on disjoint evaluation
    data. Likewise for the statistic.
    """
    r_delta = spearman(delta_half_a, delta_half_b)
    r_stat = spearman(stat_half_a, stat_half_b)

    # The correlation of interest, pooled over both halves so it uses all the
    # data rather than an arbitrary one.
    pooled_stat = list(stat_half_a) + list(stat_half_b)
    pooled_delta = list(delta_half_a) + list(delta_half_b)
    rho = spearman(pooled_stat, pooled_delta)

    all_deltas = [abs(float(d)) for d in pooled_delta
                  if d is not None and math.isfinite(float(d))]
    scale = float(np.percentile(all_deltas, 95)) if all_deltas else float("nan")
    u = ulp(baseline_loss) if math.isfinite(baseline_loss) else float("nan")
    per_ulp = scale / u if (math.isfinite(scale) and math.isfinite(u)
                            and u > 0) else float("nan")

    name, action = verdict_for(r_delta)
    converges = None
    if budget_curve and len(budget_curve) >= 3:
        vals = [budget_curve[b] for b in sorted(budget_curve)]
        finite = [v for v in vals if math.isfinite(v)]
        converges = bool(finite and (max(finite) - min(finite) > 0.2)
                         and finite[-1] > 0.3)

    return ResolvabilityResult(
        seed=seed, r_delta=r_delta, r_stat=r_stat, verdict=name, action=action,
        n_candidates=len(delta_half_a), rho_observed=rho,
        rho_corrected=disattenuate(rho, r_delta, r_stat),
        ceiling=max_observable_correlation(r_delta, r_stat),
        delta_scale=scale, ulp_at_loss=u, deltas_per_ulp=per_ulp,
        budget_curve=dict(budget_curve or {}), converges=converges)


def pooled_verdict(results: Sequence[ResolvabilityResult]) -> Dict[str, object]:
    """Combine per-seed Check 0 outcomes into one reportable verdict.

    The pooled verdict takes the BEST seed's reliability, not the mean. If the
    measurement is unresolvable even at its most favourable seed, no averaging
    rescues it.
    """
    if not results:
        return {"verdict": "unresolvable", "n_seeds": 0,
                "action": RESOLVABILITY_RULE[-1][2]}
    finite = [r.r_delta for r in results if math.isfinite(r.r_delta)]
    best = max(finite) if finite else float("nan")
    name, action = verdict_for(best)
    ceilings = [r.ceiling for r in results if math.isfinite(r.ceiling)]
    return {
        "verdict": name,
        "action": action,
        "n_seeds": len(results),
        "r_delta_best": best,
        "r_delta_per_seed": {str(r.seed): r.r_delta for r in results},
        "r_stat_per_seed": {str(r.seed): r.r_stat for r in results},
        "max_observable_correlation": max(ceilings) if ceilings else float("nan"),
        "any_seed_passed": any(r.passed for r in results),
        "claim_permitted": (
            "overlap does not predict contribution"
            if name == "reliable" else
            "overlap is a weak predictor ON THIS SAMPLE; the measurement is "
            "not reliable enough to distinguish that from attenuation"),
    }
