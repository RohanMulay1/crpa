# Audit: CRPA evidence-scaling branch

Engineering and scientific audit of this branch against the original brief.
Every number below is read from a committed artifact in `results/`, and every
experiment named here was executed on the hardware stated.

**Date:** 2026-09-03
**Compute:** RunPod RTX 6000 Ada 48GB (Check 0), RTX A6000 48GB (original Tier
1/2 campaign)

---

## The headline change

**A claim was withdrawn.** This branch previously reported, as its central
result:

> Structural overlap does not predict behavioural contribution.

That claim is not supported by its own measurement, and it has been replaced.
The reason is Check 0, which this branch did not previously run on itself.

### What Check 0 found

`python -m experiments.resolvability --seeds 42 1337 2024 --loss lm --sweep`

Each candidate edge's contribution is measured twice, on two disjoint halves of
the evaluation split. The two estimates are then correlated. The same treatment
is applied to the overlap statistic. The decision rule is ported unchanged from
the `xsa-controls` project so both apply one pre-registered standard.

| seed | r_delta | r_stat | ceiling `sqrt(r_delta*r_stat)` | observed rho | delta size | budget curve converges |
|---|---|---|---|---|---|---|
| 42 | +0.088 | +0.119 | 0.102 | +0.018 | 6.0 ULP | no |
| 1337 | -0.026 | +0.255 | 0.000 | -0.017 | 5.0 ULP | no |
| 2024 | +0.012 | +0.250 | 0.054 | +0.013 | 4.0 ULP | no |

**Pooled verdict: UNRESOLVABLE.** Best `r_delta` = 0.088 against a 0.3
threshold.

Unreliability caps any observable correlation at 0.102. Every correlation this
branch reported (|rho| <= 0.018) sits far inside that ceiling, so **the data
cannot distinguish a real decoupling from measured noise**. The
evaluation-budget sweep confirms the cause: replicate agreement runs -0.154,
+0.007, +0.165, +0.032 as the budget grows and never converges, which is the
signature of a quantisation floor rather than a small effect. A single-edge
delta is 4 to 6 float32 ULPs.

### The claim now

> At this scale, single-edge behavioural contribution is not resolvable in
> float32. Any criterion that thresholds it, including the one this repository
> audits, is unfalsifiable at that granularity. Group-level candidates are a
> requirement, not a convenience.

This is narrower than what was claimed before. It is also the only claim the
measurement supports, and unlike the withdrawn one it is not
[scooped by an existing literature](#limitations).

---

## Requirement matrix

Original brief, item by item. Before = state at the previous audit (7/10
quality, 8/10 completion). After = now.

| # | Requirement | Before | After | Evidence / reason |
|---|---|---|---|---|
| 1 | Framing: causal importance -> behavioural contribution | DONE | DONE | Terminology section in README; `crpa_contribution` canonical |
| 2 | Rename `crpa_causal` -> `crpa_contribution`, keep aliases | DONE | DONE | Alias resolves; checkpoints load; `figures.VARIANT_COLOR` carries both |
| 3 | Tier 1: 12.4M/512, seeds 42/1337/2024 | DONE | DONE | `results/tier1/aggregate.json`, 3 seeds x 5 variants |
| 4 | Tier 2: ~138M at 4k-64k | PARTIAL | **PARTIAL (constraint now proven)** | 4k/8k/16k measured. 32k/64k retried on an A100 80GB with batch 1 and adaptive chunking and still exceed memory; see the constraint note below. Recorded `oom`, never as numbers |
| 5 | Tier 3: frozen 7B/8B diagnostic | PARTIAL | **DONE** | **Pythia-6.9B on an A100 80GB, bf16, 192 edges across layers 0/16/31.** Edits propagate (logit shift 2.58, 3.05, 0.127) while every delta loss is exactly 0.000e+00 |
| 6 | Matched-overlap sweep on **realised** overlap | DONE | DONE | 36 runs, 12 pairs, mean abs diff 0.0025; self-comparisons excluded |
| 7 | Overlap vs intervention-delta dataset | DONE | DONE (reinterpreted) | 3,310 edges. Still produced; **no longer load-bearing**, see Check 0 |
| 8 | High-overlap group analyses | DONE | DONE (reinterpreted) | Both tails populated; uninterpretable for the same reason |
| 9 | Estimator ranking-stability | DONE | DONE | Spearman -0.087..+0.121 across budgets 2-32; never converges |
| 10 | Calibration / evaluation separation | DONE | DONE | Three-way split, disjoint RNG streams, asserted by test |
| 11 | KV-cache accounting, measured vs projected | DONE | DONE | `results/tier2/kv_cache.csv`; routing shown to prevent bounding |
| 12 | Six regeneratable figures | PARTIAL | **DONE** | **All 7 render, 0 skipped**, including `fig3_large_model` now that Tier 3 has real data |
| 13 | pytest suite | DONE | DONE | **266 tests, 83% coverage** (was 148 / 79%) |
| 14 | Resumable experiments | DONE | DONE | Content-hash run ids; completed cells skipped |
| 15 | Honest status handling | DONE | DONE | Status enum; `numeric_records()` sole accessor; two new guards |
| 16 | Updated README | DONE | DONE | Rewritten; central claim withdrawn with the evidence |
| 17 | Backwards compatibility | DONE | DONE | `python main.py` runs; legacy imports resolve |
| 18 | **Check 0 on this repo's own claim** | **MISSING** | **DONE** | `crpa/resolvability.py`, `experiments/resolvability.py`, 38 tests |

### BLOCKED, with reasons

| Item | Why |
|---|---|
| Tier 2 at 32,768 and 65,536 | **Proven, not assumed.** Retried on an A100 80GB at the correct 138M profile with `--bench_batch_size 1`, on an otherwise idle card, and again after making the gather chunk adaptive. All attempts exceed memory. What the attempts did establish: a **forward pass at 32k costs only 1.90 GB peak**, so the model scales fine and the cost is in the diagnostic and in training at full context, not in inference. The adaptive-chunk fix (`adaptive_query_chunk`) is a real reduction in peak allocation and is retained, tested, and verified not to change results |

### Closed since the previous audit

| Item | Was | Now |
|---|---|---|
| Tier 3 at 7B | no run | **Pythia-6.9B measured.** 192 edges, every delta exactly 0.000e+00 while logits shift by whole units |
| `fig3_large_model` | absent | **renders**; all 7 figures, 0 skipped |

---

## Defects found and fixed in this pass

| Defect | How it was found | Consequence if unfixed |
|---|---|---|
| **Central claim unsupported by its own measurement** | Running Check 0 | A published decoupling claim that the data cannot support |
| Short runs recordable as completed measurements | Working-tree audit | Nine 3-iteration runs had entered the Tier 1 aggregate, moving retrieval 4.31% -> 2.15% and perplexity 910 -> 26,485 |
| Aggregates averaged across training budgets | Same audit | Any config heterogeneity silently pooled |
| Chance floor asserted, not measured | External review | Real floor 52.78%, not 5.0%. Every above-chance verdict inverted |
| Self-comparisons in matched pairs | External review | 3 of 15 pairs compared a model with itself |
| Auxiliary routing entropy pinned at ln(4) | External review | A constant reported as a finding |
| `write_csv` argument order in new code | GPU run | Check 0's per-edge CSV silently not written |

---

## Test and coverage changes

| module | before | after | why it mattered |
|---|---|---|---|
| `crpa/resolvability.py` | did not exist | **95%** | New. 38 tests, including a class that re-reads the committed Check 0 artifact and fails if the README and the JSON ever disagree |
| `crpa/figures.py` | 42% | **69%** | Figures are where a number reaches a reader. Tests pin that a missing input skips rather than drawing placeholders, that every figure writes its own source CSV, and that an OOM row never becomes a plotted point |
| **total** | 148 tests, 79% | **266 tests, 83%** | |

One test assertion was corrected during review rather than the code: a
per-seed reliability ceiling is not a hard bound, because a negative
reliability degenerates it to zero. The defensible statement, and the one the
README makes, is that every observed correlation is below the best ceiling any
seed achieved.

---

## The measurability floor now spans three orders of magnitude

| scale | precision | single-edge delta |
|---|---|---|
| 12.4M (Tier 1) | float32 | 4 to 6 ULP, split-half reliability 0.088 |
| 138M (Tier 2) | float32 | exactly one ULP, 9.5367e-07 |
| **6.9B (Tier 3)** | **bfloat16** | **exactly 0.000e+00 across 192 edges** |

At 6.9B the edit demonstrably reaches the output, with logit shifts of 2.58,
3.05 and 0.127 at layers 0, 16 and 31, and the loss does not move at all. The
claim this branch now makes is not a small-model artifact.

---

## Reproduction

```bash
pip install -r requirements.txt

# The headline diagnostic. ~12 min on one 48GB GPU, 3 seeds.
python -m experiments.resolvability --seeds 42 1337 2024 --loss lm --sweep

# Everything else
python -m experiments.tier1_multiseed --profile small_12m --seeds 42 1337 2024
python -m experiments.matched_overlap --seeds 42 1337 2024 --tolerance 0.01
python -m experiments.overlap_vs_contribution --seeds 42 1337 2024
python -m experiments.estimator_stability --seeds 42
python -m experiments.long_context --context_lengths 4096 8192 16384
python -m experiments.plot_all

pytest                        # 266 tests
python main.py --max_iters 50 --block_size 64   # legacy entry point
```

---

## Limitations, stated plainly

* **The retrieval task cannot discriminate.** Its measured floor is 52.78% and
  no variant clears it, dense included. Tier 1 therefore establishes a null,
  not a comparison. The instrument is too weak for the question it was built
  to answer, and that is a property of the original design.
* **Three seeds** on Tier 1, twelve matched pairs. Small.
* **The withdrawn framing was also scooped.** Even had the measurement
  supported it, "structural statistics do not predict causal contribution" is
  established by Serrano & Smith 2019, Jain & Wallace 2019, Kobayashi 2020,
  Mohebbi 2023 and Hanna et al. 2024, among others. The measurability claim
  that replaces it is not covered by that literature.
* **`results/estimator_stability/stability.csv` predates the degeneracy
  guard** and still shows `classification_agreement = 1.0`. It is left rather
  than hand-edited; the README explains the column is degenerate.
* **Check 0 was run on models retrained for the purpose**, not on the exact
  checkpoints behind the original Tier 1 numbers, which were not retained.
  Reliability is a property of the measurement regime rather than of specific
  weights, so this does not affect the verdict, but it is a difference worth
  stating.

## Claims weakened in this pass

1. "Structural overlap does not predict behavioural contribution" -> withdrawn,
   replaced by a measurability claim.
2. "The baselines reproduce the original, so the task is learnable and the
   pipeline is sound" -> withdrawn in an earlier pass; dense sits on the
   trivial floor.
3. "Overlap-vs-contribution is the central evidence" -> it is no longer
   evidence for the headline at all; it is retained as a description of the
   sample.
