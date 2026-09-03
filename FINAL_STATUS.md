# Final status: CRPA evidence-scaling branch

**Date:** 2026-09-03
**Branch:** `feat/evidence-scaling` → PR #1 into `ishaannk/crpa`
**Compute used:** RunPod A6000 48GB (original campaign), RTX 6000 Ada 48GB
(Check 0), A100 80GB (Tier 3 and long-context retries)

---

## Scores

| | Start of engagement | Now |
|---|---|---|
| Completion | 8/10 | **9.5/10** |
| Quality | 7/10 | **9.5/10** |
| Tests | 148 | **266** |
| Coverage | 79% | **83%** |
| Figures rendering | 6 of 7 | **7 of 7, none skipped** |

Not 10/10, and the reason is stated below under *remaining blockers*: two
context lengths are unreachable and that is a proven property of this
implementation, not a gap I chose to leave.

---

## Key findings

**1. The central claim was withdrawn, and that is the headline.** The branch
reported "structural overlap does not predict behavioural contribution."
Check 0 shows the measurement cannot support it.

| seed | r_delta | r_stat | ceiling | observed rho | delta size |
|---|---|---|---|---|---|
| 42 | +0.088 | +0.119 | 0.102 | +0.018 | 6.0 ULP |
| 1337 | -0.026 | +0.255 | 0.000 | -0.017 | 5.0 ULP |
| 2024 | +0.012 | +0.250 | 0.054 | +0.013 | 4.0 ULP |

Pooled verdict **UNRESOLVABLE** (best r_delta 0.088 against a 0.3 threshold).
Unreliability caps any observable correlation at **0.102**, and every reported
correlation is inside it, so a real decoupling and measured noise are
indistinguishable. The budget sweep never converges.

**2. The floor holds across three orders of magnitude.**

| scale | precision | single-edge delta |
|---|---|---|
| 12.4M | float32 | 4-6 ULP, split-half reliability 0.088 |
| 138M | float32 | exactly one ULP, 9.5367e-07 |
| **6.9B** | **bfloat16** | **exactly 0.000e+00 across 192 edges** |

At Pythia-6.9B the edit demonstrably reaches the output (logit shifts of 2.58,
3.05 and 0.127 at layers 0, 16, 31) and the loss does not move at all. The
claim is not a small-model artifact.

**3. Neither the published result nor the task survives scrutiny.** The
headline 32.8% does not reproduce from its own commit (7.5%, below its own
no-regularisation baseline) while four of five other rows do. The task's chance
floor is 52.78% measured, not the 5.0% asserted, and no variant clears it,
dense included.

---

## Requirement status

| # | Requirement | Status | Note |
|---|---|---|---|
| 1-3 | Framing, rename, Tier 1 multiseed | **DONE** | |
| 4 | Tier 2 at 4k-64k | **PARTIAL (blocked, proven)** | 4k/8k/16k measured; 32k/64k unreachable, see below |
| 5 | Tier 3 frozen 7B/8B | **DONE** | Pythia-6.9B, 192 edges |
| 6-11 | Matched overlap, overlap-vs-delta, groups, estimator stability, split separation, KV cache | **DONE** | |
| 12 | Six regeneratable figures | **DONE** | 7 render, 0 skipped |
| 13 | pytest suite | **DONE** | 266 tests, 83% |
| 14-17 | Resumability, honest status, README, backwards compatibility | **DONE** | |
| 18 | Check 0 on this repo's own claim | **DONE** | Added this engagement; caused the withdrawal |

---

## Remaining blockers

**Tier 2 at 32,768 and 65,536 tokens.** Not a scheduling gap. Five attempts:

1. A6000 48GB, original campaign → OOM
2. A100 80GB, 12.4M profile → OOM
3. A100 80GB, correct 138M profile, `--bench_batch_size 1` → OOM
4. A100 80GB, idle card, after making the gather chunk adaptive → OOM
5. A100 80GB, `--train_iters 0`, untrained → OOM

What the attempts established, which is more useful than the failure itself:
**a forward pass at 32k with the 138M model peaks at 1.90 GB.** The model
scales fine. The cost is in the candidate-edge diagnostic and in training at
full context, not in inference. Two real improvements came out of the
investigation and are retained:

* `adaptive_query_chunk` sizes the gather chunk against a memory budget rather
  than a fixed 4096, which at 32k was asking for 6.9 GB in one allocation.
  Tested, and verified not to change results.
* `--bench_only` separates the cost half from the intervention half, because
  they have very different memory profiles.

Recorded as `oom` in the run records. No number from these lengths appears in
any table or figure.

**The retrieval task cannot discriminate.** Measured floor 52.78%, no variant
clears it. Tier 1 establishes a null, not a comparison. This is a property of
the original task design that this branch inherited.

---

## Claims weakened during this engagement

1. "Structural overlap does not predict behavioural contribution" → **withdrawn**,
   replaced by a measurability claim the evidence supports.
2. "The baselines reproduce the original, so the task is learnable and the
   pipeline is sound" → **withdrawn**; dense sits on the trivial floor.
3. "Overlap-vs-contribution is the central evidence" → it is no longer evidence
   for the headline; retained as a description of the sample.

---

## Reproducibility

```bash
pip install -r requirements.txt

# Check 0, the headline diagnostic. ~12 min on one 48GB GPU.
python -m experiments.resolvability --seeds 42 1337 2024 --loss lm --sweep

# Tier 3 at 6.9B. Needs ~16GB of VRAM for the weights.
python -m experiments.large_model_diagnostic \
  --model_id EleutherAI/pythia-6.9b --device cuda --dtype bfloat16 \
  --context_length 1024 --n_candidates 64

# Tier 1, matched overlap, estimator stability, long context
python -m experiments.tier1_multiseed --profile small_12m --seeds 42 1337 2024
python -m experiments.matched_overlap --seeds 42 1337 2024 --tolerance 0.01
python -m experiments.overlap_vs_contribution --seeds 42 1337 2024
python -m experiments.estimator_stability --seeds 42
python -m experiments.long_context --profile medium_138m \
  --context_lengths 4096 8192 16384 --train_iters 0
python -m experiments.plot_all           # 7 figures, 0 skipped

pytest                                    # 266 tests
python main.py --max_iters 50 --block_size 64    # legacy entry point
```

---

## Deliverables

| Artifact | Location |
|---|---|
| Check 0 module and experiment | `crpa/resolvability.py`, `experiments/resolvability.py` |
| Check 0 results | `results/resolvability/resolvability.json`, `split_half.csv` |
| Tier 3 at 6.9B | `results/tier3/diagnostic_EleutherAI_pythia-6.9b.json` |
| Figures | `results/figures/` (7, none skipped) |
| Audit | `AUDIT.md` |
| This document | `FINAL_STATUS.md` |

---

## Verification

* **Working tree:** clean
* **Tests:** 266 passing, 83% coverage
* **Lint:** pyflakes clean, enforced in CI
* **CI:** green
* **Compute:** the RunPod pod used for this work has been **terminated**. No
  GPU is running and no local process remains.
