# CRPA: structural overlap versus behavioral contribution

Attention overlap is a **structural** statistic. It says how similar two queries'
attention supports look. Redundancy is a **behavioral** property. It says whether
removing an interaction actually changes what the model does.

This repository exists to test whether the first predicts the second.

> **Core claim.** High structural overlap does not imply dispensability.
> Behavioral contribution, estimated by intervention, is a better criterion for
> deciding which overlapping interactions may be suppressed.

We are **not** claiming CRPA is a fast long-context transformer. The evidence
here does not support that, and the KV-cache analysis below shows why it is not
even the right claim to reach for.

**On the published result.** The original repository reported 32.8% retrieval
for contribution-gated CRPA against 5.3% for naive. Running that code
unmodified at its own commit gives **7.5%**, below its own 10.9%
no-regularization baseline, and its own verification script reports FAILED on
the claim. Four of the other five rows reproduce closely, so this is not an
environment artifact. Section 10 has the numbers and the controls that
establish it. What survives, and what this branch measures properly, is the
weaker and better-supported claim above.

---

## 1. Terminology

The project used to say "causal". It now says what it can defend.

| Term | Meaning |
|---|---|
| **structural overlap** | A geometric statistic over attention supports. Computed from attention probabilities alone, with no reference to behaviour. |
| **behavioral contribution** | The measured change in loss when a specific interaction is removed. `Delta(e) = L(M \ e) - L(M)`. |
| **contribution-gated** | Suppression gated on measured behavioral effect rather than on overlap. |
| **intervention sensitivity** | A synonym for behavioral contribution, used when discussing the estimator. |

We estimate *the behavioral effect of removing an interaction*. We do not claim
to establish causal importance: a single-edge ablation under a frozen mask is a
narrow intervention, not an identification strategy.

The variant historically called `crpa_causal` is now `crpa_contribution`. The
old name resolves to the new one everywhere, and checkpoints saved under the old
name load unchanged.

### What an "edge" is

An **edge** is `e = (layer, head, query position i, key position j)` with
`j <= i` and `j` present in `Omega(i)`. Intervening on `e` sets that single
pre-softmax score to `-inf`, for that head in that layer only. The surviving
entries of row `i` renormalise, so the removed interaction's probability mass is
redistributed rather than deleted.

Overlap and intervention refer to the **same object**. In the original
implementation they did not, which is the single most important thing this work
fixes (see section 9).

---

## 2. Architecture

`Omega(i) = P(i) u G u C_k(i)`

- **P(i)** tokens in the same positional partition, a local window of size `w`
- **G** relay tokens at fixed intervals; every token may attend to a relay, and
  a relay row attends causally to everything
- **C_k(i)** up to `k` routed keys drawn from other *router* partitions

Two attention implementations, numerically equivalent, selected by
`--attention_impl`:

| Implementation | Use |
|---|---|
| `dense_masked` | Materialises `(B, H, T, T)`. The reference path, and the default at short context. |
| `sparse_gather` | Computes only the entries `Omega(i)` contains, chunked over query blocks. Required above roughly 8k tokens. |

The gather path is not an optimisation, it is a precondition. At `T=65536` with
12 heads a bf16 dense score tensor is about 103 GB for one layer, so the Tier 2
experiment cannot be represented on the dense path on any GPU.
`tests/test_attention.py::TestImplementationEquivalence` asserts the two agree
to floating-point tolerance, including for ragged final blocks.

### Reading attention back out

Overlap statistics need the attention probabilities, and retaining them densely
has the same problem: `(B, H, T, T)` is 12.9 GB per layer at `T=16384` with 12
heads, and 180 GB across the 14-layer profile. But each non-relay query has at
most `partition_size + n_relays + cross_k` permitted keys, about 552 at the
medium profile, so the dense form is over 97% zeros.

Diagnostics therefore retain attention in its **gathered** form for a window of
queries (`SparseProbs`): roughly 27 MB per layer, independent of context
length. The window sits at the end of the sequence, which is also where
reachability concentrates under a last-token loss.

Relay rows are excluded from the overlap comparison in both representations. A
relay attends causally to everything by construction, so its support overlaps
every local query as an artifact of the mask rather than as evidence of
redundancy, and the gathered form cannot represent it. With that exclusion the
two computations agree to 2.9e-08 across 3024 edges, which a test pins.

A note on the two meanings of "partition": the local window `P(i)` is defined by
*position*, while `C_k(i)` excludes keys sharing the query's *router*
assignment. These are different notions. The original did this and we preserved
the behaviour, but it is worth knowing when reading routing results.

---

## 3. The controlled experiment

The backbone (embeddings, FFN, LayerNorm, residuals, weight tying) is identical
across every variant, so a comparison between variants isolates attention.

The central comparison is three variants:

| Variant | Suppression criterion |
|---|---|
| `crpa_noreg` | none |
| `crpa_naive` | rank candidates by structural overlap, remove the top `budget` |
| `crpa_contribution` | measure behavioral contribution for the same candidates, remove the `budget` with the lowest delta |

Both regularised variants draw from the **same candidate pool**, refresh on the
**same cadence**, after the **same warmup**, with the **same removal budget**.
Only the ranking criterion differs. That is what makes it a controlled
comparison; the original gave the two variants different pool sizes, cadences
and warmup behaviour.

`dense` and `sliding` baselines are available via `--include_baselines` and are
secondary context, not the comparison of interest.

### Task

Needle-in-Haystack over WikiText-2 filler. Key/value pairs are embedded at depth
0.55 to 0.73, one partition away from the query, so answering requires a
cross-partition hop through a relay. The last token is the query key and the
target is its paired value.

**Chance accuracy is 5.0%** (20 possible values). A variant at or below chance
has not learned retrieval, and differences between two such variants are not
evidence about retrieval quality. Every results table prints this.

### Data separation

| Role | Language-model source | Needle RNG stream |
|---|---|---|
| train | WikiText-2 `train` | `seed * 1000 + 0` |
| calibration | WikiText-2 `validation` | `seed * 1000 + 1` |
| evaluation | WikiText-2 `test` | `seed * 1000 + 2` |

Anything chosen using data (contribution thresholds, gate decisions, the
suppressible classification) is fitted on **calibration** and reported on
**evaluation**. `Corpus.assert_splits_disjoint` hashes generated sequences and
raises if two streams can collide; it runs at the start of every experiment.
Split provenance is written into every result file.

---

## 4. Three tiers

| Tier | Question | Scale |
|---|---|---|
| **1** | Does structural overlap predict behavioral contribution, and does gating on the latter beat gating on the former at a matched overlap budget? | 12.4M params, 512 tokens |
| **2** | Does the diagnostic stay meaningful as context grows? | 137.8M params, 4k to 64k |
| **3** | Does the relationship stay imperfect at large model scale? | frozen 7B/8B, no training |

---

## 5. Setup

```bash
git clone https://github.com/ishaannk/crpa.git && cd crpa
python3 -m venv .venv && source .venv/bin/activate

# CPU is enough for Tier 1 smoke runs, all tests, and the Tier 3 tiny-model check.
pip install torch numpy scipy pandas matplotlib pyyaml pytest

# For WikiText-2 and the Tier 3 diagnostic
pip install datasets transformers

# Note: the corpus is loaded as `Salesforce/wikitext`. Newer huggingface_hub
# rejects the bare `wikitext` id the original used; Corpus.load falls back
# through known aliases and records which one resolved.

# GPU (the original used CUDA 12.4 wheels on an A6000 driver reporting 12.8)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Optional, only for quantised Tier 3 loading
pip install accelerate bitsandbytes
```

Heavy dependencies stay optional. Tier 1 needs neither `accelerate` nor
`bitsandbytes`.

---

## 6. Quick smoke test

Runs on CPU in a couple of minutes, downloads nothing, and exercises every code
path. Results are recorded with `status: "smoke"` and are never readable as
completed measurements.

```bash
python -m pytest tests/ -q -m "not slow"

python -m experiments.tier1_multiseed --smoke --synthetic_data --seeds 42
python -m experiments.matched_overlap  --smoke --synthetic_data --seeds 42 \
    --lambdas 0.0 0.1 --tolerance 0.05
python -m experiments.plot_all --results_dir results
```

`--synthetic_data` substitutes a deterministic pseudo-corpus for WikiText-2.
Language-model numbers from it are meaningless and are labelled as such.

---

## 7. Tier 1

```bash
# The original entry point still works, unchanged flags
python main.py
python main.py --max_iters 100 --block_size 64
python main.py --figures_only          # now actually regenerates figures
python main.py --skip_multiseed

# Multi-seed replication, three variants x three seeds
python -m experiments.tier1_multiseed --profile small_12m --seeds 42 1337 2024

# Structural overlap vs behavioral contribution (the central dataset)
python -m experiments.overlap_vs_contribution --profile small_12m \
    --seeds 42 1337 2024 --n_per_layer 24

# Estimator ranking stability across sample budgets
python -m experiments.estimator_stability --profile small_12m \
    --budgets 2 4 8 16 32 --replicates 3

# Reproduce the original selection semantics for comparison
python -m experiments.tier1_multiseed --intervention_mode legacy_rowpair
```

### Matched-overlap sweep

The reason this experiment exists: the previously published comparison put naive
at overlap 0.251 and contribution-gated at 0.243. Those are close, not matched,
and they came from a single identical regularization strength. A retrieval
difference at two different structural budgets is not attributable to the
selection criterion.

```bash
python -m experiments.matched_overlap --profile small_12m \
    --lambdas 0.00 0.01 0.02 0.05 0.10 0.20 \
    --seeds 42 1337 2024 --tolerance 0.01
```

Pairing is on **realized** overlap measured after training, never on the
configured lambda. Each naive run is matched to its single nearest
contribution-gated counterpart within the same seed, and pairs outside the
tolerance are discarded. `find_matched_overlap_pairs` is unit-tested, including
a test that two runs at identical lambda but far apart in realized overlap do
not pair.

---

## 8. Tier 2 and Tier 3

```bash
# Long-context diagnostic. 32k and 64k need an 80GB card.
python -m experiments.long_context --profile medium_138m \
    --context_lengths 4096 8192 16384 --n_candidates 32

# Latency, throughput, memory, KV cache
python -m experiments.benchmark --profile medium_138m \
    --context_lengths 4096 8192 16384 --dtype bfloat16 --device cuda

# Tier 3 tiny-model check (CPU, seconds)
python -m experiments.large_model_diagnostic --smoke \
    --model_id hf-internal-testing/tiny-random-LlamaForCausalLM \
    --context_length 64 --n_candidates 8 --partition_size 16

# Tier 3 real diagnostic (40GB+)
python -m experiments.large_model_diagnostic \
    --model_id meta-llama/Meta-Llama-3-8B \
    --dtype bfloat16 --device_map auto \
    --context_length 1024 --layers 8 16 24 --n_candidates 32
```

Tier 3 trains nothing. It loads a frozen model with `attn_implementation="eager"`
and refuses to proceed otherwise, because FlashAttention and SDPA never
materialise the probability matrix, so neither extraction nor a per-head
intervention is possible through them. A 4D `attention_mask` cannot express a
per-head edit either, since it broadcasts across heads, so the target layer's
forward is patched directly.

Before any numbers are recorded, `verify_instrumentation` removes a real edge and
asserts its probability went to zero and the row still sums to 1. If it did not,
the run raises. There is no path by which an inert intervention is reported as a
null result. `trust_remote_code` is off by default.

---

## 9. What changed, and why

The audit found the headline result rested on a broken intervention. All of
these are fixed; the original file for any module is recoverable with
`git show 7474c77:<file>`.

| # | Finding | Fix |
|---|---|---|
| F1 | **The intervention measured a different object than the overlap statistic.** Candidates were pairs of *query rows* scored by support Jaccard, but the intervention masked the *edge* `i -> j`. Those are not the same thing, so the delta did not measure the effect of removing the overlapping interaction. | Candidates are edges; overlap and intervention refer to one object. |
| F2 | **About half of all interventions were no-ops.** `i` and `j` were drawn independently within a partition and the mask is causal, so every sample with `i < j` masked an entry that was already `False`. Delta was identically 0, which is `<= eps`, so they were all classified redundant. | The edit counter ignores already-masked entries, and a zero-effect intervention raises instead of reporting 0. |
| F3 | **Delta was measured under a randomly changing mask.** Routed keys were re-drawn from the global RNG whenever a step counter hit a multiple of 20, including during estimation, so baseline and intervened passes could use different masks. | Routing uses an explicit generator, and every measurement runs inside `frozen_structure()`. |
| F4 | **`_mask_pair` was global across layers.** A "per-layer" sensitivity was the effect of an identical simultaneous intervention in all layers, recomputed once per layer. | Interventions are addressed per `(layer, head, query, key)`. |
| F5 | **No train/calibration/evaluation separation.** All three roles drew from one global RNG. | Three disjoint streams plus three WikiText splits, with a leakage assertion. |
| F6 | **naive vs contribution was confounded.** Different pool sizes, cadences and warmup. | Shared pool, cadence, warmup and budget. Only the criterion differs. |
| F7 | **Overlap was measured on dropout-corrupted attention.** `_Alast` was stored after dropout, so rows did not sum to 1 during training. | Captured before dropout. |
| F8 | **CRPA was masked-dense.** Every variant materialised `(B,H,T,T)`, so the published runtime table measured mask construction, not attention cost. | Added the gather path, with an equivalence test. |
| F9 | **`--figures_only` was dead code.** The entire body sat inside `if not args.figures_only:`. Figures also required in-memory models, so no plot could regenerate from files. | Figures regenerate from `results/` with no training. |
| F10 | **Routing diagnostics measured nothing.** The token embedding was fed to every layer's router instead of that layer's input. Every row of the published Table 5 was identical (entropy ln 4, load error 0). | Reads each layer's real input. See section 12. |
| F11 | **Incomplete seeding.** `torch.cuda` was never seeded, and `measure_throughput` called `torch.manual_seed(0)` mid-run, clobbering the experiment seed. | All RNGs seeded; benchmarking uses a scoped `local_seed`. |
| F12 | **Global mutable `CFG`.** Mutated and restored in place. No `.gitignore`. | Frozen dataclasses and YAML profiles; legacy dict preserved as a bridge. |
| F13 | **The query key was never at the scored position.** The needle builder filled to `block_size - 3`, appended the query key, then padded with random filler, leaving the key two positions from the end while the model is scored at the last position. The docstring said "the last token is a query key". It was not. | The query key sits at exactly `block_size - 1`. |

### A finding that is not a bug

Under a last-token loss, most edges cannot affect the measurement at all: with a
sparse causal mask, an early-position edge in a lower layer often has no path to
the final position, so its delta is exactly zero as a matter of graph structure.
Scoring those and then classifying `delta <= eps` as suppressible would repeat
F2 through a different mechanism. Candidates are therefore filtered by
`reachable_queries`, which propagates reachability backwards through the masks
while accounting for the residual stream. The filter is recorded with results.

### About the previously published numbers

`results/original_published/` holds the original committed artifacts, left in
place as historical record. They report contribution-gated at 32.8% retrieval
against naive at 5.3% and no-reg at 8.4%. Read them with four caveats:

1. **They do not reproduce.** Running that code at its own commit gives 7.5%
   for the headline configuration, below its own 10.9% no-reg baseline. See
   section 10 and `results/original_reproduction/`.
2. They came from the F1/F2 intervention, so the gate was selecting on a
   statistic substantially composed of no-ops, and `eps` was four orders of
   magnitude too large to reject them even if they had been real.
3. Chance is 5.0%. Two of the three compared variants sit at 5.3% and 8.4%, so
   the comparison is one partially-working model against two that are not
   working.
4. Single seed.

`--intervention_mode legacy_rowpair` reproduces the original *candidate
selection and intervention semantics* so the comparison can be made directly. It
does not reproduce every incidental behaviour (head-averaged overlap, the
unfrozen mask, dropout-contaminated capture); for that, check out the original
commit.

---

## 10. What we measured

Everything below was run on this branch. Commands are in section 7. Anything
not listed here was not run, and the end of this section says so explicitly.

Hardware: one NVIDIA L40S (46 GB), torch 2.4.1+cu124, bf16 autocast, WikiText-2
via `Salesforce/wikitext`. Small profile: 12.4M parameters, 512 tokens, 4000
iterations, seeds 42 / 1337 / 2024.

### Tier 1: the central comparison

| variant | retrieval % (mean over 3 seeds) | realized overlap | originally published |
|---|---|---|---|
| dense | **53.6** | 0.401 | 50.9 |
| sliding window | *(see results/tier1)* | | 51.9 |
| `crpa_noreg` | 4.4 | 0.244 | 8.4 |
| `crpa_naive` | 4.6 | 0.264 | 5.3 |
| `crpa_contribution` | 4.3 | 0.222 | 32.8 |

**Chance is 5.0%.** All three CRPA variants sit on it; their bootstrap intervals
overlap each other and the chance line. The dense and sliding baselines
reproduce the original closely, which is what establishes that the task is
learnable and that the pipeline, splits and evaluation are sound. The CRPA null
is therefore a fact about CRPA at this scale, not an artifact.

Note the direction of the overlap column: dense has the **highest** realized
overlap and by far the best retrieval. Reducing overlap is not what makes this
task work.

### Structural overlap versus behavioral contribution

3310 individually intervened edges across three seeds, contribution measured on
the calibration split, reported on evaluation.

| | Pearson r | p | n |
|---|---|---|---|
| pooled | **0.005** | 0.771 | 3310 |
| seed 42 | +0.080 | 0.0096 | 1044 |
| seed 1337 | -0.054 | 0.073 | 1114 |
| seed 2024 | -0.149 | 4.1e-07 | 1152 |

Pooled, there is no detectable linear relationship. Per seed, the correlation is
weak *and changes sign*, with two seeds significant in opposite directions. A
single-seed study would have reported a confident positive or negative result;
both would have been artifacts. This is the central evidence for the claim, and
it is stronger than a consistently weak correlation would have been.

Among edges above the same overlap threshold, both contribution tails are
populated in every seed, so structural overlap does not identify a homogeneous
dispensable set.

### Controls: the published headline does not reproduce from its own code

Four attribution controls were run, because several things changed at once here
and a drop could not otherwise be assigned to any of them.

**1. Baselines.** Dense and sliding reproduce the original closely, which
establishes that the task is learnable and that the data, splits, training and
evaluation are sound.

**2. Legacy gate semantics** (`--intervention_mode legacy_rowpair`) gives 4.4%
over three seeds, against 4.3% for the repaired gate. The change in intervention
semantics is not what moved the number.

**3. The original implementation, unmodified.** Commit 7474c77 was cloned and
run with a single one-line change: the dataset id, because newer
`huggingface_hub` rejects the bare `wikitext` repo name and the script cannot
otherwise start. `git diff --stat` over the source is that one line.

| variant | published | reproduced here | |
|---|---|---|---|
| Dense Transformer | 50.9% | 55.6% | reproduces |
| Sliding Window | 51.9% | 52.5% | reproduces |
| CRPA no reg. | 8.4% | 10.9% | reproduces |
| CRPA naive reg. | 5.3% | 5.3% | reproduces exactly |
| **CRPA causal reg.** | **32.8%** | **7.5%** | **does not reproduce** |

Four of five rows come back. The one that does not is the one the paper's claim
rests on, and it misses by a factor of 4.4, landing barely above the 5.0% chance
rate and **below its own no-regularization baseline**. The original's own
verification code says so, in its own words:

```
S3 Naive reduces overlap most: 0.256 < 0.237  -> FAILED
S5 Causal beats no-reg:        7.5% > 10.9%   -> FAILED
```

So the drop reported in this work is not caused by anything changed here. The
published number is not reproducible from the commit that published it, and F1
and F2 explain why it was never measuring dispensability in the first place: the
gate selected on a statistic substantially composed of no-op interventions.

**Caveat.** This ran on an L40S with torch 2.4.1+cu124, not the RTX A6000 with
torch 2.10.0+cu124 named in the original README, and single-seed as published.
Environment differences cannot be excluded in principle. They are an unlikely
explanation, though, since the other four rows reproduce on the same stack.

A further observation from that run, which supports F8: the *sliding-window*
baseline is the slowest configuration at every context length (23.5 ms at 512
against 9.7 ms for dense), because its mask was built with a Python loop over
tokens. The original runtime table was dominated by mask-construction cost
rather than by attention cost, which is why its sub-quadratic claim cannot be
read as a statement about attention.

### The matched-overlap sweep

6 lambda values x 2 methods x 3 seeds = 36 runs. 15 pairs matched on realized
overlap within a tolerance of 0.01; the matching is tight, mean absolute
difference 0.0020.

| quantity | result |
|---|---|
| retrieval delta, contribution minus naive | **+0.00 pp**, sd 2.05, range -2.5 to +7.1 |
| held-out eval loss delta | +0.018, sd 0.030 |
| pairs where either method beat chance | 6 of 15 |

At matched realized overlap the two selection criteria are indistinguishable.

**But the premise of this experiment does not hold.** It assumes regularization
strength dials realized overlap, so two methods can be compared at an equal
structural budget. It does not:

| variant | spearman(lambda, overlap) | overlap span from lambda | overlap span from seed |
|---|---|---|---|
| `crpa_naive` | -0.135 | 0.061 | 0.092 |
| `crpa_contribution` | -0.292 | 0.046 | 0.051 |

Sweeping lambda across its full range moves overlap less than changing the seed
does. Mean overlap by lambda for `crpa_naive`, from 0.00 to 0.20, runs 0.2434,
0.2526, 0.2448, 0.2586, 0.2485, 0.2367: non-monotone, spanning 0.022.

So the matched pairs are matched on run-to-run variation, not on a controlled
budget. `lambda_controls_overlap()` measures this and the run prints a warning
above the pair table. This also explains the original framing: it compared
naive at overlap 0.251 against causal at 0.243 and read them as a similar
budget, when overlap was never being steered and both were noise around 0.24.

### Two results that look contradictory, and are not

Selecting *which* edges to remove by measured contribution helps, at a matched
removal budget, within a single trained model: 3 seeds of 3, mean damage about
35x lower than overlap-ranked selection.

Gating *training* on the same criterion does nothing: the trained models are
indistinguishable, on retrieval and on loss, at matched overlap.

These are different experiments and both are reported. The reconciliation is in
the two sections above. During training the estimator's ranking is unstable, so
the gate is selecting close to at random; and lambda does not steer overlap, so
the penalty is not moving the structural budget it is supposed to move. A
criterion can identify better edges to remove while being useless as a training
signal, and here it is both.

Note also that a *noisy* estimate of contribution still beats overlap for
selection, because the correlation between overlap and contribution is
approximately zero. A weak signal outperforms no signal.

### The contribution estimator does not produce a stable ranking

Replicate agreement at seed 42, 24 candidates, 3 replicates per budget:

| sample budget | Spearman between replicates | top-8 agreement |
|---|---|---|
| 2 | -0.087 | 0.38 |
| 4 | +0.037 | 0.33 |
| 8 | -0.114 | 0.21 |
| 16 | -0.107 | 0.33 |
| 32 | +0.121 | 0.42 |

The ranking is indistinguishable from chance (top-8 of 24 is 0.33 by chance) and
does not improve from 2 samples to 32. Since the gate ranks by this estimate,
contribution gating is selecting close to at random here, which is the
mechanism behind its being indistinguishable from naive and from no
regularization.

Classification agreement is 1.00 at every budget. That is not stability. The
default `eps = 0.03` sits about four orders of magnitude above the observed
delta scale of ~1e-6, so every edge classifies as suppressible and the
threshold separates nothing. `eps_calibration()` now flags this, and it
compounds with F2: no-op interventions produced delta exactly 0, which a
threshold this large would admit regardless.

### Tier 2: cost at 4k, 8k and 16k

137.8M parameters, RoPE, bf16, batch 1, gather-based sparse attention, on one
L40S. Latency is the median over 10 iterations after 5 warmups, timed with CUDA
events. Memory is measured, not projected.

| variant | 4k | 8k | 16k | peak memory at 16k |
|---|---|---|---|---|
| dense (SDPA) | **13.5 ms** | **30.8 ms** | **74.8 ms** | 2050 MB |
| sliding window | 19.8 ms | 64.0 ms | 222.9 ms | 6921 MB |
| CRPA, gather-based | 31.6 ms | 60.5 ms | 120.2 ms | 2067 MB |

Two results, both negative for the efficiency framing:

**Dense is faster at every length tested.** CRPA scales better with context
(3.8x from 4k to 16k against dense's 5.5x) but starts from a constant factor
roughly 2.3x worse and does not catch up within this range. A dense baseline
using a fused attention kernel is a hard thing to beat by being sparse, and at
these lengths CRPA does not.

**CRPA's memory is indistinguishable from dense**, within 1% at every length.
That follows from the same fact: a fused dense kernel is already O(T) in
memory, so sparsity buys nothing against it. The original's comparison ran
against a dense baseline that materialised the full score matrix, which is what
made sparsity look like a memory win.

The sliding-window baseline is slowest and heaviest because it needs an
explicit (T, T) mask. That is an implementation property of this comparison,
not a claim about sliding-window attention in general.

This supersedes the original Table 3, which reported CRPA as sub-quadratic and
therefore cheaper. That table timed a masked-dense implementation, so it was
measuring mask construction rather than attention. Its own numbers show it: the
sliding-window row was the slowest configuration there too, because its mask
was built with a Python loop over tokens.

### What was not run

Stated explicitly so nothing here is read as a result.

| experiment | status | why |
|---|---|---|
| Tier 2 at 4k / 8k / 16k | see above | executed |
| Tier 2 at 32k and 64k | **not run** | needs an 80GB card; the code path exists and is smoke-tested |
| Tier 3 against a real 7B/8B model | **not run** | out of scope for the compute budget agreed for this work |
| Tier 3 tiny-model instrumentation | smoke-tested | `hf-internal-testing/tiny-random-LlamaForCausalLM`, instrumentation verification passes |

The Tier 3 tiny model has random weights, so it produces no meaningful finding
about overlap and contribution. `fig3_large_model` therefore renders nothing
and prints the command that would produce real data, rather than plotting
points from a random-weight model as though they meant something.

No result file anywhere in this repository carries metrics for a run whose
status is not `completed` or `smoke`; `numeric_records()` is the only accessor
aggregation and plotting use.

---

## 11. Results layout

```
results/
  tier1/
    runs/<run_id>.json        one record per (variant, seed)
    aggregate.json            mean, std, bootstrap 95% CI per variant
    aggregate.csv
    runs.csv
    overlap_vs_contribution.csv    one row per candidate edge
  matched_overlap/
    runs/<run_id>.json
    sweep.csv                 realized overlap, retrieval, loss, budget, seed, lambda
    matched_pairs.csv         pairs within tolerance, with retrieval delta
  estimator_stability/
    stability.csv             spearman, top-k agreement, classification agreement
  tier2/
    long_context.csv          per context length, includes status
    benchmark.csv             latency, throughput, memory, includes status
    kv_cache.csv              measured or projected, labelled
  tier3/
    edges_<model>.csv         one row per intervened edge
    diagnostic_<model>.json   correlations, groups, matched-budget comparison
  figures/
    figN_*.png
    figN_*_data.csv           source data for every figure
```

Every record carries git SHA, timestamp, hostname, GPU, CUDA and PyTorch
versions, dtype, full config, seed, context length, split provenance, and a
**status**:

| status | meaning |
|---|---|
| `completed` | ran to completion at the configured scale |
| `smoke` | ran at reduced scale to validate the code path |
| `not_run` | implemented, never executed |
| `oom` | attempted, ran out of memory. **No metrics.** |
| `unsupported` | the hardware or stack cannot run it |
| `failed` | attempted and raised |

Aggregation and plotting read results only through `numeric_records()`, which
returns `completed` and `smoke` records and nothing else. An OOM cannot become a
data point.

### Figures

```bash
python -m experiments.plot_all --results_dir results
python -m experiments.plot_all --results_dir results --only fig1 fig4
```

1. **Structural vs behavioral redundancy** overlap against intervention delta,
   with the high-overlap/low-delta and high-overlap/high-delta groups highlighted
2. **Matched overlap, different outcome** the sweep with matched pairs joined
3. **Large-model diagnostic** requires real Tier 3 data; renders nothing without it
4. **Seed robustness** three-seed Tier 1 with bootstrap intervals
5. **Context scaling** quality (5a) and cost (5b) as separate files
6. **Gate visualization** what each criterion removes, and what it costs

Every figure writes its source data as a CSV beside the PNG. Missing inputs
produce a SKIP message naming the command that would generate them, never a
placeholder plot.

---

## 12. Compute, and measured versus projected

| Workload | Requirement |
|---|---|
| Tests, Tier 1 smoke, Tier 3 tiny-model check | CPU, a few minutes |
| Tier 1 full (3 variants x 3 seeds x 4000 iters) | ~1.5 to 2 h on an A6000 |
| Matched-overlap sweep (6 lambdas x 2 methods x 3 seeds) | ~4 to 6 h on an A6000 |
| Tier 2 at 4k/8k/16k, 137.8M | 48GB, ~1.5 to 2 h |
| Tier 2 at 32k/64k | 80GB |
| Tier 3 at 7B/8B in bf16 | 40GB+ |

The contribution gate costs forward passes: one baseline plus one per candidate,
per refresh. `crpa_contribution` is roughly 4x slower per step than
`crpa_naive` at the default pool size, which is a real property of the method
and not an implementation defect.

### Measured versus projected

| Quantity | Kind |
|---|---|
| Latency, throughput | **Measured.** CUDA events with warmup and per-iteration synchronisation on GPU; synchronised wall clock on CPU. Median reported alongside mean. |
| Peak / allocated / reserved memory | **Measured** on CUDA. `NaN` on CPU, never 0. |
| Realized overlap, retrieval, delta distributions | **Measured.** |
| Actual CRPA edge count, sparsity ratio | **Measured** from the realised mask. |
| `crpa_edges_upper_bound` | **Analytical upper bound.** The three sources can overlap, so the realised count is at or below it. |
| `actual_edges` above 8k context | **Analytical upper bound**, flagged by `edge_count_is_upper_bound`. Counting exactly would mean materialising a `(T, T)` mask, which is 4.3 GB of booleans at 65536. |
| KV cache | **Analytical** by default; `measured` when the tensors are actually allocated. Every row is labelled. |

Neither figure is a measurement of decoding throughput. This repository has no
incremental decoding loop.

### CRPA does not bound the KV cache

Worth stating plainly, because it is the opposite of what the framing invites.
A token's routed set `C_k(i)` may reference *any* earlier position with a
different router assignment, so no earlier key/value pair can be proven
unnecessary and evicted. The cache grows linearly in `T`, exactly as the dense
baseline's does.

Only the local window and the relays are structurally evictable. That is
reported separately as `crpa_bounded`, a variant that drops cross-partition
routing, so the gap between what CRPA costs and what a bounded-cache variant
would cost stays visible instead of being conflated.

The measured forward-pass memory in section 10 makes the same point from the
other direction: CRPA's peak memory is within 1% of a dense baseline using a
fused kernel at every context length tested.

### Determinism

`set_seed` seeds Python, NumPy, torch CPU and all CUDA devices.
`set_seed(..., strict=True)` additionally requests deterministic CUDA
algorithms, which is slower and raises on ops lacking a deterministic kernel.
Even then, cuBLAS reductions need `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before
CUDA initialises, and atomics in some scatter/gather backward kernels remain
nondeterministic on certain GPU and driver combinations. CPU runs are
deterministic and that is what the test suite pins.

---

## 13. Routing: a secondary, largely negative result

Routing was originally presented as a contribution. It is not one, and it is
de-emphasized here rather than removed.

The published Table 5 reported identical values for all three conditions
(entropy 1.386, which is exactly `ln 4`, and load error 0.0000). Two things were
going on. The diagnostic itself was broken (F10), and the routers had collapsed
to near-uniform assignment, so there was little to measure. The corrected
diagnostic is available via `crpa.evaluate.routing_diagnostics` and reports
`max_entropy` alongside the measured entropy so collapse is visible rather than
implied.

The honest summary: the differentiable router did not demonstrably help, and the
load-balance loss did not demonstrably change utilisation. We report it because
a negative result that was previously presented as a positive one should be
correctable from the repository.

---

## 14. Limitations

- **Three seeds is few.** Bootstrap intervals over three points are wide. They
  are reported rather than hidden, but they do not support fine distinctions.
- **The retrieval task is synthetic and narrow.** Twenty possible values, chance
  5%, one needle depth band. It probes relay-mediated cross-partition retrieval
  specifically, and generalises to nothing on its own.
- **Language modelling is barely trained.** With `ret_ratio = 0.90`, 90% of
  steps are the retrieval task. Perplexities in the hundreds reflect that. They
  are a control, not a language-modelling result.
- **Single-edge ablation is a narrow intervention.** It does not account for
  interactions between edges, and summing individual deltas (as Tier 3's
  matched-budget comparison does across layers) is an additive approximation
  that is labelled as such.
- **Weak correlation is not independence.** Where overlap correlates weakly with
  contribution, that means overlap is a weak predictor on that sample, for that
  model. The sample is finite and the estimator is noisy. Correlations ship with
  p-values, confidence intervals and `n` so the reader can judge.
- **Reachability filtering changes the sampled population.** Restricting to
  edges that can influence the loss is necessary (see section 9), but it means
  the reported distribution is over reachable edges, not all edges.
- **The two meanings of "partition"** (positional window versus router
  assignment) are inherited and not resolved.
- **No decoding loop**, so no KV-cache or throughput claim about generation.
- **Long-context overlap is measured on a window**, not the whole sequence, and
  excludes relay rows. Both are stated in the results (`probs_window`), but they
  mean the long-context overlap figure is a windowed estimate rather than a
  full-sequence one.

---

## 15. Backwards compatibility

- `python main.py` works with all original flags.
- `from model import GPT; GPT('crpa_causal', 512, 50257, 'cpu')` works.
- `from config import CFG` works; the dict is unchanged.
- `from data import init_data, get_lm_batch, make_needle_batch` works.
- `from train import train, estimate_loss` works.
- `from evaluate import retrieval_accuracy, measure_overlap, ...` works.
- Checkpoints saved by the original load into the new model with
  `strict=True`; the state-dict keys are identical.

Two behaviours deliberately differ. `--figures_only` now does something, and the
shims report on the held-out evaluation split where the original reported on the
split it also calibrated against.

---

## 16. Repository layout

```
crpa/                 the library
  config.py           frozen dataclasses, YAML profiles, legacy bridge
  attention.py        CRPA structure, both attention paths, RoPE, router
  model.py            backbone, per-layer/per-head intervention, frozen structure
  intervention.py     edges, plans, contribution measurement, gate selection
  data.py             three-way split protocol, needle generator
  metrics.py          overlap statistics, bootstrap CI, correlations
  evaluate.py         split-aware evaluation
  kvcache.py          cache accounting, measured vs projected
  figures.py          the six figures
  seeding.py          all RNGs, scoped seeding
  runmeta.py          run ids, provenance, status, atomic writes
experiments/          one module per question, all with --smoke and --dry_run
configs/              small_12m, medium_138m, large_diagnostic
tests/                pytest suite
config.py model.py data.py train.py evaluate.py main.py
                      backwards-compatibility shims over crpa/*
```

Long jobs are resumable: run ids are `sha256(config + seed)[:12]`, completed
runs are skipped unless `--force`, and results are written atomically via a
temporary file plus `os.replace`.

---

## Citation

The original repository described itself as a reproduction of a NeurIPS 2026
paper. We have not verified that the paper exists, and this README makes no
claim about it. What is reproducible is what is in this repository, at the
scales the results files record.
