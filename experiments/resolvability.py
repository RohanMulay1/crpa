"""
Check 0 applied to Tier 1's headline claim.

    python -m experiments.resolvability --seeds 42 --loss lm
    python -m experiments.resolvability --smoke

Tier 1 reports that structural overlap is a weak predictor of behavioural
contribution. That claim is only interpretable if the contribution measurement
is reliable, and this repository's own artifacts suggest it may not be: deltas
sit a handful of float32 ULPs above zero and the epsilon threshold is four
orders of magnitude above them.

This experiment measures each candidate edge's delta **twice**, on two
disjoint halves of the evaluation data, and correlates the two estimates. The
same split-half treatment is applied to the overlap statistic. Together they
give the ceiling that unreliability puts on any observable correlation.

The decision rule is ported unchanged from the xsa-controls project so both
apply the same pre-registered standard. See ``crpa/resolvability.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from crpa.config import ExperimentConfig
from crpa.data import CALIBRATION, EVAL
from crpa.intervention import (
    Candidate,
    InterventionError,
    make_lm_loss_fn,
    make_needle_loss_fn,
    measure_delta,
)
from crpa.model import GPT
from crpa.resolvability import ResolvabilityResult, assess, pooled_verdict, spearman
from crpa.runmeta import numeric_records, write_csv
from crpa.seeding import set_seed
from crpa.train import train
from experiments.common import (
    add_common_args,
    config_from_args,
    load_corpus,
    print_header,
    record_run,
    resolve_device,
    results_dir_for,
    status_for,
)
from experiments.overlap_vs_contribution import collect_candidates

EXPERIMENT = "resolvability"


def _loss_fn_for(corpus, cfg: ExperimentConfig, split: str, kind: str,
                 batch_size: int, device: str, offset: int = 0):
    """Build a fixed-batch loss function on a disjoint slice of a split.

    ``offset`` selects a different, non-overlapping set of sequences, which is
    what makes the two halves independent. Both halves come from the same
    split so the comparison is like for like.
    """
    block = cfg.model.block_size
    if kind == "lm":
        x, y = corpus.lm_batch(split, block, batch_size * (offset + 1), device)
        lo = offset * batch_size
        x, y = x[lo:lo + batch_size], y[lo:lo + batch_size]
        return make_lm_loss_fn(x, y), x.shape[0]
    x, y = corpus.needle_batch(split, block, batch_size * (offset + 1),
                               device=device)
    lo = offset * batch_size
    x, y = x[lo:lo + batch_size], y[lo:lo + batch_size]
    return make_needle_loss_fn(x, y), x.shape[0]


@torch.no_grad()
def measure_on(model: GPT, candidates: Sequence[Candidate], loss_fn
               ) -> List[float]:
    """Delta for every candidate under one fixed evaluation batch."""
    out: List[float] = []
    baseline: Optional[float] = None
    for cand in candidates:
        try:
            base, intervened, removed = measure_delta(
                model, [cand.to_edge()], loss_fn, baseline=baseline,
                strict=False)
            baseline = base
            out.append(intervened - base if removed else float("nan"))
        except InterventionError:
            out.append(float("nan"))
    return out


@torch.no_grad()
def overlap_on(model: GPT, candidates: Sequence[Candidate],
               cfg: ExperimentConfig, x: torch.Tensor) -> List[float]:
    """Recompute each candidate's support overlap under a given batch.

    Overlap is a function of the attention pattern, which is a function of the
    data, so it has its own split-half reliability. Treating it as a fixed
    structural constant would overstate how much of the observed correlation
    the statistic could possibly carry.

    Uses the same ``edge_structural_overlap`` the sampler uses, with relays
    excluded identically, so the recomputed value is comparable to the
    originally reported one.
    """
    from crpa.attention import relay_positions
    from crpa.metrics import edge_structural_overlap, top_p_support_mask

    model.eval()
    relays = relay_positions(cfg.model.block_size, cfg.model.n_relays)
    with model.frozen_structure():
        with model.capture_probabilities(True):
            model(x)
        probs = model.attention_probabilities()

        supports: Dict[tuple, torch.Tensor] = {}
        out: List[float] = []
        for cand in candidates:
            layer_probs = probs[cand.layer] if cand.layer < len(probs) else None
            if layer_probs is None:
                out.append(float("nan"))
                continue
            key = (cand.layer, cand.head)
            if key not in supports:
                A = (layer_probs[:, cand.head].mean(dim=0)
                     if cand.head is not None
                     else layer_probs.mean(dim=(0, 1)))
                supports[key] = top_p_support_mask(A, cfg.model.overlap_rho)
            try:
                out.append(edge_structural_overlap(
                    supports[key], cand.query, cand.key,
                    cfg.model.partition_size, exclude=relays))
            except Exception:
                out.append(float("nan"))
    return out


def budget_sweep(model: GPT, candidates: Sequence[Candidate], corpus,
                 cfg: ExperimentConfig, kind: str, device: str,
                 budgets: Sequence[int] = (2, 4, 8, 16),
                 n_replicates: int = 2) -> Dict[int, float]:
    """Does the delta estimate converge as evaluation data grows?

    A flat curve near zero means more data does not help, which is the
    signature of measuring a quantisation floor rather than an effect.
    """
    curve: Dict[int, float] = {}
    subset = list(candidates[:min(len(candidates), 120)])
    for b in budgets:
        reps: List[List[float]] = []
        for r in range(n_replicates):
            fn, _ = _loss_fn_for(corpus, cfg, EVAL, kind, b, device, offset=r)
            reps.append(measure_on(model, subset, fn))
        curve[b] = spearman(reps[0], reps[1]) if len(reps) == 2 else float("nan")
    return curve


def run_seed(seed: int, cfg: ExperimentConfig, corpus, device: str,
             kind: str, n_per_layer: int, batch_size: int,
             do_sweep: bool, train_iters: Optional[int]) -> tuple:
    set_seed(seed)
    run_cfg = cfg.replace(**{"train.seed": seed})
    if train_iters is not None:
        run_cfg = run_cfg.replace(**{"train.max_iters": train_iters})
    model = GPT(run_cfg.model, cfg.variant, seed=seed).to(device)
    print("  training ({} iters) ...".format(run_cfg.train.max_iters), flush=True)
    train(model, run_cfg, corpus, device, verbose=False)
    model.eval()

    print("  enumerating candidate edges ...", flush=True)
    candidates = collect_candidates(model, run_cfg, corpus, device, seed,
                                    n_per_layer=n_per_layer,
                                    batch_size=batch_size, loss=kind)
    if not candidates:
        raise RuntimeError("no candidate edges were scored for seed {}".format(seed))
    print("  {} candidates".format(len(candidates)), flush=True)

    block = run_cfg.model.block_size
    # Two disjoint evaluation batches. Same split, different sequences.
    fn_a, n_a = _loss_fn_for(corpus, run_cfg, EVAL, kind, batch_size, device, 0)
    fn_b, n_b = _loss_fn_for(corpus, run_cfg, EVAL, kind, batch_size, device, 1)

    print("  measuring delta on half A ({} seqs) ...".format(n_a), flush=True)
    delta_a = measure_on(model, candidates, fn_a)
    print("  measuring delta on half B ({} seqs) ...".format(n_b), flush=True)
    delta_b = measure_on(model, candidates, fn_b)

    if kind == "lm":
        xa, _ = corpus.lm_batch(EVAL, block, batch_size, device)
        xb, _ = corpus.lm_batch(EVAL, block, batch_size * 2, device)
        xb = xb[batch_size:]
    else:
        xa, _ = corpus.needle_batch(EVAL, block, batch_size, device=device)
        xb, _ = corpus.needle_batch(EVAL, block, batch_size, device=device)
    print("  recomputing overlap on both halves ...", flush=True)
    stat_a = overlap_on(model, candidates, run_cfg, xa)
    stat_b = overlap_on(model, candidates, run_cfg, xb)

    curve: Dict[int, float] = {}
    if do_sweep:
        print("  budget sweep ...", flush=True)
        curve = budget_sweep(model, candidates, corpus, run_cfg, kind, device)

    baseline = float(np.nanmedian(
        [c.baseline_loss for c in candidates
         if isinstance(c.baseline_loss, float)])) if candidates else float("nan")

    res = assess(delta_a, delta_b, stat_a, stat_b, seed,
                 baseline_loss=baseline, budget_curve=curve)
    rows = []
    for i, c in enumerate(candidates):
        rows.append({
            "seed": seed, "layer": c.layer,
            "head": c.head if c.head is not None else -1,
            "query": c.query, "key": c.key,
            "delta_half_a": delta_a[i], "delta_half_b": delta_b[i],
            "overlap_half_a": stat_a[i], "overlap_half_b": stat_b[i],
            "overlap_original": c.overlap,
            "delta_original": c.delta_loss,
            "baseline_loss": c.baseline_loss,
        })
    return res, rows, run_cfg


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--variant", default="crpa_contribution")
    parser.add_argument("--loss", default="lm", choices=["needle", "lm"],
                        help="'lm' is the default: the needle task is at its "
                             "trivial floor, so a retrieval delta perturbs a "
                             "predictor that never learned the task")
    parser.add_argument("--n_per_layer", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sweep", action="store_true",
                        help="also run the evaluation-budget convergence sweep")
    parser.add_argument("--train_iters", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = config_from_args(args).replace(variant=args.variant)
    device = resolve_device(args.device)
    results_dir = results_dir_for(args, "resolvability")
    seeds = args.seeds if args.seeds else list(cfg.multi_seeds)
    status = status_for(args, cfg)

    print_header("Check 0 - is the contribution measurement resolvable?")
    print("device={}  seeds={}  loss={}  variant={}".format(
        device, seeds, args.loss, cfg.variant))
    if args.dry_run:
        print("\nWould assess resolvability for seeds {}".format(seeds))
        return 0

    base_corpus = load_corpus(cfg, seed=seeds[0], synthetic=args.synthetic_data)
    results: List[ResolvabilityResult] = []
    all_rows: List[Dict[str, object]] = []

    for seed in seeds:
        print("\n--- seed {} ---".format(seed), flush=True)
        corpus = base_corpus.reseed(seed)
        run_cfg = cfg.replace(**{"train.seed": seed})
        with record_run(results_dir, EXPERIMENT, run_cfg, seed, status,
                        splits=corpus.split_metadata(
                            min(cfg.model.block_size, 256))) as rec:
            res, rows, used_cfg = run_seed(
                seed, cfg, corpus, device, args.loss, args.n_per_layer,
                args.batch_size, args.sweep, args.train_iters)
            results.append(res)
            all_rows.extend(rows)
            rec.metrics.update(res.to_dict())
            rec.metrics["loss"] = args.loss
        print(res.summary(), flush=True)

    pooled = pooled_verdict(results)
    payload = {
        "experiment": EXPERIMENT,
        "loss": args.loss,
        "per_seed": [r.to_dict() for r in results],
        "pooled": pooled,
        "decision_rule": [
            {"min_r_delta": t, "verdict": n, "action": a}
            for t, n, a in __import__(
                "crpa.resolvability", fromlist=["x"]).RESOLVABILITY_RULE],
        "provenance": "decision rule ported unchanged from xsa-controls "
                      "xsac/checks.py",
    }
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolvability.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(out / "split_half.csv", all_rows)

    print("\n" + "=" * 70)
    print("POOLED VERDICT: {}".format(pooled["verdict"].upper()))
    print("  best r_delta across seeds : {:+.3f}".format(
        pooled.get("r_delta_best", float("nan"))))
    print("  ceiling on any observable r: {:.3f}".format(
        pooled.get("max_observable_correlation", float("nan"))))
    print("  claim permitted           : {}".format(pooled["claim_permitted"]))
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
