"""
Tier 1, experiment 5 - how stable is the contribution estimator?

The gate ranks candidate edges by measured delta, so the question that matters
is not "is the estimator unbiased" but "does it produce the same *ranking*
twice". We answer it empirically.

For each sample budget n in {2, 4, 8, 16, 32} we draw ``n`` independent
calibration batches, average each candidate's delta over them, and repeat the
whole procedure ``--replicates`` times with disjoint batches. Comparing
replicates at the same budget measures estimator noise directly:

  * Spearman correlation between replicate rankings
  * top-k agreement between replicate rankings
  * agreement of the suppressible / not-suppressible classification

This is an empirical diagnostic. It is deliberately not accompanied by a
variance bound: a theoretical bound on an estimator's variance would not
establish that its induced *ranking* is stable, which is the property the gate
actually depends on.

    python -m experiments.estimator_stability --smoke
"""

from __future__ import annotations

import argparse
import itertools
import math
from typing import Dict, List, Sequence

import numpy as np
import torch

from crpa.config import ExperimentConfig
from crpa.data import CALIBRATION
from crpa.intervention import (
    Candidate,
    eps_calibration,
    InterventionPlan,
    reachable_queries,
    sample_candidate_edges,
)
from crpa.metrics import spearman, top_k_agreement
from crpa.attention import relay_positions
from crpa.model import GPT
from crpa.runmeta import write_csv, write_json
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

EXPERIMENT = "estimator_stability"
DEFAULT_BUDGETS = (2, 4, 8, 16, 32)


@torch.no_grad()
def estimate_deltas(
    model: GPT,
    candidates: Sequence[Candidate],
    corpus,
    cfg: ExperimentConfig,
    device: str,
    n_samples: int,
    batch_size: int = 8,
) -> List[float]:
    """Average each candidate's delta over ``n_samples`` calibration batches.

    Batches are drawn from the calibration stream and never rewound between
    replicates, so replicates genuinely see different data.
    """
    block = cfg.model.block_size
    totals = np.zeros(len(candidates))

    model.eval()
    with model.frozen_structure():
        for _ in range(n_samples):
            x, y = corpus.needle_batch(CALIBRATION, block, batch_size, device=device)
            logits, _ = model(x)
            base = float(
                torch.nn.functional.cross_entropy(logits[:, -1, :], y).item()
            )
            for idx, cand in enumerate(candidates):
                plan = InterventionPlan.single(cand.to_edge())
                logits_m, _ = model(x, plan=plan)
                totals[idx] += float(
                    torch.nn.functional.cross_entropy(logits_m[:, -1, :], y).item()
                ) - base
    return (totals / max(n_samples, 1)).tolist()


def ranking(deltas: Sequence[float]) -> List[int]:
    """Candidate indices ordered by ascending delta (most suppressible first)."""
    return sorted(range(len(deltas)), key=lambda i: deltas[i])


def compare_replicates(
    replicate_deltas: List[List[float]], eps: float, top_k: int
) -> Dict[str, float]:
    """Pairwise agreement statistics between replicate estimates."""
    n_rep = len(replicate_deltas)
    if n_rep < 2:
        return {"mean_spearman": float("nan"), "mean_top_k_agreement": float("nan"),
                "mean_classification_agreement": float("nan"), "n_replicates": n_rep}

    spearmans, topks, classes = [], [], []
    for a, b in itertools.combinations(range(n_rep), 2):
        da, db = replicate_deltas[a], replicate_deltas[b]
        spearmans.append(spearman(da, db))
        topks.append(top_k_agreement(ranking(da), ranking(db), top_k))
        label_a = [d <= eps for d in da]
        label_b = [d <= eps for d in db]
        classes.append(float(np.mean([x == y for x, y in zip(label_a, label_b)])))

    finite = [s for s in spearmans if math.isfinite(s)]
    return {
        "mean_spearman": float(np.mean(finite)) if finite else float("nan"),
        "std_spearman": float(np.std(finite)) if len(finite) > 1 else 0.0,
        "mean_top_k_agreement": float(np.mean(topks)),
        "mean_classification_agreement": float(np.mean(classes)),
        "n_replicates": n_rep,
        "n_comparisons": len(spearmans),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--budgets", type=int, nargs="*", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--n_candidates", type=int, default=24)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--variant", default="crpa_contribution")
    args = parser.parse_args(argv)

    cfg = config_from_args(args).replace(variant=args.variant)
    device = resolve_device(args.device)
    results_dir = results_dir_for(args, "estimator_stability")
    seeds = args.seeds if args.seeds else [list(cfg.multi_seeds)[0]]
    status = status_for(args)

    print_header("Tier 1 - contribution estimator stability")
    print("device={}  seeds={}  budgets={}  replicates={}".format(
        device, seeds, args.budgets, args.replicates))

    if args.dry_run:
        print("\nWould evaluate {} budgets x {} replicates for seeds {}".format(
            len(args.budgets), args.replicates, seeds))
        return 0

    # Loaded once; only the needle streams depend on the seed.
    base_corpus = load_corpus(cfg, seed=seeds[0], synthetic=args.synthetic_data)

    rows: List[Dict[str, object]] = []
    for seed in seeds:
        run_cfg = cfg.replace(**{"train.seed": seed})
        corpus = base_corpus.reseed(seed)

        with record_run(results_dir, EXPERIMENT, run_cfg, seed, status,
                        splits=corpus.split_metadata(min(cfg.model.block_size, 256))) as rec:
            set_seed(seed)
            model = GPT(run_cfg.model, cfg.variant, seed=seed).to(device)
            train(model, run_cfg, corpus, device, verbose=False)

            block = run_cfg.model.block_size
            x, _ = corpus.needle_batch(CALIBRATION, block, 8, device=device)
            model.eval()
            with model.frozen_structure():
                with model.capture_probabilities(True):
                    model(x)
                probs = model.attention_probabilities()
                reach = reachable_queries(model, block)
                relays = relay_positions(block, cfg.model.n_relays)
                rng = np.random.default_rng(seed + 909)
                candidates: List[Candidate] = []
                for depth in range(len(model.blocks)):
                    if probs[depth] is None:
                        continue
                    candidates += sample_candidate_edges(
                        probs[depth], depth, run_cfg.model.partition_size,
                        run_cfg.model.overlap_rho, args.n_candidates, rng,
                        reach=reach[depth], exclude_queries=relays,
                    )
                candidates = candidates[: args.n_candidates]

            if not candidates:
                raise SystemExit("no candidate edges available for stability analysis")

            print("  seed {}: {} candidates".format(seed, len(candidates)))
            seed_rows = []
            for budget in args.budgets:
                replicate_deltas = [
                    estimate_deltas(model, candidates, corpus, run_cfg, device, budget)
                    for _ in range(args.replicates)
                ]
                stats = compare_replicates(
                    replicate_deltas, run_cfg.contribution.eps, args.top_k
                )
                row = {"seed": seed, "sample_budget": budget,
                       "n_candidates": len(candidates), "top_k": args.top_k, **stats}
                seed_rows.append(row)
                rows.append(row)
                print("    budget={:>3}  spearman={:.3f}  top{}={:.2f}  "
                      "class agree={:.2f}".format(
                          budget, stats["mean_spearman"], args.top_k,
                          stats["mean_top_k_agreement"],
                          stats["mean_classification_agreement"]))
            # Report whether the suppressibility threshold means anything at
            # this delta scale. Perfect classification agreement is not
            # stability if the threshold admits every edge.
            probe = estimate_deltas(model, candidates, corpus, run_cfg, device,
                                    max(args.budgets))
            for cand, d in zip(candidates, probe):
                cand.delta_loss = d
            calib = eps_calibration(candidates, run_cfg.contribution.eps)
            if calib.get("vacuous"):
                print("    warning: {}".format(calib["note"]))
            rec.metrics = {"stability": seed_rows, "n_candidates": len(candidates),
                           "eps_calibration": calib}

    if rows:
        write_csv(results_dir / "stability.csv", rows)
        write_json(results_dir / "stability.json", {
            "experiment": EXPERIMENT,
            "rows": rows,
            "note": (
                "Empirical ranking stability. No theoretical variance bound is "
                "claimed; a bound on estimator variance would not establish "
                "ranking stability, which is what the gate depends on."
            ),
        })
        print("\nWrote {}".format(results_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
