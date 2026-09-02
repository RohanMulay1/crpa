"""
Tier 1, experiment 3 - structural overlap versus behavioral contribution.

This is the experiment the central claim rests on. For a trained model we
enumerate candidate edges, record their structural overlap, intervene on each
one individually, and record the resulting change in loss. The output is a
row-per-edge dataset:

    layer, head, query, key, overlap, baseline_loss, intervened_loss,
    delta_loss, retrieval_effect, suppressible, seed, context_length

and the correlation between overlap and delta.

Reading the correlation honestly
--------------------------------
A weak correlation means structural overlap is a weak predictor of behavioral
contribution *on this sample, for this model*. It is not proof of
independence: the sample is finite, the estimator is noisy, and a monotone
relationship could exist in a region we did not sample. The results record the
correlation, its p-value, a confidence interval and the sample size so a reader
can judge that for themselves.

Experiment 4 (high-overlap groups) follows directly: among edges of comparable
overlap, do low- and high-contribution subsets behave differently when removed?

    python -m experiments.overlap_vs_contribution --smoke
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List

import numpy as np
import torch

from crpa.config import ExperimentConfig
from crpa.data import CALIBRATION, EVAL
from crpa.evaluate import retrieval_accuracy
from crpa.intervention import (
    Candidate,
    InterventionPlan,
    make_lm_loss_fn,
    make_needle_loss_fn,
    reachable_queries,
    sample_candidate_edges,
    sample_legacy_row_pairs,
    score_candidates,
    select_contribution_gated,
    eps_calibration,
    select_naive,
    split_high_overlap_groups,
)
from crpa.metrics import correlations
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

EXPERIMENT = "overlap_vs_contribution"


def collect_candidates(
    model: GPT,
    cfg: ExperimentConfig,
    corpus,
    device: str,
    seed: int,
    n_per_layer: int = 24,
    batch_size: int = 8,
    loss: str = "needle",
) -> List[Candidate]:
    """Enumerate and score candidate edges on the calibration split.

    Args:
        loss: which behaviour to measure contribution against.

            ``needle`` last-token retrieval cross-entropy. This is what the
                gate uses, but it is only informative if the model has
                actually learned retrieval; against a model at chance it
                perturbs a near-random predictor.
            ``lm`` mean next-token cross-entropy over all positions. Always
                informative, and it makes every edge reachable by
                construction, so no reachability filtering is applied.
    """
    block = cfg.model.block_size
    if loss == "lm":
        x, y = corpus.lm_batch(CALIBRATION, block, batch_size, device)
        loss_fn = make_lm_loss_fn(x, y)
    elif loss == "needle":
        x, y = corpus.needle_batch(CALIBRATION, block, batch_size, device=device)
        loss_fn = make_needle_loss_fn(x, y)
    else:
        raise ValueError("loss must be 'needle' or 'lm', got {!r}".format(loss))
    rng = np.random.default_rng(seed + 4242)
    legacy = cfg.contribution.mode == "legacy_rowpair"

    model.eval()
    with model.frozen_structure():
        with model.capture_probabilities(True):
            model(x)
        probs = model.attention_probabilities()
        # Under an all-positions loss every edge can matter, so no
        # reachability filter is needed or appropriate.
        reach = None if (legacy or loss == "lm") else reachable_queries(model, block)
        relays = relay_positions(block, cfg.model.n_relays)

        pool: List[Candidate] = []
        for depth in range(len(model.blocks)):
            if probs[depth] is None:
                continue
            if legacy:
                pool += sample_legacy_row_pairs(
                    probs[depth], depth, cfg.model.partition_size,
                    cfg.model.overlap_rho, n_per_layer, rng,
                    min_overlap=cfg.contribution.overlap_threshold,
                )
            else:
                pool += sample_candidate_edges(
                    probs[depth], depth, cfg.model.partition_size,
                    cfg.model.overlap_rho, n_per_layer, rng,
                    min_overlap=0.0,          # keep the full overlap range
                    reach=reach[depth] if reach else None,
                    exclude_queries=relays,
                )
        scored = score_candidates(
            model, pool, loss_fn, eps=cfg.contribution.eps,
            seed=seed, context_length=block,
        )
    return scored


def measure_group_effect(
    model: GPT,
    corpus,
    cfg: ExperimentConfig,
    device: str,
    group: List[Candidate],
    baseline_retrieval: float,
    n_batches: int = 12,
) -> Dict[str, float]:
    """Remove a whole group at once and measure the damage.

    Both retrieval and language-model loss are measured on the **evaluation**
    split; the groups were formed on calibration data, so this is genuinely
    held out.

    Measuring both matters. The groups are defined by delta quantiles, so
    "the groups differ in delta" is true by construction and is not evidence.
    The non-tautological question is whether removing them damages *held-out
    behaviour* differently, and the retrieval version of that test has no power
    when the model sits at chance. The language-model version always does.
    """
    if not group:
        return {
            "n_edges": 0,
            "retrieval_after": float("nan"),
            "retrieval_drop": float("nan"),
            "lm_loss_after": float("nan"),
            "lm_loss_increase": float("nan"),
            "mean_individual_delta": float("nan"),
            "mean_overlap": float("nan"),
        }

    plan = InterventionPlan.of([c.to_edge() for c in group])
    block = cfg.model.block_size

    model.eval()
    correct = total = 0
    with model.frozen_structure():
        corpus.needles[EVAL].reset()
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = corpus.needle_batch(EVAL, block, 8, device=device)
                logits, _ = model(x, plan=plan)
                correct += int((logits[:, -1, :].argmax(dim=-1) == y).sum().item())
                total += int(y.shape[0])
        removed = model.intervened_count()

        # Held-out language-model loss, with and without the group removed.
        base_losses, cut_losses = [], []
        with torch.no_grad():
            for _ in range(n_batches):
                xb, yb = corpus.lm_batch(EVAL, block, 4, device)
                base_losses.append(float(model(xb, yb)[1].item()))
                cut_losses.append(float(model(xb, yb, plan=plan)[1].item()))

    after = 100.0 * correct / max(total, 1)
    lm_base = float(np.mean(base_losses)) if base_losses else float("nan")
    lm_cut = float(np.mean(cut_losses)) if cut_losses else float("nan")
    return {
        "n_edges": len(group),
        "n_score_positions_removed": removed,
        "retrieval_after": after,
        "retrieval_drop": baseline_retrieval - after,
        "lm_loss_baseline": lm_base,
        "lm_loss_after": lm_cut,
        "lm_loss_increase": lm_cut - lm_base,
        "mean_individual_delta": float(np.mean([c.delta_loss for c in group])),
        "mean_overlap": float(np.mean([c.overlap for c in group])),
    }


def analyse(candidates: List[Candidate], cfg: ExperimentConfig) -> Dict[str, object]:
    """Correlations plus the high-overlap group split."""
    usable = [c for c in candidates if math.isfinite(c.delta_loss)]
    overlaps = [c.overlap for c in usable]
    deltas = [c.delta_loss for c in usable]

    groups = split_high_overlap_groups(
        usable,
        cfg.contribution.high_overlap_q,
        cfg.contribution.low_contribution_q,
        cfg.contribution.high_contribution_q,
    )
    return {
        "n_candidates": len(candidates),
        "n_scored": len(usable),
        "correlation": correlations(overlaps, deltas),
        "eps_calibration": eps_calibration(usable, cfg.contribution.eps),
        "overlap_summary": {
            "min": float(np.min(overlaps)) if overlaps else float("nan"),
            "max": float(np.max(overlaps)) if overlaps else float("nan"),
            "mean": float(np.mean(overlaps)) if overlaps else float("nan"),
        },
        "delta_summary": {
            "min": float(np.min(deltas)) if deltas else float("nan"),
            "max": float(np.max(deltas)) if deltas else float("nan"),
            "mean": float(np.mean(deltas)) if deltas else float("nan"),
        },
        "group_thresholds": groups["thresholds"],
        "n_high_overlap_low_contribution": len(groups["high_overlap_low_contribution"]),
        "n_high_overlap_high_contribution": len(groups["high_overlap_high_contribution"]),
        "interpretation_note": (
            "A weak correlation indicates overlap is a weak predictor of "
            "contribution on this sample. It is not evidence of independence."
        ),
    }, groups


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--variant", default="crpa_contribution")
    parser.add_argument("--n_per_layer", type=int, default=24,
                        help="candidate edges sampled per layer per head")
    parser.add_argument("--loss", default="needle", choices=["needle", "lm"],
                        help="behaviour to measure contribution against; 'lm' is "
                             "the honest choice when the model has not learned "
                             "the retrieval task")
    parser.add_argument("--matched_budget", type=int, default=None,
                        help="removal budget for the naive-vs-contribution "
                             "comparison at a matched number of removals")
    args = parser.parse_args(argv)

    cfg = config_from_args(args).replace(variant=args.variant)
    device = resolve_device(args.device)
    results_dir = results_dir_for(args, "tier1")
    seeds = args.seeds if args.seeds else list(cfg.multi_seeds)
    status = status_for(args)

    print_header("Tier 1 - structural overlap vs behavioral contribution")
    print("device={}  seeds={}  variant={}  mode={}".format(
        device, seeds, cfg.variant, cfg.contribution.mode))

    if args.dry_run:
        print("\nWould score candidate edges for seeds {}".format(seeds))
        return 0

    # Loaded once; only the needle streams depend on the seed.
    base_corpus = load_corpus(cfg, seed=seeds[0], synthetic=args.synthetic_data)

    all_rows: List[Dict[str, object]] = []
    per_seed: Dict[int, Dict[str, object]] = {}

    for seed in seeds:
        run_cfg = cfg.replace(**{"train.seed": seed})
        print("\n--- seed {} ---".format(seed))
        corpus = base_corpus.reseed(seed)

        with record_run(results_dir, EXPERIMENT, run_cfg, seed, status,
                        splits=corpus.split_metadata(min(cfg.model.block_size, 256))) as rec:
            set_seed(seed)
            model = GPT(run_cfg.model, cfg.variant, seed=seed).to(device)
            train(model, run_cfg, corpus, device, verbose=False)

            candidates = collect_candidates(
                model, run_cfg, corpus, device, seed,
                n_per_layer=args.n_per_layer, loss=args.loss,
            )
            summary, groups = analyse(candidates, run_cfg)

            base_ret = retrieval_accuracy(
                model, corpus, run_cfg.model.block_size, device, n_batches=12
            )
            summary["baseline_retrieval"] = base_ret
            summary["contribution_loss"] = args.loss

            # Experiment 4: do structurally similar groups behave differently?
            summary["group_effects"] = {
                name: measure_group_effect(
                    model, corpus, run_cfg, device, groups[name], base_ret
                )
                for name in ("high_overlap_low_contribution",
                             "high_overlap_high_contribution")
            }

            # Matched-budget comparison of the two selection criteria.
            budget = args.matched_budget or max(1, len(candidates) // 8)
            summary["matched_budget"] = budget
            summary["selection_comparison"] = {
                "naive": measure_group_effect(
                    model, corpus, run_cfg, device,
                    select_naive(candidates, budget), base_ret
                ),
                "contribution_gated": measure_group_effect(
                    model, corpus, run_cfg, device,
                    select_contribution_gated(candidates, budget), base_ret
                ),
            }

            rec.metrics = summary
            per_seed[seed] = summary

        for cand in candidates:
            row = cand.to_row()
            row["variant"] = cfg.variant
            row["run_id"] = rec.run_id
            all_rows.append(row)

        corr = summary["correlation"]
        print("  scored {} edges | pearson r={:.3f} (p={:.3g}) | spearman r={:.3f}".format(
            summary["n_scored"], corr.get("pearson_r", float('nan')),
            corr.get("pearson_p", float('nan')), corr.get("spearman_r", float('nan'))))
        calib = summary.get("eps_calibration", {})
        if calib.get("vacuous"):
            print("  warning: {}".format(calib.get("note")))
        ge = summary["group_effects"]
        for name, label in (("high_overlap_low_contribution", "low-contribution "),
                            ("high_overlap_high_contribution", "high-contribution")):
            g = ge[name]
            # The LM-loss column is the one with power: retrieval is pinned at
            # chance, so its drop is 0.0pp for every group in every seed.
            print("  high-overlap/{}: n={:<5} mean delta={:+.2e}  "
                  "held-out LM loss {:+.3e}  retrieval drop={:.1f}pp".format(
                      label, g["n_edges"], g["mean_individual_delta"],
                      g.get("lm_loss_increase", float("nan")),
                      g["retrieval_drop"]))

    if all_rows:
        write_csv(results_dir / "overlap_vs_contribution.csv", all_rows)

        # Pooled across seeds, alongside the per-seed values. Reporting only a
        # pooled number would hide sign instability between seeds, and
        # reporting only one seed would overstate whatever that seed showed.
        pooled = correlations(
            [r["overlap"] for r in all_rows], [r["delta_loss"] for r in all_rows]
        )
        per_seed_r = {
            seed: summary["correlation"].get("pearson_r", float("nan"))
            for seed, summary in per_seed.items()
        }
        signs = {math.copysign(1.0, v) for v in per_seed_r.values()
                 if isinstance(v, float) and math.isfinite(v)}
        write_json(results_dir / "overlap_vs_contribution.json", {
            "experiment": EXPERIMENT,
            "per_seed": per_seed,
            "pooled_correlation": pooled,
            "per_seed_pearson_r": per_seed_r,
            "pearson_sign_consistent_across_seeds": len(signs) <= 1,
            "n_rows": len(all_rows),
            "interpretation_note": (
                "A weak correlation means overlap is a weak predictor on this "
                "sample. It is not evidence of independence. If the sign is not "
                "consistent across seeds, no directional claim is supported "
                "either."
            ),
        })
        if len(signs) > 1:
            print("\nNote: the correlation changes sign across seeds ({}), "
                  "so no directional claim is supported.".format(
                      {k: round(v, 3) for k, v in per_seed_r.items()}))
        print("\nWrote {} edge records to {}".format(
            len(all_rows), results_dir / "overlap_vs_contribution.csv"))
    else:
        raise SystemExit(
            "no candidate edges were scored; refusing to write an empty dataset"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
