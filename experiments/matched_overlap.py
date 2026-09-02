"""
Tier 1, experiment 2 - the matched-overlap sweep.

The question
------------
The committed results compare naive and contribution-gated suppression at
overlaps of 0.251 and 0.243. Those are *close*, not *matched*, and they were
produced at a single, identical regularization strength. A difference in
retrieval at two different structural budgets is not attributable to the
selection criterion.

This experiment sweeps regularization strength for both methods, records the
**realized** overlap each run actually achieves, and then pairs runs whose
realized overlaps agree to within a tolerance. Only then is the retrieval
difference attributable to *which* interactions were suppressed.

Matching is never done on the configured lambda. Two methods at the same lambda
land at different realized overlaps; that is the whole reason this experiment
exists.

    python -m experiments.matched_overlap --smoke
    python -m experiments.matched_overlap --profile small_12m \\
        --lambdas 0.0 0.01 0.02 0.05 0.10 0.20
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence

from crpa.config import ExperimentConfig
from crpa.evaluate import language_model_loss, measure_overlap, retrieval_accuracy
from crpa.metrics import perplexity
from crpa.model import GPT
from crpa.runmeta import load_record, numeric_records, write_csv, write_json
from crpa.seeding import set_seed
from crpa.train import train
from experiments.common import (
    add_common_args,
    config_from_args,
    describe_plan,
    load_corpus,
    make_run_id,
    print_header,
    record_run,
    resolve_device,
    results_dir_for,
    status_for,
)

EXPERIMENT = "matched_overlap"

DEFAULT_LAMBDAS = (0.00, 0.01, 0.02, 0.05, 0.10, 0.20)
SWEEP_METHODS = ("crpa_naive", "crpa_contribution")


def find_matched_overlap_pairs(
    runs: Sequence[Dict[str, object]],
    tolerance: float = 0.01,
    method_a: str = "crpa_naive",
    method_b: str = "crpa_contribution",
    within_seed: bool = True,
) -> List[Dict[str, object]]:
    """Pair runs of two methods whose *realized* overlap is within ``tolerance``.

    Each run of ``method_a`` is matched to its single nearest ``method_b``
    counterpart by realized overlap - nearest, not merely within tolerance, so
    a dense sweep cannot produce many spurious pairings. Pairs beyond the
    tolerance are discarded.

    Args:
        runs: dicts with at least ``variant``, ``seed``, ``realized_overlap``,
            ``retrieval_accuracy`` and ``lambda_red``.
        tolerance: maximum absolute difference in realized overlap.
        method_a / method_b: the two variants to pair.
        within_seed: only pair runs sharing a seed, so a pair differs in method
            and nothing else.

    Returns:
        One dict per matched pair, sorted by overlap difference ascending.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative, got {}".format(tolerance))

    def usable(r: Dict[str, object]) -> bool:
        ov = r.get("realized_overlap")
        return isinstance(ov, (int, float)) and math.isfinite(float(ov))

    a_runs = [r for r in runs if r.get("variant") == method_a and usable(r)]
    b_runs = [r for r in runs if r.get("variant") == method_b and usable(r)]

    pairs: List[Dict[str, object]] = []
    for a in a_runs:
        candidates = b_runs
        if within_seed:
            candidates = [b for b in b_runs if b.get("seed") == a.get("seed")]
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda b: abs(float(b["realized_overlap"]) - float(a["realized_overlap"])),
        )
        diff = abs(float(best["realized_overlap"]) - float(a["realized_overlap"]))
        if diff > tolerance:
            continue
        pairs.append({
            "seed": a.get("seed"),
            "overlap_tolerance": tolerance,
            "{}_overlap".format(method_a): float(a["realized_overlap"]),
            "{}_overlap".format(method_b): float(best["realized_overlap"]),
            "overlap_abs_diff": diff,
            "{}_retrieval".format(method_a): a.get("retrieval_accuracy"),
            "{}_retrieval".format(method_b): best.get("retrieval_accuracy"),
            "retrieval_delta": (
                float(best.get("retrieval_accuracy", float("nan")))
                - float(a.get("retrieval_accuracy", float("nan")))
            ),
            "{}_lambda_red".format(method_a): a.get("lambda_red"),
            "{}_lambda_red".format(method_b): best.get("lambda_red"),
            "{}_removal_budget".format(method_a): a.get("removal_budget"),
            "{}_removal_budget".format(method_b): best.get("removal_budget"),
            "{}_run_id".format(method_a): a.get("run_id"),
            "{}_run_id".format(method_b): best.get("run_id"),
        })
    pairs.sort(key=lambda p: p["overlap_abs_diff"])
    return pairs


def run_sweep_point(
    cfg: ExperimentConfig, variant: str, seed: int, lam: float,
    budget: int, device: str, corpus, verbose: bool = False,
) -> Dict[str, object]:
    """Train one (method, lambda, budget, seed) point and measure it."""
    run_cfg = cfg.replace(
        variant=variant,
        **{"train.seed": seed, "train.lambda_red": lam,
           "contribution.n_pairs": budget},
    )
    set_seed(seed)
    model = GPT(run_cfg.model, variant, seed=seed).to(device)
    history = train(model, run_cfg, corpus, device, verbose=verbose)

    block = run_cfg.model.block_size
    overlap = measure_overlap(model, corpus, run_cfg, device, n_batches=8, seed=seed)
    retrieval = retrieval_accuracy(model, corpus, block, device, n_batches=30)
    eval_loss = language_model_loss(model, corpus, block, device, n_batches=10)
    gate = history["gate_hist"][-1] if history["gate_hist"] else {}

    return {
        "variant": variant,
        "seed": seed,
        "lambda_red": lam,
        "removal_budget": budget,
        "realized_overlap": overlap,
        "retrieval_accuracy": retrieval,
        "chance_accuracy": run_cfg.data.chance_accuracy,
        "eval_loss": eval_loss,
        "eval_ppl": perplexity(eval_loss),
        "calibration_loss": history["final_calibration_loss"],
        "calibration_ppl": perplexity(history["final_calibration_loss"]),
        "gate_selected": gate.get("selected"),
        "gate_pool_size": gate.get("pool_size"),
        "gate_criterion": gate.get("criterion"),
        "gate_mean_selected_overlap": gate.get("mean_selected_overlap"),
        "contribution_delta_mean": gate.get("delta_mean"),
        "contribution_frac_below_eps": gate.get("frac_below_eps"),
        "n_gate_refreshes": len(history["gate_hist"]),
    }


def collect_runs(results_dir: Path) -> List[Dict[str, object]]:
    """Flatten completed sweep records into rows the matcher understands."""
    rows: List[Dict[str, object]] = []
    for rec in numeric_records(results_dir, EXPERIMENT):
        row = dict(rec.metrics)
        row["run_id"] = rec.run_id
        row["variant"] = rec.variant
        row["seed"] = rec.seed
        row["status"] = rec.status.value
        rows.append(row)
    return rows


def write_outputs(results_dir: Path, tolerance: float) -> Dict[str, object]:
    rows = collect_runs(results_dir)
    if rows:
        write_csv(results_dir / "sweep.csv", rows)
    pairs = find_matched_overlap_pairs(rows, tolerance=tolerance)
    if pairs:
        write_csv(results_dir / "matched_pairs.csv", pairs)
    payload = {
        "experiment": EXPERIMENT,
        "tolerance": tolerance,
        "n_runs": len(rows),
        "n_matched_pairs": len(pairs),
        "matched_pairs": pairs,
        "note": (
            "Pairs are matched on realized overlap measured after training, "
            "never on the configured lambda_red. Two methods at the same lambda "
            "reach different realized overlaps."
        ),
    }
    write_json(results_dir / "matched_pairs.json", payload)
    write_json(results_dir / "sweep.json", {"runs": rows})
    return payload


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--lambdas", type=float, nargs="*", default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--budgets", type=int, nargs="*", default=None,
                        help="removal budgets to sweep; defaults to the profile value")
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="max absolute realized-overlap difference for a match")
    parser.add_argument("--methods", nargs="*", default=list(SWEEP_METHODS))
    args = parser.parse_args(argv)

    cfg = config_from_args(args)
    device = resolve_device(args.device)
    results_dir = results_dir_for(args, "matched_overlap")
    seeds = args.seeds if args.seeds else list(cfg.multi_seeds)
    budgets = args.budgets if args.budgets else [cfg.contribution.n_pairs]

    print_header("Tier 1 - matched-overlap sweep ({})".format(cfg.profile))
    print("device={}  seeds={}  lambdas={}  budgets={}".format(
        device, seeds, args.lambdas, budgets))
    print("methods={}  tolerance={}".format(args.methods, args.tolerance))

    plan = []
    for method in args.methods:
        for lam in args.lambdas:
            for budget in budgets:
                for seed in seeds:
                    run_cfg = cfg.replace(
                        variant=method,
                        **{"train.seed": seed, "train.lambda_red": lam,
                           "contribution.n_pairs": budget},
                    )
                    plan.append((
                        make_run_id(run_cfg, seed, EXPERIMENT),
                        "{} lambda={} budget={} seed={}".format(method, lam, budget, seed),
                    ))
    if args.dry_run:
        describe_plan(plan)
        return 0

    print("\n{} sweep points".format(len(plan)))
    status = status_for(args)
    # Loaded once. Only the needle streams depend on the seed, so there is no
    # reason to re-tokenise WikiText-2 for every sweep point.
    base_corpus = load_corpus(cfg, seed=seeds[0], synthetic=args.synthetic_data)
    done = 0
    for method in args.methods:
        for lam in args.lambdas:
            for budget in budgets:
                for seed in seeds:
                    run_cfg = cfg.replace(
                        variant=method,
                        **{"train.seed": seed, "train.lambda_red": lam,
                           "contribution.n_pairs": budget},
                    )
                    rid = make_run_id(run_cfg, seed, EXPERIMENT)
                    existing = load_record(results_dir, rid)
                    done += 1
                    if existing is not None and existing.status.is_numeric and not args.force:
                        print("[{}/{}] skip {} lam={} budget={} seed={}".format(
                            done, len(plan), method, lam, budget, seed))
                        continue
                    corpus = base_corpus.reseed(seed)
                    with record_run(results_dir, EXPERIMENT, run_cfg, seed, status,
                                    splits=corpus.split_metadata(
                                        min(cfg.model.block_size, 256))) as rec:
                        rec.metrics = run_sweep_point(
                            cfg, method, seed, lam, budget, device, corpus
                        )
                    m = rec.metrics
                    print("[{}/{}] {:<18} lam={:<5} budget={} seed={:<5} "
                          "overlap={:.4f} retrieval={:.1f}%".format(
                              done, len(plan), method, lam, budget, seed,
                              m["realized_overlap"], m["retrieval_accuracy"]))

    payload = write_outputs(results_dir, args.tolerance)

    print_header("Matched pairs (realized overlap within {})".format(args.tolerance))
    if not payload["matched_pairs"]:
        print("No pairs within tolerance. Widen --tolerance or extend --lambdas;")
        print("reporting an unmatched comparison would defeat the experiment.")
    else:
        print("{:<6} {:>9} {:>9} {:>8} {:>10} {:>10} {:>9}".format(
            "seed", "naive_ov", "contr_ov", "|diff|", "naive_ret", "contr_ret", "delta"))
        for p in payload["matched_pairs"]:
            print("{:<6} {:>9.4f} {:>9.4f} {:>8.4f} {:>9.1f}% {:>9.1f}% {:>+8.1f}".format(
                p["seed"], p["crpa_naive_overlap"], p["crpa_contribution_overlap"],
                p["overlap_abs_diff"], p["crpa_naive_retrieval"],
                p["crpa_contribution_retrieval"], p["retrieval_delta"]))
    print("\nWrote {}".format(results_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
