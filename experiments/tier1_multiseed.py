"""
Tier 1, experiment 1 - multi-seed replication of the central comparison.

Runs the three central variants across seeds 42 / 1337 / 2024:

  crpa_noreg          no overlap regularization
  crpa_naive          suppress the highest-overlap interactions
  crpa_contribution   suppress the lowest-contribution interactions,
                      at the same budget from the same candidate pool

Dense and sliding baselines are available via --include_baselines but are
secondary context, not the comparison of interest.

Per seed we record calibration loss, perplexity, retrieval accuracy on the
held-out evaluation split, realized overlap, and gate diagnostics. Aggregates
carry mean, standard deviation and a bootstrap 95% interval - which, with three
seeds, is wide, and is reported precisely so that is visible.

    python -m experiments.tier1_multiseed --smoke
    python -m experiments.tier1_multiseed --profile small_12m
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import torch

from crpa.config import CENTRAL_VARIANTS, VARIANT_LABELS, ExperimentConfig
from crpa.evaluate import (
    language_model_loss,
    measure_overlap,
    retrieval_accuracy,
    routing_diagnostics,
    sparsity_report,
)
from crpa.metrics import perplexity, summarise
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

EXPERIMENT = "tier1_multiseed"


def run_one(
    cfg: ExperimentConfig,
    variant: str,
    seed: int,
    device: str,
    corpus,
    verbose: bool = True,
) -> Dict[str, object]:
    """Train and evaluate one (variant, seed) cell."""
    run_cfg = cfg.replace(variant=variant, **{"train.seed": seed})
    set_seed(seed)
    model = GPT(run_cfg.model, variant, seed=seed).to(device)

    history = train(model, run_cfg, corpus, device, verbose=verbose)

    block = run_cfg.model.block_size
    calib_loss = history["final_calibration_loss"]
    eval_loss = language_model_loss(model, corpus, block, device, n_batches=10)
    retrieval = retrieval_accuracy(model, corpus, block, device, n_batches=30)
    overlap = measure_overlap(model, corpus, run_cfg, device, n_batches=8, seed=seed)

    gate = history["gate_hist"][-1] if history["gate_hist"] else {}
    metrics: Dict[str, object] = {
        "calibration_loss": calib_loss,
        "calibration_ppl": perplexity(calib_loss),
        "eval_loss": eval_loss,
        "eval_ppl": perplexity(eval_loss),
        "retrieval_accuracy": retrieval,
        "chance_accuracy": run_cfg.data.chance_accuracy,
        "above_chance": retrieval > run_cfg.data.chance_accuracy,
        "realized_overlap": overlap,
        "n_params": model.n_params(),
        "n_gate_refreshes": len(history["gate_hist"]),
        "gate_pool_size": gate.get("pool_size"),
        "gate_selected": gate.get("selected"),
        "gate_criterion": gate.get("criterion"),
        "gate_mean_selected_overlap": gate.get("mean_selected_overlap"),
        "contribution_delta_mean": gate.get("delta_mean"),
        "contribution_frac_below_eps": gate.get("frac_below_eps"),
        "val_hist": history["val_hist"],
        "step_hist": history["step_hist"],
    }
    metrics.update(sparsity_report(model, block))
    if variant.startswith("crpa"):
        metrics["routing"] = routing_diagnostics(
            model, corpus, block, device, n_batches=4
        )
    return {"metrics": metrics, "model": model, "config": run_cfg}


def aggregate(results_dir: Path) -> Dict[str, object]:
    """Mean / std / bootstrap CI per variant, across seeds.

    Reads only records whose status marks them as real measurements, so a
    failed or skipped seed cannot silently enter an average.
    """
    records = numeric_records(results_dir, EXPERIMENT)
    by_variant: Dict[str, List] = {}
    for rec in records:
        by_variant.setdefault(rec.variant or "unknown", []).append(rec)

    fields = ["retrieval_accuracy", "realized_overlap", "eval_loss",
              "eval_ppl", "calibration_loss", "calibration_ppl"]
    out: Dict[str, object] = {"experiment": EXPERIMENT, "variants": {}}
    for variant, recs in sorted(by_variant.items()):
        entry: Dict[str, object] = {
            "label": VARIANT_LABELS.get(variant, variant),
            "seeds": sorted(r.seed for r in recs if r.seed is not None),
            "n_runs": len(recs),
            "statuses": sorted({r.status.value for r in recs}),
        }
        for field in fields:
            values = [
                r.metrics[field] for r in recs
                if isinstance(r.metrics.get(field), (int, float))
                and math.isfinite(float(r.metrics[field]))
            ]
            entry[field] = summarise(values)
        out["variants"][variant] = entry
    return out


def write_outputs(results_dir: Path) -> None:
    """Write aggregate.json and the flat CSVs."""
    agg = aggregate(results_dir)
    write_json(results_dir / "aggregate.json", agg)

    rows = []
    for variant, entry in agg["variants"].items():
        row = {"variant": variant, "label": entry["label"], "n_runs": entry["n_runs"],
               "seeds": ";".join(str(s) for s in entry["seeds"]),
               "statuses": ";".join(entry["statuses"])}
        for field, stats in entry.items():
            if isinstance(stats, dict) and "mean" in stats:
                row["{}_mean".format(field)] = stats["mean"]
                row["{}_std".format(field)] = stats["std"]
                row["{}_ci_low".format(field)] = stats["ci_low"]
                row["{}_ci_high".format(field)] = stats["ci_high"]
        rows.append(row)
    if rows:
        write_csv(results_dir / "aggregate.csv", rows)

    per_run = []
    for rec in numeric_records(results_dir, EXPERIMENT):
        per_run.append({
            "run_id": rec.run_id, "variant": rec.variant, "seed": rec.seed,
            "status": rec.status.value, "context_length": rec.context_length,
            **{k: v for k, v in rec.metrics.items() if isinstance(v, (int, float, bool, str))},
        })
    if per_run:
        write_csv(results_dir / "runs.csv", per_run)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--variants", nargs="*", default=list(CENTRAL_VARIANTS))
    parser.add_argument("--include_baselines", action="store_true",
                        help="also run dense and sliding as secondary context")
    parser.add_argument("--save_checkpoints", action="store_true")
    args = parser.parse_args(argv)

    cfg = config_from_args(args)
    device = resolve_device(args.device)
    results_dir = results_dir_for(args, "tier1")
    seeds = args.seeds if args.seeds else list(cfg.multi_seeds)
    variants = list(args.variants)
    if args.include_baselines:
        variants = ["dense", "sliding"] + variants

    print_header("Tier 1 - multi-seed replication ({})".format(cfg.profile))
    print("device={}  seeds={}  variants={}".format(device, seeds, variants))
    print("intervention mode={}  attention impl={}".format(
        cfg.contribution.mode, cfg.model.attention_impl))
    print("chance retrieval accuracy = {:.1f}%".format(cfg.data.chance_accuracy))

    plan = [
        (make_run_id(cfg.replace(variant=v, **{"train.seed": s}), s, EXPERIMENT),
         "{} seed={}".format(v, s))
        for v in variants for s in seeds
    ]
    if args.dry_run:
        describe_plan(plan)
        return 0

    # Loaded once: the language-model splits do not depend on the seed, only
    # the needle streams do, and those are cheap to rebuild.
    corpus = load_corpus(cfg, seed=seeds[0], synthetic=args.synthetic_data)
    status = status_for(args)

    for variant in variants:
        for seed in seeds:
            run_cfg = cfg.replace(variant=variant, **{"train.seed": seed})
            rid = make_run_id(run_cfg, seed, EXPERIMENT)
            existing = load_record(results_dir, rid)
            if existing is not None and existing.status.is_numeric and not args.force:
                print("\n[skip] {} seed={} already complete ({})".format(
                    variant, seed, rid))
                continue

            print("\n--- {} | seed {} | {} ---".format(variant, seed, rid))
            # Each seed gets its own corpus so needle streams are seed-specific.
            seed_corpus = corpus.reseed(seed)
            with record_run(results_dir, EXPERIMENT, run_cfg, seed, status,
                            splits=seed_corpus.split_metadata(
                                min(cfg.model.block_size, 256))) as rec:
                result = run_one(run_cfg, variant, seed, device, seed_corpus)
                rec.metrics = result["metrics"]
                rec.dtype = str(next(result["model"].parameters()).dtype)
                if args.save_checkpoints:
                    ckpt = Path("checkpoints")
                    ckpt.mkdir(exist_ok=True)
                    torch.save(result["model"].state_dict(),
                               ckpt / "{}_{}_{}.pt".format(variant, seed, rid))
            m = rec.metrics
            print("  retrieval={:.1f}%  overlap={:.3f}  eval_ppl={:.2f}  {}".format(
                m["retrieval_accuracy"], m["realized_overlap"], m["eval_ppl"],
                "ABOVE chance" if m["above_chance"] else "at/below chance"))

    write_outputs(results_dir)
    print_header("Aggregate")
    agg = aggregate(results_dir)
    print("{:<26} {:>16} {:>14} {:>12}".format(
        "variant", "retrieval %", "overlap", "eval ppl"))
    for variant, entry in agg["variants"].items():
        r, o, p = entry["retrieval_accuracy"], entry["realized_overlap"], entry["eval_ppl"]
        print("{:<26} {:>7.1f} +/- {:<5.1f} {:>7.3f}+/-{:<5.3f} {:>11.2f}".format(
            entry["label"], r["mean"], r["std"], o["mean"], o["std"], p["mean"]))
    print("\nWrote {}".format(results_dir))
    print("Chance = {:.1f}%. Variants at or below chance have not learned "
          "retrieval; differences among them are not evidence about retrieval "
          "quality.".format(cfg.data.chance_accuracy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
