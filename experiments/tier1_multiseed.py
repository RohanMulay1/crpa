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
from crpa.data import EVAL
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
    # Measured by simulating the generator, not asserted from vocabulary size.
    # Only n_needles value tokens appear in a sequence, so "guess a value token
    # you can see" already scores about 52%. Judging against the 5% uniform
    # figure inverts the conclusion for every strong variant.
    floor = corpus.needles[EVAL].measure_chance_floor(block, n=2000)

    metrics: Dict[str, object] = {
        "calibration_loss": calib_loss,
        "calibration_ppl": perplexity(calib_loss),
        "eval_loss": eval_loss,
        "eval_ppl": perplexity(eval_loss),
        "retrieval_accuracy": retrieval,
        "uniform_chance": floor["uniform"],
        "chance_floor_context_value": floor["context_value"],
        "chance_floor_last_value": floor["last_value"],
        "chance_accuracy": floor["strongest"],
        "above_chance": retrieval > floor["strongest"],
        "margin_over_floor_pp": retrieval - floor["strongest"],
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


def _measured_floor(records) -> Dict[str, float]:
    """Simulate the generator to find what a non-retrieving model scores.

    The floor is a property of the data generator and the config, not of any
    trained model, so it can be recovered from a stored record long after the
    run. That is what makes it possible to correct results written before the
    floor was measured.
    """
    from crpa.config import DataConfig
    from crpa.data import EVAL as _EVAL
    from crpa.data import NeedleGenerator

    for rec in records:
        data_cfg = (rec.config or {}).get("data")
        if not data_cfg:
            continue
        try:
            cfg = DataConfig(**{
                k: (tuple(v) if isinstance(v, list) else v)
                for k, v in data_cfg.items()
            })
        except TypeError:
            continue
        block = rec.context_length or 512
        gen = NeedleGenerator(cfg, _EVAL, rec.seed or 42)
        return gen.measure_chance_floor(block, n=2000)
    return {}


def _record_iters(rec) -> object:
    """The training budget a record was produced under, or None."""
    cfg = rec.config or {}
    train = cfg.get("train") if isinstance(cfg, dict) else None
    if isinstance(train, dict):
        return train.get("max_iters")
    return None


def _drop_inconsistent_budgets(records):
    """Keep only records trained for the modal number of iterations.

    Averaging a 3-iteration run with a 2000-iteration run produces a number
    that describes neither. This has happened: nine short records once entered
    this aggregate and moved reported retrieval from 4.31% to 2.15% and
    perplexity from 910 to 26,485, because status was the only guard and the
    short runs had been recorded as ``completed``.

    Status is now also checked at the source (``status_for``), but a second,
    independent guard belongs here, because this one catches any config
    heterogeneity rather than only the short-run case that was found first.
    The minority budget is dropped and named, never silently averaged.
    """
    counts: Dict[object, int] = {}
    for rec in records:
        counts[_record_iters(rec)] = counts.get(_record_iters(rec), 0) + 1
    if len(counts) <= 1:
        return list(records), []
    modal = max(counts.items(), key=lambda kv: (kv[1], kv[0] or 0))[0]
    keep, dropped = [], []
    for rec in records:
        (keep if _record_iters(rec) == modal else dropped).append(rec)
    if dropped:
        print(
            "[aggregate] refusing to average across training budgets: keeping "
            "{} record(s) at max_iters={} and dropping {} at {}. A run trained "
            "for a different number of steps is a different experiment."
            .format(len(keep), modal, len(dropped),
                    sorted({str(_record_iters(r)) for r in dropped})))
    return keep, dropped


def aggregate(results_dir: Path) -> Dict[str, object]:
    """Mean / std / bootstrap CI per variant, across seeds.

    Reads only records whose status marks them as real measurements, so a
    failed or skipped seed cannot silently enter an average.
    """
    records = numeric_records(results_dir, EXPERIMENT)
    records, dropped = _drop_inconsistent_budgets(records)
    by_variant: Dict[str, List] = {}
    for rec in records:
        by_variant.setdefault(rec.variant or "unknown", []).append(rec)

    # Records written before the floor was measured carry the misleading 5%
    # uniform figure. Recompute from the config each record stores, so old
    # results are corrected on aggregation instead of silently kept.
    measured_floor = _measured_floor(records)

    fields = ["retrieval_accuracy", "realized_overlap", "eval_loss",
              "eval_ppl", "calibration_loss", "calibration_ppl",
              # Carried through so figures and tables can draw the chance line.
              # With this task at 5.0%, whether a bar clears it is the first
              # thing a reader needs to see.
              "chance_accuracy"]
    out: Dict[str, object] = {
        "experiment": EXPERIMENT,
        "measured_chance_floor": measured_floor,
        "runs_dropped_for_budget_mismatch": len(dropped),
        "variants": {},
    }
    for variant, recs in sorted(by_variant.items()):
        entry: Dict[str, object] = {
            "label": VARIANT_LABELS.get(variant, variant),
            "seeds": sorted(r.seed for r in recs if r.seed is not None),
            "n_runs": len(recs),
            "statuses": sorted({r.status.value for r in recs}),
        }
        if measured_floor:
            ret_mean = summarise([
                r.metrics["retrieval_accuracy"] for r in recs
                if isinstance(r.metrics.get("retrieval_accuracy"), (int, float))
            ])["mean"]
            entry["measured_chance_floor"] = measured_floor["strongest"]
            entry["margin_over_floor_pp"] = ret_mean - measured_floor["strongest"]
            entry["beats_floor"] = bool(ret_mean > measured_floor["strongest"])
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
    print("NOTE: the floor is measured by simulating the generator, not taken "
          "from vocabulary\nsize. Only {} value tokens appear per sequence, so "
          "'guess a value token you can\nsee' already scores about {:.0f}%.".format(
              cfg.data.n_needles, 100.0 / max(cfg.data.n_needles, 1)))

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
    status = status_for(args, cfg)

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
            print("  retrieval={:.1f}%  floor={:.1f}%  margin={:+.1f}pp  "
                  "overlap={:.3f}  eval_ppl={:.2f}  {}".format(
                      m["retrieval_accuracy"], m["chance_accuracy"],
                      m["margin_over_floor_pp"], m["realized_overlap"],
                      m["eval_ppl"],
                      "ABOVE floor" if m["above_chance"] else "at/below floor"))

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
    print("The floor is the strongest trivial strategy, measured by simulating "
          "the generator.\nA variant at or below it has not learned retrieval, "
          "and differences among such\nvariants are not evidence about retrieval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
