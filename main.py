"""
main.py - the original entry point, preserved.

    python main.py                      # full Tier 1 comparison
    python main.py --max_iters 100 --block_size 64
    python main.py --figures_only       # regenerate figures from results/, no training
    python main.py --skip_multiseed

Every original flag still works. Three behaviours changed, deliberately:

1. ``--figures_only`` now does something. In the original every code path,
   including figure generation, sat inside ``if not args.figures_only:``, so
   the flag produced nothing at all. Figures now regenerate from the files
   under ``results/`` without loading a model.
2. The variant named ``crpa_causal`` is now ``crpa_contribution``. The old name
   still works everywhere and old checkpoints still load. What changed is the
   mechanism: see ``crpa/intervention.py``. Pass
   ``--intervention_mode legacy_rowpair`` for the original selection semantics.
3. Retrieval and overlap are reported on a held-out **evaluation** split. The
   original calibrated and reported on the same data.

For the full study - multi-seed aggregates, the matched-overlap sweep, the
overlap-versus-contribution dataset, long context, and the frozen large-model
diagnostic - use the ``experiments`` package. ``python main.py`` runs the
subset that corresponds to the original script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from crpa.config import CENTRAL_VARIANTS, VARIANT_LABELS, load_profile
from crpa.runmeta import atomic_write_text


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CRPA experiments (original entry point)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--max_iters", type=int, default=None)
    p.add_argument("--block_size", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--figures_only", action="store_true",
                   help="regenerate figures from results/ without training")
    p.add_argument("--skip_multiseed", action="store_true",
                   help="run seed 42 only instead of all three seeds")
    p.add_argument("--include_baselines", action="store_true",
                   help="also run the dense and sliding baselines")
    p.add_argument("--intervention_mode", default="edge",
                   choices=["edge", "legacy_rowpair"])
    p.add_argument("--profile", default="small_12m")
    p.add_argument("--results_dir", default="results/tier1")
    p.add_argument("--synthetic_data", action="store_true",
                   help="deterministic pseudo-corpus; for CPU smoke runs only")
    return p.parse_args(argv)


def write_legacy_tables(results_dir: Path) -> None:
    """Emit the original ``table2_main.txt`` / ``table4_ablation.txt`` layout.

    Reads the structured aggregate rather than recomputing, so the text tables
    and the JSON can never disagree.

    These overwrite ``results/table*.txt`` on each run, as the original did.
    The artifacts published with the original implementation are preserved
    separately under ``results/original_published/`` so a run cannot destroy
    the historical record.
    """
    from experiments.tier1_multiseed import aggregate

    agg = aggregate(results_dir)
    variants = agg["variants"]
    if not variants:
        print("  (no completed runs; skipping legacy tables)")
        return

    out = Path("results")
    lines = ["TABLE 2 - Main Results (mean over seeds)", "=" * 72,
             "  {:<28} {:>9} {:>12} {:>10}".format("Model", "PPL", "Ret.Acc", "Overlap"),
             "  " + "-" * 62]
    for name, entry in variants.items():
        lines.append("  {:<28} {:>9.2f} {:>11.1f}% {:>10.3f}".format(
            VARIANT_LABELS.get(name, name),
            entry["eval_ppl"]["mean"], entry["retrieval_accuracy"]["mean"],
            entry["realized_overlap"]["mean"]))
    lines += ["", "Reported on the held-out evaluation split.",
              "Chance retrieval accuracy is 5.0%; values at or below it indicate",
              "retrieval was not learned."]
    atomic_write_text(out / "table2_main.txt", "\n".join(lines))

    lines = ["TABLE 4 - Overlap regularization ablation", "=" * 72,
             "  {:<28} {:>10} {:>12} {:>10}".format(
                 "Variant", "Overlap", "Ret.Acc", "PPL"),
             "  " + "-" * 62]
    for name in CENTRAL_VARIANTS:
        entry = variants.get(name)
        if not entry:
            continue
        lines.append("  {:<28} {:>10.3f} {:>11.1f}% {:>10.2f}".format(
            VARIANT_LABELS.get(name, name),
            entry["realized_overlap"]["mean"], entry["retrieval_accuracy"]["mean"],
            entry["eval_ppl"]["mean"]))
    lines += ["", "Overlaps here are whatever each configuration happened to reach.",
              "To compare the two suppression criteria at a *matched* overlap",
              "budget, use: python -m experiments.matched_overlap"]
    atomic_write_text(out / "table4_ablation.txt", "\n".join(lines))
    print("  Wrote results/table2_main.txt and results/table4_ablation.txt")


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = Path(args.results_dir)

    if args.figures_only:
        # Previously a no-op: the whole body sat inside `if not figures_only`.
        from experiments.plot_all import main as plot_main

        print("Regenerating figures from results/ (no training)\n")
        return plot_main(["--results_dir", "results"])

    from experiments.tier1_multiseed import main as tier1_main

    cfg = load_profile(args.profile)
    seeds = ["42"] if args.skip_multiseed else [str(s) for s in cfg.multi_seeds]

    forwarded = [
        "--profile", args.profile,
        "--device", args.device,
        "--results_dir", str(results_dir),
        "--intervention_mode", args.intervention_mode,
        "--seeds", *seeds,
    ]
    if args.max_iters is not None:
        forwarded += ["--max_iters", str(args.max_iters)]
    if args.block_size is not None:
        forwarded += ["--block_size", str(args.block_size)]
    if args.include_baselines:
        forwarded += ["--include_baselines"]
    if args.synthetic_data:
        forwarded += ["--synthetic_data"]

    rc = tier1_main(forwarded)
    if rc != 0:
        return rc

    print("\nWriting legacy text tables...")
    write_legacy_tables(results_dir)

    print("\nRegenerating figures...")
    from experiments.plot_all import main as plot_main

    plot_main(["--results_dir", "results"])

    print("\nDone. For the rest of the study:")
    print("  python -m experiments.matched_overlap         # matched-overlap sweep")
    print("  python -m experiments.overlap_vs_contribution # the central dataset")
    print("  python -m experiments.estimator_stability     # estimator diagnostics")
    print("  python -m experiments.long_context            # Tier 2")
    print("  python -m experiments.large_model_diagnostic  # Tier 3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
