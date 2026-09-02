"""
Regenerate every figure from structured results. No training, no model loading.

Figures whose inputs are absent are reported as SKIPPED with the command that
would produce them. They are never drawn from placeholder data, so a missing
experiment looks missing.

    python -m experiments.plot_all --results_dir results
    python -m experiments.plot_all --results_dir results --only fig1 fig4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from crpa.figures import FIGURES, FigureSkipped


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results_dir", default="results",
                        help="root containing tier1/, tier2/, tier3/, matched_overlap/")
    parser.add_argument("--out_dir", default=None,
                        help="defaults to <results_dir>/figures")
    parser.add_argument("--only", nargs="*", default=None,
                        help="figure name prefixes to render, e.g. fig1 fig4")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if any figure was skipped")
    args = parser.parse_args(argv)

    results_root = Path(args.results_dir)
    if not results_root.exists():
        raise SystemExit("results directory not found: {}".format(results_root))
    out_dir = Path(args.out_dir) if args.out_dir else results_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = {
        name: fn for name, fn in FIGURES.items()
        if not args.only or any(name.startswith(p) for p in args.only)
    }
    if not selected:
        raise SystemExit("no figures matched --only {}".format(args.only))

    print("Regenerating figures from {}".format(results_root.resolve()))
    print("Output: {}\n".format(out_dir.resolve()))

    rendered, skipped = 0, 0
    for name, fn in selected.items():
        try:
            result = fn(results_root, out_dir)
        except FigureSkipped as exc:
            skipped += 1
            print("  SKIP    {:<34} {}".format(name, exc))
            continue
        paths = result if isinstance(result, list) else [result]
        for path in paths:
            rendered += 1
            print("  OK      {:<34} {}".format(name, path.name))

    print("\n{} rendered, {} skipped".format(rendered, skipped))
    if skipped:
        print("Skipped figures have no data. That is reported, not filled in.")
    if args.strict and skipped:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
