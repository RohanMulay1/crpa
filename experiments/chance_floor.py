"""Measure the strongest trivial strategy for the needle task.

This is deliberately a standalone, seed-complete measurement so reports do
not have to recover the floor from trained-model records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from crpa.config import DataConfig
from crpa.data import EVAL, NeedleGenerator


def measure(seeds, block_size: int, n: int) -> dict:
    rows = []
    for seed in seeds:
        result = NeedleGenerator(DataConfig(), EVAL, seed).measure_chance_floor(
            block_size, n=n)
        rows.append({"seed": seed, **result})
    strongest = np.asarray([row["strongest"] for row in rows], dtype=float)
    return {
        "experiment": "chance_floor",
        "status": "completed",
        "method": "simulate each trivial strategy on generated evaluation examples",
        "block_size": block_size,
        "n_per_seed": n,
        "seeds": list(seeds),
        "rows": rows,
        "strongest_mean_percent": float(strongest.mean()),
        "strongest_sd_percent": float(strongest.std(ddof=1)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 1337, 2024, 7, 99])
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--n", type=int, default=25_000)
    parser.add_argument("--out", type=Path,
                        default=Path("results/chance_floor.json"))
    args = parser.parse_args(argv)
    payload = measure(args.seeds, args.block_size, args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote {}: {:.4f}% +/- {:.4f}% (n={} x {})".format(
        args.out, payload["strongest_mean_percent"],
        payload["strongest_sd_percent"], args.n, len(args.seeds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
