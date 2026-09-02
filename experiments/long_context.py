"""
Tier 2 - does the diagnostic stay meaningful as context grows?

The question is not whether the model "scales". It is whether the
overlap-versus-contribution distinction survives at longer context: do
high-overlap edges keep separating into low- and high-contribution groups, and
does the contribution ranking stay stable, at 4k, 8k, 16k, 32k and 64k?

For each context length we record retrieval, realized overlap, the
distribution of intervention deltas, ranking stability, the fraction of
candidates in each high-overlap group, and the sparsity ratio. Latency,
throughput and memory come from :mod:`experiments.benchmark`.

Compute honesty
---------------
Lengths that do not fit are recorded with ``status='oom'`` and no numbers. A
length that was never attempted is recorded ``status='not_run'``. Neither ever
becomes a data point.

    python -m experiments.long_context --smoke
    python -m experiments.long_context --profile medium_138m \\
        --context_lengths 4096 8192 16384
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Optional

import numpy as np
import torch

from crpa.config import ExperimentConfig
from crpa.data import CALIBRATION
from crpa.evaluate import measure_overlap, retrieval_accuracy, sparsity_report
from crpa.intervention import (
    make_needle_loss_fn,
    reachable_queries,
    sample_candidate_edges,
    score_candidates,
    split_high_overlap_groups,
)
from crpa.kvcache import attention_edge_counts, kv_cache_table
from crpa.metrics import spearman
from crpa.model import GPT
from crpa.runmeta import RunRecord, Status, environment, save_record, write_csv, write_json
from crpa.seeding import set_seed
from crpa.train import train
from experiments.common import (
    add_common_args,
    config_from_args,
    load_corpus,
    make_run_id,
    print_header,
    resolve_device,
    results_dir_for,
    status_for,
)

EXPERIMENT = "long_context"


def diagnose_at_length(
    cfg: ExperimentConfig,
    context_length: int,
    seed: int,
    device: str,
    corpus,
    n_candidates: int = 32,
    train_iters: Optional[int] = None,
    batch_size: int = 2,
) -> Dict[str, object]:
    """Run the overlap/contribution diagnostic at one context length."""
    run_cfg = cfg.replace(**{
        "model.block_size": context_length,
        "train.seed": seed,
        "train.max_iters": train_iters if train_iters is not None else cfg.train.max_iters,
    })
    set_seed(seed)
    model = GPT(run_cfg.model, cfg.variant, seed=seed).to(device)

    if run_cfg.train.max_iters > 0:
        train(model, run_cfg, corpus, device, verbose=False)

    x, y = corpus.needle_batch(CALIBRATION, context_length, batch_size, device=device)
    loss_fn = make_needle_loss_fn(x, y)

    model.eval()
    with model.frozen_structure():
        # Probability capture is O(T^2) memory; only affordable at diagnostic
        # sizes, which is why candidate counts here are modest.
        with model.capture_probabilities(True):
            model(x)
        probs = model.attention_probabilities()
        reach = reachable_queries(model, context_length)
        rng = np.random.default_rng(seed + 31337)
        pool = []
        for depth in range(len(model.blocks)):
            if probs[depth] is None:
                continue
            pool += sample_candidate_edges(
                probs[depth], depth, run_cfg.model.partition_size,
                run_cfg.model.overlap_rho, n_candidates, rng,
                reach=reach[depth],
            )
        pool = pool[:n_candidates]
        scored = score_candidates(
            model, pool, loss_fn, eps=run_cfg.contribution.eps,
            seed=seed, context_length=context_length,
        )
        # A second independent estimate, so ranking stability is measurable
        # here rather than assumed from the Tier 1 result.
        x2, y2 = corpus.needle_batch(CALIBRATION, context_length, batch_size, device=device)
        scored2 = score_candidates(
            model, [c for c in pool], make_needle_loss_fn(x2, y2),
            eps=run_cfg.contribution.eps, seed=seed, context_length=context_length,
        )

    deltas = [c.delta_loss for c in scored if math.isfinite(c.delta_loss)]
    by_key = {(c.layer, c.head, c.query, c.key): c.delta_loss for c in scored2}
    paired = [(c.delta_loss, by_key[(c.layer, c.head, c.query, c.key)])
              for c in scored
              if (c.layer, c.head, c.query, c.key) in by_key
              and math.isfinite(c.delta_loss)]

    groups = split_high_overlap_groups(
        scored, run_cfg.contribution.high_overlap_q,
        run_cfg.contribution.low_contribution_q,
        run_cfg.contribution.high_contribution_q,
    )
    n_scored = max(len(scored), 1)

    metrics: Dict[str, object] = {
        "context_length": context_length,
        "n_candidates_scored": len(scored),
        "retrieval_accuracy": retrieval_accuracy(
            model, corpus, context_length, device, n_batches=8, bs=batch_size
        ),
        "realized_overlap": measure_overlap(
            model, corpus, run_cfg, device, n_batches=3, bs=1, seed=seed
        ),
        "delta_mean": float(np.mean(deltas)) if deltas else float("nan"),
        "delta_std": float(np.std(deltas)) if deltas else float("nan"),
        "delta_p50": float(np.percentile(deltas, 50)) if deltas else float("nan"),
        "delta_p90": float(np.percentile(deltas, 90)) if deltas else float("nan"),
        "delta_max": float(np.max(deltas)) if deltas else float("nan"),
        "ranking_stability_spearman": (
            spearman([a for a, _ in paired], [b for _, b in paired])
            if len(paired) > 2 else float("nan")
        ),
        "frac_high_overlap_low_contribution":
            len(groups["high_overlap_low_contribution"]) / n_scored,
        "frac_high_overlap_high_contribution":
            len(groups["high_overlap_high_contribution"]) / n_scored,
        "group_thresholds": groups["thresholds"],
    }
    metrics.update(sparsity_report(model, context_length))
    metrics.update(attention_edge_counts(run_cfg.model, context_length))
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return metrics


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--context_lengths", type=int, nargs="*", default=None)
    parser.add_argument("--n_candidates", type=int, default=32)
    parser.add_argument("--train_iters", type=int, default=None,
                        help="training steps at each length; 0 diagnoses an "
                             "untrained model, which is stated in the results")
    parser.add_argument("--bench_batch_size", type=int, default=2)
    parser.add_argument("--variant", default="crpa_contribution")
    args = parser.parse_args(argv)

    cfg = config_from_args(args).replace(variant=args.variant)
    device = resolve_device(args.device)
    results_dir = results_dir_for(args, "tier2")
    lengths = args.context_lengths or list(cfg.scale_lens)
    if args.smoke:
        lengths = [128, 256]
    seeds = args.seeds if args.seeds else [list(cfg.multi_seeds)[0]]
    status = status_for(args)

    print_header("Tier 2 - diagnostic stability across context length")
    print("device={}  lengths={}  seeds={}  variant={}".format(
        device, lengths, seeds, cfg.variant))
    env = environment()
    print("gpu={}  torch={}  cuda={}".format(
        env["gpu"]["name"] or "none (CPU)", env["torch"], env["cuda"]))
    if args.train_iters == 0:
        print("train_iters=0: diagnosing untrained models. Recorded as such; "
              "these are structural observations, not trained-model results.")

    if args.dry_run:
        print("\nWould diagnose {} length(s) x {} seed(s)".format(len(lengths), len(seeds)))
        return 0

    # Loaded once; only the needle streams depend on the seed.
    base_corpus = load_corpus(cfg, seed=seeds[0], synthetic=args.synthetic_data)

    rows: List[Dict[str, object]] = []
    for seed in seeds:
        corpus = base_corpus.reseed(seed)
        for length in lengths:
            run_cfg = cfg.replace(**{"model.block_size": length, "train.seed": seed})
            rid = make_run_id(run_cfg, seed, EXPERIMENT)
            record = RunRecord(
                run_id=rid, experiment=EXPERIMENT, status=status,
                config=run_cfg.to_dict(), seed=seed, context_length=length,
                variant=cfg.variant, splits=corpus.split_metadata(256),
                dtype=str(torch.get_default_dtype()).replace("torch.", ""),
            )
            try:
                record.metrics = diagnose_at_length(
                    cfg, length, seed, device, corpus,
                    n_candidates=args.n_candidates,
                    train_iters=args.train_iters,
                    batch_size=args.bench_batch_size,
                )
                row = {"seed": seed, "status": record.status.value,
                       **{k: v for k, v in record.metrics.items()
                          if isinstance(v, (int, float, bool, str))}}
                rows.append(row)
                print("  ctx={:<7} retrieval={:>5.1f}%  overlap={:.3f}  "
                      "delta_p90={:.2e}  rank_stab={:.3f}  sparsity={:.4f}".format(
                          length, record.metrics["retrieval_accuracy"],
                          record.metrics["realized_overlap"],
                          record.metrics["delta_p90"],
                          record.metrics["ranking_stability_spearman"],
                          record.metrics["sparsity_ratio"]))
            except torch.cuda.OutOfMemoryError as exc:
                record.status = Status.OOM
                record.error = str(exc)[:2000]
                record.metrics = {}
                rows.append({"seed": seed, "context_length": length,
                             "status": Status.OOM.value})
                print("  ctx={:<7} OOM - recorded as oom, no numbers".format(length))
                torch.cuda.empty_cache()
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                record.status = Status.OOM
                record.error = str(exc)[:2000]
                record.metrics = {}
                rows.append({"seed": seed, "context_length": length,
                             "status": Status.OOM.value})
                print("  ctx={:<7} OOM - recorded as oom, no numbers".format(length))
            save_record(results_dir, record)

    kv_rows = kv_cache_table(cfg.model, lengths, dtype="bfloat16", batch_size=1)
    if rows:
        write_csv(results_dir / "long_context.csv", rows)
    write_csv(results_dir / "kv_cache_projection.csv", kv_rows)
    write_json(results_dir / "long_context.json", {
        "experiment": EXPERIMENT,
        "environment": env,
        "context_lengths_requested": lengths,
        "rows": rows,
        "kv_cache": kv_rows,
        "note": (
            "Rows with status 'oom' carry no metrics. Context lengths absent "
            "from this file were not attempted and must be reported as not run."
        ),
    })
    print("\nWrote {}".format(results_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
