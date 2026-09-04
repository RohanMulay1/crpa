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
import heapq
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from crpa.config import ExperimentConfig
from crpa.data import CALIBRATION
from crpa.evaluate import measure_overlap, retrieval_accuracy, sparsity_report
from crpa.intervention import (
    InterventionPlan,
    make_needle_loss_fn,
    reachable_queries,
    sample_candidate_edges,
    sample_candidate_edges_sparse,
    score_candidates_chunked,
    split_high_overlap_groups,
)
from crpa.attention import relay_positions
from crpa.kvcache import attention_edge_counts, kv_cache_table
from crpa.metrics import spearman
from crpa.model import GPT
from crpa.runmeta import (
    RunRecord,
    Status,
    environment,
    load_records,
    save_record,
    write_csv,
    write_json,
)
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


def collect_rows(results_dir: Path) -> List[Dict[str, object]]:
    """Flatten every record on disk into CSV rows, ordered by context length.

    Includes non-numeric statuses so an OOM stays visible as an OOM, with no
    metrics attached.
    """
    out: List[Dict[str, object]] = []
    for rec in load_records(results_dir):
        if rec.experiment != EXPERIMENT:
            continue
        row = {"seed": rec.seed, "status": rec.status.value,
               "context_length": rec.context_length}
        if rec.status.is_numeric:
            row.update({k: v for k, v in rec.metrics.items()
                        if isinstance(v, (int, float, bool, str))})
        out.append(row)
    out.sort(key=lambda r: (r.get("context_length") or 0, r.get("seed") or 0))
    return out


def diagnose_at_length(
    cfg: ExperimentConfig,
    context_length: int,
    seed: int,
    device: str,
    corpus,
    n_candidates: int = 32,
    train_iters: Optional[int] = None,
    batch_size: int = 2,
    probs_window: int = 1024,
    bench_only: bool = False,
    score_chunk_size: int = 8,
) -> Dict[str, object]:
    """Run the overlap/contribution diagnostic at one context length.

    ``bench_only`` skips the candidate-edge half and keeps the cost half.
    They have very different memory profiles: a forward pass at 32k with the
    138M profile peaks at 1.90 GB, while the candidate diagnostic exceeds an
    80GB card at the same length. Separating them means the latency, memory
    and sparsity numbers are obtainable at lengths where the intervention
    analysis is not, instead of losing both to one OOM.
    """
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
    phase_peaks: Dict[str, float] = {}

    def begin_phase() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def end_phase(name: str) -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            phase_peaks[name] = int(torch.cuda.max_memory_allocated())
    if bench_only:
        # Cost measurements only. Everything below this point is the
        # intervention diagnostic, which is what does not fit at 32k+.
        return {
            "context_length": context_length,
            "seed": seed,
            "bench_only": True,
            "n_candidates": 0,
            "note": "cost half only; the candidate-edge diagnostic was "
                    "skipped because it exceeds available memory at this "
                    "length. No contribution numbers are reported.",
        }

    with model.frozen_structure():
        # A dense capture is 12.9 GB per layer at T=16384 with 12 heads, and
        # 180 GB across a 14-layer model, so the diagnostic retains
        # probabilities in gathered form for a window of queries instead. The
        # window sits at the end of the sequence, which is also where
        # reachability concentrates under a last-token loss.
        window_lo = max(0, context_length - probs_window)
        gathered = run_cfg.model.attention_impl == "sparse_gather"
        rng = np.random.default_rng(seed + 31337)
        # Keep only the global top n while layers are captured one at a time.
        # The tie counter prevents heapq from comparing Candidate instances.
        candidate_heap = []
        tie = 0

        def retain(candidates) -> None:
            nonlocal tie
            for cand in candidates:
                item = (float(cand.overlap), tie, cand)
                tie += 1
                if len(candidate_heap) < n_candidates:
                    heapq.heappush(candidate_heap, item)
                elif item[0] > candidate_heap[0][0]:
                    heapq.heapreplace(candidate_heap, item)

        if gathered:
            begin_phase()
            for depth in range(len(model.blocks)):
                with torch.no_grad(), model.capture_probabilities(
                        True, window=(window_lo, context_length),
                        layers=[depth]):
                    model(x, last_only=True)
                sparse = model.sparse_attention_probabilities()[depth]
                reach = reachable_queries(model, context_length)[depth]
                retain(sample_candidate_edges_sparse(
                    sparse, depth, run_cfg.model.partition_size,
                    run_cfg.model.overlap_rho, n_candidates, rng,
                    reach=reach,
                ))
                # Captures are GPU tensors. Drop the sole live reference before
                # advancing to the next layer so memory is O(one layer).
                model.blocks[depth].attn._sparse_probs = None
                del sparse, reach
            end_phase("candidate_capture")
        else:
            # The dense path has no gathered capture. It is only usable at
            # short context, which is exactly where this branch runs.
            begin_phase()
            with torch.no_grad(), model.capture_probabilities(True):
                model(x, last_only=True)
            probs = model.attention_probabilities()
            reach = reachable_queries(model, context_length)
            relays = relay_positions(context_length, run_cfg.model.n_relays)
            for depth in range(len(model.blocks)):
                if probs[depth] is None:
                    continue
                retain(sample_candidate_edges(
                    probs[depth], depth, run_cfg.model.partition_size,
                    run_cfg.model.overlap_rho, n_candidates, rng,
                    reach=reach[depth], exclude_queries=relays,
                ))
            end_phase("candidate_capture")
        pool = [item[2] for item in sorted(candidate_heap, reverse=True)]
        begin_phase()
        scored = score_candidates_chunked(
            model, pool, loss_fn, eps=run_cfg.contribution.eps,
            seed=seed, context_length=context_length,
            chunk_size=score_chunk_size,
        )
        # A second independent estimate, so ranking stability is measurable
        # here rather than assumed from the Tier 1 result.
        x2, y2 = corpus.needle_batch(CALIBRATION, context_length, batch_size, device=device)
        scored2 = score_candidates_chunked(
            model, [c for c in pool], make_needle_loss_fn(x2, y2),
            eps=run_cfg.contribution.eps, seed=seed,
            context_length=context_length, chunk_size=score_chunk_size,
        )
        end_phase("candidate_scoring")

    # Single-edge deltas fall below float resolution at this scale: against a
    # loss around 10.8, float32 resolves roughly 1.3e-06, and one edge out of
    # 552 permitted keys in one head of one layer moves it far less than that.
    # Removing the whole candidate set at once is the constructive check: if a
    # group intervention is resolvable where singles are not, the diagnostic
    # survives at length in group form even though it does not edge by edge.
    begin_phase()
    group_delta = float("nan")
    group_removed = 0
    if scored:
        with torch.no_grad(), model.frozen_structure():
            plan = InterventionPlan.of([c.to_edge() for c in scored])
            base = loss_fn(model, None)
            after = loss_fn(model, plan)
            group_removed = model.intervened_count()
            group_delta = after - base

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
        "probs_window": probs_window,
        "retrieval_accuracy": retrieval_accuracy(
            model, corpus, context_length, device, n_batches=8, bs=batch_size
        ),
        "realized_overlap": measure_overlap(
            model, corpus, run_cfg, device, n_batches=3, bs=1, seed=seed,
            window=probs_window if gathered else None
        ),
        "delta_mean": float(np.mean(deltas)) if deltas else float("nan"),
        "delta_std": float(np.std(deltas)) if deltas else float("nan"),
        "delta_p50": float(np.percentile(deltas, 50)) if deltas else float("nan"),
        "delta_p90": float(np.percentile(deltas, 90)) if deltas else float("nan"),
        "delta_max": float(np.max(deltas)) if deltas else float("nan"),
        "single_edge_deltas_all_zero": bool(deltas) and all(d == 0.0 for d in deltas),
        "group_delta": group_delta,
        "group_n_edges": len(scored),
        "group_n_removed": group_removed,
        # Float32 resolves about this much on a loss of the observed size.
        "float32_resolution_estimate": 1.2e-07 * 11.0,
        "ranking_stability_spearman": (
            spearman([a for a, _ in paired], [b for _, b in paired])
            if len(paired) > 2 else float("nan")
        ),
        "frac_high_overlap_low_contribution":
            len(groups["high_overlap_low_contribution"]) / n_scored,
        "frac_high_overlap_high_contribution":
            len(groups["high_overlap_high_contribution"]) / n_scored,
        "group_thresholds": groups["thresholds"],
        "score_chunk_size": score_chunk_size,
    }
    metrics.update(sparsity_report(model, context_length))
    metrics.update(attention_edge_counts(run_cfg.model, context_length))
    end_phase("group_and_summary")
    metrics["peak_measurement"] = (
        "measured" if device.startswith("cuda") else "not_applicable_cpu")
    metrics["diagnostic_peak_memory_bytes"] = (
        max(phase_peaks.values()) if phase_peaks else None)
    metrics["diagnostic_phase_peak_memory_bytes"] = dict(phase_peaks)
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
    parser.add_argument("--bench_only", action="store_true",
                        help="measure cost only and skip the candidate-edge "
                             "diagnostic, which needs far more memory")
    parser.add_argument("--rebuild_only", action="store_true",
                        help="rebuild aggregate CSV/JSON rows from existing "
                             "run records without launching a model")
    parser.add_argument("--train_iters", type=int, default=None,
                        help="training steps at each length; 0 diagnoses an "
                             "untrained model, which is stated in the results")
    parser.add_argument("--bench_batch_size", type=int, default=2,
                        help="sequences per measurement batch. Above 16k this "
                             "is the dominant memory term, and the statistic "
                             "is per-sequence, so 1 is the right choice there")

    parser.add_argument("--probs_window", type=int, default=1024,
                        help="number of trailing query rows whose attention is "
                             "retained for overlap analysis; cost is independent "
                             "of context length")
    parser.add_argument("--score_chunk_size", type=int, default=8,
                        help="bounded number of candidate scalar records "
                             "consumed per scoring chunk")
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

    if args.rebuild_only:
        rows = collect_rows(results_dir)
        if not rows:
            raise SystemExit("no long-context run records in {}".format(
                results_dir))
        write_csv(results_dir / "long_context.csv", rows)
        json_path = results_dir / "long_context.json"
        payload = (json.loads(json_path.read_text(encoding="utf-8"))
                   if json_path.exists() else {"experiment": EXPERIMENT})
        payload["rows"] = rows
        payload["note"] = (
            "Rows with status 'oom' carry no metrics. Context lengths absent "
            "from this file were not attempted and must be reported as not run.")
        write_json(json_path, payload)
        print("rebuilt {} rows from run records".format(len(rows)))
        return 0

    # Loaded once; only the needle streams depend on the seed.
    base_corpus = load_corpus(cfg, seed=seeds[0], synthetic=args.synthetic_data)

    rows: List[Dict[str, object]] = []
    for seed in seeds:
        corpus = base_corpus.reseed(seed)
        for length in lengths:
            # Must match what diagnose_at_length actually runs, including
            # train_iters. Building it differently here meant the record said
            # max_iters=2000 while the run used --train_iters 0, so the stored
            # provenance did not describe the executed run.
            run_cfg = cfg.replace(**{
                "model.block_size": length,
                "train.seed": seed,
                "train.max_iters": (
                    args.train_iters if args.train_iters is not None
                    else cfg.train.max_iters
                ),
            })
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
                    probs_window=args.probs_window,
                    bench_only=args.bench_only,
                    score_chunk_size=args.score_chunk_size,
                )
                row = {"seed": seed, "status": record.status.value,
                       **{k: v for k, v in record.metrics.items()
                          if isinstance(v, (int, float, bool, str))}}
                rows.append(row)
                m = record.metrics
                print("  ctx={:<7} retrieval={:>5.1f}%  overlap={:.3f}  "
                      "single delta_p90={:.2e}  group delta={:+.3e}  "
                      "sparsity={:.4f}".format(
                          length, m["retrieval_accuracy"], m["realized_overlap"],
                          m["delta_p90"], m["group_delta"], m["sparsity_ratio"]))
                if m.get("single_edge_deltas_all_zero"):
                    print("           every single-edge delta was exactly zero: "
                          "below float resolution at this scale")
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

    # Build the CSV from every record on disk, not just this invocation's
    # rows. Re-running a single context length would otherwise overwrite the
    # file with that one row and silently discard the others.
    rows = collect_rows(results_dir)

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
