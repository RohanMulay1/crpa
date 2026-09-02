"""
Tier 2 - latency, throughput and memory benchmarking.

GPU timing uses CUDA events with explicit synchronisation and a warmup phase,
because ``time.perf_counter`` around an unsynchronised CUDA call measures kernel
launch, not kernel execution. The original implementation synchronised only
once at the end of the whole loop, so its warmup and its first iterations were
folded into the measurement.

Out of memory is recorded as ``status='oom'``. It is never converted into a
number, and never silently retried at a smaller size.

    python -m experiments.benchmark --smoke
    python -m experiments.benchmark --profile medium_138m \\
        --context_lengths 4096 8192 16384
"""

from __future__ import annotations

import argparse
import gc
import math
import time
from typing import Dict, List

import torch

from crpa.config import ExperimentConfig
from crpa.kvcache import attention_edge_counts, kv_cache_table
from crpa.model import GPT
from crpa.runmeta import Status, environment, write_csv, write_json
from crpa.seeding import local_seed
from experiments.common import (
    add_common_args,
    config_from_args,
    print_header,
    resolve_device,
    results_dir_for,
)

EXPERIMENT = "benchmark"


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _reset_memory(device: str) -> None:
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def memory_report(device: str) -> Dict[str, float]:
    """Allocated, reserved and peak memory. NaN on CPU, never zero."""
    if not device.startswith("cuda"):
        return {
            "allocated_mb": float("nan"),
            "reserved_mb": float("nan"),
            "peak_allocated_mb": float("nan"),
            "peak_reserved_mb": float("nan"),
        }
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1e6,
        "reserved_mb": torch.cuda.memory_reserved() / 1e6,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
    }


@torch.no_grad()
def time_forward(
    model: GPT,
    x: torch.Tensor,
    device: str,
    n_warmup: int = 5,
    n_iters: int = 20,
) -> Dict[str, float]:
    """Median and mean forward latency in milliseconds.

    Uses CUDA events on GPU and a synchronised wall clock on CPU. Reports the
    median as well as the mean, because a single stall skews the mean and the
    median is the more honest summary of steady-state latency.
    """
    model.eval()
    for _ in range(n_warmup):
        model(x)
    _sync(device)

    samples: List[float] = []
    if device.startswith("cuda"):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(n_iters):
            start.record()
            model(x)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
    else:
        for _ in range(n_iters):
            t0 = time.perf_counter()
            model(x)
            samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    mid = len(samples) // 2
    median = samples[mid] if len(samples) % 2 else 0.5 * (samples[mid - 1] + samples[mid])
    mean = sum(samples) / len(samples)
    tokens = int(x.numel())
    return {
        "latency_ms_median": median,
        "latency_ms_mean": mean,
        "latency_ms_min": samples[0],
        "latency_ms_max": samples[-1],
        "tokens_per_forward": tokens,
        "tokens_per_sec": tokens / (median / 1000.0) if median > 0 else float("nan"),
        "n_warmup": n_warmup,
        "n_iters": n_iters,
    }


def benchmark_point(
    cfg: ExperimentConfig,
    variant: str,
    context_length: int,
    device: str,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 1,
    n_warmup: int = 5,
    n_iters: int = 20,
    seed: int = 0,
) -> Dict[str, object]:
    """Benchmark one (variant, context length) point.

    Returns a dict carrying its own ``status``: ``completed`` on success,
    ``oom`` if CUDA ran out of memory. The caller must not read metrics from a
    non-completed row.
    """
    model_cfg = cfg.replace(**{"model.block_size": context_length}).model

    row: Dict[str, object] = {
        "variant": variant,
        "context_length": context_length,
        "batch_size": batch_size,
        "dtype": str(dtype).replace("torch.", ""),
        "attention_impl": model_cfg.attention_impl,
        "position": model_cfg.position,
        "n_layer": model_cfg.n_layer,
        "n_head": model_cfg.n_head,
        "n_embd": model_cfg.n_embd,
        "partition_size": model_cfg.partition_size,
        "n_relays": model_cfg.n_relays,
        "cross_k": model_cfg.cross_k,
        "analytic_params_m": model_cfg.n_params() / 1e6,
    }
    row.update(attention_edge_counts(model_cfg, context_length))

    _reset_memory(device)
    model = None
    try:
        with local_seed(seed):
            model = GPT(model_cfg, variant, seed=seed).to(device=device, dtype=dtype)
            x = torch.randint(
                0, model_cfg.vocab_size, (batch_size, context_length), device=device
            )
            # Probability capture would materialise (B,H,T,T); benchmarking
            # must measure the sparse path, not defeat it.
            with model.capture_probabilities(False):
                row.update(time_forward(model, x, device, n_warmup, n_iters))
        row.update(memory_report(device))
        row["actual_params_m"] = model.n_params() / 1e6
        row["status"] = Status.COMPLETED.value
    except torch.cuda.OutOfMemoryError as exc:
        row["status"] = Status.OOM.value
        row["error"] = str(exc)[:400]
        # Explicitly blank, so no reader can mistake a default for a measurement.
        for key in ("latency_ms_median", "tokens_per_sec", "peak_allocated_mb"):
            row[key] = None
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            row["status"] = Status.OOM.value
            row["error"] = str(exc)[:400]
            for key in ("latency_ms_median", "tokens_per_sec", "peak_allocated_mb"):
                row[key] = None
        else:
            raise
    finally:
        del model
        _reset_memory(device)
    return row


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--context_lengths", type=int, nargs="*", default=None)
    parser.add_argument("--variants", nargs="*",
                        default=["dense", "sliding", "crpa_contribution"])
    parser.add_argument("--bench_batch_size", type=int, default=1)
    parser.add_argument("--n_warmup", type=int, default=5)
    parser.add_argument("--n_iters", type=int, default=20)
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "bfloat16", "float16"])
    args = parser.parse_args(argv)

    cfg = config_from_args(args)
    device = resolve_device(args.device)
    results_dir = results_dir_for(args, "tier2")
    lengths = args.context_lengths or list(cfg.scale_lens)
    if args.smoke:
        lengths = [128, 256]
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]

    print_header("Tier 2 - latency / throughput / memory benchmark")
    print("device={}  dtype={}  lengths={}".format(device, args.dtype, lengths))
    print("variants={}  batch={}".format(args.variants, args.bench_batch_size))
    env = environment()
    print("gpu={}  torch={}  cuda={}".format(
        env["gpu"]["name"] or "none (CPU)", env["torch"], env["cuda"]))

    if args.dry_run:
        print("\nWould benchmark {} points".format(len(lengths) * len(args.variants)))
        return 0

    rows: List[Dict[str, object]] = []
    for length in lengths:
        for variant in args.variants:
            row = benchmark_point(
                cfg, variant, length, device, dtype=dtype,
                batch_size=args.bench_batch_size,
                n_warmup=args.n_warmup, n_iters=args.n_iters,
            )
            rows.append(row)
            if row["status"] == Status.OOM.value:
                print("  {:<20} ctx={:<7} OOM (recorded as oom, not as a number)".format(
                    variant, length))
            else:
                print("  {:<20} ctx={:<7} {:>9.2f} ms  {:>12,.0f} tok/s  "
                      "peak {:>8}".format(
                          variant, length, row["latency_ms_median"],
                          row["tokens_per_sec"],
                          "n/a" if math.isnan(row["peak_allocated_mb"])
                          else "{:.0f} MB".format(row["peak_allocated_mb"])))

    kv_rows = kv_cache_table(
        cfg.model, lengths, dtype="bfloat16",
        batch_size=args.bench_batch_size,
        measure_up_to=0 if device == "cpu" else max(lengths),
        device=device,
    )

    write_csv(results_dir / "benchmark.csv", rows)
    write_csv(results_dir / "kv_cache.csv", kv_rows)
    write_json(results_dir / "benchmark.json", {
        "experiment": EXPERIMENT,
        "environment": env,
        "config": cfg.to_dict(),
        "rows": rows,
        "kv_cache": kv_rows,
    })

    print_header("KV cache")
    print("{:<16} {:>10} {:>14} {:>14} {:>12}".format(
        "scheme", "ctx", "cached pos", "size", "measurement"))
    for r in kv_rows:
        print("{:<16} {:>10} {:>14,} {:>11.1f} MB {:>12}".format(
            r["scheme"], r["context_length"], r["cached_positions"],
            r["megabytes"], r["measurement"]))
    print("\nNote: CRPA does not bound the KV cache - routed keys may reference "
          "any earlier\nposition. crpa_bounded shows what dropping routing "
          "would buy.")
    print("\nWrote {}".format(results_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
