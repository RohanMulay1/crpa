"""
experiments.common - shared plumbing for every experiment entry point.

Keeps CLI construction, device selection, corpus loading and run bookkeeping in
one place so the individual experiment modules stay about their science.
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import torch

from crpa.config import ExperimentConfig, load_profile
from crpa.data import Corpus
from crpa.runmeta import RunRecord, Status, run_id, save_record

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Flags every experiment accepts."""
    parser.add_argument("--profile", default="small_12m",
                        help="config profile name in configs/, or a path to a YAML file")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--results_dir", default=None,
                        help="defaults to results/<experiment name>")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="override the profile's seed list")
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--intervention_mode", default=None,
                        choices=["edge", "legacy_rowpair"],
                        help="'legacy_rowpair' reproduces the original repository "
                             "semantics; see crpa.intervention for why its deltas "
                             "are not interpretable")
    parser.add_argument("--attention_impl", default=None,
                        choices=["dense_masked", "sparse_gather"])
    parser.add_argument("--smoke", action="store_true",
                        help="tiny configuration that exercises the code path only; "
                             "results are recorded with status='smoke'")
    parser.add_argument("--dry_run", action="store_true",
                        help="print what would run, execute nothing")
    parser.add_argument("--force", action="store_true",
                        help="recompute runs that already have a completed record")
    parser.add_argument("--legacy_needle_position", action="store_true",
                        help="reproduce the original needle construction, which "
                             "left the query key two positions before the scored "
                             "position; use to attribute the effect of that fix")
    parser.add_argument("--synthetic_data", action="store_true",
                        help="use a deterministic pseudo-corpus instead of WikiText-2; "
                             "language-model numbers from it are meaningless and are "
                             "labelled as such")
    return parser


def resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda requested but torch reports no CUDA device. "
            "Refusing to silently fall back to CPU, which would make timing "
            "and memory results incomparable."
        )
    return choice


SMOKE_OVERRIDES: Dict[str, object] = {
    "model.n_embd": 64,
    "model.n_head": 4,
    "model.n_layer": 3,
    "model.block_size": 128,
    "model.partition_size": 32,
    "model.n_relays": 3,
    "model.cross_k": 3,
    "model.vocab_size": 50257,
    "train.batch_size": 4,
    "train.max_iters": 30,
    "train.eval_iters": 2,
    "train.eval_interval": 15,
    "train.warmup_steps": 5,
    "train.checkpoint_every": 0,
    "contribution.warmup_steps": 5,
    "contribution.interval": 10,
    "contribution.n_pairs": 4,
}


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Build the effective configuration from a profile plus CLI overrides."""
    cfg = load_profile(args.profile)
    overrides: Dict[str, object] = {}
    if args.smoke:
        overrides.update(SMOKE_OVERRIDES)
    if args.max_iters is not None:
        overrides["train.max_iters"] = args.max_iters
    if args.block_size is not None:
        overrides["model.block_size"] = args.block_size
    if args.intervention_mode is not None:
        overrides["contribution.mode"] = args.intervention_mode
    if args.attention_impl is not None:
        overrides["model.attention_impl"] = args.attention_impl
    if getattr(args, "legacy_needle_position", False):
        overrides["data.query_key_at_end"] = False
    return cfg.replace(**overrides) if overrides else cfg


def results_dir_for(args: argparse.Namespace, default_name: str) -> Path:
    path = Path(args.results_dir) if args.results_dir else RESULTS_ROOT / default_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_corpus(cfg: ExperimentConfig, seed: int, synthetic: bool,
                verbose: bool = True) -> Corpus:
    """Load data and immediately assert the three splits cannot collide."""
    corpus = Corpus(cfg.data, seed=seed)
    if synthetic:
        corpus.load_synthetic(vocab_size=cfg.model.vocab_size)
    else:
        corpus.load(verbose=verbose)
    corpus.assert_splits_disjoint(block_size=min(cfg.model.block_size, 256))
    return corpus


def status_for(args: argparse.Namespace) -> Status:
    """Smoke runs are recorded as smoke, never as completed."""
    return Status.SMOKE if args.smoke else Status.COMPLETED


def make_run_id(cfg: ExperimentConfig, seed: int, experiment: str, extra: str = "") -> str:
    return run_id(cfg.canonical_json(), seed, "{}|{}".format(experiment, extra))


@contextmanager
def record_run(
    results_dir: Path,
    experiment: str,
    cfg: ExperimentConfig,
    seed: int,
    status: Status,
    extra: str = "",
    **fields: object,
) -> Iterator[RunRecord]:
    """Create, populate and persist a :class:`RunRecord`.

    An exception inside the block is recorded with ``FAILED`` (or ``OOM`` for a
    CUDA out-of-memory) and re-raised. That way a crashed run leaves an honest
    artifact rather than no artifact, and its metrics can never be read as
    measurements.
    """
    rid = make_run_id(cfg, seed, experiment, extra)
    record = RunRecord(
        run_id=rid,
        experiment=experiment,
        status=status,
        config=cfg.to_dict(),
        seed=seed,
        context_length=cfg.model.block_size,
        variant=cfg.variant,
        **fields,
    )
    started = time.time()
    try:
        yield record
    except torch.cuda.OutOfMemoryError as exc:
        record.status = Status.OOM
        record.error = str(exc)[:2000]
        record.metrics = {}
        record.duration_s = time.time() - started
        save_record(results_dir, record)
        raise
    except Exception as exc:
        record.status = Status.FAILED
        record.error = "{}: {}".format(type(exc).__name__, str(exc)[:2000])
        record.metrics = {}
        record.duration_s = time.time() - started
        save_record(results_dir, record)
        raise
    else:
        record.duration_s = time.time() - started
        save_record(results_dir, record)


def print_header(title: str, width: int = 72) -> None:
    print("\n" + "=" * width)
    print(" " + title)
    print("=" * width)


def describe_plan(rows: List[Tuple[str, str]]) -> None:
    """Print what a --dry_run would have executed."""
    print("\nPlanned runs (dry run - nothing executed):")
    for rid, label in rows:
        print("  {}  {}".format(rid, label))
    print("  {} run(s) total".format(len(rows)))
