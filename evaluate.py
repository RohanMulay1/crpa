"""
evaluate.py - backwards-compatibility shim.

The implementation moved to :mod:`crpa.evaluate`, whose functions take an
explicit split role. This module preserves the original signatures, which had
no split argument.

Where the original reported on its ``'val'`` split, these wrappers report on the
held-out **evaluation** split. That is a deliberate change: the original
calibrated thresholds and reported final numbers on the same data.

The original file is available in git history:

    git show 7474c77:evaluate.py
"""

from __future__ import annotations

from typing import Tuple

import torch

import data as _data
from config import CFG
from crpa.config import from_legacy_cfg
from crpa.data import EVAL
from crpa.evaluate import language_model_loss  # noqa: F401  (re-exported)
from crpa.evaluate import measure_overlap as _measure_overlap
from crpa.evaluate import retrieval_accuracy as _retrieval_accuracy
from crpa.evaluate import routing_diagnostics as _routing_diagnostics
from crpa.seeding import local_seed

__all__ = ["retrieval_accuracy", "retrieval_by_depth", "measure_overlap",
           "measure_throughput", "routing_diagnostics", "language_model_loss"]


def retrieval_accuracy(model, block_size: int, device: str,
                       n_batches: int = 30, bs: int = 8) -> float:
    """Top-1 Needle-in-Haystack accuracy on the held-out evaluation split."""
    return _retrieval_accuracy(
        model, _data._require_corpus(), block_size, device,
        role=EVAL, n_batches=n_batches, bs=bs,
    )


def retrieval_by_depth(model, block_size: int, depth: float, device: str,
                       n_batches: int = 20, bs: int = 8) -> float:
    """Retrieval accuracy with the needle pinned at ``depth``."""
    return _retrieval_accuracy(
        model, _data._require_corpus(), block_size, device,
        role=EVAL, n_batches=n_batches, bs=bs, needle_depth=depth,
    )


def measure_overlap(model, block_size: int, device: str, n_batches: int = 12) -> float:
    """Mean within-partition attention overlap."""
    cfg = from_legacy_cfg(CFG, variant=model.variant,
                          **{"model.block_size": block_size,
                             "model.vocab_size": model.cfg.vocab_size})
    return _measure_overlap(
        model, _data._require_corpus(), cfg, device,
        role=EVAL, n_batches=n_batches, bs=4,
    )


def measure_throughput(model_cls, variant: str, ctx: int, vocab_size: int,
                       device: str, n_runs: int = 60, bs: int = 8) -> float:
    """Forward latency in milliseconds.

    Two fixes relative to the original: timing is synchronised per iteration
    rather than once around the whole loop, and the benchmark seed is scoped so
    it no longer clobbers the surrounding experiment's RNG state (the original
    called ``torch.manual_seed(0)`` globally, mid-run).
    """
    from experiments.benchmark import time_forward

    with local_seed(0):
        model = model_cls(variant, ctx, vocab_size, device).to(device).eval()
        x = torch.randint(0, vocab_size, (bs, ctx), device=device)
        with model.capture_probabilities(False):
            stats = time_forward(model, x, device, n_warmup=5, n_iters=n_runs)
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return stats["latency_ms_median"]


def routing_diagnostics(model, block_size: int, device: str,
                        n_batches: int = 20) -> Tuple[float, float, float]:
    """Returns ``(empty_partition_pct, routing_entropy, load_error)``.

    The original fed the token embedding into every layer's router instead of
    that layer's actual input, which is why every row of the published Table 5
    was identical. This reads each layer's real input.
    """
    out = _routing_diagnostics(
        model, _data._require_corpus(), block_size, device,
        role=EVAL, n_batches=n_batches,
    )
    return (out["empty_partition_pct"], out["routing_entropy"], out["load_error"])
