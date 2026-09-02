"""
train.py - backwards-compatibility shim.

The implementation moved to :mod:`crpa.train`. This module preserves the
original signatures:

    from train import train, estimate_loss
    model, val_hist, step_hist = train(model, block_size, device)

The original file is available in git history:

    git show 7474c77:train.py
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import data as _data
from config import CFG
from crpa.config import from_legacy_cfg
from crpa.train import estimate_loss as _estimate_loss
from crpa.train import make_lr_schedule  # noqa: F401  (re-exported)
from crpa.train import train as _train

__all__ = ["train", "estimate_loss", "make_lr_schedule", "experiment_config_for"]


def experiment_config_for(model, block_size: int, max_iters: Optional[int] = None):
    """Assemble an :class:`~crpa.config.ExperimentConfig` from the legacy globals."""
    cfg = from_legacy_cfg(
        CFG,
        variant=model.variant,
        **{
            "model.block_size": block_size,
            "model.vocab_size": model.cfg.vocab_size,
            "train.max_iters": max_iters or CFG["max_iters"],
        },
    )
    return cfg


def estimate_loss(model, block_size: int, device: str) -> Dict[str, float]:
    """Mean train/val language-model loss.

    Returns the original ``{'train': ..., 'val': ...}`` keys; ``'val'`` is the
    calibration split under the new protocol.
    """
    corpus = _data._require_corpus()
    out = _estimate_loss(model, corpus, block_size, device, CFG["eval_iters"])
    return {"train": out["train"], "val": out["calibration"]}


def train(model, block_size: int, device: str,
          max_iters: Optional[int] = None,
          verbose: bool = True) -> Tuple[object, List[float], List[int]]:
    """Train one variant. Returns ``(model, val_hist, step_hist)`` as before."""
    corpus = _data._require_corpus()
    cfg = experiment_config_for(model, block_size, max_iters)
    history = _train(model, cfg, corpus, device, max_iters=max_iters, verbose=verbose)
    return model, history["val_hist"], history["step_hist"]
