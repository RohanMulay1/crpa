"""
data.py - backwards-compatibility shim.

The implementation moved to :mod:`crpa.data`, which enforces a three-way
train / calibration / evaluation split. This module preserves the original
module-level API:

    from data import init_data, get_lm_batch, make_needle_batch

Split mapping used by the shim
------------------------------
The original had only ``'train'`` and ``'val'``. They map onto the new roles as
``train -> train`` and ``val -> calibration``. Note that the original used its
``'val'`` split for both threshold calibration and final reporting; new code
should call :mod:`crpa.data` directly and report on ``evaluation``.

The original file is available in git history:

    git show 7474c77:data.py
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from config import CFG
from crpa.config import DataConfig
from crpa.data import CALIBRATION, EVAL, TRAIN, Corpus

__all__ = ["init_data", "get_lm_batch", "make_needle_batch", "corpus",
           "tokenizer", "vocab_size", "train_data", "val_data"]

# Populated by init_data(), mirroring the original module globals.
corpus: Optional[Corpus] = None
tokenizer = None
vocab_size: Optional[int] = None
train_data: Optional[torch.Tensor] = None
val_data: Optional[torch.Tensor] = None

_LEGACY_ROLE = {"train": TRAIN, "val": CALIBRATION, "test": EVAL}


def data_config_from_cfg() -> DataConfig:
    """Build a :class:`~crpa.config.DataConfig` from the legacy ``CFG`` dict."""
    return DataConfig(
        filler_range=tuple(CFG["filler_range"]),
        key_range=tuple(CFG["key_range"]),
        val_range=tuple(CFG["val_range"]),
        needle_depths=tuple(CFG["needle_depths"]),
    )


def init_data(device: str = "cpu", seed: Optional[int] = None):
    """Load WikiText-2 and the needle generators. Mirrors the original signature."""
    global corpus, tokenizer, vocab_size, train_data, val_data
    corpus = Corpus(data_config_from_cfg(), seed=seed if seed is not None else CFG["seed"])
    corpus.load(verbose=True)
    corpus.assert_splits_disjoint(block_size=256)
    tokenizer = corpus.tokenizer
    vocab_size = corpus.vocab_size
    train_data = corpus._lm[TRAIN]
    val_data = corpus._lm[CALIBRATION]
    return tokenizer, vocab_size, train_data, val_data


def _require_corpus() -> Corpus:
    if corpus is None:
        raise RuntimeError("call init_data() before requesting batches")
    return corpus


def get_lm_batch(split: str, block_size: int, bs: Optional[int] = None,
                 device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Language-modelling batch. ``split`` accepts 'train', 'val' or 'test'."""
    return _require_corpus().lm_batch(
        _LEGACY_ROLE.get(split, split), block_size, bs or CFG["batch_size"], device
    )


def make_needle_batch(block_size: int, bs: Optional[int] = None, n_needles: int = 2,
                      needle_depth: Optional[float] = None, device: str = "cpu",
                      split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
    """Needle-in-Haystack batch.

    The original drew every role from one global RNG. This shim defaults to the
    training stream; pass ``split='val'`` or ``'test'`` for the calibration and
    evaluation streams respectively.
    """
    return _require_corpus().needle_batch(
        _LEGACY_ROLE.get(split, split), block_size, bs or CFG["batch_size"],
        n_needles, needle_depth, device,
    )
