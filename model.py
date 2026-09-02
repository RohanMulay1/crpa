"""
model.py - backwards-compatibility shim.

The implementation moved to :mod:`crpa.model` and :mod:`crpa.attention`. This
module preserves the original import surface and call signatures so existing
code and checkpoints keep working:

    from model import GPT
    m = GPT('crpa_causal', 512, 50257, 'cpu')

State-dict keys are unchanged, so checkpoints saved by the original
implementation load into the new model with ``strict=True``.

The original file is available in git history:

    git show 7474c77:model.py
"""

from __future__ import annotations

from typing import Optional

import torch

from config import CFG
from crpa.attention import (  # noqa: F401  (re-exported for compatibility)
    DifferentiableRouter,
    build_crpa_mask,
)
from crpa.config import ModelConfig, resolve_variant
from crpa.model import FFN, Block, CRPAAttention  # noqa: F401
from crpa.model import GPT as _GPT

__all__ = ["GPT", "Block", "FFN", "CRPAAttention", "DifferentiableRouter",
           "build_crpa_mask", "model_config_from_cfg"]


def model_config_from_cfg(block_size: int, vocab_size: int) -> ModelConfig:
    """Build a :class:`~crpa.config.ModelConfig` from the legacy ``CFG`` dict."""
    return ModelConfig(
        n_embd=CFG["n_embd"],
        n_head=CFG["n_head"],
        n_layer=CFG["n_layer"],
        dropout=CFG["dropout"],
        vocab_size=vocab_size,
        block_size=block_size,
        position="learned",
        partition_size=CFG["partition_size"],
        n_relays=CFG["n_relays"],
        cross_k=CFG["cross_k"],
        route_temp=CFG["route_temp"],
        overlap_rho=CFG["overlap_rho"],
        attention_impl="dense_masked",
    )


class GPT(_GPT):
    """Original positional signature: ``GPT(variant, block_size, vocab_size, device)``.

    ``crpa_causal`` resolves to ``crpa_contribution``; see
    :mod:`crpa.intervention` for what changed in the mechanism it names.
    """

    def __init__(
        self,
        variant: str,
        block_size: int,
        vocab_size: int,
        device: "str | torch.device" = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            model_config_from_cfg(block_size, vocab_size),
            resolve_variant(variant),
            seed=seed if seed is not None else CFG.get("seed"),
        )
