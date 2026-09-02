"""
crpa.train - the co-training loop and the redundancy gate.

Co-training
-----------
``ret_ratio`` of steps train the Needle-in-Haystack retrieval task and the rest
train language modelling on WikiText-2. Retrieval has to be learned during
training, not merely probed at evaluation time.

The gate
--------
Both regularised variants refresh their penalty set on the same cadence, from
the same candidate pool, with the same removal budget:

``crpa_naive``
    Rank the pool by structural overlap, take the top ``budget``.

``crpa_contribution``
    Measure behavioral contribution for every candidate in the pool, rank by
    ascending delta, take the lowest ``budget``.

Only the ranking criterion differs. The original implementation gave the two
variants different pool sizes, different refresh cadences and different warmup
behaviour, so any difference in outcome confounded "which edges were removed"
with "how many, and from when".

All gate decisions read the **calibration** split. Nothing here touches
evaluation data.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from crpa.config import ExperimentConfig
from crpa.data import CALIBRATION, TRAIN, Corpus
from crpa.intervention import (
    Candidate,
    candidates_to_penalty_pairs,
    make_needle_loss_fn,
    reachable_queries,
    sample_candidate_edges,
    sample_legacy_row_pairs,
    score_candidates,
    select_contribution_gated,
    select_naive,
)
from crpa.attention import relay_positions
from crpa.model import GPT


def make_lr_schedule(
    optimizer: torch.optim.Optimizer, warmup: int, total: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup into cosine decay."""

    def fn(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    corpus: Corpus,
    block_size: int,
    device: str,
    eval_iters: int,
    roles: Tuple[str, ...] = (TRAIN, CALIBRATION),
    bs: int = 8,
) -> Dict[str, float]:
    """Mean language-model loss on the requested splits."""
    was_training = model.training
    model.eval()
    out: Dict[str, float] = {}
    try:
        for role in roles:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = corpus.lm_batch(role, block_size, bs, device)
                _, loss = model(x, y, loss_chunk=2048)
                losses[k] = loss.item()
            out[role] = float(losses.mean().item())
    finally:
        if was_training:
            model.train()
    return out


@torch.no_grad()
def refresh_gate(
    model: GPT,
    cfg: ExperimentConfig,
    corpus: Corpus,
    block_size: int,
    device: str,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Recompute which interactions the redundancy penalty acts on.

    Returns diagnostics describing what the gate saw and chose, so a run's
    behaviour is auditable after the fact rather than inferred from its loss.
    """
    variant = model.variant
    if variant not in ("crpa_naive", "crpa_contribution"):
        return {}

    budget = cfg.contribution.n_pairs
    pool_size = max(budget * 3, budget + 4)
    legacy = cfg.contribution.mode == "legacy_rowpair"

    # Gate decisions are made on calibration data only.
    x, y = corpus.needle_batch(CALIBRATION, block_size, bs=8, device=device)
    loss_fn = make_needle_loss_fn(x, y)

    was_training = model.training
    model.eval()
    stats: Dict[str, object] = {}
    try:
        with model.frozen_structure():
            with model.capture_probabilities(True):
                model(x)
            probs = model.attention_probabilities()
            reach = None if legacy else reachable_queries(model, block_size)
            relays = relay_positions(block_size, cfg.model.n_relays)

            pool: List[Candidate] = []
            for depth in range(len(model.blocks)):
                if probs[depth] is None:
                    continue
                if legacy:
                    pool += sample_legacy_row_pairs(
                        probs[depth], depth, cfg.model.partition_size,
                        cfg.model.overlap_rho, pool_size, rng,
                        min_overlap=cfg.contribution.overlap_threshold,
                    )
                else:
                    pool += sample_candidate_edges(
                        probs[depth], depth, cfg.model.partition_size,
                        cfg.model.overlap_rho, pool_size, rng,
                        min_overlap=cfg.contribution.overlap_threshold,
                        reach=reach[depth] if reach else None,
                        exclude_queries=relays,
                    )

            if not pool:
                stats["pool_size"] = 0
                stats["selected"] = 0
                return stats

            # Cap the pool globally. Without this it grows as
            # layers x heads x pool_size, which both inflates the estimator's
            # cost and hands the two variants pools of different sizes.
            pool.sort(key=lambda c: -c.overlap)
            pool = pool[:pool_size]

            if variant == "crpa_naive":
                # Structural criterion only - no interventions are run, which
                # is the whole point of the baseline.
                selected = select_naive(pool, budget)
                stats["criterion"] = "structural_overlap"
            else:
                scored = score_candidates(
                    model, pool, loss_fn, eps=cfg.contribution.eps,
                    seed=cfg.train.seed, context_length=block_size,
                )
                if not scored:
                    stats["pool_size"] = len(pool)
                    stats["selected"] = 0
                    stats["note"] = "no candidate produced a measurable intervention"
                    return stats
                selected = select_contribution_gated(scored, budget)
                stats["criterion"] = "behavioral_contribution"
                deltas = [c.delta_loss for c in scored if math.isfinite(c.delta_loss)]
                if deltas:
                    stats["delta_mean"] = float(np.mean(deltas))
                    stats["delta_max"] = float(np.max(deltas))
                    stats["frac_below_eps"] = float(
                        np.mean([d <= cfg.contribution.eps for d in deltas])
                    )
                stats["n_scored"] = len(scored)

            # Route each selected edge to the layer that owns it.
            by_layer: Dict[int, List[Candidate]] = {}
            for cand in selected:
                by_layer.setdefault(cand.layer, []).append(cand)
            for depth, blk in enumerate(model.blocks):
                blk.attn.set_penalty_pairs(
                    candidates_to_penalty_pairs(by_layer.get(depth, []))
                )

            stats["pool_size"] = len(pool)
            stats["selected"] = len(selected)
            stats["mean_selected_overlap"] = float(
                np.mean([c.overlap for c in selected])
            ) if selected else float("nan")
    finally:
        if was_training:
            model.train()
    return stats


def train(
    model: GPT,
    cfg: ExperimentConfig,
    corpus: Corpus,
    device: str,
    max_iters: Optional[int] = None,
    verbose: bool = True,
    on_checkpoint: Optional[Callable[[int, GPT], None]] = None,
) -> Dict[str, object]:
    """Train one variant.

    Returns a history dict with validation-loss checkpoints and gate
    diagnostics. The model is modified in place.
    """
    max_iters = max_iters or cfg.train.max_iters
    block_size = cfg.model.block_size
    use_amp = "cuda" in str(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    sched = make_lr_schedule(opt, cfg.train.warmup_steps, max_iters)

    # Dedicated stream so gate sampling cannot be perturbed by, or perturb,
    # anything else in the run.
    gate_rng = np.random.default_rng(cfg.train.seed + 77)
    py_rng = random.Random(cfg.train.seed + 13)

    val_hist: List[float] = []
    step_hist: List[int] = []
    gate_hist: List[Dict[str, object]] = []

    gated = model.variant in ("crpa_naive", "crpa_contribution")
    if verbose:
        print(
            "  variant={} | {:.1f}M params | block={} | impl={}".format(
                model.variant, model.n_params() / 1e6, block_size,
                cfg.model.attention_impl,
            )
        )

    model.train()
    for step in range(max_iters):
        # Both regularised variants refresh on the same cadence and only after
        # the same warmup, so their comparison stays controlled.
        if (
            gated
            and step >= cfg.contribution.warmup_steps
            and step % cfg.contribution.interval == 0
        ):
            stats = refresh_gate(model, cfg, corpus, block_size, device, gate_rng)
            if stats:
                stats["step"] = step
                gate_hist.append(stats)

        with torch.autocast(
            device_type="cuda" if use_amp else "cpu",
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            if py_rng.random() < cfg.train.ret_ratio:
                x, y_ret = corpus.needle_batch(
                    TRAIN, block_size, cfg.train.batch_size, device=device
                )
                logits, _ = model(x, lambda_bal=cfg.train.lambda_bal,
                                  lambda_red=cfg.train.lambda_red)
                loss = F.cross_entropy(logits[:, -1, :], y_ret)
                # Auxiliary terms are folded in by GPT.forward only when
                # targets are supplied, so add them explicitly here.
                loss = loss + _aux_losses(model, cfg)
            else:
                x, y = corpus.lm_batch(TRAIN, block_size, cfg.train.batch_size, device)
                _, loss = model(
                    x, y, lambda_bal=cfg.train.lambda_bal,
                    lambda_red=cfg.train.lambda_red,
                )

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % cfg.train.eval_interval == 0 or step == max_iters - 1:
            L = estimate_loss(
                model, corpus, block_size, device, cfg.train.eval_iters
            )
            val_hist.append(L[CALIBRATION])
            step_hist.append(step)
            if verbose:
                print(
                    "    step {:>5}  train={:.4f}  calib={:.4f}".format(
                        step, L[TRAIN], L[CALIBRATION]
                    )
                )

        if (
            on_checkpoint is not None
            and cfg.train.checkpoint_every > 0
            and step > 0
            and step % cfg.train.checkpoint_every == 0
        ):
            on_checkpoint(step, model)

    return {
        "val_hist": val_hist,
        "step_hist": step_hist,
        "gate_hist": gate_hist,
        "final_calibration_loss": val_hist[-1] if val_hist else float("nan"),
    }


def _aux_losses(model: GPT, cfg: ExperimentConfig) -> torch.Tensor:
    """Load-balance and redundancy terms for the retrieval branch.

    ``GPT.forward`` only folds auxiliary losses into its return value when
    targets are supplied. The retrieval branch computes its own last-token
    loss, so the auxiliary terms are recomputed from the values each attention
    layer cached during the forward pass.
    """
    device = next(model.parameters()).device
    if model._last_aux is None:
        return torch.zeros((), device=device)
    lb, lr = model._last_aux
    return cfg.train.lambda_bal * lb + cfg.train.lambda_red * lr
