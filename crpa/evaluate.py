"""
crpa.evaluate - split-aware evaluation.

Every function here takes an explicit split role. Final numbers are reported on
:data:`crpa.data.EVAL`, which nothing else in the pipeline reads.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from crpa.config import ExperimentConfig
from crpa.data import EVAL, Corpus
from crpa.metrics import mean_pairwise_overlap, mean_pairwise_overlap_sparse
from crpa.model import GPT


@torch.no_grad()
def retrieval_accuracy(
    model: GPT,
    corpus: Corpus,
    block_size: int,
    device: str,
    role: str = EVAL,
    n_batches: int = 30,
    bs: int = 8,
    needle_depth: Optional[float] = None,
) -> float:
    """Top-1 Needle-in-Haystack accuracy, in percent.

    Compare against ``cfg.data.chance_accuracy`` (5.0% by default): a variant
    at or below chance has not learned retrieval, and differences between two
    such variants are not evidence about retrieval quality.
    """
    was_training = model.training
    model.eval()
    correct = total = 0
    try:
        corpus.needles[role].reset()
        for _ in range(n_batches):
            x, y = corpus.needle_batch(
                role, block_size, bs, needle_depth=needle_depth, device=device
            )
            # Only the final position is scored, so project only that one.
            # The full (B, T, vocab) projection is 6.6 GB at T=16384.
            logits, _ = model(x, last_only=True)
            correct += int((logits[:, -1, :].argmax(dim=-1) == y).sum().item())
            total += int(y.shape[0])
    finally:
        if was_training:
            model.train()
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def retrieval_by_depth(
    model: GPT,
    corpus: Corpus,
    block_size: int,
    device: str,
    depths: Sequence[float],
    role: str = EVAL,
    n_batches: int = 20,
    bs: int = 8,
) -> Dict[float, float]:
    """Retrieval accuracy as a function of where the needle sits."""
    return {
        float(d): retrieval_accuracy(
            model, corpus, block_size, device, role=role,
            n_batches=n_batches, bs=bs, needle_depth=d,
        )
        for d in depths
    }


@torch.no_grad()
def measure_overlap(
    model: GPT,
    corpus: Corpus,
    cfg: ExperimentConfig,
    device: str,
    role: str = EVAL,
    n_batches: int = 12,
    bs: int = 4,
    seed: int = 0,
    window: Optional[int] = None,
) -> float:
    """Mean within-partition attention overlap - the *realized* overlap.

    This is the quantity the matched-overlap sweep matches on. Matching on the
    configured ``lambda_red`` instead would compare two different structural
    budgets and attribute the difference to the selection criterion.

    Args:
        window: retain attention for only this many trailing query rows, in
            gathered form. Required above a few thousand tokens: a dense
            capture is 12.9 GB per layer at T=16384 with 12 heads. Defaults to
            the dense path, which is exact and fine at 512.
    """
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    values: List[float] = []
    T = cfg.model.block_size
    capture_window = (max(0, T - window), T) if window else None
    try:
        for _ in range(n_batches):
            x, _ = corpus.lm_batch(role, T, bs, device)
            if capture_window is None:
                with model.capture_probabilities(True):
                    model(x, last_only=True)
                    for probs in model.attention_probabilities():
                        if probs is None:
                            continue
                        values.append(mean_pairwise_overlap(
                            probs.mean(dim=(0, 1)),
                            cfg.model.overlap_rho, cfg.model.partition_size,
                            n_samples=8, rng=rng,
                        ))
            else:
                # Reduce each captured layer to a scalar before capturing the
                # next. Retaining every layer's gathered window simultaneously
                # is the diagnostic allocation that exhausted 80 GiB.
                for depth in range(len(model.blocks)):
                    with model.capture_probabilities(
                            True, window=capture_window, layers=[depth]):
                        model(x, last_only=True)
                    sparse = model.sparse_attention_probabilities()[depth]
                    if sparse is not None:
                        values.append(mean_pairwise_overlap_sparse(
                            sparse, cfg.model.overlap_rho,
                            cfg.model.partition_size, n_samples=8, rng=rng,
                        ))
                    model.blocks[depth].attn._sparse_probs = None
                    del sparse
    finally:
        if was_training:
            model.train()
    return float(np.mean(values)) if values else 0.0


@torch.no_grad()
def language_model_loss(
    model: GPT,
    corpus: Corpus,
    block_size: int,
    device: str,
    role: str = EVAL,
    n_batches: int = 20,
    bs: int = 8,
    loss_chunk: int = 2048,
) -> float:
    """Mean next-token cross-entropy on one split.

    ``loss_chunk`` bounds the logits materialised at once, so this is usable at
    long context where the full projection would not fit.
    """
    was_training = model.training
    model.eval()
    losses = []
    try:
        for _ in range(n_batches):
            x, y = corpus.lm_batch(role, block_size, bs, device)
            _, loss = model(x, y, loss_chunk=loss_chunk)
            losses.append(float(loss.item()))
    finally:
        if was_training:
            model.train()
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def routing_diagnostics(
    model: GPT, corpus: Corpus, block_size: int, device: str,
    role: str = EVAL, n_batches: int = 20, bs: int = 8,
) -> Dict[str, float]:
    """Router utilisation statistics.

    Fixes a bug in the original: it fed the *token embedding* into every
    layer's router rather than that layer's actual input, so the numbers it
    produced described nothing. Every row of the published Table 5 was
    identical (entropy ln 4, load error 0) as a result.

    Routing remains a secondary, largely negative result and is reported as
    such rather than as a contribution.
    """
    was_training = model.training
    model.eval()
    per_layer: List[torch.Tensor] = []
    try:
        for _ in range(n_batches):
            x, _ = corpus.lm_batch(role, block_size, bs, device)
            # Re-run the stack, capturing each layer's genuine input.
            h = model.tok_emb(x)
            if model.pos_emb is not None:
                h = h + model.pos_emb(torch.arange(x.shape[1], device=x.device))
            h = model.drop(h)
            for blk in model.blocks:
                if hasattr(blk.attn, "router"):
                    soft, _ = blk.attn.router(blk.ln1(h))
                    per_layer.append(soft.detach().reshape(-1, soft.shape[-1]).cpu())
                out, _, _ = blk.attn(blk.ln1(h))
                h = h + out
                h = h + blk.ffn(blk.ln2(h))
    finally:
        if was_training:
            model.train()

    if not per_layer:
        return {"empty_partition_pct": float("nan"), "routing_entropy": float("nan"),
                "load_error": float("nan"), "n_partitions": 0}

    from crpa.metrics import pinned_at_extremum

    S = torch.cat(per_layer)
    avg = S.mean(0)
    entropy = float(-(S * (S + 1e-9).log()).sum(-1).mean().item())
    max_entropy = float(math.log(S.shape[-1]))
    pinned = pinned_at_extremum(entropy, 0.0, max_entropy)
    return {
        "empty_partition_pct": float((avg < 0.01).float().mean().item() * 100),
        "routing_entropy": entropy,
        "load_error": float(avg.std().item()),
        "n_partitions": int(S.shape[-1]),
        "max_entropy": max_entropy,
        # Uniform routing maximises entropy. If it is pinned there the router
        # is doing nothing and the number is not a finding.
        "entropy_pinned_at_uniform": pinned["at_max"],
        "routing_note": pinned["note"],
    }


#: Above this context length, counting permitted edges by materialising the
#: (T, T) mask costs more memory than the measurement is worth (4.3 GB of
#: booleans at 65536), so the analytic upper bound is reported instead.
MAX_EXACT_EDGE_COUNT_T = 8192


def sparsity_report(model: GPT, block_size: int) -> Dict[str, float]:
    """Permitted attention entries versus the dense causal upper bound."""
    theoretical = block_size * (block_size + 1) // 2
    if block_size > MAX_EXACT_EDGE_COUNT_T:
        from crpa.kvcache import attention_edge_counts

        bound = attention_edge_counts(model.cfg, block_size)
        return {
            "theoretical_causal_edges": theoretical,
            "actual_edges": bound["crpa_edges_upper_bound"],
            "sparsity_ratio": bound["sparsity_ratio_upper_bound"],
            "edge_count_is_upper_bound": True,
        }
    counts = []
    for blk in model.blocks:
        structure = blk.attn._structure
        if structure is not None and structure.T == block_size:
            counts.append(structure.edge_count())
    if not counts:
        return {
            "theoretical_causal_edges": theoretical,
            "actual_edges": float("nan"),
            "sparsity_ratio": float("nan"),
        }
    actual = float(np.mean(counts))
    return {
        "theoretical_causal_edges": theoretical,
        "actual_edges": actual,
        "sparsity_ratio": actual / theoretical,
        "edge_count_is_upper_bound": False,
    }
