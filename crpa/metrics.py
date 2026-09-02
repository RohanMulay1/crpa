"""
crpa.metrics - structural overlap statistics, evaluation metrics, and the
statistics used to summarise them.

The distinction this module exists to keep clean:

*Structural overlap* is a geometric property of attention supports. It is
computed here from attention probabilities alone, with no reference to model
behaviour.

*Behavioral contribution* is measured in :mod:`crpa.intervention` by actually
removing something and observing the loss change.

Keeping them in separate modules is deliberate: the research question is
whether the first predicts the second, so nothing in the overlap computation
may consult a measured delta.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
#  Structural overlap
# ---------------------------------------------------------------------------

def top_p_support_mask(A: torch.Tensor, rho: float) -> torch.Tensor:
    """Boolean mask of the smallest key set covering ``rho`` of each row's mass.

    Vectorised equivalent of the original per-row Python implementation: rank
    ``r`` is retained iff the cumulative mass *strictly before* it is below
    ``rho``, which always keeps at least the largest entry.

    Args:
        A: ``(..., T)`` attention probabilities; rows should sum to 1.
        rho: mass fraction defining the support.

    Returns:
        Boolean tensor shaped like ``A``.
    """
    sorted_a, idx = A.sort(dim=-1, descending=True)
    cum_before = sorted_a.cumsum(dim=-1) - sorted_a
    keep_sorted = cum_before < rho
    return torch.zeros_like(A, dtype=torch.bool).scatter(-1, idx, keep_sorted)


def jaccard_matrix(support: torch.Tensor) -> torch.Tensor:
    """All-pairs Jaccard similarity between support sets.

    Args:
        support: ``(T, T)`` boolean support mask, one row per query.

    Returns:
        ``(T, T)`` float matrix. O(T^2) memory - use
        :func:`jaccard_pairs` above a few thousand tokens.
    """
    S = support.float()
    inter = S @ S.T
    sizes = S.sum(dim=-1)
    union = sizes.unsqueeze(1) + sizes.unsqueeze(0) - inter
    return inter / union.clamp_min(1.0)


def jaccard_pairs(
    support: torch.Tensor, idx_i: torch.Tensor, idx_j: torch.Tensor
) -> torch.Tensor:
    """Jaccard similarity for specific query pairs, without the full matrix."""
    Si = support[idx_i]
    Sj = support[idx_j]
    inter = (Si & Sj).sum(dim=-1).float()
    union = (Si | Sj).sum(dim=-1).float()
    return inter / union.clamp_min(1.0)


def edge_structural_overlap(
    support: torch.Tensor,
    query: int,
    key: int,
    partition_size: int,
    exclude: Optional[Sequence[int]] = None,
) -> float:
    """Structural overlap of the edge ``query -> key``.

    Defined as the largest support-Jaccard between ``query`` and any *other*
    query in the same partition that also attends to ``key``:

        overlap(i -> j) = max over i' != i, i' in P(i), j in S(i')
                              of  Jaccard(S(i), S(i'))

    Reading: the edge is structurally redundant to the extent that key ``j`` is
    already covered by another query whose attention support resembles ``i``'s.
    Zero means no other query in the partition reaches this key at all, so the
    edge is structurally unique regardless of whether it matters behaviorally.

    This is the statistic the repaired pipeline uses, because it refers to the
    *same object* the intervention removes. The original code scored pairs of
    query rows but then intervened on an edge, so the two were never comparable.

    ``exclude`` drops query positions from the comparison. Relay positions are
    always passed here: a relay attends causally to everything by construction,
    so its support overlaps every local query as an artifact of the mask rather
    than as evidence of redundancy. Excluding them also makes this agree
    exactly with :func:`edge_structural_overlap_sparse`, which cannot represent
    a relay's support at all.
    """
    T = support.shape[0]
    start = (query // partition_size) * partition_size
    stop = min(start + partition_size, T)

    others = torch.arange(start, stop, device=support.device)
    others = others[(others != query) & support[start:stop, key]]
    if exclude is not None and len(exclude):
        drop = torch.tensor(list(exclude), device=support.device, dtype=others.dtype)
        others = others[~torch.isin(others, drop)]
    if others.numel() == 0:
        return 0.0

    Si = support[query].unsqueeze(0)
    So = support[others]
    inter = (Si & So).sum(dim=-1).float()
    union = (Si | So).sum(dim=-1).float()
    return float((inter / union.clamp_min(1.0)).max().item())


def support_key_sets(
    probs_2d: torch.Tensor, key_idx: torch.Tensor, rho: float
) -> List[set]:
    """Top-rho support of each query, as sets of *absolute* key indices.

    Operates on the gathered representation, so cost is independent of context
    length. ``key_idx`` maps each slot to an absolute key, with ``-1`` marking
    an unused slot.

    Args:
        probs_2d: ``(n_q, C)`` probabilities over each query's key list.
        key_idx: ``(n_q, C)`` absolute key index per slot.
        rho: mass fraction defining the support.
    """
    keep = top_p_support_mask(probs_2d, rho) & (key_idx >= 0)
    return [
        set(key_idx[row][keep[row]].tolist())
        for row in range(probs_2d.shape[0])
    ]


def jaccard_sets(a: set, b: set) -> float:
    """Jaccard similarity of two key sets."""
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def edge_structural_overlap_sparse(
    supports: Sequence[set],
    query_lo: int,
    local_query: int,
    key: int,
    partition_size: int,
    is_relay: Optional[torch.Tensor] = None,
) -> float:
    """Structural overlap of an edge, from gathered supports.

    Same definition as :func:`edge_structural_overlap`: the largest
    support-Jaccard between this query and any other query in the same
    partition that also attends to ``key``. Only the representation differs.

    ``is_relay`` marks rows whose true support spans every causal key. They are
    excluded, matching ``edge_structural_overlap(..., exclude=relay_positions)``
    exactly. Without this the two computations disagree, because the gathered
    form holds only a truncated view of a relay's support.
    """
    absolute = query_lo + local_query
    start = (absolute // partition_size) * partition_size
    stop = start + partition_size

    mine = supports[local_query]
    best = 0.0
    for other in range(len(supports)):
        if other == local_query:
            continue
        other_abs = query_lo + other
        if not (start <= other_abs < stop):
            continue
        if is_relay is not None and bool(is_relay[other]):
            continue
        if key not in supports[other]:
            continue
        best = max(best, jaccard_sets(mine, supports[other]))
    return best


def mean_pairwise_overlap_sparse(
    sparse,
    rho: float,
    partition_size: int,
    n_samples: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Within-partition row-pair Jaccard, from gathered probabilities.

    Same statistic as :func:`mean_pairwise_overlap`, computed on the
    representation that is affordable at long context. Relay rows are excluded
    because the gathered form does not represent their full support.
    """
    if sparse is None or sparse.n_queries < 2:
        return 0.0
    rng = rng or np.random.default_rng(0)
    n_heads = int(sparse.probs.shape[1])
    values: List[float] = []

    for head in range(n_heads):
        supports = support_key_sets(sparse.head(head), sparse.key_idx, rho)
        by_partition: Dict[int, List[int]] = {}
        for local in range(sparse.n_queries):
            if bool(sparse.is_relay[local]) or not supports[local]:
                continue
            block = (sparse.query_lo + local) // partition_size
            by_partition.setdefault(block, []).append(local)

        for members in by_partition.values():
            if len(members) < 2:
                continue
            for _ in range(n_samples):
                i = members[int(rng.integers(0, len(members)))]
                j = members[int(rng.integers(0, len(members)))]
                if i != j:
                    values.append(jaccard_sets(supports[i], supports[j]))
    return float(np.mean(values)) if values else 0.0


def mean_pairwise_overlap(
    A: torch.Tensor,
    rho: float,
    partition_size: int,
    n_samples: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Average within-partition row-pair Jaccard - the headline overlap number.

    Kept identical in definition to the original ``measure_overlap`` so the
    reported quantity stays comparable with previously published tables. Only
    the sampling is now driven by an explicit generator.

    Cross-partition pairs are excluded because they share only relay tokens by
    construction, so including them would measure the mask, not the model.
    """
    rng = rng or np.random.default_rng(0)
    support = top_p_support_mask(A, rho)
    T = A.shape[0]

    idx_i: List[int] = []
    idx_j: List[int] = []
    for p in range(math.ceil(T / partition_size)):
        start = p * partition_size
        stop = min(start + partition_size, T)
        if stop - start < 2:
            continue
        for _ in range(n_samples):
            i, j = int(rng.integers(start, stop)), int(rng.integers(start, stop))
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    if not idx_i:
        return 0.0
    vals = jaccard_pairs(
        support,
        torch.tensor(idx_i, device=A.device),
        torch.tensor(idx_j, device=A.device),
    )
    return float(vals.mean().item())


# ---------------------------------------------------------------------------
#  Summary statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    With only three seeds a bootstrap CI is wide and should be read as such;
    it is reported because it is honest about that, not because three seeds
    make a tight interval.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_resamples, arr.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)))


def summarise(values: Sequence[float], confidence: float = 0.95) -> Dict[str, float]:
    """Mean, standard deviation, n and a bootstrap CI."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0,
                "ci_low": float("nan"), "ci_high": float("nan")}
    lo, hi = bootstrap_ci(arr, confidence=confidence)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
        "ci_low": lo,
        "ci_high": hi,
    }


def correlations(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Pearson and Spearman correlation with p-values and a Fisher-z CI.

    A weak correlation here means overlap is a weak *linear* or *monotone*
    predictor of contribution on this sample. It is not evidence of
    independence, and must not be described as such.
    """
    from scipy import stats

    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    finite = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[finite], ya[finite]
    n = int(xa.size)
    out: Dict[str, float] = {"n": n}
    if n < 3 or np.allclose(xa, xa[0]) or np.allclose(ya, ya[0]):
        out.update(pearson_r=float("nan"), pearson_p=float("nan"),
                   spearman_r=float("nan"), spearman_p=float("nan"),
                   pearson_ci_low=float("nan"), pearson_ci_high=float("nan"))
        return out

    pr = stats.pearsonr(xa, ya)
    sr = stats.spearmanr(xa, ya)
    out["pearson_r"] = float(pr[0])
    out["pearson_p"] = float(pr[1])
    out["spearman_r"] = float(sr[0])
    out["spearman_p"] = float(sr[1])

    # Fisher z interval for the Pearson coefficient.
    z = np.arctanh(np.clip(pr[0], -0.999999, 0.999999))
    se = 1.0 / math.sqrt(max(n - 3, 1))
    out["pearson_ci_low"] = float(np.tanh(z - 1.96 * se))
    out["pearson_ci_high"] = float(np.tanh(z + 1.96 * se))
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation, or NaN when it is undefined."""
    from scipy import stats

    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    if xa.size < 2 or np.allclose(xa, xa[0]) or np.allclose(ya, ya[0]):
        return float("nan")
    return float(stats.spearmanr(xa, ya)[0])


def top_k_agreement(rank_a: Sequence[int], rank_b: Sequence[int], k: int) -> float:
    """Fraction of the top ``k`` items shared by two rankings."""
    if k <= 0:
        return float("nan")
    a = set(list(rank_a)[:k])
    b = set(list(rank_b)[:k])
    return len(a & b) / float(k)


def perplexity(loss: float) -> float:
    """Perplexity from a mean cross-entropy, guarded against overflow."""
    if not math.isfinite(loss):
        return float("nan")
    return math.exp(min(loss, 700.0))
