"""
crpa.intervention - behavioral contribution measured by actually removing things.

The unit of analysis
--------------------
An **edge** is ``e = (layer, head, query i, key j)`` with ``j <= i`` and
``j`` present in ``Omega(i)``. Intervening on ``e`` sets that single score to
``-inf`` before the softmax, for that head in that layer only. The surviving
entries of row ``i`` renormalise to 1, so the removed interaction's probability
mass is redistributed rather than simply deleted.

Behavioral contribution is then

    Delta(e) = L(M \\ e) - L(M)

evaluated on the **calibration** split with the sparsity structure frozen.

Why this differs from the original implementation
-------------------------------------------------
The original sampled *pairs of query rows* ``(i, j)`` scored by support Jaccard,
then intervened by masking the *edge* ``i -> j``. Those are different objects,
so the delta did not measure the effect of removing the overlapping
interaction. Worse, ``i`` and ``j`` were drawn independently within a partition
and the mask is causal, so every sample with ``i < j`` masked an entry that was
already ``False``. Those are no-ops with delta identically zero, and since the
classifier kept anything with ``delta <= eps`` they were all labelled redundant.

``ContributionConfig.mode='legacy_rowpair'`` reproduces that behaviour so the
previously published numbers remain obtainable. ``mode='edge'`` is the repaired
default.

Every measurement here refuses to report a delta unless the forward pass
confirms the intervention actually removed score mass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from crpa.metrics import edge_structural_overlap, jaccard_pairs, top_p_support_mask


@dataclass(frozen=True)
class Edge:
    """A single attention interaction.

    ``head=None`` addresses every head at that layer, which is what the
    original global mask did; the repaired pipeline always names a head.
    """

    layer: int
    head: Optional[int]
    query: int
    key: int

    def as_triple(self) -> Tuple[Optional[int], int, int]:
        return (self.head, self.query, self.key)


@dataclass
class InterventionPlan:
    """A set of edges to remove in one forward pass."""

    edges: Tuple[Edge, ...] = ()

    def for_layer(self, layer: int) -> List[Tuple[Optional[int], int, int]]:
        return [e.as_triple() for e in self.edges if e.layer == layer]

    @classmethod
    def single(cls, edge: Edge) -> "InterventionPlan":
        return cls(edges=(edge,))

    @classmethod
    def of(cls, edges: Sequence[Edge]) -> "InterventionPlan":
        return cls(edges=tuple(edges))

    def __len__(self) -> int:
        return len(self.edges)


@dataclass
class Candidate:
    """One scored candidate edge, before and after intervention."""

    layer: int
    head: Optional[int]
    query: int
    key: int
    overlap: float
    row_pair_overlap: float = float("nan")
    baseline_loss: float = float("nan")
    intervened_loss: float = float("nan")
    delta_loss: float = float("nan")
    retrieval_effect: float = float("nan")
    n_intervened: int = 0
    suppressible: Optional[bool] = None
    seed: Optional[int] = None
    context_length: Optional[int] = None

    def to_row(self) -> Dict[str, object]:
        return {
            "layer": self.layer,
            "head": self.head if self.head is not None else -1,
            "query": self.query,
            "key": self.key,
            "overlap": self.overlap,
            "row_pair_overlap": self.row_pair_overlap,
            "baseline_loss": self.baseline_loss,
            "intervened_loss": self.intervened_loss,
            "delta_loss": self.delta_loss,
            "retrieval_effect": self.retrieval_effect,
            "n_intervened": self.n_intervened,
            "suppressible": self.suppressible,
            "seed": self.seed,
            "context_length": self.context_length,
        }

    def to_edge(self) -> Edge:
        return Edge(self.layer, self.head, self.query, self.key)


class InterventionError(RuntimeError):
    """Raised when an intervention did not remove anything.

    Reporting ``delta = 0.0`` in that case would silently turn a no-op into a
    scientific claim of dispensability, which is exactly the failure mode the
    original implementation had.
    """


# ---------------------------------------------------------------------------
#  Reachability
# ---------------------------------------------------------------------------

def reachable_queries(
    model: torch.nn.Module, T: int, target: Optional[int] = None
) -> List[torch.Tensor]:
    """Query positions whose layer output can influence ``target``.

    When contribution is measured with a *last-token* loss - which is what the
    needle retrieval task uses - most edges cannot affect the measurement at
    all. Under a sparse mask, an early-position edge in a lower layer often has
    no causal path to the final position, so its delta is exactly zero as a
    matter of graph structure, not behaviour.

    Scoring such edges and then classifying ``delta <= eps`` as "suppressible"
    would repeat the original implementation's central error through a
    different mechanism: an absence of any possible effect read as evidence of
    dispensability. Candidates are therefore filtered to reachable queries, and
    the filter is recorded with the results.

    Propagation accounts for the residual stream: if position ``i`` matters
    after layer ``L``, then before layer ``L`` both ``i`` itself and every key
    ``i`` attends to matter.

    Args:
        model: a :class:`crpa.model.GPT` that has already run a forward pass,
            so its masks exist.
        T: sequence length.
        target: position whose prediction defines the loss; defaults to the
            last position.

    Returns:
        One boolean ``(T,)`` tensor per layer: the queries at that layer whose
        outputs can influence ``target``.
    """
    device = next(model.parameters()).device
    target = T - 1 if target is None else target
    reach = torch.zeros(T, dtype=torch.bool, device=device)
    reach[target] = True

    per_layer: List[Optional[torch.Tensor]] = [None] * len(model.blocks)
    for depth in reversed(range(len(model.blocks))):
        per_layer[depth] = reach.clone()
        reach = reach | model.blocks[depth].attn.attended_keys(reach)
    return [r for r in per_layer if r is not None]


def filter_reachable(
    candidates: Sequence[Candidate], reach: Sequence[torch.Tensor]
) -> List[Candidate]:
    """Drop candidates that cannot influence the measured loss by construction."""
    out: List[Candidate] = []
    for cand in candidates:
        if cand.layer < len(reach) and bool(reach[cand.layer][cand.query]):
            out.append(cand)
    return out


# ---------------------------------------------------------------------------
#  Candidate selection
# ---------------------------------------------------------------------------

def sample_candidate_edges(
    probs: torch.Tensor,
    layer: int,
    partition_size: int,
    rho: float,
    n_candidates: int,
    rng: np.random.Generator,
    min_overlap: float = 0.0,
    per_head: bool = True,
    reach: Optional[torch.Tensor] = None,
    exclude_queries: Optional[Sequence[int]] = None,
) -> List[Candidate]:
    """Sample edges that are present in the mask and score their overlap.

    Args:
        probs: ``(B, H, T, T)`` attention probabilities from one layer.
        layer: this layer's index.
        partition_size: window size defining P(i).
        rho: support mass fraction.
        n_candidates: how many edges to return, highest overlap first.
        rng: explicit generator, so selection is reproducible.
        min_overlap: discard candidates below this structural overlap.
        per_head: score each head separately (the repaired behaviour) rather
            than averaging heads together, which destroys head structure.
        reach: optional ``(T,)`` mask of queries that can influence the loss;
            see :func:`reachable_queries`. Sampling directly from it is far
            more efficient than sampling everywhere and discarding.
        exclude_queries: positions to drop from the overlap comparison.
            Relay positions belong here: a relay attends causally to
            everything, so its support overlaps every local query as an
            artifact of the mask. Excluding them also makes this agree
            exactly with the gathered-form computation, which cannot
            represent a relay's support.

    Returns:
        Candidates sorted by descending structural overlap.
    """
    if probs is None:
        return []
    B, H, T, _ = probs.shape
    heads = range(H) if per_head else [None]
    out: List[Candidate] = []

    if reach is not None:
        query_pool = reach.nonzero(as_tuple=True)[0].tolist()
        if not query_pool:
            return []
    else:
        query_pool = list(range(T))

    # Oversample, because a randomly drawn (query, key) is often not in the mask.
    attempts = max(n_candidates * 8, 64)

    for head in heads:
        A = probs[:, head].mean(dim=0) if head is not None else probs.mean(dim=(0, 1))
        support = top_p_support_mask(A, rho)
        seen: set = set()
        scored: List[Candidate] = []

        for _ in range(attempts):
            i = int(query_pool[int(rng.integers(0, len(query_pool)))])
            # Only keys inside the query's own support are meaningful
            # candidates: an edge carrying negligible mass is trivially
            # removable and would bias the sample.
            row = support[i].nonzero(as_tuple=True)[0]
            row = row[row <= i]
            # A query with a single permitted key (position 0 always has one)
            # cannot be intervened on: removing it would empty the row rather
            # than remove one interaction.
            if row.numel() < 2:
                continue
            j = int(row[int(rng.integers(0, row.numel()))])
            if (i, j) in seen:
                continue
            seen.add((i, j))
            ov = edge_structural_overlap(support, i, j, partition_size,
                                         exclude=exclude_queries)
            if ov < min_overlap:
                continue
            scored.append(Candidate(layer=layer, head=head, query=i, key=j, overlap=ov))

        scored.sort(key=lambda c: -c.overlap)
        out.extend(scored[:n_candidates])

    out.sort(key=lambda c: -c.overlap)
    return out[:n_candidates] if not per_head else out


def sample_candidate_edges_sparse(
    sparse,
    layer: int,
    partition_size: int,
    rho: float,
    n_candidates: int,
    rng: np.random.Generator,
    min_overlap: float = 0.0,
    reach: Optional[torch.Tensor] = None,
) -> List[Candidate]:
    """Sample candidate edges from gathered attention probabilities.

    The long-context counterpart of :func:`sample_candidate_edges`. Identical
    in definition; it just reads the representation that is affordable above a
    few thousand tokens.

    Relay rows are skipped: their true support spans every causal key, which
    the gathered form does not represent, so their overlap is not computable
    here and a wrong value would be worse than an omission.
    """
    from crpa.metrics import edge_structural_overlap_sparse, support_key_sets

    if sparse is None or sparse.n_queries == 0:
        return []

    n_heads = int(sparse.probs.shape[1])
    out: List[Candidate] = []
    attempts = max(n_candidates * 8, 64)

    for head in range(n_heads):
        supports = support_key_sets(sparse.head(head), sparse.key_idx, rho)

        eligible = [
            local for local in range(sparse.n_queries)
            if not bool(sparse.is_relay[local])
            and supports[local]
            and (reach is None or bool(reach[sparse.query_lo + local]))
        ]
        if not eligible:
            continue

        seen: set = set()
        scored: List[Candidate] = []
        for _ in range(attempts):
            local = eligible[int(rng.integers(0, len(eligible)))]
            absolute = sparse.query_lo + local
            keys = sorted(k for k in supports[local] if k <= absolute)
            if len(keys) < 2:
                continue
            key = keys[int(rng.integers(0, len(keys)))]
            if (absolute, key) in seen:
                continue
            seen.add((absolute, key))
            overlap = edge_structural_overlap_sparse(
                supports, sparse.query_lo, local, key, partition_size,
                is_relay=sparse.is_relay,
            )
            if overlap < min_overlap:
                continue
            scored.append(Candidate(layer=layer, head=head, query=absolute,
                                    key=key, overlap=overlap))
        scored.sort(key=lambda c: -c.overlap)
        out.extend(scored[:n_candidates])

    out.sort(key=lambda c: -c.overlap)
    return out


def sample_legacy_row_pairs(
    probs: torch.Tensor,
    layer: int,
    partition_size: int,
    rho: float,
    n_pairs: int,
    rng: np.random.Generator,
    min_overlap: float = 0.2,
) -> List[Candidate]:
    """Reproduce the original candidate sampler, for ``mode='legacy_rowpair'``.

    Samples *pairs of query rows* within a partition, scored by support
    Jaccard, and reports them as if they were edges ``(i -> j)``. Retained only
    so previously published numbers can be regenerated; see the module
    docstring for why the resulting deltas are not interpretable.
    """
    if probs is None:
        return []
    A = probs.mean(dim=(0, 1))
    support = top_p_support_mask(A, rho)
    T = A.shape[0]
    pairs: List[Candidate] = []

    for p in range(math.ceil(T / partition_size)):
        start = p * partition_size
        stop = min(start + partition_size, T)
        if stop - start < 2:
            continue
        for _ in range(n_pairs * 2):
            i, j = int(rng.integers(start, stop)), int(rng.integers(start, stop))
            if i == j:
                continue
            ov = float(
                jaccard_pairs(
                    support,
                    torch.tensor([i], device=A.device),
                    torch.tensor([j], device=A.device),
                )[0].item()
            )
            if ov > min_overlap:
                pairs.append(
                    Candidate(layer=layer, head=None, query=i, key=j,
                              overlap=ov, row_pair_overlap=ov)
                )
    pairs.sort(key=lambda c: -c.overlap)
    return pairs[:n_pairs]


# ---------------------------------------------------------------------------
#  Measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_delta(
    model: torch.nn.Module,
    edges: Sequence[Edge],
    loss_fn: Callable[[torch.nn.Module, Optional[InterventionPlan]], float],
    baseline: Optional[float] = None,
    strict: bool = True,
) -> Tuple[float, float, int]:
    """Measure the behavioral effect of removing ``edges``.

    The structure is frozen for the whole measurement, so the baseline and the
    intervened pass share one sparsity pattern.

    Args:
        model: the model under test.
        edges: edges to remove.
        loss_fn: maps ``(model, plan)`` to a scalar loss. Supplying this rather
            than a batch lets callers choose the retrieval loss, the LM loss, or
            anything else, without this module knowing about data.
        baseline: reuse a previously measured baseline instead of recomputing.
        strict: raise :class:`InterventionError` when nothing was removed.

    Returns:
        ``(baseline_loss, intervened_loss, n_removed)``.
    """
    was_training = model.training
    model.eval()
    try:
        with model.frozen_structure():
            base = loss_fn(model, None) if baseline is None else baseline
            plan = InterventionPlan.of(edges)
            intervened = loss_fn(model, plan)
            removed = model.intervened_count()
    finally:
        if was_training:
            model.train()

    if removed == 0 and strict:
        raise InterventionError(
            "intervention on {} edge(s) removed no attention mass. The edge is "
            "absent from the mask, so its delta is not a measurement of "
            "dispensability.".format(len(edges))
        )
    return base, intervened, removed


@torch.no_grad()
def score_candidates(
    model: torch.nn.Module,
    candidates: Sequence[Candidate],
    loss_fn: Callable[[torch.nn.Module, Optional[InterventionPlan]], float],
    eps: float,
    seed: Optional[int] = None,
    context_length: Optional[int] = None,
    skip_no_ops: bool = True,
) -> List[Candidate]:
    """Fill in the behavioral fields of each candidate.

    Candidates whose intervention removes nothing are dropped when
    ``skip_no_ops`` is set. They are not recorded as zero-delta observations,
    because a no-op is an absence of evidence, not evidence of dispensability.
    """
    scored: List[Candidate] = []
    was_training = model.training
    model.eval()
    try:
        with model.frozen_structure():
            baseline = loss_fn(model, None)
            for cand in candidates:
                plan = InterventionPlan.single(cand.to_edge())
                intervened = loss_fn(model, plan)
                removed = model.intervened_count()
                if removed == 0:
                    if skip_no_ops:
                        continue
                    cand.n_intervened = 0
                    cand.suppressible = None
                    scored.append(cand)
                    continue
                cand.baseline_loss = baseline
                cand.intervened_loss = intervened
                cand.delta_loss = intervened - baseline
                cand.n_intervened = removed
                cand.suppressible = bool(cand.delta_loss <= eps)
                cand.seed = seed
                cand.context_length = context_length
                scored.append(cand)
    finally:
        if was_training:
            model.train()
    return scored


def make_needle_loss_fn(
    x: torch.Tensor, y: torch.Tensor
) -> Callable[[torch.nn.Module, Optional[InterventionPlan]], float]:
    """Last-token cross-entropy on a fixed needle batch.

    Fixing the batch matters: if the batch were resampled between the baseline
    and the intervened pass, the difference would be dominated by batch noise.
    """

    def loss_fn(model: torch.nn.Module, plan: Optional[InterventionPlan]) -> float:
        logits, _ = model(x, plan=plan)
        return float(F.cross_entropy(logits[:, -1, :], y).item())

    return loss_fn


def make_lm_loss_fn(
    x: torch.Tensor, y: torch.Tensor
) -> Callable[[torch.nn.Module, Optional[InterventionPlan]], float]:
    """Mean next-token cross-entropy on a fixed language-modelling batch."""

    def loss_fn(model: torch.nn.Module, plan: Optional[InterventionPlan]) -> float:
        logits, _ = model(x, plan=plan)
        return float(
            F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
            ).item()
        )

    return loss_fn


# ---------------------------------------------------------------------------
#  Gate selection: naive vs contribution-gated, at a matched budget
# ---------------------------------------------------------------------------

def select_naive(candidates: Sequence[Candidate], budget: int) -> List[Candidate]:
    """Highest structural overlap first. Ignores behaviour entirely."""
    return sorted(candidates, key=lambda c: -c.overlap)[:budget]


def select_contribution_gated(
    candidates: Sequence[Candidate], budget: int
) -> List[Candidate]:
    """Lowest measured behavioral contribution first, among the same pool.

    Given the same candidate pool and the same budget as :func:`select_naive`,
    the only difference is the ranking criterion. That is what makes the two
    a controlled comparison: any difference in outcome is attributable to
    *which* interactions were removed, not how many.

    Ranking is on the **signed** delta, not its magnitude. The objective is to
    minimise the damage done by removing ``budget`` edges, so an edge whose
    removal lowers the loss (delta < 0) is a better removal than one whose
    removal leaves it unchanged. This matches the ``delta <= eps``
    suppressibility rule, which is likewise one-sided. Ranking on ``|delta|``
    would answer a different question - "which edges matter least in either
    direction" - and would decline to remove an edge that is actively harmful.
    """
    scored = [c for c in candidates if math.isfinite(c.delta_loss)]
    return sorted(scored, key=lambda c: c.delta_loss)[:budget]


def split_high_overlap_groups(
    candidates: Sequence[Candidate],
    high_overlap_q: float = 0.75,
    low_contribution_q: float = 0.25,
    high_contribution_q: float = 0.75,
) -> Dict[str, List[Candidate]]:
    """Partition candidates into the two groups the central claim contrasts.

    Group A: high structural overlap, low behavioral contribution.
    Group B: high structural overlap, high behavioral contribution.

    Both groups look alike structurally. If intervening on them produces
    different behavioral damage, structural overlap is not sufficient to
    decide dispensability.

    Thresholds are quantiles of the observed distributions, so they adapt to
    the run rather than being hard-coded, and are recorded with the results.
    """
    usable = [c for c in candidates if math.isfinite(c.delta_loss)]
    if not usable:
        return {"high_overlap_low_contribution": [], "high_overlap_high_contribution": [],
                "thresholds": {}}

    overlaps = np.array([c.overlap for c in usable])
    ov_thr = float(np.quantile(overlaps, high_overlap_q))
    high_ov = [c for c in usable if c.overlap >= ov_thr]
    if not high_ov:
        return {"high_overlap_low_contribution": [], "high_overlap_high_contribution": [],
                "thresholds": {}}

    # Contribution thresholds are quantiles *within* the high-overlap subset.
    # Taking them over all candidates would answer a different question - the
    # claim is about how structurally similar edges differ behaviorally, so the
    # split must be made among edges that are already structurally similar.
    high_deltas = np.array([c.delta_loss for c in high_ov])
    lo_thr = float(np.quantile(high_deltas, low_contribution_q))
    hi_thr = float(np.quantile(high_deltas, high_contribution_q))
    return {
        "high_overlap_low_contribution": [c for c in high_ov if c.delta_loss <= lo_thr],
        "high_overlap_high_contribution": [c for c in high_ov if c.delta_loss >= hi_thr],
        "thresholds": {
            "overlap": ov_thr,
            "low_contribution": lo_thr,
            "high_contribution": hi_thr,
            "high_overlap_q": high_overlap_q,
            "low_contribution_q": low_contribution_q,
            "high_contribution_q": high_contribution_q,
        },
    }


def candidates_to_penalty_pairs(
    candidates: Sequence[Candidate],
) -> List[Tuple[int, int]]:
    """Map selected edges onto the row pairs the redundancy loss acts on.

    The redundancy penalty is a differentiable surrogate operating on pairs of
    attention rows; an edge ``i -> j`` contributes the pair ``(i, j)``. This is
    the same surrogate the original used, so the training signal is unchanged -
    what changed is *which* pairs reach it.
    """
    return [(c.query, c.key) for c in candidates]
