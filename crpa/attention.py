"""
crpa.attention - CRPA sparse attention with two interchangeable implementations.

Omega(i) = P(i) u G u C_k(i)
  P(i)  contiguous partition window containing i
  G     relay positions; every token may attend to a relay, and a relay row
        attends to everything causally available
  C_k(i) up to k routed keys drawn from other *router* partitions

Two implementations
-------------------
``dense_masked``
    Materialises ``(B, H, T, T)`` scores and a ``(T, T)`` boolean mask. This is
    the original implementation, kept as the reference and the default at short
    context because it is simple and obviously correct.

``sparse_gather``
    Computes only the entries Omega(i) actually contains, chunked over query
    blocks. Required above roughly 8k tokens: at T=65536 with 12 heads a bf16
    dense score tensor is ~103 GB, so the dense path cannot represent the Tier 2
    experiment at all.

Both paths consume the *same* :class:`CRPAStructure` indices, so
``tests/test_attention_equivalence.py`` can assert they agree numerically.

A note on the two meanings of "partition"
-----------------------------------------
The local window P(i) is defined by *position* (contiguous blocks of
``partition_size``), while the routed set C_k(i) excludes keys sharing the
query's *router* assignment. These are different notions of a partition. The
original implementation did this and we preserve the behaviour, but it is a
modelling inconsistency worth knowing about when reading routing results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

NEG_INF = float("-inf")

#: Query-block chunk size for the gather path. Bounds peak score memory.
DEFAULT_QUERY_CHUNK = 4096

#: Chunk size used when sampling routed keys, so the (chunk, T) scratch
#: tensor stays bounded at long context.
_ROUTE_SAMPLE_CHUNK = 1024


def relay_positions(T: int, n_relays: int) -> List[int]:
    """Evenly spaced relay positions, matching the original ``_relay_pos``."""
    step = max(T // (n_relays + 1), 1)
    return [step * (i + 1) for i in range(n_relays) if step * (i + 1) < T]


def _sample_cross_indices(
    T: int,
    hard_asgn: Optional[torch.Tensor],
    cross_k: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample up to ``cross_k`` causal keys from other router partitions.

    Returns a ``(T, cross_k)`` long tensor; ``-1`` marks an unused slot.

    Selecting the top-k of i.i.d. uniform scores over the valid set - which is
    what the original code did - is distributionally identical to drawing k
    distinct keys uniformly without replacement. We keep the top-k formulation
    so results match, but chunk it over queries: the original materialised a
    ``(T, T)`` scratch tensor, which is 17 GB at T=65536.
    """
    if hard_asgn is None or cross_k <= 0:
        return torch.full((T, cross_k if cross_k > 0 else 0), -1, dtype=torch.long, device=device)

    k = min(cross_k, T)
    out = torch.full((T, k), -1, dtype=torch.long, device=device)
    positions = torch.arange(T, device=device)

    for start in range(0, T, _ROUTE_SAMPLE_CHUNK):
        stop = min(start + _ROUTE_SAMPLE_CHUNK, T)
        rows = positions[start:stop]

        # Valid iff strictly causal and assigned to a different router partition.
        causal = positions.unsqueeze(0) < rows.unsqueeze(1)
        different = hard_asgn.unsqueeze(0) != hard_asgn[rows].unsqueeze(1)
        valid = causal & different

        scores = torch.rand((stop - start, T), device=device, generator=generator)
        # Strictly positive on valid entries so 0 unambiguously means "invalid".
        scores = scores.clamp_min(torch.finfo(scores.dtype).tiny) * valid.to(scores.dtype)

        top_scores, top_idx = scores.topk(k, dim=1)
        out[start:stop] = torch.where(top_scores > 0, top_idx, torch.full_like(top_idx, -1))

    return out


@dataclass
class SparseProbs:
    """Attention probabilities kept in their gathered form.

    A dense ``(B, H, T, T)`` capture is unusable at long context: at T=16384
    with 12 heads that is 12.9 GB for a single layer, and 180 GB for a
    14-layer model. But each non-relay query has at most
    ``partition_size + n_relays + cross_k`` permitted keys - about 552 at the
    medium profile - so the dense form is over 97% zeros.

    Keeping the gathered form instead costs about 27 MB per layer for a
    1024-query window, independent of context length.

    Attributes:
        probs: ``(B, H, n_q, C)`` probabilities over each query's key list.
        key_idx: ``(n_q, C)`` absolute key index per slot; ``-1`` is unused.
        query_lo: absolute position of local row 0.
        is_relay: ``(n_q,)`` marks relay rows, whose true support spans every
            causal key and is therefore not represented here.
    """

    probs: torch.Tensor
    key_idx: torch.Tensor
    query_lo: int
    is_relay: torch.Tensor

    @property
    def n_queries(self) -> int:
        return int(self.probs.shape[2])

    def head(self, h: int) -> torch.Tensor:
        """Batch-averaged probabilities for one head, ``(n_q, C)``."""
        return self.probs[:, h].mean(dim=0)


@dataclass
class CRPAStructure:
    """The concrete sparsity pattern for one forward pass.

    Holding this explicitly is what lets the dense and gather paths be compared:
    they are two ways of evaluating attention over the *same* index sets.
    """

    T: int
    partition_size: int
    relay_pos: torch.Tensor      # (g,) long
    cross_idx: torch.Tensor      # (T, k) long, -1 = unused
    device: torch.device

    @property
    def n_blocks(self) -> int:
        return math.ceil(self.T / self.partition_size)

    @property
    def n_relays(self) -> int:
        return int(self.relay_pos.numel())

    @property
    def cross_k(self) -> int:
        return int(self.cross_idx.shape[1]) if self.cross_idx.ndim == 2 else 0

    def dense_mask(self, causal: bool = True) -> torch.Tensor:
        """Materialise the ``(T, T)`` boolean mask this structure describes."""
        T, p = self.T, self.partition_size
        mask = torch.zeros(T, T, dtype=torch.bool, device=self.device)

        for b in range(self.n_blocks):
            s, e = b * p, min((b + 1) * p, T)
            mask[s:e, s:e] = True

        if self.n_relays:
            mask[:, self.relay_pos] = True
            mask[self.relay_pos, :] = True

        if self.cross_k:
            rows = torch.arange(T, device=self.device).unsqueeze(1).expand_as(self.cross_idx)
            keep = self.cross_idx >= 0
            mask[rows[keep], self.cross_idx[keep]] = True

        if causal:
            mask &= torch.tril(torch.ones(T, T, dtype=torch.bool, device=self.device))
        mask.fill_diagonal_(True)
        return mask

    def edge_count(self) -> int:
        """Number of attention entries actually permitted, for sparsity reporting."""
        return int(self.dense_mask().sum().item())

    def attended_mask(self, queries: torch.Tensor) -> torch.Tensor:
        """Union of keys attended to by a set of query positions.

        Computed from the structure rather than a materialised ``(T, T)`` mask,
        so reachability analysis remains possible at long context.

        Args:
            queries: ``(T,)`` boolean mask selecting query positions.

        Returns:
            ``(T,)`` boolean mask of keys any selected query can reach.
        """
        T, p = self.T, self.partition_size
        out = torch.zeros(T, dtype=torch.bool, device=self.device)
        idx = queries.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return out

        # Relay rows see everything causally available, so a single selected
        # relay dominates the union up to its own position.
        if self.n_relays:
            is_relay = torch.isin(idx, self.relay_pos)
            if bool(is_relay.any()):
                out[: int(idx[is_relay].max().item()) + 1] = True

        # Partition-local windows, truncated causally at each query.
        for b in torch.unique(idx // p).tolist():
            start = int(b) * p
            stop = min(start + p, T)
            hi = int(idx[(idx >= start) & (idx < stop)].max().item())
            out[start: min(stop, hi + 1)] = True

        # Relays are visible to every query at or after them.
        if self.n_relays:
            visible = self.relay_pos <= int(idx.max().item())
            out[self.relay_pos[visible]] = True

        # Routed keys.
        if self.cross_k:
            routed = self.cross_idx[idx]
            routed = routed[routed >= 0]
            if routed.numel():
                out[routed] = True

        return out

    def allowed_keys(self, query: int) -> torch.Tensor:
        """The sorted key indices query ``query`` may attend to.

        Used by the intervention layer to verify that a targeted edge is
        genuinely present before claiming its removal had an effect.
        """
        T, p = self.T, self.partition_size
        if self.n_relays and bool((self.relay_pos == query).any()):
            return torch.arange(query + 1, device=self.device)

        b = query // p
        keys = set(range(b * p, min((b + 1) * p, T)))
        keys.update(int(r) for r in self.relay_pos.tolist())
        if self.cross_k:
            keys.update(int(j) for j in self.cross_idx[query].tolist() if j >= 0)
        keys = sorted(j for j in keys if j <= query)
        return torch.tensor(keys, dtype=torch.long, device=self.device)


def build_crpa_structure(
    T: int,
    partition_size: int,
    n_relays: int,
    cross_k: int,
    hard_asgn: Optional[torch.Tensor],
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> CRPAStructure:
    """Build the sparsity pattern. Both attention paths start here."""
    rp = relay_positions(T, n_relays)
    return CRPAStructure(
        T=T,
        partition_size=partition_size,
        relay_pos=torch.tensor(rp, dtype=torch.long, device=device),
        cross_idx=_sample_cross_indices(T, hard_asgn, cross_k, device, generator),
        device=device,
    )


def build_crpa_mask(
    T: int,
    p_size: int,
    relay_pos: Optional[Sequence[int]],
    hard_asgn: Optional[torch.Tensor] = None,
    cross_k: int = 4,
    causal: bool = True,
    device: "str | torch.device" = "cpu",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Boolean CRPA mask. Signature preserved from the original implementation.

    ``generator`` is new: routed-key sampling previously drew from the global
    RNG on every rebuild, which meant an intervention's baseline and masked
    forward passes could be evaluated under two different masks.
    """
    device = torch.device(device)
    n_relays = len(relay_pos) if relay_pos else 0
    structure = CRPAStructure(
        T=T,
        partition_size=p_size,
        relay_pos=torch.tensor(list(relay_pos or []), dtype=torch.long, device=device),
        cross_idx=_sample_cross_indices(T, hard_asgn, cross_k, device, generator),
        device=device,
    )
    # n_relays is implied by relay_pos; kept for signature symmetry.
    del n_relays
    return structure.dense_mask(causal=causal)


def sliding_mask(T: int, window: int, device: "str | torch.device" = "cpu") -> torch.Tensor:
    """Banded causal mask of width ``window``, vectorised.

    The original built this with a Python loop over T, which is untenable at
    long context.
    """
    device = torch.device(device)
    idx = torch.arange(T, device=device)
    delta = idx.unsqueeze(1) - idx.unsqueeze(0)
    return (delta >= 0) & (delta < window)


# ---------------------------------------------------------------------------
#  Rotary position embeddings (long-context profiles)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """RoPE, used by profiles that must not pay a learned table at 64k tokens.

    A learned ``nn.Embedding(65536, 768)`` would add ~50M parameters and make
    the advertised parameter count depend on context length.
    """

    def __init__(self, d_head: int, base: float = 10000.0) -> None:
        super().__init__()
        if d_head % 2 != 0:
            raise ValueError("RoPE requires an even head dimension, got {}".format(d_head))
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_T = 0
        self._cos: Optional[torch.Tensor] = None
        self._sin: Optional[torch.Tensor] = None

    def _build(self, T: int, device: torch.device, dtype: torch.dtype) -> None:
        t = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos = emb.cos().to(dtype)
        self._sin = emb.sin().to(dtype)
        self._cached_T = T

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to ``(B, H, T, d)`` queries and keys."""
        T = q.shape[-2]
        if self._cos is None or self._cached_T < T or self._cos.dtype != q.dtype:
            self._build(max(T, self._cached_T), q.device, q.dtype)
        cos = self._cos[:T].unsqueeze(0).unsqueeze(0)
        sin = self._sin[:T].unsqueeze(0).unsqueeze(0)
        return (q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin)


# ---------------------------------------------------------------------------
#  Routing
# ---------------------------------------------------------------------------

class DifferentiableRouter(nn.Module):
    """Maps each token to one of ``n_partitions`` router partitions.

    Soft (differentiable) during training, hard argmax for mask construction.
    Unchanged from the original beyond typing and documentation.
    """

    def __init__(self, d_model: int, n_partitions: int, temp: float = 0.7) -> None:
        super().__init__()
        self.T = temp
        self.n = n_partitions
        r_dim = max(d_model // 8, 16)
        self.Wr = nn.Linear(d_model, r_dim, bias=False)
        self.cents = nn.Parameter(0.02 * torch.randn(n_partitions, r_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.Wr(x)
        sc = r @ self.cents.T / self.T
        soft = torch.softmax(sc, dim=-1)
        return soft, soft.argmax(dim=-1)

    def load_balance_loss(self, soft: torch.Tensor) -> torch.Tensor:
        """``m * sum_p (mean_i pi_ip)^2`` - penalises uneven partition use."""
        avg = soft.mean(dim=(0, 1))
        return self.n * (avg ** 2).sum()


# ---------------------------------------------------------------------------
#  Attention kernels
# ---------------------------------------------------------------------------

def _apply_edge_interventions(
    scores: torch.Tensor,
    edges: Sequence[Tuple[Optional[int], int, int]],
    key_index: Optional[torch.Tensor] = None,
    query_offset: int = 0,
) -> int:
    """Set ``-inf`` at the requested (head, query, key) score positions.

    Args:
        scores: ``(B, H, Tq, Tk)`` score tensor to modify in place.
        edges: ``(head, query, key)`` triples; ``head=None`` targets all heads.
        key_index: when the last axis is a *gathered* key list rather than
            absolute positions, a ``(Tq, Tk)`` map from slot to absolute key.
        query_offset: absolute index of row 0 of ``scores``.

    Returns:
        The number of ``(head, query, key)`` edges that carried finite mass and
        were removed - independent of batch size. Entries already masked are
        *not* counted: setting an already ``-inf`` score to ``-inf`` changes
        nothing, and counting it would let a no-op be reported as a successful
        intervention with delta zero, which is the exact failure mode of the
        original implementation.
    """
    touched = 0
    Tq = scores.shape[-2]
    for head, q, k in edges:
        local_q = q - query_offset
        if not (0 <= local_q < Tq):
            continue
        if key_index is None:
            slots = [k] if 0 <= k < scores.shape[-1] else []
        else:
            slots = (key_index[local_q] == k).nonzero(as_tuple=True)[0].tolist()
        for slot in slots:
            if head is None:
                live_here = torch.isfinite(scores[:, :, local_q, slot]).any(dim=0)
                # Never remove a row's last permitted key: the row would
                # softmax to NaN, and the "intervention" would be a deletion of
                # the query rather than of one interaction.
                row_live = torch.isfinite(scores[:, :, local_q, :]).sum(dim=-1)
                safe = live_here & (row_live > 1).all(dim=0)
                n_live = int(safe.sum().item())
                if n_live == 0:
                    continue
                scores[:, safe, local_q, slot] = NEG_INF
                touched += n_live
            else:
                if not bool(torch.isfinite(scores[:, head, local_q, slot]).any()):
                    continue
                row_live = torch.isfinite(scores[:, head, local_q, :]).sum(dim=-1)
                if not bool((row_live > 1).all()):
                    # Query has exactly one permitted key (position 0 always
                    # does). Removing it is not a measurable intervention, so
                    # it is skipped and not counted, and the caller refuses to
                    # report a delta.
                    continue
                scores[:, head, local_q, slot] = NEG_INF
                touched += 1
    return touched


def dense_masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    dropout: Optional[nn.Dropout] = None,
    edges: Sequence[Tuple[Optional[int], int, int]] = (),
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Reference attention over an explicit ``(T, T)`` boolean mask.

    Masking happens *before* the softmax, so a removed entry loses its
    probability mass and the surviving entries in that row renormalise to 1.
    This is the property ``tests/test_intervention.py`` pins.

    Returns ``(output, probabilities_pre_dropout, n_intervened)``.
    """
    scale = q.shape[-1] ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), NEG_INF)

    touched = _apply_edge_interventions(scores, edges) if edges else 0

    probs = torch.softmax(scores, dim=-1)
    # A fully-masked row would produce NaN. The diagonal is always allowed, so
    # this should be unreachable; assert rather than silently zero-filling as
    # the original did, which would have hidden a genuinely broken mask.
    if torch.isnan(probs).any():
        raise RuntimeError(
            "attention produced NaN: some query row has no permitted key. "
            "This indicates a malformed mask or an intervention that removed a "
            "row's last remaining edge."
        )

    out = (dropout(probs) if dropout is not None else probs) @ v
    return out, probs, touched


def sparse_gather_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    structure: CRPAStructure,
    dropout: Optional[nn.Dropout] = None,
    edges: Sequence[Tuple[Optional[int], int, int]] = (),
    query_chunk: int = DEFAULT_QUERY_CHUNK,
    return_probs: bool = False,
    probs_window: Optional[Tuple[int, int]] = None,
    sparse_probs_out: Optional[List["SparseProbs"]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
    """Attention evaluated only over Omega(i), chunked over query blocks.

    Each non-relay query attends to ``partition_size + n_relays + cross_k``
    candidate keys at most, so cost is O(T * (w + g + k)) rather than O(T^2).
    Relay rows attend causally to everything and are computed separately, then
    written over the general result - their permitted set is a superset, so the
    overwrite is exact.

    ``return_probs`` materialises the dense ``(B, H, T, T)`` probability tensor
    for overlap analysis. That defeats the memory saving, so it is opt-in and
    only usable at short context.

    ``probs_window=(lo, hi)`` instead collects :class:`SparseProbs` for those
    query rows into ``sparse_probs_out``, which is what long-context
    diagnostics use: it is independent of T and roughly 500x smaller.
    """
    B, H, T, d = q.shape
    p = structure.partition_size
    scale = d ** -0.5
    device = q.device

    out = torch.zeros_like(q)
    probs_full = (
        torch.zeros(B, H, T, T, dtype=q.dtype, device=device) if return_probs else None
    )
    touched = 0

    relay = structure.relay_pos
    g = structure.n_relays
    kk = structure.cross_k

    k_relay = k[:, :, relay, :] if g else None       # (B,H,g,d)
    v_relay = v[:, :, relay, :] if g else None

    positions = torch.arange(T, device=device)

    # Round the chunk up to a whole number of partition blocks.
    block_chunk = max(1, query_chunk // p)
    for b_start in range(0, structure.n_blocks, block_chunk):
        b_stop = min(b_start + block_chunk, structure.n_blocks)
        q_lo, q_hi = b_start * p, min(b_stop * p, T)
        n_q = q_hi - q_lo
        if n_q <= 0:
            continue

        q_chunk = q[:, :, q_lo:q_hi, :]                       # (B,H,n_q,d)
        q_pos = positions[q_lo:q_hi]

        # -- source 1: the query's own partition block ----------------------
        # Keys are the block span; queries in different blocks of this chunk
        # see different spans, so build an explicit per-query key index.
        blk_of_q = q_pos // p
        blk_key_idx = (blk_of_q.unsqueeze(1) * p) + torch.arange(p, device=device).unsqueeze(0)
        blk_valid = (blk_key_idx < T) & (blk_key_idx <= q_pos.unsqueeze(1))

        # -- source 2: relays ------------------------------------------------
        if g:
            rel_key_idx = relay.unsqueeze(0).expand(n_q, g)
            rel_valid = rel_key_idx <= q_pos.unsqueeze(1)
        else:
            rel_key_idx = torch.zeros(n_q, 0, dtype=torch.long, device=device)
            rel_valid = torch.zeros(n_q, 0, dtype=torch.bool, device=device)

        # -- source 3: routed cross-partition keys ---------------------------
        if kk:
            cross_key_idx = structure.cross_idx[q_lo:q_hi]
            cross_valid = (cross_key_idx >= 0) & (cross_key_idx <= q_pos.unsqueeze(1))
            cross_key_idx = cross_key_idx.clamp_min(0)
        else:
            cross_key_idx = torch.zeros(n_q, 0, dtype=torch.long, device=device)
            cross_valid = torch.zeros(n_q, 0, dtype=torch.bool, device=device)

        # A key reachable through more than one source must be counted once, or
        # softmax would double-weight it. Rather than an O(C^2) all-pairs
        # comparison - 1.25 GB of booleans at 64k - exploit the structure: the
        # block source is a contiguous range and the other two sources are tiny,
        # so only relay-vs-block, cross-vs-block and cross-vs-relay can collide.
        blk_lo = (q_pos // p) * p
        blk_hi = blk_lo + p
        if g:
            in_block = (rel_key_idx >= blk_lo.unsqueeze(1)) & (rel_key_idx < blk_hi.unsqueeze(1))
            rel_valid = rel_valid & ~in_block
        if kk:
            in_block = (cross_key_idx >= blk_lo.unsqueeze(1)) & (
                cross_key_idx < blk_hi.unsqueeze(1)
            )
            cross_valid = cross_valid & ~in_block
            if g:
                # (n_q, kk, g) - small, since g and kk are both O(10).
                hits_relay = (cross_key_idx.unsqueeze(2) == rel_key_idx.unsqueeze(1)).any(dim=2)
                cross_valid = cross_valid & ~hits_relay
            # topk returns distinct indices, so the routed set is self-disjoint.

        key_idx = torch.cat([blk_key_idx, rel_key_idx, cross_key_idx], dim=1)   # (n_q, C)
        valid = torch.cat([blk_valid, rel_valid, cross_valid], dim=1)

        # Scores: block part is blocked, relay and cross parts are gathers.
        n_blk = blk_key_idx.shape[1]
        full_blocks = (b_stop * p) <= T
        q_blocks = q_chunk.reshape(B, H, b_stop - b_start, p, d) if full_blocks else None
        if q_blocks is not None:
            k_blocks = k[:, :, b_start * p: b_stop * p, :].reshape(B, H, b_stop - b_start, p, d)
            s_blk = (q_blocks @ k_blocks.transpose(-2, -1)).reshape(B, H, n_q, p) * scale
        else:
            # Ragged final block: fall back to an explicit gather.
            k_blk = k.gather(
                2, blk_key_idx.clamp(0, T - 1).reshape(1, 1, -1, 1).expand(B, H, n_q * n_blk, d)
            ).reshape(B, H, n_q, n_blk, d)
            s_blk = torch.einsum("bhqd,bhqcd->bhqc", q_chunk, k_blk) * scale

        parts = [s_blk]
        if g:
            parts.append((q_chunk @ k_relay.transpose(-2, -1)) * scale)
        if kk:
            k_cross = k.gather(
                2, cross_key_idx.reshape(1, 1, -1, 1).expand(B, H, n_q * kk, d)
            ).reshape(B, H, n_q, kk, d)
            parts.append(torch.einsum("bhqd,bhqcd->bhqc", q_chunk, k_cross) * scale)

        scores = torch.cat(parts, dim=-1)                       # (B,H,n_q,C)
        scores = scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), NEG_INF)

        if edges:
            # Relay rows are recomputed and overwritten below, so applying
            # their edges here too would double-count the intervention.
            general_edges = [
                e for e in edges if not (g and bool((relay == e[1]).any()))
            ]
            if general_edges:
                touched += _apply_edge_interventions(
                    scores, general_edges,
                    key_index=key_idx.masked_fill(~valid, -1), query_offset=q_lo,
                )

        probs = torch.softmax(scores, dim=-1)
        if torch.isnan(probs).any():
            raise RuntimeError(
                "sparse attention produced NaN at queries [{}, {}): a row has no "
                "permitted key".format(q_lo, q_hi)
            )

        p_use = dropout(probs) if dropout is not None else probs

        # Recombine: block part blocked, relay part shared, cross part gathered.
        acc = torch.zeros(B, H, n_q, d, dtype=q.dtype, device=device)
        p_blk = p_use[..., :n_blk]
        if q_blocks is not None:
            v_blocks = v[:, :, b_start * p: b_stop * p, :].reshape(B, H, b_stop - b_start, p, d)
            acc += (p_blk.reshape(B, H, b_stop - b_start, p, p) @ v_blocks).reshape(B, H, n_q, d)
        else:
            v_blk = v.gather(
                2, blk_key_idx.clamp(0, T - 1).reshape(1, 1, -1, 1).expand(B, H, n_q * n_blk, d)
            ).reshape(B, H, n_q, n_blk, d)
            acc += torch.einsum("bhqc,bhqcd->bhqd", p_blk, v_blk)

        cursor = n_blk
        if g:
            acc += p_use[..., cursor:cursor + g] @ v_relay
            cursor += g
        if kk:
            v_cross = v.gather(
                2, cross_key_idx.reshape(1, 1, -1, 1).expand(B, H, n_q * kk, d)
            ).reshape(B, H, n_q, kk, d)
            acc += torch.einsum("bhqc,bhqcd->bhqd", p_use[..., cursor:cursor + kk], v_cross)

        out[:, :, q_lo:q_hi, :] = acc

        if probs_full is not None:
            scatter_idx = key_idx.masked_fill(~valid, 0)
            contrib = probs.masked_fill(~valid.unsqueeze(0).unsqueeze(0), 0.0)
            probs_full[:, :, q_lo:q_hi, :].scatter_add_(
                3, scatter_idx.unsqueeze(0).unsqueeze(0).expand(B, H, n_q, key_idx.shape[1]),
                contrib,
            )

        if probs_window is not None and sparse_probs_out is not None:
            w_lo, w_hi = probs_window
            lo = max(q_lo, w_lo)
            hi = min(q_hi, w_hi)
            if lo < hi:
                sl = slice(lo - q_lo, hi - q_lo)
                rows = positions[lo:hi]
                is_relay = (
                    torch.isin(rows, relay) if g
                    else torch.zeros(hi - lo, dtype=torch.bool, device=device)
                )
                sparse_probs_out.append(SparseProbs(
                    probs=probs[:, :, sl, :].detach().clone(),
                    key_idx=key_idx[sl].masked_fill(~valid[sl], -1).detach().clone(),
                    query_lo=lo,
                    is_relay=is_relay,
                ))

    # Relay rows attend causally to everything; recompute and overwrite.
    if g:
        rel_q = q[:, :, relay, :]
        s_rel = (rel_q @ k.transpose(-2, -1)) * scale
        rel_causal = positions.unsqueeze(0) <= relay.unsqueeze(1)
        s_rel = s_rel.masked_fill(~rel_causal.unsqueeze(0).unsqueeze(0), NEG_INF)
        if edges:
            rel_edges = [
                (h, int((relay == qq).nonzero()[0, 0]), kkey)
                for (h, qq, kkey) in edges
                if bool((relay == qq).any())
            ]
            if rel_edges:
                touched += _apply_edge_interventions(s_rel, rel_edges)
        p_rel = torch.softmax(s_rel, dim=-1)
        if torch.isnan(p_rel).any():
            raise RuntimeError("relay-row attention produced NaN")
        out[:, :, relay, :] = (dropout(p_rel) if dropout is not None else p_rel) @ v
        if probs_full is not None:
            probs_full[:, :, relay, :] = p_rel

    return out, probs_full, touched
