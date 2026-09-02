"""
crpa.model - the transformer backbone and the CRPA attention layer.

The backbone (embeddings, FFN, LayerNorm, residuals, weight tying) is identical
across every variant, so the comparison between variants isolates attention.
That property is inherited from the original implementation and preserved.

What changed relative to the original, and why:

* ``_Alast`` is captured *before* dropout. Previously overlap statistics were
  computed on dropout-corrupted rows that did not sum to 1.
* Interventions are addressed per ``(layer, head, query, key)``. Previously a
  single ``GPT._mask_pair`` was broadcast to every layer, so a "per-layer"
  measurement was really the effect of an identical simultaneous intervention
  in all layers, recomputed once per layer.
* The sparsity structure can be *frozen*, so an intervention's baseline and
  masked forward passes are guaranteed to run under the same mask. Previously
  the routed-key draw was re-rolled from the global RNG whenever the step
  counter hit a multiple of 20 - including during estimation - so deltas
  absorbed pure mask noise.
* ``crpa_naive`` and ``crpa_contribution`` now share a candidate pool, a
  refresh cadence and a removal budget. Only the *ranking criterion* differs,
  which is what makes them a controlled comparison.
"""

from __future__ import annotations

import contextlib
import math
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from crpa.attention import (
    CRPAStructure,
    DifferentiableRouter,
    RotaryEmbedding,
    build_crpa_structure,
    dense_masked_attention,
    sliding_mask,
    sparse_gather_attention,
)
from crpa.config import ModelConfig, resolve_variant

#: How often the routed-key structure is rebuilt during training.
STRUCTURE_REFRESH_STEPS = 20


class FFN(nn.Module):
    """Position-wise feed-forward block."""

    def __init__(self, d: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(),
            nn.Linear(4 * d, d), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CRPAAttention(nn.Module):
    """Attention restricted to Omega(i) = P(i) u G u C_k(i).

    Args:
        cfg: architecture configuration.
        variant: canonical variant name.
        layer_idx: this layer's depth index, used to address interventions.
    """

    def __init__(self, cfg: ModelConfig, variant: str, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.variant = resolve_variant(variant)
        self.layer_idx = layer_idx
        self.n_head = cfg.n_head
        self.d_head = cfg.d_head
        self.p_size = cfg.partition_size
        self.n_rel = cfg.n_relays
        self.cross_k = cfg.cross_k
        self.rho = cfg.overlap_rho

        d = cfg.n_embd
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

        self.rope = RotaryEmbedding(cfg.d_head) if cfg.position == "rope" else None

        if "crpa" in self.variant:
            n_parts = math.ceil(cfg.block_size / self.p_size)
            self.router = DifferentiableRouter(d, n_parts, cfg.route_temp)

        # Pairs of query rows whose attention supports the redundancy penalty
        # pushes apart. Populated by the training loop; see crpa.intervention.
        self._penalty_pairs: List[Tuple[int, int]] = []
        # Measured behavioral contribution, keyed by (head, query, key).
        self._contribution: Dict[Tuple[Optional[int], int, int], float] = {}

        self._structure: Optional[CRPAStructure] = None
        self._dense_mask: Optional[torch.Tensor] = None
        self._step = 0
        self._frozen = False
        self._Alast: Optional[torch.Tensor] = None
        # Capture is opt-in. Defaulting it on made every forward pass outside a
        # capture context materialise a dense (B, H, T, T) tensor on the gather
        # path - 3 GB at 8k and 12 GB at 16k with 12 heads - which is exactly
        # what that path exists to avoid. Use GPT.capture_probabilities().
        self._capture_probs = False
        self._last_intervened = 0
        # When set, attention probabilities are retained in their gathered
        # form for this query window instead of densely. Required above a few
        # thousand tokens: see crpa.attention.SparseProbs.
        self._probs_window: Optional[Tuple[int, int]] = None
        self._sparse_probs: Optional[List] = None

    # -- structure management ------------------------------------------------
    def freeze_structure(self, frozen: bool = True) -> None:
        """Stop rebuilding the sparsity pattern.

        Every intervention runs inside this, so the baseline and the masked
        forward pass are evaluated over an identical mask.
        """
        self._frozen = frozen

    def reset_structure(self) -> None:
        self._structure = None
        self._dense_mask = None

    def _needs_rebuild(self, T: int) -> bool:
        if self._frozen and self._structure is not None and self._structure.T == T:
            return False
        if self._structure is None or self._structure.T != T:
            return True
        return self._step % STRUCTURE_REFRESH_STEPS == 0

    def _get_structure(
        self, x: torch.Tensor, generator: Optional[torch.Generator]
    ) -> CRPAStructure:
        T = x.shape[1]
        if self._needs_rebuild(T):
            hard = None
            if hasattr(self, "router"):
                with torch.no_grad():
                    _, hard = self.router(x)
                hard = hard[0].detach()
            self._structure = build_crpa_structure(
                T, self.p_size, self.n_rel, self.cross_k, hard, x.device, generator
            )
            self._dense_mask = None
        return self._structure

    def _get_dense_mask(self, x: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
        T = x.shape[1]
        if self.variant == "dense":
            if self._dense_mask is None or self._dense_mask.shape[0] != T:
                self._dense_mask = torch.tril(
                    torch.ones(T, T, dtype=torch.bool, device=x.device)
                )
            return self._dense_mask
        if self.variant == "sliding":
            if self._dense_mask is None or self._dense_mask.shape[0] != T:
                self._dense_mask = sliding_mask(T, self.p_size, x.device)
            return self._dense_mask
        structure = self._get_structure(x, generator)
        if self._dense_mask is None or self._dense_mask.shape[0] != T:
            self._dense_mask = structure.dense_mask()
        return self._dense_mask

    # -- shape helpers -------------------------------------------------------
    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_head, self.d_head).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, d = x.shape
        return x.transpose(1, 2).reshape(B, T, H * d)

    # -- forward -------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edges: Sequence[Tuple[Optional[int], int, int]] = (),
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run attention.

        Args:
            x: ``(B, T, D)`` layer input.
            edges: ``(head, query, key)`` triples to remove, already filtered to
                this layer. ``head=None`` targets every head.
            generator: RNG for routed-key sampling.

        Returns:
            ``(output, load_balance_loss, redundancy_loss)``.
        """
        B, T, D = x.shape
        if not self._frozen:
            self._step += 1

        q = self._split(self.Wq(x))
        k = self._split(self.Wk(x))
        v = self._split(self.Wv(x))
        if self.rope is not None:
            q, k = self.rope(q, k)

        dropout = self.drop if self.training else None
        use_gather = (
            self.cfg.attention_impl == "sparse_gather" and self.variant.startswith("crpa")
        )

        # The dense and sliding baselines have no CRPA structure to exploit, but
        # a real implementation of either would use a fused kernel rather than
        # materialising (B,H,T,T). Benchmarking them on the naive path would
        # flatter CRPA by comparing against a straw man, so use SDPA whenever
        # nothing needs the explicit probability matrix.
        use_sdpa = (
            self.variant in ("dense", "sliding")
            and not edges
            and not self._capture_probs
        )

        if use_gather:
            structure = self._get_structure(x, generator)
            self._sparse_probs = [] if self._probs_window is not None else None
            # The redundancy penalty operates on attention rows, so a gated
            # variant needs them during training whether or not a diagnostic
            # asked for a capture.
            needs_probs_for_loss = bool(
                self.training
                and self._penalty_pairs
                and self.variant in ("crpa_naive", "crpa_contribution")
            )
            out_h, probs, touched = sparse_gather_attention(
                q, k, v, structure, dropout=dropout, edges=edges,
                return_probs=(
                    (self._capture_probs and self._probs_window is None)
                    or needs_probs_for_loss
                ),
                probs_window=self._probs_window,
                sparse_probs_out=self._sparse_probs,
            )
        elif use_sdpa:
            p_drop = self.cfg.dropout if self.training else 0.0
            if self.variant == "dense":
                out_h = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, dropout_p=p_drop
                )
            else:
                out_h = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=self._get_dense_mask(x, generator),
                    dropout_p=p_drop,
                )
            probs, touched = None, 0
        else:
            mask = self._get_dense_mask(x, generator)
            out_h, probs, touched = dense_masked_attention(
                q, k, v, mask, dropout=dropout, edges=edges,
            )

        self._last_intervened = touched
        # Captured pre-dropout, so rows are genuine probability distributions.
        self._Alast = probs.detach() if (probs is not None and self._capture_probs) else None

        out = self.Wo(self._merge(out_h))

        zero = torch.zeros((), device=x.device, dtype=out.dtype)
        Lb = Lr = zero
        if hasattr(self, "router"):
            soft, _ = self.router(x)
            Lb = self.router.load_balance_loss(soft)
            if self.variant in ("crpa_naive", "crpa_contribution") and self._penalty_pairs:
                Lr = self.redundancy_loss(probs)
        return out, Lb, Lr

    def redundancy_loss(self, probs: Optional[torch.Tensor]) -> torch.Tensor:
        """Differentiable penalty pushing apart the supports of selected pairs.

        The pairs come from :mod:`crpa.intervention`. ``crpa_naive`` and
        ``crpa_contribution`` receive pools of identical size on an identical
        cadence; only which pairs are chosen differs.
        """
        if probs is None or not self._penalty_pairs:
            return torch.zeros((), device=self.Wq.weight.device)
        A = probs.mean(dim=(0, 1))  # (T, T), averaged over batch and heads
        T = A.shape[0]
        pairs = [(i, j) for i, j in self._penalty_pairs if i < T and j < T]
        if not pairs:
            return torch.zeros((), device=probs.device)
        idx_i = torch.tensor([i for i, _ in pairs], device=probs.device)
        idx_j = torch.tensor([j for _, j in pairs], device=probs.device)
        return (A[idx_i] * A[idx_j]).sum(dim=-1).mean()

    def set_penalty_pairs(self, pairs: Sequence[Tuple[int, int]]) -> None:
        self._penalty_pairs = list(pairs)

    def attended_keys(self, queries: torch.Tensor) -> torch.Tensor:
        """Union of keys the given query positions attend to in this layer.

        Used for reachability analysis. Falls back to the dense mask for the
        ``dense`` and ``sliding`` baselines, which have no CRPA structure.
        """
        if self._structure is not None and self.variant.startswith("crpa"):
            return self._structure.attended_mask(queries)
        if self._dense_mask is None:
            raise RuntimeError(
                "attended_keys requires a forward pass first: no mask has been built"
            )
        return self._dense_mask[queries].any(dim=0)


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, cfg: ModelConfig, variant: str, layer_idx: int) -> None:
        super().__init__()
        self.attn = CRPAAttention(cfg, variant, layer_idx)
        self.ffn = FFN(cfg.n_embd, cfg.dropout)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(
        self,
        x: torch.Tensor,
        edges: Sequence[Tuple[Optional[int], int, int]] = (),
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        o, lb, lr = self.attn(self.ln1(x), edges, generator)
        x = x + o
        x = x + self.ffn(self.ln2(x))
        return x, lb, lr


class GPT(nn.Module):
    """Decoder-only transformer with a switchable attention variant."""

    def __init__(self, cfg: ModelConfig, variant: str, seed: Optional[int] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.variant = resolve_variant(variant)
        self.block_size = cfg.block_size

        d = cfg.n_embd
        self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        self.pos_emb = nn.Embedding(cfg.block_size, d) if cfg.position == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [Block(cfg, self.variant, i) for i in range(cfg.n_layer)]
        )
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)

        # Explicit generator for routed-key sampling, so the sparsity pattern
        # does not depend on unrelated consumers of the global RNG.
        self._route_seed = seed
        self._generator: Optional[torch.Generator] = None

        # Deprecated shim for the original global pair mask. Setting it applies
        # the same edge to every layer and every head, which is what the
        # original did; new code should pass an InterventionPlan instead.
        self._mask_pair: Optional[Tuple[int, int]] = None

        # (load_balance, redundancy) from the most recent forward pass.
        self._last_aux: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # -- structure control ---------------------------------------------------
    def _get_generator(self, device: torch.device) -> Optional[torch.Generator]:
        if self._route_seed is None:
            return None
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(int(self._route_seed))
        return self._generator

    @contextlib.contextmanager
    def frozen_structure(self) -> Iterator[None]:
        """Hold the sparsity pattern fixed for the duration of the block.

        Every behavioral-contribution measurement runs inside this, so the
        baseline and the intervened forward pass share one mask.
        """
        for blk in self.blocks:
            blk.attn.freeze_structure(True)
        try:
            yield
        finally:
            for blk in self.blocks:
                blk.attn.freeze_structure(False)

    @contextlib.contextmanager
    def capture_probabilities(
        self, enabled: bool = True, window: Optional[Tuple[int, int]] = None
    ) -> Iterator[None]:
        """Toggle retention of attention probabilities.

        Args:
            enabled: whether to retain probabilities at all.
            window: ``(lo, hi)`` query range to retain in *gathered* form.
                Without it the capture is dense ``(B, H, T, T)``, which is
                12.9 GB per layer at T=16384 with 12 heads and cannot be used
                for long-context diagnostics. With it the cost is independent
                of context length.
        """
        previous = [(blk.attn._capture_probs, blk.attn._probs_window)
                    for blk in self.blocks]
        for blk in self.blocks:
            blk.attn._capture_probs = enabled
            blk.attn._probs_window = window if enabled else None
        try:
            yield
        finally:
            for blk, (prev_enabled, prev_window) in zip(self.blocks, previous):
                blk.attn._capture_probs = prev_enabled
                blk.attn._probs_window = prev_window

    def attention_probabilities(self) -> List[Optional[torch.Tensor]]:
        """Per-layer dense attention from the most recent forward pass."""
        return [blk.attn._Alast for blk in self.blocks]

    def sparse_attention_probabilities(self) -> List[Optional["object"]]:
        """Per-layer gathered attention from the most recent windowed capture.

        Each entry is a :class:`crpa.attention.SparseProbs`, or ``None`` if the
        layer captured nothing.
        """
        out = []
        for blk in self.blocks:
            chunks = blk.attn._sparse_probs
            if not chunks:
                out.append(None)
                continue
            if len(chunks) == 1:
                out.append(chunks[0])
                continue
            # A window spanning several query chunks: concatenate. The slot
            # count C is the same for every chunk, so this is well defined.
            from crpa.attention import SparseProbs

            out.append(SparseProbs(
                probs=torch.cat([c.probs for c in chunks], dim=2),
                key_idx=torch.cat([c.key_idx for c in chunks], dim=0),
                query_lo=chunks[0].query_lo,
                is_relay=torch.cat([c.is_relay for c in chunks], dim=0),
            ))
        return out

    def intervened_count(self) -> int:
        """Total score positions removed by the most recent forward pass.

        Zero means no intervention actually took place. Callers must refuse to
        report a delta in that case rather than reporting 0.0 as a measurement.
        """
        return sum(blk.attn._last_intervened for blk in self.blocks)

    # -- forward -------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        plan: Optional["object"] = None,
        lambda_bal: float = 0.0,
        lambda_red: float = 0.0,
        last_only: bool = False,
        loss_chunk: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Args:
            idx: ``(B, T)`` token ids.
            targets: ``(B, T)`` next-token targets; when given, a loss is returned.
            plan: an :class:`crpa.intervention.InterventionPlan`, or ``None``.
            lambda_bal / lambda_red: auxiliary loss weights.
            last_only: project only the final position through the output head.
                The retrieval task is scored there and nowhere else, and the
                full projection is ``(B, T, vocab)`` - 6.6 GB at T=16384 with a
                50k vocabulary, which is what exhausts memory at long context.
            loss_chunk: compute the language-model loss in chunks of this many
                positions, so the logits for a long sequence never exist all at
                once. 0 disables chunking.
        """
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(
                "sequence length {} exceeds block_size {}".format(T, self.block_size)
            )
        gen = self._get_generator(idx.device)

        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)

        # Legacy shim: a bare (i, j) applies to every layer and head.
        legacy_edges: Sequence[Tuple[Optional[int], int, int]] = ()
        if plan is None and self._mask_pair is not None:
            legacy_edges = ((None, self._mask_pair[0], self._mask_pair[1]),)

        Lb = torch.zeros((), device=idx.device)
        Lr = torch.zeros((), device=idx.device)
        for depth, blk in enumerate(self.blocks):
            if plan is not None:
                edges = plan.for_layer(depth)
            else:
                edges = legacy_edges
            x, lb, lr = blk(x, edges, gen)
            Lb = Lb + lb
            Lr = Lr + lr

        h = self.ln_f(x)

        # Retained with their graph so a caller computing its own task loss
        # (the retrieval branch does) can still add the auxiliary terms.
        self._last_aux = (Lb, Lr)

        if last_only and targets is None:
            # (B, 1, vocab). Callers that only read logits[:, -1, :] get the
            # same answer for a fraction of the memory.
            return self.head(h[:, -1:, :]), None

        if targets is not None and loss_chunk > 0 and T > loss_chunk:
            # Accumulate the mean cross-entropy without materialising logits
            # for the whole sequence at once.
            total = torch.zeros((), device=idx.device, dtype=torch.float32)
            for start in range(0, T, loss_chunk):
                stop = min(start + loss_chunk, T)
                part = self.head(h[:, start:stop, :])
                total = total + F.cross_entropy(
                    part.reshape(-1, part.shape[-1]).float(),
                    targets[:, start:stop].reshape(-1),
                    reduction="sum",
                )
                del part
            token_loss = total / float(B * T)
            return None, token_loss + lambda_bal * Lb + lambda_red * Lr

        logits = self.head(h)
        loss = None
        if targets is not None:
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
            loss = token_loss + lambda_bal * Lb + lambda_red * Lr
        return logits, loss

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def structures(self) -> List[Optional[CRPAStructure]]:
        return [blk.attn._structure for blk in self.blocks]
