"""
crpa.kvcache - key/value cache accounting for autoregressive decoding.

This repository has no incremental decoding loop, so the figures produced here
are **analytical estimates** unless ``measure=True`` is passed, which actually
allocates the tensors and reports their real footprint. Every record carries a
``measurement`` field saying which it is. Nothing here may be reported as a
measured decoding cost.

An honest caveat about CRPA
---------------------------
CRPA does **not** bound the KV cache in its published form. A token's routed
set C_k(i) is chosen from *any* earlier position whose router assignment
differs, so no earlier key/value pair can be proven unnecessary and evicted.
The cache therefore grows linearly in T exactly as the dense baseline's does.

Only the local window and the relays are structurally evictable. We report that
separately as ``crpa_bounded`` - a variant that drops cross-partition routing -
so the difference between "what CRPA costs" and "what a bounded-cache variant
of CRPA would cost" stays visible rather than being quietly conflated.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import torch

from crpa.config import ModelConfig

#: Bytes per element for the dtypes benchmarks use.
DTYPE_BYTES = {
    "float32": 4, "fp32": 4,
    "bfloat16": 2, "bf16": 2,
    "float16": 2, "fp16": 2,
    "float8": 1, "fp8": 1,
}


def dtype_bytes(dtype: "str | torch.dtype") -> int:
    """Element size in bytes for a dtype name or a torch dtype."""
    if isinstance(dtype, torch.dtype):
        return torch.empty((), dtype=dtype).element_size()
    key = str(dtype).lower().replace("torch.", "")
    if key not in DTYPE_BYTES:
        raise ValueError("unknown dtype {!r}; expected one of {}".format(
            dtype, sorted(set(DTYPE_BYTES))))
    return DTYPE_BYTES[key]


@dataclass
class KVCacheEstimate:
    """Cache footprint for one attention scheme at one context length."""

    scheme: str
    context_length: int
    cached_positions: int
    bytes_total: int
    bytes_per_token: float
    megabytes: float
    gigabytes: float
    measurement: str          # "analytical" or "measured"
    dtype: str
    batch_size: int
    n_layer: int
    n_head: int
    d_head: int
    evictable: bool
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def cached_positions(scheme: str, context_length: int, cfg: ModelConfig) -> int:
    """How many positions must be retained under each scheme.

    ``dense``          every earlier position
    ``sliding``        the last ``partition_size`` positions
    ``crpa``           every earlier position - routing is unrestricted, so
                       nothing can be evicted
    ``crpa_bounded``   local window plus relays, a variant that gives up
                       cross-partition routing in exchange for a bounded cache
    """
    if scheme == "dense":
        return context_length
    if scheme == "sliding":
        return min(context_length, cfg.partition_size)
    if scheme == "crpa":
        return context_length
    if scheme == "crpa_bounded":
        return min(context_length, cfg.partition_size + cfg.n_relays)
    raise ValueError("unknown attention scheme {!r}".format(scheme))


def estimate_kv_cache(
    scheme: str,
    context_length: int,
    cfg: ModelConfig,
    dtype: str = "bfloat16",
    batch_size: int = 1,
    measure: bool = False,
    device: str = "cpu",
) -> KVCacheEstimate:
    """Cache footprint for one scheme.

    The analytical formula is

        bytes = 2 (K and V) * batch * n_layer * n_head * d_head
                * cached_positions * bytes_per_element

    With ``measure=True`` the tensors are actually allocated and their real
    ``nbytes`` reported. That is a genuine measurement of the footprint, though
    still not a measurement of decoding throughput, which this repository does
    not implement.
    """
    n_cached = cached_positions(scheme, context_length, cfg)
    elem = dtype_bytes(dtype)
    evictable = scheme in ("sliding", "crpa_bounded")

    note = ""
    if scheme == "crpa":
        note = (
            "CRPA does not bound the KV cache: routed keys C_k(i) may reference "
            "any earlier position, so no entry is provably evictable. Compare "
            "with crpa_bounded, which drops routing to gain a bounded cache."
        )
    elif scheme == "crpa_bounded":
        note = (
            "Hypothetical variant without cross-partition routing. Bounded cache, "
            "but it gives up the mechanism CRPA relies on for long-range retrieval."
        )

    if measure:
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}[dtype]
        k = torch.zeros(
            (batch_size, cfg.n_layer, cfg.n_head, n_cached, cfg.d_head),
            dtype=torch_dtype, device=device,
        )
        v = torch.zeros_like(k)
        total = int(k.numel() * k.element_size() + v.numel() * v.element_size())
        del k, v
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        measurement = "measured"
    else:
        total = int(
            2 * batch_size * cfg.n_layer * cfg.n_head * cfg.d_head * n_cached * elem
        )
        measurement = "analytical"

    return KVCacheEstimate(
        scheme=scheme,
        context_length=context_length,
        cached_positions=n_cached,
        bytes_total=total,
        bytes_per_token=total / max(context_length, 1),
        megabytes=total / 1e6,
        gigabytes=total / 1e9,
        measurement=measurement,
        dtype=dtype,
        batch_size=batch_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        d_head=cfg.d_head,
        evictable=evictable,
        note=note,
    )


def kv_cache_table(
    cfg: ModelConfig,
    context_lengths: List[int],
    schemes: Optional[List[str]] = None,
    dtype: str = "bfloat16",
    batch_size: int = 1,
    measure_up_to: int = 0,
    device: str = "cpu",
) -> List[Dict[str, object]]:
    """Cache footprints across schemes and context lengths.

    Args:
        measure_up_to: actually allocate for context lengths at or below this,
            and project beyond it. Rows are labelled accordingly, so a
            projection is never mistaken for an allocation.
    """
    schemes = schemes or ["dense", "sliding", "crpa", "crpa_bounded"]
    rows: List[Dict[str, object]] = []
    for T in context_lengths:
        for scheme in schemes:
            rows.append(
                estimate_kv_cache(
                    scheme, T, cfg, dtype=dtype, batch_size=batch_size,
                    measure=(T <= measure_up_to), device=device,
                ).to_dict()
            )
    return rows


def attention_edge_counts(cfg: ModelConfig, context_length: int) -> Dict[str, float]:
    """Permitted attention entries per layer, dense versus CRPA.

    Analytical, and an upper bound for CRPA: the three sources can overlap, so
    the realised count is at or below ``crpa_edges``. The measured count is
    available from :func:`crpa.evaluate.sparsity_report`.
    """
    T = context_length
    dense = T * (T + 1) / 2
    per_query = min(cfg.partition_size, T) + cfg.n_relays + cfg.cross_k
    crpa = T * per_query + cfg.n_relays * T  # relay rows attend causally to all
    return {
        "context_length": T,
        "dense_causal_edges": dense,
        "crpa_edges_upper_bound": min(crpa, dense),
        "sparsity_ratio_upper_bound": min(crpa, dense) / dense,
        "edges_per_query": per_query,
    }
