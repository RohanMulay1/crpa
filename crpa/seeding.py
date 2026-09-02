"""
crpa.seeding - deterministic seeding across every RNG the project touches.

The original ``main.set_seed`` seeded ``random``, NumPy and torch CPU but not
``torch.cuda``, and ``evaluate.measure_throughput`` called
``torch.manual_seed(0)`` mid-run, silently clobbering the experiment seed.
Both are fixed here.

Determinism caveats (documented in the README):
  * ``torch.use_deterministic_algorithms(True)`` makes most CUDA kernels
    reproducible but raises on ops with no deterministic implementation.
    It is opt-in via ``strict=True`` rather than the default.
  * cuBLAS reductions need ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` set *before*
    CUDA initialises; :func:`set_seed` sets it but cannot retroactively apply
    it if CUDA is already up.
  * Atomics in scatter/gather backward kernels remain nondeterministic even
    under the strict flag on some GPU/driver combinations.
"""

from __future__ import annotations

import contextlib
import os
import random
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch


def set_seed(seed: int, strict: bool = False) -> None:
    """Seed Python, NumPy, torch CPU and every visible CUDA device.

    Args:
        seed: the seed to apply.
        strict: also request deterministic CUDA algorithms. Slower, and raises
            on operations lacking a deterministic kernel.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if strict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state() -> Dict[str, Any]:
    """Capture every RNG state, for checkpoint/resume."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore states captured by :func:`rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@contextlib.contextmanager
def local_seed(seed: Optional[int]) -> Iterator[None]:
    """Temporarily seed every RNG, then restore the previous state.

    Used by benchmarking so timing runs cannot perturb the experiment's stream.
    ``seed=None`` is a no-op, so callers need not branch.
    """
    if seed is None:
        yield
        return
    saved = rng_state()
    try:
        set_seed(seed)
        yield
    finally:
        restore_rng_state(saved)


def torch_generator(seed: int, device: str = "cpu") -> torch.Generator:
    """An explicit generator, so a draw cannot depend on global RNG order.

    CRPA's cross-partition routing uses one of these. In the original code the
    routing draw came from the global stream and was re-rolled on every mask
    rebuild, which meant an intervention's baseline and masked forward could be
    evaluated under two different masks.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen
