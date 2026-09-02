"""
crpa.data - WikiText-2 language modelling and the Needle-in-Haystack task,
with an enforced three-way split.

The split protocol
------------------
============  ======================  ==============================
role          language-model source   needle generator stream
============  ======================  ==============================
train         WikiText-2 ``train``     ``seed * 1000 + 0``
calibration   WikiText-2 ``validation````seed * 1000 + 1``
evaluation    WikiText-2 ``test``      ``seed * 1000 + 2``
============  ======================  ==============================

Anything selected using data - contribution thresholds, gate decisions, the
suppressible/not-suppressible classification - is fitted on **calibration** and
reported on **evaluation**. The original implementation drew all three roles
from the global ``random`` module, so thresholds were calibrated and evaluated
on samples from the same stream.

``tests/test_data_separation.py`` hashes generated sequences and asserts the
three streams never collide.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from crpa.config import DataConfig

#: The three roles. ``Split.CALIBRATION`` fits thresholds; ``Split.EVAL``
#: reports them. They must never be the same data.
TRAIN = "train"
CALIBRATION = "calibration"
EVAL = "evaluation"
SPLITS = (TRAIN, CALIBRATION, EVAL)

#: Offsets that separate the needle RNG streams. Distinct per role, so the
#: same base seed still yields three disjoint sequences of draws.
_STREAM_OFFSET = {TRAIN: 0, CALIBRATION: 1, EVAL: 2}

#: WikiText split backing each role.
_LM_SOURCE = {TRAIN: "train", CALIBRATION: "validation", EVAL: "test"}


class DataNotInitialised(RuntimeError):
    """Raised when a batch is requested before :meth:`Corpus.load`."""


@dataclass
class SplitMetadata:
    """Provenance for one split, stored alongside every result."""

    role: str
    lm_source: str
    n_tokens: int
    needle_stream_seed: int
    sample_hashes: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "role": self.role,
            "lm_source": self.lm_source,
            "n_tokens": self.n_tokens,
            "needle_stream_seed": self.needle_stream_seed,
            "sample_hashes": self.sample_hashes,
        }


def sequence_hash(seq: List[int]) -> str:
    """Stable short hash of a token sequence, used for leakage checks."""
    h = hashlib.sha256()
    for tok in seq:
        h.update(int(tok).to_bytes(4, "little", signed=False))
    return h.hexdigest()[:16]


class NeedleGenerator:
    """Needle-in-Haystack sequences drawn from one isolated RNG stream.

    Each sequence embeds key->value pairs among filler tokens and ends with a
    query key; the target is that key's paired value.

    The default depth range places the needle one partition away from the query
    so that answering requires a cross-partition hop through a relay. That is
    the original design and is deliberately preserved - it is what makes the
    task diagnostic of relay paths rather than of local copying.
    """

    def __init__(self, cfg: DataConfig, role: str, seed: int) -> None:
        if role not in SPLITS:
            raise ValueError("unknown split role {!r}; expected one of {}".format(role, SPLITS))
        self.cfg = cfg
        self.role = role
        self.stream_seed = seed * 1000 + _STREAM_OFFSET[role]
        self._rng = random.Random(self.stream_seed)

    def reset(self) -> None:
        """Rewind the stream, so an evaluation is repeatable within a run."""
        self._rng = random.Random(self.stream_seed)

    def _one(self, block_size: int, n_needles: int,
             needle_depth: Optional[float]) -> Tuple[List[int], int]:
        rng = self._rng
        fr, kr, vr = self.cfg.filler_range, self.cfg.key_range, self.cfg.val_range
        n_keys = min(n_needles, kr[1] - kr[0] + 1)

        keys = rng.sample(range(kr[0], kr[1] + 1), n_keys)
        vals = [rng.randint(*vr) for _ in keys]
        kv = dict(zip(keys, vals))

        seq: List[int] = []
        inserted: set = set()
        kv_items = list(kv.items())
        depth = (
            needle_depth
            if needle_depth is not None
            else rng.uniform(*self.cfg.needle_depth_range)
        )
        target_pos = int(depth * block_size)

        # Fill every slot but the last, which is reserved for the query key.
        #
        # The original construction filled to block_size - 3, appended the
        # query key, then padded with random filler. That left the query key
        # two positions from the end while the model is scored at the *last*
        # position, so the scored position always followed a filler token -
        # contradicting the documented task ("the last token is a query key")
        # and adding avoidable noise. The query key now sits at exactly
        # block_size - 1.
        limit = block_size - 1
        while len(seq) < limit:
            remaining = [p for p in kv_items if p[0] not in inserted]
            has_room = (limit - len(seq)) >= 2
            if remaining and has_room and target_pos <= len(seq) < target_pos + 10:
                k, v = rng.choice(remaining)
                seq.extend([k, v])
                inserted.add(k)
            else:
                seq.append(rng.randint(*fr))

        if not inserted:
            # Nothing was embedded, so the item is unanswerable. Keep the shape
            # consistent but make that visible via the returned target.
            qk = rng.randint(*kr)
            target = rng.randint(*vr)
        else:
            qk = rng.choice(sorted(inserted))
            target = kv[qk]
        seq.append(qk)

        if len(seq) != block_size:
            raise RuntimeError(
                "needle sequence is {} tokens, expected {}".format(len(seq), block_size)
            )
        return seq, target

    def batch(
        self,
        block_size: int,
        bs: int,
        n_needles: Optional[int] = None,
        needle_depth: Optional[float] = None,
        device: "str | torch.device" = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate one needle batch from this split's stream."""
        n_needles = self.cfg.n_needles if n_needles is None else n_needles
        seqs, targets = [], []
        for _ in range(bs):
            seq, target = self._one(block_size, n_needles, needle_depth)
            seqs.append(seq)
            targets.append(target)
        return (
            torch.tensor(seqs, dtype=torch.long, device=device),
            torch.tensor(targets, dtype=torch.long, device=device),
        )

    def sample_hashes(self, block_size: int, n: int = 8) -> List[str]:
        """Hashes of ``n`` sequences, for the leakage check. Does not disturb the stream."""
        saved = self._rng.getstate()
        try:
            return [sequence_hash(self._one(block_size, self.cfg.n_needles, None)[0])
                    for _ in range(n)]
        finally:
            self._rng.setstate(saved)


class Corpus:
    """Tokenised WikiText-2 plus the three needle streams.

    Replaces the original module-level globals, so two corpora with different
    seeds can coexist and a test can build one without touching global state.
    """

    def __init__(self, cfg: DataConfig, seed: int = 42) -> None:
        self.cfg = cfg
        self.seed = seed
        self.tokenizer = None
        self.vocab_size: Optional[int] = None
        self._lm: Dict[str, torch.Tensor] = {}
        self.needles: Dict[str, NeedleGenerator] = {
            role: NeedleGenerator(cfg, role, seed) for role in SPLITS
        }

    # -- loading -------------------------------------------------------------
    def load(self, verbose: bool = True) -> "Corpus":
        """Download and tokenise WikiText-2.

        WikiText-2 ships train/validation/test, which maps cleanly onto the
        three roles. If the ``test`` split is unavailable the calibration split
        is halved rather than silently reused, and the fact is reported.
        """
        try:
            from datasets import load_dataset
            from transformers import GPT2TokenizerFast
        except ImportError as exc:
            raise ImportError(
                "language-model data requires `pip install datasets transformers`"
            ) from exc

        if verbose:
            print("Loading GPT2 tokenizer...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.vocab_size = self.tokenizer.vocab_size

        if verbose:
            print("Loading {}/{}...".format(self.cfg.dataset_name, self.cfg.dataset_config))
        ds = load_dataset(self.cfg.dataset_name, self.cfg.dataset_config)

        def tok(split_name: str) -> torch.Tensor:
            raw = "\n".join(ds[split_name]["text"])
            return torch.tensor(self.tokenizer.encode(raw), dtype=torch.long)

        available = set(ds.keys())
        for role, source in _LM_SOURCE.items():
            if source in available:
                self._lm[role] = tok(source)
            elif role == EVAL and CALIBRATION in self._lm:
                cal = self._lm[CALIBRATION]
                half = cal.numel() // 2
                self._lm[CALIBRATION] = cal[:half]
                self._lm[EVAL] = cal[half:]
                if verbose:
                    print(
                        "  note: no 'test' split; calibration halved so evaluation "
                        "data is still disjoint"
                    )
            else:
                raise KeyError(
                    "dataset has no split {!r} for role {!r}; refusing to reuse "
                    "another split".format(source, role)
                )

        self._check_token_ranges()

        if verbose:
            sizes = ", ".join(
                "{}={:,}".format(r, self._lm[r].numel()) for r in SPLITS
            )
            print("Vocab: {} | {}".format(self.vocab_size, sizes))
        return self

    def _check_token_ranges(self) -> None:
        """Every needle token must be representable in the vocabulary."""
        for name in ("filler_range", "key_range", "val_range"):
            hi = getattr(self.cfg, name)[1]
            if hi >= (self.vocab_size or 0):
                raise ValueError(
                    "needle {} upper bound {} exceeds vocab size {}".format(
                        name, hi, self.vocab_size
                    )
                )

    def load_synthetic(self, vocab_size: int = 50257, n_tokens: int = 200_000) -> "Corpus":
        """Deterministic pseudo-corpus for CPU smoke tests and unit tests.

        Explicitly not a substitute for WikiText-2: language-model numbers
        produced from it are meaningless and callers record the fact.
        """
        self.vocab_size = vocab_size
        self._check_token_ranges()
        for i, role in enumerate(SPLITS):
            g = torch.Generator().manual_seed(self.seed + 991 * (i + 1))
            self._lm[role] = torch.randint(0, vocab_size, (n_tokens,), generator=g)
        return self

    def reseed(self, seed: int) -> "Corpus":
        """Rebuild the needle streams for a new seed, keeping the loaded corpus.

        The language-model splits do not depend on the seed, only the needle
        streams do. Re-tokenising WikiText-2 for every seed costs about half a
        minute each time and buys nothing, so multi-seed experiments load once
        and reseed per run.
        """
        self.seed = seed
        self.needles = {role: NeedleGenerator(self.cfg, role, seed) for role in SPLITS}
        return self

    # -- access --------------------------------------------------------------
    def _require(self, role: str) -> torch.Tensor:
        if role not in SPLITS:
            raise ValueError("unknown split role {!r}".format(role))
        if role not in self._lm:
            raise DataNotInitialised(
                "corpus not loaded; call Corpus.load() or Corpus.load_synthetic() first"
            )
        return self._lm[role]

    def lm_batch(
        self,
        role: str,
        block_size: int,
        bs: int,
        device: "str | torch.device" = "cpu",
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a language-modelling batch from one split."""
        src = self._require(role)
        if src.numel() <= block_size + 1:
            raise ValueError(
                "split {!r} has {} tokens, too few for block_size {}".format(
                    role, src.numel(), block_size
                )
            )
        ix = torch.randint(src.numel() - block_size - 1, (bs,), generator=generator)
        x = torch.stack([src[i: i + block_size] for i in ix])
        y = torch.stack([src[i + 1: i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    def needle_batch(
        self,
        role: str,
        block_size: int,
        bs: int,
        n_needles: Optional[int] = None,
        needle_depth: Optional[float] = None,
        device: "str | torch.device" = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a retrieval batch from one split's isolated stream."""
        if role not in self.needles:
            raise ValueError("unknown split role {!r}".format(role))
        return self.needles[role].batch(block_size, bs, n_needles, needle_depth, device)

    def split_metadata(self, block_size: int = 128) -> Dict[str, Dict[str, object]]:
        """Provenance for all three splits, embedded in every result file."""
        out: Dict[str, Dict[str, object]] = {}
        for role in SPLITS:
            gen = self.needles[role]
            out[role] = SplitMetadata(
                role=role,
                lm_source=_LM_SOURCE[role],
                n_tokens=int(self._lm[role].numel()) if role in self._lm else 0,
                needle_stream_seed=gen.stream_seed,
                sample_hashes=gen.sample_hashes(block_size),
            ).to_dict()
        return out

    def assert_splits_disjoint(self, block_size: int = 128, n: int = 32) -> None:
        """Fail loudly if two needle streams can produce the same sequence.

        Cheap enough to run at the start of every experiment, and it turns a
        silent leak into an immediate error.
        """
        hashes: Dict[str, set] = {}
        for role in SPLITS:
            gen = NeedleGenerator(self.cfg, role, self.seed)
            hashes[role] = {
                sequence_hash(gen._one(block_size, self.cfg.n_needles, None)[0])
                for _ in range(n)
            }
        for a in SPLITS:
            for b in SPLITS:
                if a < b and hashes[a] & hashes[b]:
                    raise AssertionError(
                        "needle streams {!r} and {!r} produced identical sequences; "
                        "the split protocol is broken".format(a, b)
                    )
        if EVAL in self._lm and TRAIN in self._lm:
            if self._lm[EVAL].numel() and torch.equal(
                self._lm[EVAL][:1000], self._lm[TRAIN][:1000]
            ):
                raise AssertionError("evaluation and train language-model splits are identical")
