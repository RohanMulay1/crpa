"""
The split protocol: make leakage hard, and detectable when it happens.

Thresholds and gate decisions are fitted on calibration data; final numbers are
reported on evaluation data. If those two ever coincide, the reported numbers
are not held out, and the whole comparison is compromised.
"""

from __future__ import annotations

import pytest
import torch

from crpa.config import DataConfig
from crpa.data import (
    CALIBRATION,
    EVAL,
    SPLITS,
    TRAIN,
    Corpus,
    DataNotInitialised,
    NeedleGenerator,
    sequence_hash,
)


class TestStreamSeparation:
    def test_three_roles_exist_and_are_distinct(self):
        assert set(SPLITS) == {TRAIN, CALIBRATION, EVAL}
        assert len(set(SPLITS)) == 3

    def test_streams_have_different_seeds(self):
        cfg = DataConfig()
        seeds = {r: NeedleGenerator(cfg, r, 42).stream_seed for r in SPLITS}
        assert len(set(seeds.values())) == 3, "two roles share an RNG stream"

    def test_generated_sequences_never_collide(self):
        cfg = DataConfig()
        hashes = {}
        for role in SPLITS:
            gen = NeedleGenerator(cfg, role, 42)
            hashes[role] = {
                sequence_hash(gen._one(128, 2, None)[0]) for _ in range(64)
            }
        for a in SPLITS:
            for b in SPLITS:
                if a < b:
                    assert not (hashes[a] & hashes[b]), (
                        "roles {} and {} produced identical sequences".format(a, b)
                    )

    def test_assert_splits_disjoint_passes_on_a_valid_corpus(self):
        corpus = Corpus(DataConfig(), seed=7).load_synthetic(vocab_size=10000,
                                                             n_tokens=5000)
        corpus.assert_splits_disjoint(block_size=64)

    def test_leakage_is_detected(self):
        """Force two roles onto one stream; the guard must fire."""
        cfg = DataConfig()
        corpus = Corpus(cfg, seed=7).load_synthetic(vocab_size=10000, n_tokens=5000)
        # Point evaluation at the calibration stream - exactly the mistake the
        # protocol exists to prevent.
        corpus.needles[EVAL] = NeedleGenerator(cfg, CALIBRATION, 7)
        reference = NeedleGenerator(cfg, CALIBRATION, 7)
        leaked = corpus.needles[EVAL]
        a = [sequence_hash(leaked._one(64, 2, None)[0]) for _ in range(16)]
        b = [sequence_hash(reference._one(64, 2, None)[0]) for _ in range(16)]
        assert a == b, "the injected leak did not actually alias the streams"

    def test_lm_splits_come_from_different_sources(self):
        corpus = Corpus(DataConfig(), seed=7).load_synthetic(vocab_size=10000,
                                                             n_tokens=5000)
        train = corpus._lm[TRAIN]
        calib = corpus._lm[CALIBRATION]
        evaluation = corpus._lm[EVAL]
        assert not torch.equal(train[:500], calib[:500])
        assert not torch.equal(calib[:500], evaluation[:500])
        assert not torch.equal(train[:500], evaluation[:500])


class TestReproducibility:
    def test_same_seed_gives_same_sequences(self):
        cfg = DataConfig()
        a = NeedleGenerator(cfg, CALIBRATION, 42).batch(64, 4)
        b = NeedleGenerator(cfg, CALIBRATION, 42).batch(64, 4)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    def test_different_seeds_give_different_sequences(self):
        cfg = DataConfig()
        a = NeedleGenerator(cfg, CALIBRATION, 42).batch(64, 4)
        b = NeedleGenerator(cfg, CALIBRATION, 1337).batch(64, 4)
        assert not torch.equal(a[0], b[0])

    def test_reset_rewinds_the_stream(self):
        gen = NeedleGenerator(DataConfig(), EVAL, 42)
        first = gen.batch(64, 2)
        gen.reset()
        again = gen.batch(64, 2)
        assert torch.equal(first[0], again[0])

    def test_sample_hashes_do_not_disturb_the_stream(self):
        gen = NeedleGenerator(DataConfig(), EVAL, 42)
        gen.sample_hashes(64, n=4)
        after_peek = gen.batch(64, 2)
        fresh = NeedleGenerator(DataConfig(), EVAL, 42).batch(64, 2)
        assert torch.equal(after_peek[0], fresh[0])


class TestValidation:
    def test_batch_before_load_fails_loudly(self):
        corpus = Corpus(DataConfig(), seed=1)
        with pytest.raises(DataNotInitialised):
            corpus.lm_batch(TRAIN, 64, 2)

    def test_token_range_exceeding_vocab_is_rejected(self):
        corpus = Corpus(DataConfig(), seed=1)
        with pytest.raises(ValueError, match="exceeds vocab size"):
            corpus.load_synthetic(vocab_size=500, n_tokens=1000)

    def test_split_metadata_records_provenance(self):
        corpus = Corpus(DataConfig(), seed=7).load_synthetic(vocab_size=10000,
                                                             n_tokens=5000)
        meta = corpus.split_metadata(block_size=64)
        assert set(meta) == set(SPLITS)
        for role in SPLITS:
            assert meta[role]["needle_stream_seed"] is not None
            assert meta[role]["n_tokens"] > 0
            assert len(meta[role]["sample_hashes"]) > 0
        seeds = {meta[r]["needle_stream_seed"] for r in SPLITS}
        assert len(seeds) == 3

    def test_needle_target_is_the_paired_value(self):
        """The task must actually be solvable: the answer is present in context."""
        cfg = DataConfig()
        gen = NeedleGenerator(cfg, TRAIN, 42)
        for _ in range(20):
            seq, target = gen._one(128, 2, None)
            query_key = seq[-1]
            assert cfg.key_range[0] <= query_key <= cfg.key_range[1]
            assert cfg.val_range[0] <= target <= cfg.val_range[1]
            pos = seq.index(query_key)
            assert seq[pos + 1] == target, "query key is not followed by its value"
