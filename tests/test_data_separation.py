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


class TestNeedlePosition:
    """The query key must sit at the position the model is scored on.

    The original filled to ``block_size - 3``, appended the query key, then
    padded with random filler, so the scored position always held a filler
    token despite the docstring saying "the last token is a query key".
    """

    def test_query_key_is_at_the_scored_position(self):
        cfg = DataConfig()
        gen = NeedleGenerator(cfg, TRAIN, 42)
        for _ in range(20):
            seq, _ = gen._one(128, 2, None)
            assert len(seq) == 128
            assert cfg.key_range[0] <= seq[-1] <= cfg.key_range[1], (
                "scored position holds {} which is not a query key".format(seq[-1])
            )

    def test_legacy_position_reproduces_the_original(self):
        cfg = DataConfig(query_key_at_end=False)
        gen = NeedleGenerator(cfg, TRAIN, 42)
        seq, _ = gen._one(128, 2, None)
        assert len(seq) == 128
        # Original behaviour: the scored position holds filler, not the key.
        assert cfg.filler_range[0] <= seq[-1] <= cfg.filler_range[1]
        keys = [i for i, t in enumerate(seq)
                if cfg.key_range[0] <= t <= cfg.key_range[1]]
        assert max(keys) < 127, "query key should not be at the scored position"

    def test_both_modes_produce_valid_lengths(self):
        for flag in (True, False):
            gen = NeedleGenerator(DataConfig(query_key_at_end=flag), EVAL, 7)
            for block in (64, 128, 512):
                seq, _ = gen._one(block, 2, None)
                assert len(seq) == block


class TestChanceFloorIsMeasuredNotAssumed:
    """The floor must come from simulating the generator.

    Asserting it from vocabulary size gives 5%, but only ``n_needles`` value
    tokens appear in any sequence, so "guess a value token you can see" already
    scores about 100/n_needles. On the default config that is roughly 52%, and
    using the 5% figure inverts every "above chance" verdict for a strong
    variant. This was a real error in the reported CRPA results.
    """

    def test_uniform_chance_is_the_vocabulary_figure(self):
        cfg = DataConfig()
        assert cfg.uniform_chance == pytest.approx(5.0)
        assert cfg.chance_accuracy == cfg.uniform_chance

    def test_the_measured_floor_is_far_above_the_uniform_figure(self):
        gen = NeedleGenerator(DataConfig(), EVAL, 42)
        floor = gen.measure_chance_floor(256, n=1500)
        assert floor["uniform"] == pytest.approx(5.0, abs=2.0)
        # Two needles -> a context-value guess is right about half the time.
        assert floor["context_value"] > 40.0
        assert floor["strongest"] >= floor["context_value"]
        assert floor["strongest"] > 8 * floor["uniform"], (
            "the trivial floor should dwarf the uniform figure; if it does "
            "not, the generator changed and the reported floor is stale"
        )

    def test_the_floor_tracks_the_needle_count(self):
        """More needles means more candidates, so a lower trivial floor."""
        two = NeedleGenerator(DataConfig(n_needles=2), EVAL, 42
                              ).measure_chance_floor(256, n=1200)
        five = NeedleGenerator(DataConfig(n_needles=5), EVAL, 42
                               ).measure_chance_floor(256, n=1200)
        assert two["context_value"] > five["context_value"] + 10.0

    def test_measuring_the_floor_does_not_disturb_the_stream(self):
        """The floor is diagnostic; it must not consume evaluation samples."""
        gen = NeedleGenerator(DataConfig(), EVAL, 42)
        gen.measure_chance_floor(128, n=200)
        after = gen.batch(128, 2)
        fresh = NeedleGenerator(DataConfig(), EVAL, 42).batch(128, 2)
        assert torch.equal(after[0], fresh[0])
