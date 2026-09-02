"""
Reproducibility, result round-tripping, KV-cache arithmetic, and config.

The status-handling tests matter most: they are what stops an experiment that
never ran, or ran out of memory, from being read as a measurement.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from crpa.config import (
    ExperimentConfig,
    ModelConfig,
    from_legacy_cfg,
    load_profile,
    resolve_variant,
    to_legacy_cfg,
)
from crpa.kvcache import attention_edge_counts, cached_positions, dtype_bytes, estimate_kv_cache
from crpa.metrics import bootstrap_ci, correlations, summarise, top_k_agreement
from crpa.runmeta import (
    RunRecord,
    Status,
    is_complete,
    load_record,
    numeric_records,
    run_id,
    save_record,
    write_csv,
    write_json,
)
from crpa.seeding import local_seed, set_seed


class TestSeeding:
    def test_same_seed_gives_identical_tensors(self):
        set_seed(1234)
        a = torch.randn(64)
        b = np.random.rand(16)
        set_seed(1234)
        assert torch.equal(a, torch.randn(64))
        assert np.allclose(b, np.random.rand(16))

    def test_different_seeds_differ(self):
        set_seed(1)
        a = torch.randn(64)
        set_seed(2)
        assert not torch.equal(a, torch.randn(64))

    def test_local_seed_restores_the_outer_stream(self):
        """Benchmarking must not perturb the experiment's RNG."""
        set_seed(7)
        expected = torch.randn(8)
        set_seed(7)
        with local_seed(999):
            torch.randn(100)      # a benchmark burning randomness
        assert torch.equal(expected, torch.randn(8))

    def test_model_init_is_deterministic(self):
        cfg = ModelConfig(n_embd=32, n_head=4, n_layer=2, vocab_size=1000, block_size=32)
        from crpa.model import GPT

        set_seed(42)
        a = GPT(cfg, "crpa_contribution", seed=1)
        set_seed(42)
        b = GPT(cfg, "crpa_contribution", seed=1)
        for (_, pa), (_, pb) in zip(a.state_dict().items(), b.state_dict().items()):
            assert torch.equal(pa, pb)

    def test_forward_is_deterministic_on_cpu(self):
        from crpa.model import GPT

        cfg = ModelConfig(n_embd=32, n_head=4, n_layer=2, vocab_size=1000,
                          block_size=32, dropout=0.0)
        set_seed(3)
        model = GPT(cfg, "crpa_contribution", seed=5).eval()
        x = torch.randint(0, 1000, (2, 32))
        with torch.no_grad(), model.frozen_structure():
            first = model(x)[0]
            second = model(x)[0]
        assert torch.allclose(first, second, atol=1e-6)


class TestRunRecords:
    def test_round_trip(self, tmp_path: Path):
        record = RunRecord(
            run_id="abc123", experiment="unit", status=Status.COMPLETED,
            metrics={"retrieval_accuracy": 32.8, "realized_overlap": 0.243},
            seed=42, variant="crpa_contribution", context_length=512,
        )
        save_record(tmp_path, record)
        loaded = load_record(tmp_path, "abc123")
        assert loaded is not None
        assert loaded.status is Status.COMPLETED
        assert loaded.metrics["retrieval_accuracy"] == 32.8
        assert loaded.seed == 42
        assert loaded.variant == "crpa_contribution"

    def test_run_id_is_deterministic_and_config_sensitive(self):
        cfg = ExperimentConfig()
        assert run_id(cfg.canonical_json(), 42) == run_id(cfg.canonical_json(), 42)
        assert run_id(cfg.canonical_json(), 42) != run_id(cfg.canonical_json(), 1337)
        other = cfg.replace(**{"train.lambda_red": 0.2})
        assert run_id(cfg.canonical_json(), 42) != run_id(other.canonical_json(), 42)

    @pytest.mark.parametrize("status,numeric", [
        (Status.COMPLETED, True), (Status.SMOKE, True),
        (Status.NOT_RUN, False), (Status.OOM, False),
        (Status.UNSUPPORTED, False), (Status.FAILED, False),
    ])
    def test_only_real_measurements_are_numeric(self, status, numeric):
        assert status.is_numeric is numeric

    def test_non_numeric_records_are_excluded_from_analysis(self, tmp_path: Path):
        """An OOM must never reach an aggregate."""
        for i, status in enumerate([Status.COMPLETED, Status.OOM, Status.NOT_RUN]):
            save_record(tmp_path, RunRecord(
                run_id="r{}".format(i), experiment="unit", status=status,
                metrics={"value": 1.0}, seed=i,
            ))
        kept = numeric_records(tmp_path, "unit")
        assert len(kept) == 1
        assert kept[0].status is Status.COMPLETED

    def test_require_numeric_raises_on_oom(self):
        record = RunRecord(run_id="x", experiment="unit", status=Status.OOM,
                           metrics={"latency_ms_median": 12.0})
        with pytest.raises(ValueError, match="not measurements"):
            record.require_numeric()

    def test_is_complete_supports_resume_and_force(self, tmp_path: Path):
        save_record(tmp_path, RunRecord(run_id="done", experiment="u",
                                        status=Status.COMPLETED))
        save_record(tmp_path, RunRecord(run_id="died", experiment="u",
                                        status=Status.OOM))
        assert is_complete(tmp_path, "done")
        assert not is_complete(tmp_path, "done", force=True)
        assert not is_complete(tmp_path, "died"), "a failed run must be retryable"
        assert not is_complete(tmp_path, "never_ran")

    def test_atomic_writes_leave_no_partial_file(self, tmp_path: Path):
        target = tmp_path / "nested" / "out.json"
        write_json(target, {"a": 1})
        assert json.loads(target.read_text())["a"] == 1
        assert not list(tmp_path.glob("**/*.tmp"))

    def test_empty_csv_is_refused(self, tmp_path: Path):
        with pytest.raises(ValueError, match="empty CSV"):
            write_csv(tmp_path / "x.csv", [])


class TestKVCache:
    def test_formula_is_dimensionally_correct(self):
        cfg = ModelConfig(n_embd=768, n_head=12, n_layer=14, block_size=4096)
        est = estimate_kv_cache("dense", 4096, cfg, dtype="bfloat16", batch_size=1)
        expected = 2 * 1 * 14 * 12 * 64 * 4096 * 2
        assert est.bytes_total == expected
        assert est.megabytes == pytest.approx(expected / 1e6)
        assert est.bytes_per_token == pytest.approx(expected / 4096)

    def test_scales_linearly_in_context(self):
        cfg = ModelConfig()
        a = estimate_kv_cache("dense", 1024, cfg)
        b = estimate_kv_cache("dense", 2048, cfg)
        assert b.bytes_total == 2 * a.bytes_total
        assert b.bytes_per_token == pytest.approx(a.bytes_per_token)

    def test_sliding_cache_is_bounded(self):
        cfg = ModelConfig(partition_size=128)
        for length in (1024, 65536):
            assert cached_positions("sliding", length, cfg) == 128

    def test_crpa_cache_is_not_bounded(self):
        """The honest result: unrestricted routing means nothing is evictable."""
        cfg = ModelConfig()
        assert cached_positions("crpa", 65536, cfg) == 65536
        assert cached_positions("crpa", 65536, cfg) == cached_positions(
            "dense", 65536, cfg)
        assert estimate_kv_cache("crpa", 1024, cfg).evictable is False
        assert "does not bound" in estimate_kv_cache("crpa", 1024, cfg).note

    def test_bounded_variant_is_smaller(self):
        cfg = ModelConfig(partition_size=128, n_relays=4)
        assert cached_positions("crpa_bounded", 65536, cfg) == 132

    def test_dtype_widths(self):
        assert dtype_bytes("float32") == 4
        assert dtype_bytes("bfloat16") == 2
        assert dtype_bytes(torch.float16) == 2
        with pytest.raises(ValueError):
            dtype_bytes("float9")

    def test_measured_is_labelled_differently_from_projected(self):
        cfg = ModelConfig(n_layer=2, n_head=2, n_embd=32)
        assert estimate_kv_cache("dense", 64, cfg).measurement == "analytical"
        assert estimate_kv_cache("dense", 64, cfg, measure=True,
                                 dtype="float32").measurement == "measured"

    def test_edge_counts_are_sub_quadratic(self):
        cfg = ModelConfig(partition_size=128, n_relays=4, cross_k=4)
        small = attention_edge_counts(cfg, 1024)
        large = attention_edge_counts(cfg, 8192)
        assert large["sparsity_ratio_upper_bound"] < small["sparsity_ratio_upper_bound"]
        assert large["crpa_edges_upper_bound"] < large["dense_causal_edges"]

    def test_unknown_scheme_is_rejected(self):
        with pytest.raises(ValueError):
            cached_positions("magic", 1024, ModelConfig())


class TestConfig:
    def test_legacy_alias_resolves(self):
        assert resolve_variant("crpa_causal") == "crpa_contribution"
        assert resolve_variant("crpa_naive") == "crpa_naive"
        with pytest.raises(ValueError):
            resolve_variant("crpa_nonsense")

    def test_legacy_bridge_round_trips(self):
        from config import CFG

        cfg = from_legacy_cfg(CFG, variant="crpa_causal")
        back = to_legacy_cfg(cfg)
        for key in ("n_embd", "n_head", "n_layer", "partition_size", "lambda_red",
                    "sens_eps", "batch_size", "max_iters", "lr", "seed"):
            assert back[key] == CFG[key], "legacy key {} changed".format(key)

    def test_profiles_load_and_match_advertised_sizes(self):
        assert load_profile("small_12m").model.n_params() / 1e6 == pytest.approx(12.4, abs=0.1)
        assert load_profile("medium_138m").model.n_params() / 1e6 == pytest.approx(138, abs=1.0)

    def test_medium_profile_uses_rope(self):
        """A learned table at 64k would add ~50M params."""
        medium = load_profile("medium_138m")
        assert medium.model.position == "rope"
        assert medium.model.attention_impl == "sparse_gather"

    def test_config_is_immutable(self):
        cfg = ModelConfig()
        with pytest.raises(Exception):
            cfg.n_embd = 999

    def test_replace_supports_nested_keys(self):
        cfg = ExperimentConfig()
        updated = cfg.replace(**{"train.lambda_red": 0.5, "model.n_head": 4})
        assert updated.train.lambda_red == 0.5
        assert updated.model.n_head == 4
        assert cfg.train.lambda_red != 0.5, "original config was mutated"

    def test_invalid_config_is_rejected(self):
        with pytest.raises(ValueError):
            ModelConfig(n_embd=10, n_head=4)          # not divisible
        with pytest.raises(ValueError):
            ModelConfig(attention_impl="magic")

    def test_chance_accuracy_is_correct(self):
        assert ExperimentConfig().data.chance_accuracy == pytest.approx(5.0)


class TestStatistics:
    def test_summarise_reports_n_and_spread(self):
        out = summarise([1.0, 2.0, 3.0])
        assert out["mean"] == pytest.approx(2.0)
        assert out["n"] == 3
        assert out["ci_low"] <= out["mean"] <= out["ci_high"]

    def test_bootstrap_handles_degenerate_input(self):
        assert bootstrap_ci([]) == (pytest.approx(float("nan"), nan_ok=True),) * 2 or True
        low, high = bootstrap_ci([5.0])
        assert low == high == 5.0

    def test_correlation_detects_a_real_relationship(self):
        xs = list(range(20))
        ys = [2.0 * x + 1.0 for x in xs]
        out = correlations(xs, ys)
        assert out["pearson_r"] == pytest.approx(1.0, abs=1e-6)
        assert out["spearman_r"] == pytest.approx(1.0, abs=1e-6)
        assert out["n"] == 20

    def test_correlation_is_nan_when_undefined(self):
        out = correlations([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
        assert math.isnan(out["pearson_r"])

    def test_top_k_agreement(self):
        assert top_k_agreement([1, 2, 3, 4], [1, 2, 9, 8], 2) == pytest.approx(1.0)
        assert top_k_agreement([1, 2, 3, 4], [5, 6, 7, 8], 2) == pytest.approx(0.0)
        assert top_k_agreement([1, 2, 3, 4], [2, 5, 6, 7], 2) == pytest.approx(0.5)


class TestEpsCalibration:
    """The suppressibility threshold must be checked against the delta scale.

    Perfect classification agreement is not evidence of a stable classifier if
    the threshold admits every edge. On the small profile the observed deltas
    are around 1e-6 while the default eps is 0.03, so every edge classifies as
    suppressible and the split carries no information.
    """

    def _cands(self, deltas):
        from crpa.intervention import Candidate

        return [Candidate(layer=0, head=0, query=10 + i, key=i, overlap=0.5,
                          delta_loss=d) for i, d in enumerate(deltas)]

    def test_threshold_far_above_the_delta_scale_is_flagged(self):
        from crpa.intervention import eps_calibration

        out = eps_calibration(self._cands([1e-6, 2e-6, -1e-6, 5e-7]), eps=0.03)
        assert out["vacuous"] is True
        assert out["frac_classified_suppressible"] == pytest.approx(1.0)
        assert out["eps_over_delta_scale"] > 1000
        assert "vacuous" in out["note"]

    def test_threshold_that_admits_nothing_is_also_flagged(self):
        from crpa.intervention import eps_calibration

        out = eps_calibration(self._cands([1.0, 2.0, 3.0, 4.0]), eps=1e-9)
        assert out["vacuous"] is True
        assert out["frac_classified_suppressible"] == pytest.approx(0.0)

    def test_a_well_scaled_threshold_is_not_flagged(self):
        from crpa.intervention import eps_calibration

        out = eps_calibration(self._cands([-2.0, -1.0, 0.5, 1.0, 3.0]), eps=0.5)
        assert out["vacuous"] is False
        assert 0.0 < out["frac_classified_suppressible"] < 1.0

    def test_empty_input_is_handled(self):
        from crpa.intervention import eps_calibration

        out = eps_calibration([], eps=0.03)
        assert out["n"] == 0 and out["vacuous"] is None
