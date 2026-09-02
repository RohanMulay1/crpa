"""
CPU smoke tests for every entry point, plus the Tier 3 tiny-model diagnostic.

These are deliberately tiny. They check that each command runs end to end and
writes well-formed, honestly-labelled results - not that any scientific claim
holds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crpa.runmeta import Status, load_records


def _args(results_dir: Path, *extra: str):
    return [
        "--smoke", "--synthetic_data", "--device", "cpu",
        "--results_dir", str(results_dir), *extra,
    ]


class TestEntryPoints:
    def test_tier1_multiseed(self, tmp_path: Path):
        from experiments.tier1_multiseed import main

        out = tmp_path / "tier1"
        assert main(_args(out, "--seeds", "42")) == 0
        assert (out / "aggregate.json").exists()
        assert (out / "aggregate.csv").exists()

        records = load_records(out)
        assert records, "no run records written"
        for record in records:
            assert record.status is Status.SMOKE, (
                "a --smoke run must never be recorded as completed"
            )
            assert record.git_sha
            assert record.env["torch"]
            assert set(record.splits) == {"train", "calibration", "evaluation"}

    def test_matched_overlap(self, tmp_path: Path):
        from experiments.matched_overlap import main

        out = tmp_path / "matched"
        assert main(_args(out, "--seeds", "42", "--lambdas", "0.0", "0.1",
                          "--tolerance", "0.05")) == 0
        assert (out / "sweep.csv").exists()
        payload = json.loads((out / "matched_pairs.json").read_text())
        assert payload["n_runs"] == 4
        for pair in payload["matched_pairs"]:
            assert pair["overlap_abs_diff"] <= payload["tolerance"]

    def test_overlap_vs_contribution(self, tmp_path: Path):
        from experiments.overlap_vs_contribution import main

        out = tmp_path / "tier1"
        assert main(_args(out, "--seeds", "42", "--n_per_layer", "4")) == 0
        assert (out / "overlap_vs_contribution.csv").exists()
        payload = json.loads((out / "overlap_vs_contribution.json").read_text())
        assert payload["n_rows"] > 0
        summary = payload["per_seed"]["42"]
        assert "correlation" in summary
        assert "group_effects" in summary
        assert "selection_comparison" in summary

    def test_estimator_stability(self, tmp_path: Path):
        from experiments.estimator_stability import main

        out = tmp_path / "stability"
        assert main(_args(out, "--seeds", "42", "--budgets", "2", "4",
                          "--replicates", "2", "--n_candidates", "4",
                          "--top_k", "2")) == 0
        payload = json.loads((out / "stability.json").read_text())
        budgets = {row["sample_budget"] for row in payload["rows"]}
        assert budgets == {2, 4}

    def test_long_context(self, tmp_path: Path):
        from experiments.long_context import main

        out = tmp_path / "tier2"
        assert main(_args(out, "--seeds", "42", "--n_candidates", "4")) == 0
        assert (out / "long_context.csv").exists()
        assert (out / "kv_cache_projection.csv").exists()

    def test_benchmark(self, tmp_path: Path):
        from experiments.benchmark import main

        out = tmp_path / "tier2"
        assert main([
            "--smoke", "--device", "cpu", "--results_dir", str(out),
            "--n_iters", "2", "--n_warmup", "1",
            "--variants", "crpa_contribution",
        ]) == 0
        assert (out / "benchmark.csv").exists()
        assert (out / "kv_cache.csv").exists()

    def test_dry_run_executes_nothing(self, tmp_path: Path):
        from experiments.tier1_multiseed import main

        out = tmp_path / "dry"
        assert main(_args(out, "--dry_run", "--seeds", "42")) == 0
        assert not (out / "runs").exists(), "--dry_run wrote results"

    def test_resume_skips_completed_runs(self, tmp_path: Path, capsys):
        from experiments.tier1_multiseed import main

        out = tmp_path / "resume"
        main(_args(out, "--seeds", "42", "--variants", "crpa_noreg"))
        capsys.readouterr()
        main(_args(out, "--seeds", "42", "--variants", "crpa_noreg"))
        assert "[skip]" in capsys.readouterr().out


class TestFigures:
    def test_missing_data_is_skipped_not_invented(self, tmp_path: Path):
        from crpa.figures import FigureSkipped, fig_large_model

        with pytest.raises(FigureSkipped, match="requires a real frozen-model run"):
            fig_large_model(tmp_path, tmp_path / "figures")

    def test_plot_all_runs_and_reports_skips(self, tmp_path: Path, capsys):
        from experiments.plot_all import main as plot_main
        from experiments.tier1_multiseed import main as tier1_main

        results = tmp_path / "results"
        tier1_main(_args(results / "tier1", "--seeds", "42", "--variants", "crpa_noreg"))
        assert plot_main(["--results_dir", str(results)]) == 0

        captured = capsys.readouterr().out
        assert "fig4_seed_robustness" in captured
        assert (results / "figures" / "fig4_seed_robustness.png").exists()
        # Source data is written beside every rendered figure.
        assert (results / "figures" / "fig4_seed_robustness_data.csv").exists()

    def test_strict_mode_fails_on_skips(self, tmp_path: Path):
        from experiments.plot_all import main as plot_main

        (tmp_path / "tier1").mkdir(parents=True)
        assert plot_main(["--results_dir", str(tmp_path), "--strict"]) == 1


class TestLargeModelDiagnostic:
    """Runs against a genuinely tiny Hugging Face causal LM."""

    @pytest.mark.slow
    def test_tiny_model_end_to_end(self, tmp_path: Path):
        pytest.importorskip("transformers")
        from experiments.large_model_diagnostic import main

        try:
            rc = main([
                "--smoke", "--model_id", "hf-internal-testing/tiny-random-LlamaForCausalLM",
                "--device", "cpu", "--dtype", "float32",
                "--context_length", "64", "--n_candidates", "4",
                "--partition_size", "16", "--results_dir", str(tmp_path),
            ])
        except OSError as exc:
            pytest.skip("tiny model unavailable (offline?): {}".format(exc))

        assert rc == 0
        edge_files = list(tmp_path.glob("edges_*.csv"))
        assert edge_files, "no edge dataset written"
        payload = json.loads(next(tmp_path.glob("diagnostic_*.json")).read_text())
        summary = payload["summary"]
        # Instrumentation must have been proven, not assumed.
        for verification in summary["instrumentation_verification"].values():
            assert verification["verified"] is True
            assert verification["edge_prob_after"] < 1e-6
            assert abs(verification["row_sum_after"] - 1.0) < 1e-2
            assert verification["n_edits_applied"] > 0

    @pytest.mark.slow
    def test_an_inert_intervention_reports_zero_edits(self):
        """The edit counter must be honest, so a no-op cannot pass as a result."""
        pytest.importorskip("transformers")
        import torch

        from experiments.large_model_diagnostic import AttentionProbe, load_model

        try:
            model, _ = load_model("hf-internal-testing/tiny-random-LlamaForCausalLM",
                                  dtype="float32")
        except OSError as exc:
            pytest.skip("tiny model unavailable (offline?): {}".format(exc))

        x = torch.randint(0, int(model.config.vocab_size), (1, 32))
        with AttentionProbe(model, 0) as probe:
            probe.reset()
            probe.edges = [(0, 999, 999)]      # out of range for a 32-token input
            with torch.no_grad():
                model(x)
            assert probe.n_intervened == 0, (
                "an out-of-range edge was counted as an applied intervention"
            )

            probe.reset()
            probe.edges = [(0, 31, 5)]         # a real edge
            with torch.no_grad():
                model(x)
            assert probe.n_intervened == 1

    @pytest.mark.slow
    def test_probe_only_captures_its_own_layer(self):
        """Capture must be scoped, or a later layer overwrites the edited one."""
        pytest.importorskip("transformers")
        import torch

        from experiments.large_model_diagnostic import (
            AttentionProbe,
            decoder_layers,
            load_model,
        )

        try:
            model, _ = load_model("hf-internal-testing/tiny-random-LlamaForCausalLM",
                                  dtype="float32")
        except OSError as exc:
            pytest.skip("tiny model unavailable (offline?): {}".format(exc))
        if len(decoder_layers(model)) < 2:
            pytest.skip("model has too few layers to test capture scoping")

        x = torch.randint(0, int(model.config.vocab_size), (1, 32))
        with AttentionProbe(model, 0) as probe:
            probe.reset()
            probe.edges = [(0, 31, 5)]
            with torch.no_grad():
                model(x)
            # If capture leaked to layer 1, this entry would be non-zero.
            assert float(probe.captured[0, 0, 31, 5]) == 0.0
            row_sum = float(probe.captured[0, 0, 31].sum())
            assert abs(row_sum - 1.0) < 1e-3
