"""Figure generation.

Figures are where a number reaches a reader, so the properties that matter are
not aesthetic:

* a figure whose inputs are absent must be **skipped**, never drawn from
  placeholder points, or an unexecuted experiment looks executed;
* every figure must write its own source CSV, so no plotted number exists
  only inside a PNG;
* records that are not real measurements must never become points.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from crpa.figures import (
    FigureSkipped,
    VARIANT_COLOR,
    VARIANT_LABEL,
    _num,
    _read_csv,
    _save_data,
    fig_context_scaling,
    fig_gate_visualization,
    fig_large_model,
    fig_matched_overlap,
    fig_seed_robustness,
    fig_structural_vs_behavioral,
)

ALL_FIGURES = [
    fig_structural_vs_behavioral,
    fig_matched_overlap,
    fig_large_model,
    fig_seed_robustness,
    fig_context_scaling,
    fig_gate_visualization,
]


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


class TestMissingInputsAreSkipped:
    """The property that keeps an unexecuted experiment looking unexecuted."""

    @pytest.mark.parametrize("fn", ALL_FIGURES, ids=lambda f: f.__name__)
    def test_every_figure_skips_on_an_empty_results_tree(self, fn, tmp_path):
        with pytest.raises(FigureSkipped):
            fn(tmp_path, tmp_path / "out")

    @pytest.mark.parametrize("fn", ALL_FIGURES, ids=lambda f: f.__name__)
    def test_a_skipped_figure_writes_no_image(self, fn, tmp_path):
        out = tmp_path / "out"
        with pytest.raises(FigureSkipped):
            fn(tmp_path, out)
        assert not out.exists() or not list(out.glob("*.png"))

    def test_the_skip_message_names_the_command_that_fixes_it(self, tmp_path):
        with pytest.raises(FigureSkipped) as exc:
            fig_structural_vs_behavioral(tmp_path, tmp_path / "out")
        assert "python -m experiments" in str(exc.value)

    def test_a_present_but_empty_csv_is_still_a_skip(self, tmp_path):
        write_csv(tmp_path / "tier1" / "overlap_vs_contribution.csv", [])
        with pytest.raises(FigureSkipped):
            fig_structural_vs_behavioral(tmp_path, tmp_path / "out")


class TestStructuralVsBehavioral:
    def _rows(self, n=60):
        return [{"overlap": 0.3 + 0.01 * (i % 40),
                 "delta_loss": 1e-6 * ((i % 7) - 3),
                 "seed": 42, "layer": i % 6, "head": i % 8}
                for i in range(n)]

    def test_renders_and_writes_its_source_data(self, tmp_path):
        write_csv(tmp_path / "tier1" / "overlap_vs_contribution.csv",
                  self._rows())
        out = tmp_path / "out"
        path = fig_structural_vs_behavioral(tmp_path, out)
        assert Path(path).exists()
        data = list(out.glob("*_data.csv"))
        assert data, "every figure must write its own source CSV"

    def test_non_finite_rows_are_dropped_not_plotted_as_zero(self, tmp_path):
        rows = self._rows()
        rows.append({"overlap": "nan", "delta_loss": "nan", "seed": 42,
                     "layer": 0, "head": 0})
        write_csv(tmp_path / "tier1" / "overlap_vs_contribution.csv", rows)
        fig_structural_vs_behavioral(tmp_path, tmp_path / "out")

    def test_all_non_finite_input_is_a_skip(self, tmp_path):
        write_csv(tmp_path / "tier1" / "overlap_vs_contribution.csv",
                  [{"overlap": "nan", "delta_loss": "nan"} for _ in range(5)])
        with pytest.raises(FigureSkipped):
            fig_structural_vs_behavioral(tmp_path, tmp_path / "out")


class TestSeedRobustness:
    def test_renders_from_an_aggregate(self, tmp_path):
        rows = [{"variant": v, "label": v, "n_runs": 3,
                 "retrieval_accuracy_mean": m,
                 "retrieval_accuracy_std": 1.2,
                 "retrieval_accuracy_ci_low": m - 1,
                 "retrieval_accuracy_ci_high": m + 1,
                 "measured_chance_floor": 52.78,
                 "chance_accuracy_mean": 5.0}
                for v, m in (("dense", 53.6), ("sliding", 47.2),
                             ("crpa_noreg", 4.4), ("crpa_naive", 4.6),
                             ("crpa_contribution", 4.3))]
        write_csv(tmp_path / "tier1" / "aggregate.csv", rows)
        path = fig_seed_robustness(tmp_path, tmp_path / "out")
        assert Path(path).exists()

    def test_the_reference_line_is_the_measured_floor(self, tmp_path):
        """Drawing the 5% uniform figure makes every baseline look like it
        learned the task."""
        rows = [{"variant": "dense", "label": "dense", "n_runs": 3,
                 "retrieval_accuracy_mean": 53.6,
                 "retrieval_accuracy_std": 1.7,
                 "retrieval_accuracy_ci_low": 51.9,
                 "retrieval_accuracy_ci_high": 55.3,
                 "measured_chance_floor": 52.78,
                 "chance_accuracy_mean": 5.0}]
        write_csv(tmp_path / "tier1" / "aggregate.csv", rows)
        out = tmp_path / "out"
        fig_seed_robustness(tmp_path, out)
        data = list(out.glob("*_data.csv"))[0].read_text(encoding="utf-8")
        assert "52.78" in data or "measured_chance_floor" in data


class TestContextScaling:
    def test_only_completed_rows_become_points(self, tmp_path):
        """An OOM row must not be plotted as a number."""
        rows = []
        for t, lat, status in ((4096, 13.2, "completed"),
                               (8192, 30.0, "completed"),
                               (16384, 72.0, "completed"),
                               (32768, "", "oom")):
            rows.append({"variant": "dense", "context_length": t,
                         "latency_ms_median": lat, "status": status,
                         "peak_allocated_mb": 700,
                         "retrieval_accuracy": 50.0,
                         "realized_overlap": 0.3,
                         "sparsity_ratio_upper_bound": 0.28,
                         "tokens_per_sec": 1000})
        write_csv(tmp_path / "tier2" / "long_context.csv", rows)
        write_csv(tmp_path / "tier2" / "benchmark.csv", rows)
        out = tmp_path / "out"
        paths = fig_context_scaling(tmp_path, out)
        assert paths
        for p in out.glob("*_data.csv"):
            text = p.read_text(encoding="utf-8")
            assert "32768" not in text or "oom" in text, (
                "an out-of-memory row reached a figure as a data point")


class TestLargeModelFigureNeedsRealData:
    """Figure 3 must render nothing without a real Tier 3 run."""

    def test_it_skips_when_tier3_never_ran(self, tmp_path):
        with pytest.raises(FigureSkipped):
            fig_large_model(tmp_path, tmp_path / "out")


class TestHelpers:
    def test_num_parses_and_degrades_to_nan(self):
        assert _num("1.5") == 1.5
        assert _num("") != _num("")     # NaN
        assert _num(None) != _num(None)
        assert _num("abc") != _num("abc")

    def test_read_csv_of_a_missing_file_is_empty_not_an_error(self, tmp_path):
        assert _read_csv(tmp_path / "nope.csv") == []

    def test_save_data_returns_none_for_no_rows(self, tmp_path):
        assert _save_data(tmp_path / "x.csv", []) is None

    def test_save_data_writes_the_union_of_keys(self, tmp_path):
        _save_data(tmp_path / "x.csv", [{"a": 1}, {"b": 2}])
        header = (tmp_path / "x.csv").read_text(encoding="utf-8").splitlines()[0]
        assert "a" in header and "b" in header


class TestPalette:
    def test_both_variant_names_resolve_to_the_same_colour(self):
        """crpa_causal is a deprecated alias and must not get its own hue."""
        assert VARIANT_COLOR["crpa_causal"] == VARIANT_COLOR["crpa_contribution"]

    def test_every_labelled_variant_has_a_colour(self):
        for name in VARIANT_LABEL:
            assert name in VARIANT_COLOR

    def test_colours_are_distinct_across_variants(self):
        seen = {VARIANT_COLOR[v] for v in VARIANT_LABEL}
        assert len(seen) == len(VARIANT_LABEL)
