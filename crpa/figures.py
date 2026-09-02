"""
crpa.figures - publication figures, each regenerated from structured results.

Design rules this module follows
--------------------------------
* Every figure writes its **source data** as a CSV next to the PNG. That is the
  table view, and it is also what makes the "relief rule" satisfiable for the
  one palette slot below 3:1 contrast on the light surface.
* A figure whose inputs are missing is **skipped with a message**. It is never
  drawn from placeholder or synthesised points - an absent experiment must look
  absent.
* Records whose status is not a real measurement (``not_run``, ``oom``,
  ``failed``) are filtered out upstream by
  :func:`crpa.runmeta.numeric_records`, so they cannot appear as points.
* No dual axes anywhere. Where two quantities have different scales they get
  separate panels or separate files.

Palette
-------
Categorical slots 1-3 of the reference palette, which clear the all-pairs CVD
and normal-vision floors in light mode - the binding constraint, since the
central figures are scatters. Baseline variants use slots 7 and 8, which is
safe because they only ever appear in adjacent-pair forms (bars and lines).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reference palette, light mode. Validated with scripts/validate_palette.js:
# all-pairs CVD dE 9.2, normal-vision dE 24.0 for the first three slots.
SERIES = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "violet": "#4a3aa7",
    "red": "#e34948",
}

VARIANT_COLOR = {
    "crpa_noreg": SERIES["blue"],
    "crpa_naive": SERIES["orange"],
    "crpa_contribution": SERIES["aqua"],
    "crpa_causal": SERIES["aqua"],
    "dense": SERIES["violet"],
    "sliding": SERIES["red"],
}

VARIANT_LABEL = {
    "crpa_noreg": "No regularization",
    "crpa_naive": "Naive (overlap-ranked)",
    "crpa_contribution": "Contribution-gated",
    "dense": "Dense",
    "sliding": "Sliding window",
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3de"


def _style(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    """Recessive grid and axes; text in ink tokens, never in a series color."""
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK_SECONDARY)
    ax.set_title(title, fontsize=11, color=INK, pad=12, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5, length=0)
    ax.set_facecolor(SURFACE)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def _save_data(path: Path, rows: Sequence[Dict[str, object]]) -> Optional[Path]:
    """Write the figure's source data - the table view the relief rule needs."""
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _num(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return out


class FigureSkipped(Exception):
    """Raised when a figure's inputs are absent. Never draw a placeholder."""


# ---------------------------------------------------------------------------
#  Figure 1 - structural overlap vs behavioral contribution
# ---------------------------------------------------------------------------

def fig_structural_vs_behavioral(results_root: Path, out_dir: Path) -> Path:
    """The central scatter: does overlap predict contribution?"""
    src = results_root / "tier1" / "overlap_vs_contribution.csv"
    rows = _read_csv(src)
    if not rows:
        raise FigureSkipped(
            "missing {}. Run: python -m experiments.overlap_vs_contribution".format(src)
        )

    overlap = np.array([_num(r["overlap"]) for r in rows])
    delta = np.array([_num(r["delta_loss"]) for r in rows])
    ok = np.isfinite(overlap) & np.isfinite(delta)
    overlap, delta = overlap[ok], delta[ok]
    if overlap.size == 0:
        raise FigureSkipped("no finite (overlap, delta) pairs in {}".format(src))

    ov_thr = float(np.quantile(overlap, 0.75))
    hi = overlap >= ov_thr
    hi_delta = delta[hi]
    lo_c = float(np.quantile(hi_delta, 0.25)) if hi_delta.size else float("nan")
    hi_c = float(np.quantile(hi_delta, 0.75)) if hi_delta.size else float("nan")

    group_a = hi & (delta <= lo_c)   # high overlap, low contribution
    group_b = hi & (delta >= hi_c)   # high overlap, high contribution
    other = ~(group_a | group_b)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    _style(ax, "Structural overlap (support Jaccard)",
           "Behavioral contribution  (delta loss when removed)",
           "Structural overlap does not determine behavioral contribution")

    ax.scatter(overlap[other], delta[other], s=26, c=SERIES["blue"],
               alpha=0.45, linewidths=0.5, edgecolors=SURFACE,
               label="All other edges", zorder=2)
    ax.scatter(overlap[group_a], delta[group_a], s=64, c=SERIES["orange"],
               linewidths=1.2, edgecolors=SURFACE,
               label="High overlap, low contribution", zorder=4)
    ax.scatter(overlap[group_b], delta[group_b], s=64, c=SERIES["aqua"],
               linewidths=1.2, edgecolors=SURFACE, marker="D",
               label="High overlap, high contribution", zorder=4)

    ax.axvline(ov_thr, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    # Anchored to the bottom-left of the line: the top of a high-overlap
    # column is exactly where the highlighted high-contribution points sit.
    ax.annotate("high-overlap\nthreshold (75th pct)", xy=(ov_thr, ax.get_ylim()[0]),
                xytext=(-6, 10), textcoords="offset points",
                fontsize=7.5, color=INK_MUTED, va="bottom", ha="right")

    if overlap.size > 2:
        from crpa.metrics import correlations
        stats = correlations(overlap.tolist(), delta.tolist())
        ax.annotate(
            "Pearson r = {:.3f}  (p = {:.3g})\nSpearman r = {:.3f}   n = {}".format(
                stats.get("pearson_r", float("nan")), stats.get("pearson_p", float("nan")),
                stats.get("spearman_r", float("nan")), stats.get("n", 0)),
            xy=(0.985, 0.03), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=8.5, color=INK_SECONDARY,
            bbox=dict(facecolor=SURFACE, edgecolor=GRID, boxstyle="round,pad=0.5"))

    ax.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor=SURFACE,
              edgecolor=GRID, labelcolor=INK_SECONDARY)
    fig.text(0.01, -0.045,
             "Both highlighted groups sit above the same overlap threshold. "
             "They separate on measured behaviour, not on structure.",
             fontsize=8, color=INK_MUTED)

    _save_data(out_dir / "fig1_structural_vs_behavioral_data.csv", rows)
    return _save(fig, out_dir / "fig1_structural_vs_behavioral.png")


# ---------------------------------------------------------------------------
#  Figure 2 - matched overlap, different outcome
# ---------------------------------------------------------------------------

def fig_matched_overlap(results_root: Path, out_dir: Path) -> Path:
    """Retrieval against realized overlap, with matched pairs joined."""
    src = results_root / "matched_overlap" / "sweep.csv"
    rows = _read_csv(src)
    if not rows:
        raise FigureSkipped(
            "missing {}. Run: python -m experiments.matched_overlap".format(src))

    pairs = _read_csv(results_root / "matched_overlap" / "matched_pairs.csv")

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    _style(ax, "Realized attention overlap (measured after training)",
           "Retrieval accuracy (%)",
           "At a matched overlap budget, which edges are removed decides the outcome")

    for variant in ("crpa_naive", "crpa_contribution"):
        sel = [r for r in rows if r.get("variant") == variant]
        if not sel:
            continue
        xs = np.array([_num(r["realized_overlap"]) for r in sel])
        ys = np.array([_num(r["retrieval_accuracy"]) for r in sel])
        order = np.argsort(xs)
        ax.plot(xs[order], ys[order], color=VARIANT_COLOR[variant], linewidth=2.0,
                marker="o", markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=1.2, label=VARIANT_LABEL[variant], zorder=3)

    for pair in pairs:
        x1, x2 = _num(pair.get("crpa_naive_overlap")), _num(pair.get("crpa_contribution_overlap"))
        y1, y2 = _num(pair.get("crpa_naive_retrieval")), _num(pair.get("crpa_contribution_retrieval"))
        if not all(math.isfinite(v) for v in (x1, x2, y1, y2)):
            continue
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-", color=INK_MUTED,
                                    linewidth=1.1, linestyle=(0, (3, 2))), zorder=2)

    chance = [_num(r.get("chance_accuracy")) for r in rows if r.get("chance_accuracy")]
    if chance and math.isfinite(chance[0]):
        ax.axhline(chance[0], color=INK_MUTED, linewidth=1.2, linestyle=(0, (5, 3)), zorder=1)
        ax.annotate("chance ({:.0f}%)".format(chance[0]),
                    xy=(ax.get_xlim()[0], chance[0]), xytext=(4, 4),
                    textcoords="offset points", fontsize=7.5, color=INK_MUTED)

    ax.legend(loc="best", fontsize=8.5, frameon=True, facecolor=SURFACE,
              edgecolor=GRID, labelcolor=INK_SECONDARY)
    fig.text(0.01, -0.045,
             "Dashed connectors join runs whose realized overlap matches within "
             "tolerance. Matching is on measured overlap, never on configured lambda.",
             fontsize=8, color=INK_MUTED)

    _save_data(out_dir / "fig2_matched_overlap_data.csv", rows)
    if pairs:
        _save_data(out_dir / "fig2_matched_pairs_data.csv", pairs)
    return _save(fig, out_dir / "fig2_matched_overlap.png")


# ---------------------------------------------------------------------------
#  Figure 3 - large-model diagnostic (requires real Tier 3 data)
# ---------------------------------------------------------------------------

def fig_large_model(results_root: Path, out_dir: Path) -> Path:
    """Tier 3 overlap vs contribution. Renders nothing without real data."""
    tier3 = results_root / "tier3"
    sources = sorted(tier3.glob("edges_*.csv")) if tier3.exists() else []
    rows: List[Dict[str, str]] = []
    for src in sources:
        rows.extend(_read_csv(src))
    if not rows:
        raise FigureSkipped(
            "no Tier 3 edge data under {}. This figure requires a real frozen-model "
            "run: python -m experiments.large_model_diagnostic --model_id <id>. "
            "It is deliberately not drawn from synthetic points.".format(tier3))

    overlap = np.array([_num(r["overlap"]) for r in rows])
    delta = np.array([_num(r["delta_loss"]) for r in rows])
    layers = np.array([_num(r.get("layer", 0)) for r in rows])
    ok = np.isfinite(overlap) & np.isfinite(delta)
    overlap, delta, layers = overlap[ok], delta[ok], layers[ok]

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    _style(ax, "Structural overlap (support Jaccard)",
           "Behavioral contribution  (delta loss when removed)",
           "Frozen large-model diagnostic: overlap vs contribution")

    unique = sorted(set(layers.tolist()))[:3]  # all-pairs cap: three slots
    slots = [SERIES["blue"], SERIES["orange"], SERIES["aqua"]]
    for color, layer in zip(slots, unique):
        sel = layers == layer
        ax.scatter(overlap[sel], delta[sel], s=44, c=color, alpha=0.75,
                   linewidths=0.6, edgecolors=SURFACE,
                   label="Layer {:.0f}".format(layer), zorder=3)
    extra = layers > (unique[-1] if unique else 0)
    if extra.any():
        ax.scatter(overlap[extra], delta[extra], s=30, c=INK_MUTED, alpha=0.5,
                   linewidths=0.5, edgecolors=SURFACE, label="Other layers", zorder=2)

    from crpa.metrics import correlations
    stats = correlations(overlap.tolist(), delta.tolist())
    ax.annotate("Pearson r = {:.3f}  (p = {:.3g})\nSpearman r = {:.3f}   n = {}".format(
        stats.get("pearson_r", float("nan")), stats.get("pearson_p", float("nan")),
        stats.get("spearman_r", float("nan")), stats.get("n", 0)),
        xy=(0.985, 0.03), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=8.5, color=INK_SECONDARY,
        bbox=dict(facecolor=SURFACE, edgecolor=GRID, boxstyle="round,pad=0.5"))

    ax.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor=SURFACE,
              edgecolor=GRID, labelcolor=INK_SECONDARY)
    _save_data(out_dir / "fig3_large_model_data.csv", rows)
    return _save(fig, out_dir / "fig3_large_model.png")


# ---------------------------------------------------------------------------
#  Figure 4 - seed robustness
# ---------------------------------------------------------------------------

def fig_seed_robustness(results_root: Path, out_dir: Path) -> Path:
    """Three-seed Tier 1 results with bootstrap intervals."""
    src = results_root / "tier1" / "aggregate.csv"
    rows = _read_csv(src)
    if not rows:
        raise FigureSkipped(
            "missing {}. Run: python -m experiments.tier1_multiseed".format(src))

    order = ["crpa_noreg", "crpa_naive", "crpa_contribution"]
    rows = [r for r in rows if r.get("variant") in order]
    rows.sort(key=lambda r: order.index(r["variant"]))
    if not rows:
        raise FigureSkipped("aggregate contains none of the central variants")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4))

    for ax, field, ylabel, title in (
        (ax1, "retrieval_accuracy", "Retrieval accuracy (%)", "Retrieval, 3 seeds"),
        (ax2, "realized_overlap", "Realized overlap", "Overlap, 3 seeds"),
    ):
        _style(ax, "", ylabel, title)
        xs = np.arange(len(rows))
        means = np.array([_num(r.get("{}_mean".format(field))) for r in rows])
        lows = np.array([_num(r.get("{}_ci_low".format(field))) for r in rows])
        highs = np.array([_num(r.get("{}_ci_high".format(field))) for r in rows])
        err = np.vstack([
            np.clip(means - lows, 0, None),
            np.clip(highs - means, 0, None),
        ])
        colors = [VARIANT_COLOR[r["variant"]] for r in rows]
        ax.bar(xs, means, width=0.56, color=colors, edgecolor=SURFACE, linewidth=2.0,
               zorder=3)
        ax.errorbar(xs, means, yerr=err, fmt="none", ecolor=INK_SECONDARY,
                    elinewidth=1.4, capsize=5, capthick=1.4, zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels([VARIANT_LABEL[r["variant"]] for r in rows],
                           fontsize=8, color=INK_SECONDARY)
        for x, mean in zip(xs, means):
            if math.isfinite(mean):
                ax.annotate("{:.3g}".format(mean), xy=(x, mean),
                            xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=8.5, color=INK)

    chance = _num(rows[0].get("chance_accuracy_mean", "nan"))
    if math.isfinite(chance):
        ax1.axhline(chance, color=INK_MUTED, linewidth=1.4, linestyle=(0, (5, 3)),
                    zorder=5)
        ax1.annotate("chance ({:.0f}%)".format(chance),
                     xy=(ax1.get_xlim()[1], chance), xytext=(-4, 4),
                     textcoords="offset points", ha="right", va="bottom",
                     fontsize=8, color=INK_MUTED)

    fig.suptitle("Seed robustness: mean with bootstrap 95% interval over 3 seeds",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.text(0.01, -0.04,
             "Three seeds give a wide interval. It is shown rather than hidden.",
             fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    _save_data(out_dir / "fig4_seed_robustness_data.csv", rows)
    return _save(fig, out_dir / "fig4_seed_robustness.png")


# ---------------------------------------------------------------------------
#  Figure 5 - context scaling (two files: quality, then cost)
# ---------------------------------------------------------------------------

def fig_context_scaling(results_root: Path, out_dir: Path) -> List[Path]:
    """Diagnostic quality and runtime cost across context length."""
    diag = _read_csv(results_root / "tier2" / "long_context.csv")
    bench = _read_csv(results_root / "tier2" / "benchmark.csv")
    if not diag and not bench:
        raise FigureSkipped(
            "missing Tier 2 results. Run: python -m experiments.long_context "
            "and python -m experiments.benchmark")

    written: List[Path] = []

    if diag:
        usable = [r for r in diag if r.get("status") in ("completed", "smoke")]
        if usable:
            fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.0))
            panels = [
                ("retrieval_accuracy", "Retrieval accuracy (%)", "Retrieval", SERIES["blue"]),
                ("realized_overlap", "Realized overlap", "Overlap", SERIES["orange"]),
                ("ranking_stability_spearman", "Spearman (replicate rankings)",
                 "Contribution ranking stability", SERIES["aqua"]),
            ]
            xs = [_num(r["context_length"]) for r in usable]
            for ax, (field, ylabel, title, color) in zip(axes, panels):
                _style(ax, "Context length (tokens)", ylabel, title)
                ys = [_num(r.get(field)) for r in usable]
                pts = sorted(zip(xs, ys))
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color,
                        linewidth=2.0, marker="o", markersize=8,
                        markeredgecolor=SURFACE, markeredgewidth=1.2)
                ax.set_xscale("log", base=2)
            fig.suptitle("Does the diagnostic stay meaningful as context grows?",
                         fontsize=11, color=INK, x=0.02, ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            _save_data(out_dir / "fig5a_context_quality_data.csv", usable)
            written.append(_save(fig, out_dir / "fig5a_context_quality.png"))

    if bench:
        ok = [r for r in bench if r.get("status") == "completed"]
        oom = [r for r in bench if r.get("status") == "oom"]
        if ok:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4))
            _style(ax1, "Context length (tokens)", "Forward latency (ms, median)",
                   "Latency")
            _style(ax2, "Context length (tokens)", "Peak allocated memory (MB)",
                   "Memory")
            for variant in sorted({r["variant"] for r in ok}):
                sel = sorted([r for r in ok if r["variant"] == variant],
                             key=lambda r: _num(r["context_length"]))
                xs = [_num(r["context_length"]) for r in sel]
                ax1.plot(xs, [_num(r["latency_ms_median"]) for r in sel],
                         color=VARIANT_COLOR.get(variant, INK_MUTED), linewidth=2.0,
                         marker="o", markersize=8, markeredgecolor=SURFACE,
                         markeredgewidth=1.2,
                         label=VARIANT_LABEL.get(variant, variant))
                mem = [_num(r.get("peak_allocated_mb")) for r in sel]
                if any(math.isfinite(v) for v in mem):
                    ax2.plot(xs, mem, color=VARIANT_COLOR.get(variant, INK_MUTED),
                             linewidth=2.0, marker="o", markersize=8,
                             markeredgecolor=SURFACE, markeredgewidth=1.2,
                             label=VARIANT_LABEL.get(variant, variant))
            for ax in (ax1, ax2):
                ax.set_xscale("log", base=2)
                # Memory is unavailable on CPU, so that panel can legitimately
                # have no series; a legend call there would warn and draw an
                # empty box.
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=8.5, frameon=True, facecolor=SURFACE,
                              edgecolor=GRID, labelcolor=INK_SECONDARY)
                else:
                    ax.annotate("no measurements on this device\n"
                                "(GPU memory counters unavailable on CPU)",
                                xy=(0.5, 0.5), xycoords="axes fraction",
                                ha="center", va="center", fontsize=9,
                                color=INK_MUTED)
            if oom:
                lengths = sorted({int(_num(r["context_length"])) for r in oom})
                fig.text(0.01, -0.04,
                         "Out of memory (no measurement recorded) at: {}".format(
                             ", ".join(str(v) for v in lengths)),
                         fontsize=8, color=INK_MUTED)
            fig.suptitle("Forward cost across context length",
                         fontsize=11, color=INK, x=0.02, ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            _save_data(out_dir / "fig5b_context_performance_data.csv", bench)
            written.append(_save(fig, out_dir / "fig5b_context_performance.png"))

    if not written:
        raise FigureSkipped("Tier 2 files exist but contain no completed rows")
    return written


# ---------------------------------------------------------------------------
#  Figure 6 - what the gate actually does
# ---------------------------------------------------------------------------

def fig_gate_visualization(results_root: Path, out_dir: Path) -> Path:
    """Which candidates each criterion removes, and what it costs."""
    src = results_root / "tier1" / "overlap_vs_contribution.csv"
    rows = _read_csv(src)
    if not rows:
        raise FigureSkipped(
            "missing {}. Run: python -m experiments.overlap_vs_contribution".format(src))

    recs = [
        {"overlap": _num(r["overlap"]), "delta": _num(r["delta_loss"])}
        for r in rows
        if math.isfinite(_num(r["overlap"])) and math.isfinite(_num(r["delta_loss"]))
    ]
    if not recs:
        raise FigureSkipped("no finite candidates in {}".format(src))

    budget = max(1, len(recs) // 8)
    naive_ids = {id(r) for r in sorted(recs, key=lambda r: -r["overlap"])[:budget]}
    contrib_ids = {id(r) for r in sorted(recs, key=lambda r: r["delta"])[:budget]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.8),
                                   gridspec_kw={"width_ratios": [1.55, 1]})

    _style(ax1, "Structural overlap", "Behavioral contribution (delta loss)",
           "Same budget, different selections")
    kept = [r for r in recs if id(r) not in naive_ids and id(r) not in contrib_ids]
    ax1.scatter([r["overlap"] for r in kept], [r["delta"] for r in kept],
                s=24, c=INK_MUTED, alpha=0.35, linewidths=0.5,
                edgecolors=SURFACE, label="Not selected", zorder=2)
    ax1.scatter([r["overlap"] for r in recs if id(r) in naive_ids],
                [r["delta"] for r in recs if id(r) in naive_ids],
                s=70, c=SERIES["orange"], marker="s", linewidths=1.2,
                edgecolors=SURFACE, label="Removed by naive", zorder=4)
    ax1.scatter([r["overlap"] for r in recs if id(r) in contrib_ids],
                [r["delta"] for r in recs if id(r) in contrib_ids],
                s=70, c=SERIES["aqua"], marker="D", linewidths=1.2,
                edgecolors=SURFACE, label="Removed by contribution gate", zorder=4)
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor=SURFACE,
               edgecolor=GRID, labelcolor=INK_SECONDARY)

    naive_cost = sum(r["delta"] for r in recs if id(r) in naive_ids)
    contrib_cost = sum(r["delta"] for r in recs if id(r) in contrib_ids)
    _style(ax2, "", "Total delta loss incurred (sum over removed edges)",
           "Cost of the removal")
    # Edge in the series colour rather than the surface, so a bar whose value is
    # near zero still renders as a visible mark instead of vanishing.
    bars = ax2.bar([0, 1], [naive_cost, contrib_cost], width=0.5,
                   color=[SERIES["orange"], SERIES["aqua"]],
                   edgecolor=[SERIES["orange"], SERIES["aqua"]],
                   linewidth=1.6, zorder=3)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Naive", "Contribution-gated"], fontsize=9,
                        color=INK_SECONDARY)
    ax2.axhline(0, color=GRID, linewidth=1.0)
    for bar, value in zip(bars, (naive_cost, contrib_cost)):
        ax2.annotate("{:+.2e}".format(value),
                     xy=(bar.get_x() + bar.get_width() / 2, value),
                     xytext=(0, 7 if value >= 0 else -14), textcoords="offset points",
                     ha="center", fontsize=8.5, color=INK)

    fig.text(0.01, -0.04,
             "Both criteria remove {} edges. Lower total delta means the removed "
             "interactions mattered less.".format(budget),
             fontsize=8, color=INK_MUTED)
    fig.tight_layout()
    _save_data(out_dir / "fig6_gate_visualization_data.csv", rows)
    return _save(fig, out_dir / "fig6_gate_visualization.png")


FIGURES = {
    "fig1_structural_vs_behavioral": fig_structural_vs_behavioral,
    "fig2_matched_overlap": fig_matched_overlap,
    "fig3_large_model": fig_large_model,
    "fig4_seed_robustness": fig_seed_robustness,
    "fig5_context_scaling": fig_context_scaling,
    "fig6_gate_visualization": fig_gate_visualization,
}
