#!/usr/bin/env python3
"""Nature-style rank-aggregation dot plot for benchmark results (combo or single-pert).

Creates a rank-aggregation visualization with:
- Individual rank points colored by dataset and shaped by metric
- Datasets side-by-side at each rank (symmetric x-offsets centered on the rank); metrics stacked per dataset
- Calibrated metrics (``c`` prefix, e.g. ``cwmse``) use the same marker as their uncalibrated analogues; ranks are always lower-is-better
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import AbstractSet, Dict, List, Optional, Tuple

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.legend_handler import HandlerBase
import numpy as np
import pandas as pd

# Nature-style plotting conventions
_NATURE_FONT = "Arial"
_NATURE_BASE_SIZE = 7
_NATURE_DPI = 450

ORACLE_MODELS = {
    "technical_duplicate",
    "epistasis_specific",
    "target_scaling_oracle",
    "synthetic_perturbation_oracle",
    "linear_regression_wmse",
    "onehot_linear_wmse",
    # "interpolated_duplicate",  # Excluded as requested
}

# Lightweight / classical models (distinct from deep learning on the y-axis)
SIMPLE_MODELS = frozenset(
    {
        "target_scaling",
        "synthetic_perturbation",
        "linear_regression",
        "onehot_linear",
        "epistasis_shared",
    }
)

BASELINE_MODELS = {
    "dataset_mean",
    "additive",
    "control_mean",
    "control",
}

# Single-gene / single-pert benchmark datasets (use non-combo row filter; no "_" heuristic)
SINGLE_PERT_DATASETS = frozenset(
    {
        "adamson16",
        "frangieh21",
        "replogle22k562",
        "nadig25hepg2",
        "nadig25hpeg2",  # alias for common typo vs HepG2
    }
)

# Map common CLI typos / aliases to canonical names (must match substring in outputs/benchmark_* dirs)
DATASET_ALIASES: Dict[str, str] = {
    "nadig25hpeg2": "nadig25hepg2",
    "repogle22k562": "replogle22k562",  # missing "l" vs replogle
}


def canonical_dataset(ds: str) -> str:
    """Return canonical dataset key used in benchmark directory names and SINGLE_PERT_DATASETS."""
    return DATASET_ALIASES.get(ds, ds)


def ordered_unique_canonical(datasets: List[str]) -> List[str]:
    """Preserve order while deduplicating aliases (e.g. repogle vs replogle)."""
    seen: set[str] = set()
    out: List[str] = []
    for d in datasets:
        c = canonical_dataset(d)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def all_single_pert_datasets(datasets: List[str]) -> bool:
    """True if every canonical dataset is a single-perturbation benchmark."""
    canon = ordered_unique_canonical(datasets)
    if not canon:
        return False
    return all(d in SINGLE_PERT_DATASETS for d in canon)


# Horizontal gap between rank tick centers (1 = legacy default for combo and single ≤3).
# Wider spacing only for single-pert plots with >3 datasets (many side-by-side markers).
RANK_COLUMN_SPACING_LEGACY = 1.0
RANK_COLUMN_SPACING_WIDE_SINGLE_MANY = 1.38


def effective_rank_column_spacing(
    datasets: List[str],
    user_override: float | None,
) -> float:
    """Legacy 1.0 unless user overrides or single-pert with >3 datasets (wide spacing)."""
    if user_override is not None:
        return user_override
    canon = ordered_unique_canonical(datasets)
    if all_single_pert_datasets(datasets) and len(canon) > 3:
        return RANK_COLUMN_SPACING_WIDE_SINGLE_MANY
    return RANK_COLUMN_SPACING_LEGACY

# Dataset colors — avoid bright tab green (oracle #2ca02c). wessels: dark forest green (distinct from oracle / baseline blue).
DATASET_COLORS = {
    "wessels23": "#1b5e20",   # dark green
    "norman19": "#ff7f0e",    # orange
    "replogle20": "#9467bd",  # purple (not green)
    "adamson16": "#17becf",   # cyan
    "frangieh21": "#d62728",  # red
    "replogle22k562": "#e377c2",  # pink
    "nadig25hepg2": "#8c564b",  # brown
    "nadig25hpeg2": "#8c564b",
}

# Metric markers (shapes)
METRIC_MARKERS = {
    "mse": "o",        # circle
    "wmse": "o",       # circle
    "pearson_deltapert": "D",  # diamond
    "weighted_pearson_deltapert": "D",  # diamond
    "pds": "s",        # square
    "pds_wmse": "s",   # square
    "pds_pearson_deltapert": "s",  # square
    "pds_weighted_pearson_deltapert": "s",  # square
    "pds_r2_deltapert": "s",      # square
    "pds_weighted_r2_deltapert": "s",  # square
    "weighted_r2_deltapert": "s",  # square
}

# Metric display names
METRIC_DISPLAY = {
    "mse": "MSE",
    "wmse": "wMSE",
    "pearson_deltapert": "PearsonDeltaPert",
    "weighted_pearson_deltapert": "wPearsonΔPert",
    "pds": "PDS",
    "pds_wmse": "wPDS",
    "pds_pearson_deltapert": "PDS-PearsonΔPert",
    "pds_weighted_pearson_deltapert": "PDS-wPearsonΔPert",
    "pds_r2_deltapert": "PDS-R²ΔPert",
    "pds_weighted_r2_deltapert": "PDS-wR²ΔPert",
    "weighted_r2_deltapert": "wR²",
}

# Metric ranking direction: True => higher is better, False => lower is better
METRIC_HIGHER_BETTER = {
    "mse": False,
    "wmse": False,
    "pearson_deltapert": True,
    "weighted_pearson_deltapert": True,
    "pds": True,
    "pds_wmse": True,
    "pds_pearson_deltapert": True,
    "pds_weighted_pearson_deltapert": True,
    "pds_r2_deltapert": True,
    "pds_weighted_r2_deltapert": True,
    "weighted_r2_deltapert": True,
}


def uncalibrated_metric_key(metric: str) -> str:
    """Map calibrated names to their uncalibrated analogue (leading ``c`` only)."""
    if metric.startswith("c") and len(metric) > 1:
        return metric[1:]
    return metric


def metric_marker_for_plot(metric: str) -> str:
    """Same marker as the uncalibrated analogue (METRIC_MARKERS + fallbacks)."""
    u = uncalibrated_metric_key(metric)
    for key in (u, metric):
        if key in METRIC_MARKERS:
            return METRIC_MARKERS[key]
    if "weighted_r2" in u:
        return "s"
    if "pds" in u:
        return "s"
    if "pearson" in u or "r2" in u:
        return "D"
    return "o"


def metric_display_for_plot(metric: str) -> str:
    """Legend text; calibrated metrics get a ``c`` prefix on the base display name."""
    u = uncalibrated_metric_key(metric)
    if u == metric:
        return METRIC_DISPLAY.get(metric, metric.replace("_", " ").title())
    base = METRIC_DISPLAY.get(u, u.replace("_", " ").title())
    return f"c{base}"


def metric_rank_higher_is_better(metric: str) -> bool:
    """Ranking direction. Calibrated metrics (``c…`` names) are always lower-is-better."""
    if metric.startswith("c") and len(metric) > 1:
        return False
    u = uncalibrated_metric_key(metric)
    if u in METRIC_HIGHER_BETTER:
        return METRIC_HIGHER_BETTER[u]
    if metric in METRIC_HIGHER_BETTER:
        return METRIC_HIGHER_BETTER[metric]
    return ("pearson" in u or "r2" in u or "pds" in u)

# Models to exclude
EXCLUDED_MODELS = {"interpolated_duplicate", "linear_regression_wmse", "onehot_linear_wmse"}

# Figure size (inches): combo / mixed benchmarks vs single-pert-only (shorter).
_FIG_WIDTH_IN = 6.75
_FIG_HEIGHT_DEFAULT_IN = 5.5
_FIG_HEIGHT_SINGLE_PERT_IN = 4.45

# Y-axis / legend: model name styling (must stay in sync)
STYLE_ORACLE = "#2ca02c"
STYLE_LEARNED_BASELINE = "#0d47a1"  # simple / classical learned models
STYLE_UNLEARNED_BASELINE = "#666666"
STYLE_SOTA = "#000000"

# Model-type legend lines: horizontal nudge (0 = align with dataset text; 1 = full symbol-column shift)
_MODEL_LEGEND_SHIFT_FRAC = 0.28


def get_category(model_name: str) -> str:
    """Return category: Oracle, Simple, Baseline, or Model."""
    normalized = "control_mean" if model_name == "control" else model_name
    if normalized in ORACLE_MODELS:
        return "Oracle"
    if normalized in SIMPLE_MODELS:
        return "Simple"
    if normalized in BASELINE_MODELS:
        return "Baseline"
    return "Model"


def format_model_name(model_name: str) -> str:
    """Transform model names to nice display format."""
    name_map = {
        "presage": "PRESAGE",
        "pdae": "PDAE",
        "gears": "GEARS",
        "sclambda": "scLambda",
        "scgpt": "scGPT",
        "linear_regression": "Linear Regression",
        "onehot_linear": "Linear Regression",
        "linear_regression_wmse": "Linear Regression (wMSE)",
        "onehot_linear_wmse": "Linear Regression (wMSE)",
        "epistasis_shared": "Epistasis (Global)",
        "epistasis_specific": "Epistasis (Specific)",
        "technical_duplicate": "Technical Duplicate",
        "interpolated_duplicate": "Interpolated Duplicate",
        "dataset_mean": "Dataset Mean",
        "control_mean": "Control Mean",
        "additive": "Additive",
        "target_scaling": "Target Scaling",
        "synthetic_perturbation": "Target Scaling",
        "target_scaling_oracle": "Target Scaling (Oracle)",
        "synthetic_perturbation_oracle": "Target Scaling (Oracle)",
    }
    return name_map.get(model_name, model_name.replace("_", " ").title())


def _is_combo(p: str) -> bool:
    """Combo benchmarks: keep perturbations that look like multi-gene / combo targets."""
    return any(s in str(p) for s in ("+", "__", "|", ";")) or "_" in str(p)


def _is_single_pert_row(p: str) -> bool:
    """Single-pert benchmarks: exclude only explicit multi-target separators (not bare _)."""
    s = str(p)
    return not any(x in s for x in ("+", "__", "|", ";"))


def _bench_matches_dataset(bench_name: str, ds: str) -> bool:
    ds = canonical_dataset(ds)
    if ds == "nadig25hepg2":
        return "nadig25hepg2" in bench_name or "nadig25hpeg2" in bench_name
    return ds in bench_name


def _find_dirs(base: Path, datasets: List[str]) -> Dict[str, List[Path]]:
    """Find benchmark directories for each dataset."""
    results = {}
    for ds in datasets:
        dirs = []
        for pattern in ["benchmark_*", "_benchmark_*"]:
            for bench in (base / "outputs").glob(pattern):
                if not _bench_matches_dataset(bench.name, ds):
                    continue
                for sub in sorted(bench.iterdir(), reverse=True):
                    if sub.is_dir() and (sub / "detailed_metrics.csv").exists():
                        dirs.append(sub)
                        break
                if dirs:
                    break
            if dirs:
                break
        results[ds] = dirs
    return results


def _load_data(
    bench_dir: Path,
    metric: str,
    dataset: str,
    extra_excluded_models: Optional[AbstractSet[str]] = None,
) -> pd.DataFrame:
    csv = bench_dir / "detailed_metrics.csv"
    if not csv.exists():
        return pd.DataFrame()
    dataset = canonical_dataset(dataset)
    df = pd.read_csv(csv)
    df = df[df["metric"] == metric]
    if df.empty:
        return pd.DataFrame(columns=["model", "perturbation", "value"])
    if dataset in SINGLE_PERT_DATASETS:
        df = df[df["perturbation"].apply(_is_single_pert_row)]
    else:
        df = df[df["perturbation"].apply(_is_combo)]
    if df.empty:
        return pd.DataFrame(columns=["model", "perturbation", "value"])
    excluded = set(EXCLUDED_MODELS)
    if extra_excluded_models:
        excluded |= set(extra_excluded_models)
    df = df[~df["model"].isin(excluded)]
    return df[["model", "perturbation", "value"]].reset_index(drop=True)


def prepare_rank_data(
    bench_dirs: Dict[str, List[Path]],
    datasets: List[str],
    metrics: List[str],
    extra_excluded_models: Optional[AbstractSet[str]] = None,
) -> pd.DataFrame:
    """Load data and prepare rank information."""
    all_data = []

    for metric in metrics:
        for ds in datasets:
            if ds not in bench_dirs or not bench_dirs[ds]:
                continue
            bench_dir = bench_dirs[ds][0]
            df = _load_data(bench_dir, metric, ds, extra_excluded_models)
            if df.empty:
                continue
            # Compute mean value per model for this dataset-metric combination
            means = df.groupby("model")["value"].mean().reset_index()
            means["dataset"] = canonical_dataset(ds)
            means["metric"] = metric
            all_data.append(means)

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)

    # Compute ranks within each (Dataset, Metric) group
    combined["rank"] = np.nan
    for (ds, metric), group in combined.groupby(["dataset", "metric"]):
        higher_better = metric_rank_higher_is_better(metric)
        combined.loc[group.index, "rank"] = group["value"].rank(
            method="min",
            ascending=not higher_better,
        )

    # Add category and formatting
    combined["category"] = combined["model"].apply(get_category)
    combined["model_display"] = combined["model"].apply(format_model_name)

    return combined


def plot_rank_aggregation(
    df: pd.DataFrame,
    datasets: List[str],
    metrics: List[str],
    output_path: Path,
    formats: Tuple[str, ...] = ("png", "pdf"),
    rank_column_spacing: float = RANK_COLUMN_SPACING_LEGACY,
    rank_changes: Dict[str, int] | None = None,
):
    """Create Nature-style Rank-Aggregation Dot Plot."""

    rc = rank_column_spacing

    # Set Nature style - match stripplot font
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [_NATURE_FONT, "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": _NATURE_BASE_SIZE,
        "axes.titlesize": _NATURE_BASE_SIZE,
        "xtick.labelsize": _NATURE_BASE_SIZE,
        "ytick.labelsize": _NATURE_BASE_SIZE,
        "legend.fontsize": _NATURE_BASE_SIZE - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Median rank per model determines y-axis order (best = top)
    median_ranks_order = df.groupby("model_display", observed=True)["rank"].median().sort_values(ascending=True)
    model_order = median_ranks_order.index.tolist()

    # Create categorical type for proper ordering
    df["model_display"] = pd.Categorical(df["model_display"], categories=model_order, ordered=True)

    # Determine max rank for x-axis limit
    max_rank = int(df["rank"].max())

    # Create figure (slightly shorter when only single-pert datasets — fewer combo-only models)
    _fh = _FIG_HEIGHT_SINGLE_PERT_IN if all_single_pert_datasets(datasets) else _FIG_HEIGHT_DEFAULT_IN
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH_IN, _fh))

    # Y position mapping
    y_positions = {model: i for i, model in enumerate(model_order)}

    # Horizontal spacing between dataset columns (group centered on each rank tick)
    dx = 0.34
    datasets_canon = ordered_unique_canonical(datasets)
    ds_order = {ds: i for i, ds in enumerate(datasets_canon)}
    max_n_ds = max(len(datasets_canon), 1)
    half_span = (max_n_ds - 1) / 2 * dx

    # Pre-calculate all symbol positions first
    vertical_spacing = 0.26  # spacing between stacked points (metrics within dataset)
    df_sorted = df.sort_values(["model_display", "rank", "dataset", "metric"]).reset_index(drop=True)
    df_sorted["position_index"] = df_sorted.groupby(
        ["model_display", "rank", "dataset"], observed=True
    ).cumcount()
    df_sorted["group_size"] = df_sorted.groupby(
        ["model_display", "rank", "dataset"], observed=True
    )["position_index"].transform("count")

    # Calculate final y position for each point
    def calc_y_offset(row):
        y_base = y_positions[row["model_display"]]
        position_index = row["position_index"]
        n_same = row["group_size"]
        if n_same > 1:
            total_height = (n_same - 1) * vertical_spacing
            offset = (position_index * vertical_spacing) - (total_height / 2)
        else:
            offset = 0
        return y_base + offset

    df_sorted["y_pos"] = df_sorted.apply(calc_y_offset, axis=1)

    # Per-(model, rank): symmetric x-offsets so the cluster is centered on the rank tick
    offset_rows: List[Tuple[str, float, str, float]] = []
    for (model_disp, rank), g in df_sorted.groupby(
        ["model_display", "rank"], observed=True
    ):
        present = sorted(
            g["dataset"].unique(),
            key=lambda d: (ds_order.get(d, len(datasets_canon)), str(d)),
        )
        n = len(present)
        if n == 0:
            continue
        if n == 1:
            offset_rows.append((model_disp, rank, present[0], 0.0))
        else:
            offs = np.linspace(-(n - 1) / 2 * dx, (n - 1) / 2 * dx, n)
            for d, o in zip(present, offs):
                offset_rows.append((model_disp, rank, d, float(o)))
    off_df = pd.DataFrame(
        offset_rows,
        columns=["model_display", "rank", "dataset", "dx_off"],
    )
    df_sorted = df_sorted.merge(off_df, on=["model_display", "rank", "dataset"], how="left")
    df_sorted["dx_off"] = df_sorted["dx_off"].fillna(0.0)
    df_sorted["x_pos"] = df_sorted["rank"] * rc + df_sorted["dx_off"]

    median_rank_by_model = df.groupby("model_display", observed=True)["rank"].median()

    # Axis limits and y inversion before background layers and points
    ax.set_xlim(rc * 0.5 - half_span, max_rank * rc + rc * 0.5 + half_span)
    ax.set_ylim(-0.7, len(model_order) - 0.3)
    ax.invert_yaxis()
    ax.set_axisbelow(True)

    # Grey band at median rank per model row (behind symbols).
    # Median can be *.5 when an even count of non-NaN ranks (e.g. missing cells) —
    # snap to nearest integer so the band aligns with a rank column, not between ticks.
    median_band_half = 0.5
    row_half = 0.42
    for model_disp in model_order:
        i = y_positions[model_disp]
        med_raw = float(median_rank_by_model.loc[model_disp])
        med_center = int(np.floor(med_raw + 0.5))
        mc_x = med_center * rc
        ax.fill_betweenx(
            [i - row_half, i + row_half],
            mc_x - median_band_half * rc,
            mc_x + median_band_half * rc,
            facecolor="#d0d0d0",
            edgecolor="none",
            alpha=0.4,
            zorder=0.5,
        )

    # Vertical lines between rank bins (boundaries at k+0.5), not through rank centers
    # Boundaries between rank bins; last line at max_rank + 0.5 closes the last bin
    for x_sep in np.arange(1.5, max_rank + 1, 1.0):
        ax.axvline(
            x_sep * rc,
            color="#d4d4d4",
            linestyle="-",
            linewidth=0.45,
            alpha=0.55,
            zorder=1,
        )

    # Plot symbols
    for _, row in df_sorted.iterrows():
        color = DATASET_COLORS.get(row["dataset"], "#333333")
        marker = metric_marker_for_plot(str(row["metric"]))

        ax.scatter(
            row["x_pos"],
            row["y_pos"],
            c=color,
            marker=marker,
            s=20,  # marker size
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )

    ax.set_xticks([r * rc for r in range(1, max_rank + 1)])
    ax.set_xticklabels([str(i) for i in range(1, max_rank + 1)], fontsize=_NATURE_BASE_SIZE)
    # X-axis label with downward arrow - match y-axis label style
    ax.set_xlabel("Rank ($\\downarrow$)", fontsize=_NATURE_BASE_SIZE)
    ax.set_ylabel("")

    # Style x-axis ticks - Nature style
    ax.tick_params(axis="x", which="major", length=2.5, width=0.6, pad=2)
    ax.spines["bottom"].set_linewidth(0.6)

    # Set y-ticks and labels
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order, fontsize=_NATURE_BASE_SIZE)
    ax.spines["left"].set_linewidth(0.6)

    # Style y-tick labels by category
    model_to_cat = dict(zip(df["model_display"], df["category"]))
    for label in ax.get_yticklabels():
        text = label.get_text()
        cat = model_to_cat.get(text, "Model")
        if cat == "Oracle":
            label.set_color(STYLE_ORACLE)
            label.set_fontweight("bold")
            label.set_fontstyle("normal")
        elif cat == "Simple":
            label.set_color(STYLE_LEARNED_BASELINE)
            label.set_fontweight("bold")
            label.set_fontstyle("normal")
        elif cat == "Baseline":
            label.set_color(STYLE_UNLEARNED_BASELINE)
            label.set_fontstyle("italic")
            label.set_fontweight("normal")
        else:
            label.set_color(STYLE_SOTA)
            label.set_fontweight("bold")
            label.set_fontstyle("normal")

    # Single legend: datasets, metrics, and model-name colour key (text-only, flush-left)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    class _FlushTextHandler(HandlerBase):
        """Legend handler that collapses the handle column to zero width."""

        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            handlebox.width = 0
            return []

    def _legend_spacer() -> Line2D:
        return Line2D([], [], linestyle="None", label=" ", color="none", linewidth=0)

    # Dataset handles (small squares)
    dataset_handles = [
        Line2D(
            [], [],
            marker="s",
            color="w",
            markerfacecolor=DATASET_COLORS.get(ds, "#333333"),
            markeredgecolor=DATASET_COLORS.get(ds, "#333333"),
            markersize=5,
            label=ds,
            linestyle="None",
        )
        for ds in datasets_canon
    ]

    # Metric handles (small shapes; calibrated metrics reuse uncalibrated marker)
    metric_handles = [
        Line2D(
            [], [],
            marker=metric_marker_for_plot(m),
            color="w",
            markerfacecolor="#555555",
            markeredgecolor="#555555",
            markersize=5,
            label=metric_display_for_plot(m),
            linestyle="None",
        )
        for m in metrics
    ]

    # Placeholder handles for model-name colour key (handler collapses handle column)
    model_category_handles = [
        Patch(facecolor="none", edgecolor="none", linewidth=0, label="Oracle"),
        Patch(facecolor="none", edgecolor="none", linewidth=0, label="Deep Learning"),
        Patch(facecolor="none", edgecolor="none", linewidth=0, label="Learned Baseline"),
        Patch(facecolor="none", edgecolor="none", linewidth=0, label="Unlearned Baseline"),
    ]

    all_handles = (
        model_category_handles
        + [_legend_spacer()]
        + dataset_handles
        + [_legend_spacer()]
        + metric_handles
    )

    handler_map = {h: _FlushTextHandler() for h in model_category_handles}

    leg = ax.legend(
        handles=all_handles,
        handler_map=handler_map,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fancybox=True,
        shadow=False,
        facecolor="white",
        edgecolor="gray",
        framealpha=0.95,
        fontsize=_NATURE_BASE_SIZE,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.5,
        borderpad=0.4,
    )

    # Style the first four legend texts (model-type key) to match y-axis model name colours
    model_legend_styles = [
        (STYLE_ORACLE, "bold", "normal"),
        (STYLE_SOTA, "bold", "normal"),
        (STYLE_LEARNED_BASELINE, "bold", "normal"),
        (STYLE_UNLEARNED_BASELINE, "normal", "italic"),
    ]
    for text, (color, weight, fontstyle) in zip(leg.get_texts()[:4], model_legend_styles):
        text.set_color(color)
        text.set_fontweight(weight)
        text.set_fontstyle(fontstyle)

    plt.tight_layout()
    plt.subplots_adjust(right=0.72)

    # Nudge model-category labels left (fraction of full symbol-column shift; tune
    # _MODEL_LEGEND_SHIFT_FRAC between 0 and 1).
    fig.canvas.draw()
    label_texts = leg.get_texts()
    if len(label_texts) >= 4:
        fs = float(label_texts[0].get_fontsize())
        dx_pts = (
            (float(leg.handlelength) + float(leg.handletextpad))
            * fs
            * _MODEL_LEGEND_SHIFT_FRAC
        )
        shift = mtransforms.ScaledTranslation(-dx_pts / 72.0, 0, fig.dpi_scale_trans)
        for t in label_texts[:4]:
            t.set_transform(t.get_transform() + shift)

    # Rank change annotations placed to the left of each y-tick label
    if rank_changes:
        renderer = fig.canvas.get_renderer()
        inv_data = ax.transData.inverted()
        for label in ax.get_yticklabels():
            model_disp = label.get_text()
            delta = rank_changes.get(model_disp, 0)
            if delta == 0:
                continue
            txt = f"↑+{delta} " if delta > 0 else f"↓{delta} "
            bbox = label.get_window_extent(renderer)
            x_left, y_mid = inv_data.transform(
                (bbox.x0, (bbox.y0 + bbox.y1) / 2)
            )
            ax.text(
                x_left, y_mid, txt,
                transform=ax.transData,
                fontsize=_NATURE_BASE_SIZE - 1,
                color="#888888",
                va="center",
                ha="right",
                clip_on=False,
            )

    # Save
    for fmt in formats:
        out = output_path.with_suffix(f".{fmt}")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=_NATURE_DPI, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Nature-style rank-aggregation dot plot for benchmark results (combo or single-pert)."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["wessels23", "norman19", "replogle20"]
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["wmse", "weighted_pearson_deltapert", "pds_wmse"]
    )
    parser.add_argument("--base-path", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/_benchmark_dataset/dataset_ranking.png"),
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument(
        "--rank-column-spacing",
        type=float,
        default=None,
        metavar="S",
        help=(
            "Horizontal distance between rank tick centers (1 = legacy). "
            "Default: auto — wide spacing only for single-pert with >3 datasets; "
            "otherwise 1.0. Marker size and intra-cluster dx unchanged."
        ),
    )
    parser.add_argument(
        "--reference-metrics",
        nargs="+",
        default=None,
        help=(
            "Reference metrics for rank change annotations. When set, the plot "
            "shows small arrows next to method names indicating how the y-axis "
            "position changed relative to the ranking produced by these metrics "
            "(e.g., pass weighted metrics when plotting unweighted results)."
        ),
    )
    parser.add_argument(
        "--exclude-models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Additional model ids to drop (e.g. scgpt). Applied with the built-in excluded list.",
    )
    args = parser.parse_args()

    print("Searching for benchmark directories...")
    bench_dirs = _find_dirs(args.base_path, args.datasets)
    for ds, dirs in bench_dirs.items():
        print(f"  {ds}: {len(dirs)} directories found")
        if not dirs:
            print(
                f"    WARNING: No outputs/benchmark_* directory matched for '{ds}' "
                f"(canonical: '{canonical_dataset(ds)}'). That dataset will have no points."
            )

    extra_excl = frozenset(args.exclude_models) if args.exclude_models else None
    print("\nLoading and computing ranks...")
    df = prepare_rank_data(bench_dirs, args.datasets, args.metrics, extra_excl)

    if df.empty:
        print("ERROR: No data loaded")
        sys.exit(1)

    print(f"Loaded {len(df)} data points")
    print(f"Models: {df['model'].nunique()}")
    print(f"Datasets: {df['dataset'].nunique()}")
    print(f"Metrics: {df['metric'].nunique()}")

    # Show median ranks (same ordering as the plot)
    median_ranks = df.groupby("model_display", observed=True)["rank"].median().sort_values()
    print("\nMedian ranks (best to worst):")
    for model, rank in median_ranks.items():
        cat = get_category(df[df["model_display"] == model]["model"].iloc[0])
        print(f"  {model} ({cat}): {rank:.2f}")

    # Compute rank changes relative to reference metrics (e.g., weighted → unweighted)
    rank_changes = None
    if args.reference_metrics:
        print(f"\nComputing reference ranking with metrics: {args.reference_metrics}")
        ref_df = prepare_rank_data(
            bench_dirs, args.datasets, args.reference_metrics, extra_excl
        )
        if not ref_df.empty:
            cur_median = df.groupby("model_display", observed=True)["rank"].median().sort_values()
            ref_median = ref_df.groupby("model_display", observed=True)["rank"].median().sort_values()
            cur_pos = {m: i + 1 for i, m in enumerate(cur_median.index)}
            ref_pos = {m: i + 1 for i, m in enumerate(ref_median.index)}
            rank_changes = {}
            for model_disp in cur_pos:
                if model_disp in ref_pos:
                    rank_changes[model_disp] = ref_pos[model_disp] - cur_pos[model_disp]
            print("Rank changes (current vs reference):")
            for model, change in sorted(rank_changes.items(), key=lambda x: -x[1]):
                if change > 0:
                    print(f"  {model}: ↑+{change}")
                elif change < 0:
                    print(f"  {model}: ↓{change}")
                else:
                    print(f"  {model}: —")

    rc = effective_rank_column_spacing(args.datasets, args.rank_column_spacing)
    plot_rank_aggregation(
        df,
        args.datasets,
        args.metrics,
        args.output,
        tuple(args.formats),
        rank_column_spacing=rc,
        rank_changes=rank_changes,
    )
    print(f"\nDone! Saved to {args.output}")


if __name__ == "__main__":
    main()
