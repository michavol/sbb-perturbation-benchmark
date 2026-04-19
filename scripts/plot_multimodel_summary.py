#!/usr/bin/env python3
"""
Generate beautiful summary plots for multi-model benchmarking results.

Creates:
1. Heatmap of model × metric with annotated mean values
2. Table with mean ± SEM for all models/metrics
3. Z-score heatmap showing relative performance vs. mean

Usage:
    python scripts/plot_multimodel_summary.py <path_to_detailed_metrics.csv>
    
Example:
    python scripts/plot_multimodel_summary.py outputs/benchmark_*/2025-*/detailed_metrics.csv

Omit ``scgpt`` from all plots:
    python scripts/plot_multimodel_summary.py path/to/detailed_metrics.csv --exclude-models scgpt

By default the ``interpolated_duplicate`` baseline is omitted from heatmaps, stripplots,
and tables; pass ``--include-interpolated-duplicate`` to show it. Auxiliary scatter plots
that compare duplicates still load the full CSV.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path
import sys
import argparse
from typing import List, Optional
from scipy.stats import ttest_rel, bootstrap, spearmanr
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


# ---------------------------------------------------------------------------
# Nature-style plotting conventions (matching analyses/perturbation_discrimination)
# ---------------------------------------------------------------------------
_NATURE_FONT = "Arial"
_NATURE_AXIS_LABEL_SIZE = 7
_NATURE_TICK_SIZE = 6
_NATURE_LEGEND_SIZE = 6
_NATURE_TITLE_SIZE = 7
_NATURE_FIG_SIZE = (3.5, 2.35)       # ~89 mm wide (single panel)
_NATURE_HEATMAP_FIG_SIZE = (7.2, 3.8)  # two-column width
_NATURE_DPI = 450


def _apply_nature_rc() -> None:
    """Set Matplotlib rcParams to Nature-journal conventions."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [_NATURE_FONT, "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": _NATURE_AXIS_LABEL_SIZE,
            "axes.titlesize": _NATURE_TITLE_SIZE,
            "xtick.labelsize": _NATURE_TICK_SIZE,
            "ytick.labelsize": _NATURE_TICK_SIZE,
            "legend.fontsize": _NATURE_LEGEND_SIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _savefig(path: Path, facecolor: str = "white", edgecolor: str = "none", **kwargs) -> None:
    """Save figure as both PNG (DPI 450) and PDF."""
    path = Path(path)
    plt.savefig(path, dpi=_NATURE_DPI, bbox_inches="tight",
                facecolor=facecolor, edgecolor=edgecolor, **kwargs)
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight",
                facecolor=facecolor, edgecolor=edgecolor)


# Metric display names and whether higher is better
METRIC_INFO = {
    'mse': {'display': 'MSE', 'higher_better': False},
    'wmse': {'display': 'wMSE', 'higher_better': False},
    'cmse': {'display': 'cMSE', 'higher_better': False},
    'cwmse': {'display': 'cwMSE', 'higher_better': False},
    'fmse': {'display': 'fMSE', 'higher_better': False},
    'fcmse': {'display': 'fcMSE', 'higher_better': False},
    'fpearson': {'display': 'fPearson', 'higher_better': True},
    'cpearson': {'display': 'cPearson', 'higher_better': False},
    'fcpearson': {'display': 'fcPearson', 'higher_better': False},
    'fr2': {'display': 'fR²', 'higher_better': True},
    'cr2': {'display': 'cR²', 'higher_better': False},
    'fcr2': {'display': 'fcR²', 'higher_better': False},
    # 'pearson_deltactrl': {'display': 'Pearson(Δ Ctrl)', 'higher_better': True},
    # 'pearson_deltactrl_degs': {'display': 'Pearson(Δ Ctrl DEG)', 'higher_better': True},
    'pearson_deltapert': {'display': 'Pearson Δ Pert', 'higher_better': True},
    'pearson_deltapert_degs': {'display': 'Pearson Δ Pert DEG', 'higher_better': True},
    'fpearson_deltactrl': {'display': 'fPearson Δ Ctrl', 'higher_better': True},
    'weighted_pearson_deltactrl': {'display': 'wPearson Δ Ctrl', 'higher_better': True},
    'cpearson_deltactrl': {'display': 'cPearson Δ Ctrl', 'higher_better': False},
    'cweighted_pearson_deltactrl': {'display': 'cwPearson Δ Ctrl', 'higher_better': False},
    'fcpearson_deltactrl': {'display': 'fcPearson Δ Ctrl', 'higher_better': False},
    'fpearson_deltapert': {'display': 'fPearson Δ Pert', 'higher_better': True},
    'weighted_pearson_deltapert': {'display': 'wPearson Δ Pert', 'higher_better': True},
    'cpearson_deltapert': {'display': 'cPearson Δ Pert', 'higher_better': False},
    'cweighted_pearson_deltapert': {'display': 'cwPearson Δ Pert', 'higher_better': False},
    'fcpearson_deltapert': {'display': 'fcPearson Δ Pert', 'higher_better': False},
    # 'r2_deltactrl': {'display': 'R² Δ Ctrl', 'higher_better': True},
    # 'r2_deltactrl_degs': {'display': 'R² Δ Ctrl DEG', 'higher_better': True},
    'r2_deltapert': {'display': 'R² Δ Pert', 'higher_better': True},
    'r2_deltapert_degs': {'display': 'R² Δ Pert DEG', 'higher_better': True},
    'fr2_deltactrl': {'display': 'fR² Δ Ctrl', 'higher_better': True},
    'cr2_deltactrl': {'display': 'cR² Δ Ctrl', 'higher_better': False},
    'fcr2_deltactrl': {'display': 'fcR² Δ Ctrl', 'higher_better': False},
    'fr2_deltapert': {'display': 'fR² Δ Pert', 'higher_better': True},
    'cr2_deltapert': {'display': 'cR² Δ Pert', 'higher_better': False},
    'fcr2_deltapert': {'display': 'fcR² Δ Pert', 'higher_better': False},
    # 'weighted_r2_deltactrl': {'display': 'wR² Δ Ctrl', 'higher_better': True},
    'weighted_r2_deltapert': {'display': 'wR² Δ Pert', 'higher_better': True},
    'cweighted_r2_deltactrl': {'display': 'cwR² Δ Ctrl', 'higher_better': False},
    'cweighted_r2_deltapert': {'display': 'cwR² Δ Pert', 'higher_better': False},
    'pds': {'display': 'PDS', 'higher_better': True},
    'pds_wmse': {'display': 'PDS (wMSE)', 'higher_better': True},
    'pds_pearson_deltapert': {'display': 'PDS (Pearson Δ Pert)', 'higher_better': True},
    'pds_weighted_pearson_deltapert': {'display': 'PDS (wPearson Δ Pert)', 'higher_better': True},
    'pds_r2_deltapert': {'display': 'PDS (R² Δ Pert)', 'higher_better': True},
    'pds_weighted_r2_deltapert': {'display': 'PDS (wR² Δ Pert)', 'higher_better': True},
    'cpds': {'display': 'cPDS', 'higher_better': False},
    'cpds_wmse': {'display': 'cPDS (wMSE)', 'higher_better': False},
    'cpds_pearson_deltapert': {'display': 'cPDS (Pearson Δ Pert)', 'higher_better': False},
    'cpds_weighted_pearson_deltapert': {'display': 'cPDS (wPearson Δ Pert)', 'higher_better': False},
    'cpds_r2_deltapert': {'display': 'cPDS (R² Δ Pert)', 'higher_better': False},
    'cpds_weighted_r2_deltapert': {'display': 'cPDS (wR² Δ Pert)', 'higher_better': False},
}

# Model families — keep in sync with scripts/plot_dataset_ranking.py (y-axis / legend key)
ORACLE_MODELS = frozenset(
    {
        "technical_duplicate",
        "epistasis_specific",
        "target_scaling_oracle",
        "synthetic_perturbation_oracle",
        "linear_regression_wmse",
        "onehot_linear_wmse",
    }
)

SIMPLE_MODELS = frozenset(
    {
        "target_scaling",
        "synthetic_perturbation",
        "linear_regression",
        "onehot_linear",
        "epistasis_shared",
    }
)

BASELINE_MODELS = frozenset(
    {
        "dataset_mean",
        "additive",
        "control_mean",
        "control",
    }
)

# Match plot_dataset_ranking.py: Oracle / Deep Learning / Learned Baseline / Unlearned Baseline
MODEL_CATEGORY_COLORS = {
    "oracle": "#2ca02c",
    "deep_learning": "#000000",
    "learned_baseline": "#0d47a1",
    "unlearned_baseline": "#666666",
}

# Legend text nudge for model-type key (match scripts/plot_dataset_ranking._MODEL_LEGEND_SHIFT_FRAC)
_STRIPPLOT_MODEL_LEGEND_SHIFT_FRAC = 0.28


def add_stripplot_model_category_legend(
    ax,
    *,
    mean_line_handle: Line2D,
    median_line_handle: Line2D,
    fontsize: int = 12,
) -> None:
    """Model-type key: text-only coloured labels like plot_dataset_ranking, then mean/median."""

    class _FlushTextHandler(HandlerBase):
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            handlebox.width = 0
            return []

    model_category_handles = [
        Patch(facecolor="none", edgecolor="none", linewidth=0, label="Oracle"),
        Patch(facecolor="none", edgecolor="none", linewidth=0, label="Deep Learning"),
        Patch(facecolor="none", edgecolor="none", linewidth=0, label="Learned Baseline"),
        Patch(facecolor="none", edgecolor="none", linewidth=0, label="Unlearned Baseline"),
    ]
    handler_map = {h: _FlushTextHandler() for h in model_category_handles}
    all_handles = model_category_handles + [mean_line_handle, median_line_handle]
    leg = ax.legend(
        handles=all_handles,
        handler_map=handler_map,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.9,
        fontsize=fontsize,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.5,
        borderpad=0.4,
    )
    mc = MODEL_CATEGORY_COLORS
    model_legend_styles = [
        (mc["oracle"], "bold", "normal"),
        (mc["deep_learning"], "bold", "normal"),
        (mc["learned_baseline"], "bold", "normal"),
        (mc["unlearned_baseline"], "normal", "italic"),
    ]
    for text, (color, weight, fontstyle) in zip(leg.get_texts()[:4], model_legend_styles):
        text.set_color(color)
        text.set_fontweight(weight)
        text.set_fontstyle(fontstyle)

    fig = ax.figure
    fig.canvas.draw()
    label_texts = leg.get_texts()
    if len(label_texts) >= 4:
        fs = float(label_texts[0].get_fontsize())
        dx_pts = (
            (float(leg.handlelength) + float(leg.handletextpad))
            * fs
            * _STRIPPLOT_MODEL_LEGEND_SHIFT_FRAC
        )
        shift = mtransforms.ScaledTranslation(-dx_pts / 72.0, 0, fig.dpi_scale_trans)
        for t in label_texts[:4]:
            t.set_transform(t.get_transform() + shift)


def normalize_model_name(model_name: str) -> str:
    """Normalize historical aliases to canonical model ids."""
    return 'control_mean' if model_name == 'control' else model_name


def get_model_category(model_name: str) -> str:
    """Return category aligned with plot_dataset_ranking.get_category (four-way split)."""
    normalized = normalize_model_name(model_name)
    if normalized in ORACLE_MODELS:
        return "oracle"
    if normalized in SIMPLE_MODELS:
        return "learned_baseline"
    if normalized in BASELINE_MODELS:
        return "unlearned_baseline"
    return "deep_learning"


def stripplot_alpha_for_dataset(
    dataset_name: str,
    n_perturbations: int,
    *,
    base_alpha: float,
    dense_factor: float = 0.72,
    dense_threshold: int = 500,
) -> float:
    """Slightly lower point alpha for very large nadig*/replogle* benchmarks (overlapping dots)."""
    ds = dataset_name.lower()
    if ("nadig" in ds or "replogle" in ds) and n_perturbations > dense_threshold:
        return float(max(0.15, base_alpha * dense_factor))
    return float(base_alpha)


def stripplot_metric_df_for_model(metric_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Drop rows where the metric is undefined for that model/baseline (stripplots only).

    cPDS (Pearson Δ Pert) is not defined for the dataset-mean baseline; omit those points.
    """
    _pds_deltapert_calibrated = {
        "cpds_pearson_deltapert",
        "cpds_weighted_pearson_deltapert",
        "cpds_r2_deltapert",
        "cpds_weighted_r2_deltapert",
    }
    if metric in _pds_deltapert_calibrated:
        return metric_df[metric_df["model"] != "dataset_mean"].copy()
    return metric_df


def is_calibrated_metric(metric: str) -> bool:
    """Return True for calibrated metrics (c*/fc* families)."""
    metric = metric.lower()
    return metric.startswith('c') or metric.startswith('fc')


def is_uncalibrated_r2_metric(metric: str) -> bool:
    """Return True for non-calibrated R2/weighted-R2 delta metrics."""
    metric = metric.lower()
    return (
        (
            metric.startswith('r2_')
            or metric.startswith('weighted_r2_')
            or metric.startswith('fr2_')
        )
        and not metric.startswith('c')
    )


def apply_calibrated_axis_compression(ax, metric: str, target_compressed_fraction: float = 0.3):
    """Compress y-values above 1.0 so [0,1] remains visually dominant.

    For calibrated metrics, values above 1 are shown with a piecewise-linear
    y-transform. The high-value region is compressed to roughly
    `target_compressed_fraction` of the total y-axis height.
    """
    if not is_calibrated_metric(metric):
        return

    y_min, y_max = ax.get_ylim()
    if y_max <= 1.0:
        return

    base_span = max(1e-6, 1.0 - y_min)
    high_span = max(1e-6, y_max - 1.0)

    # Solve alpha so transformed high-span takes desired fraction of the axis.
    target_ratio = target_compressed_fraction / max(1e-6, 1.0 - target_compressed_fraction)
    alpha = target_ratio * base_span / high_span
    alpha = float(np.clip(alpha, 0.02, 1.0))

    def forward(y):
        y = np.asarray(y, dtype=float)
        out = y.copy()
        mask = y > 1.0
        out[mask] = 1.0 + alpha * (y[mask] - 1.0)
        return out

    def inverse(y):
        y = np.asarray(y, dtype=float)
        out = y.copy()
        mask = y > 1.0
        out[mask] = 1.0 + (y[mask] - 1.0) / alpha
        return out

    ax.set_yscale('function', functions=(forward, inverse))

    # Keep [0,1] readable with denser ticks; add sparse ticks above 1.
    ticks = []
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        if y_min <= t <= y_max:
            ticks.append(t)
    for t in [1.5, 2.0, 3.0, 5.0, 10.0]:
        if t <= y_max:
            ticks.append(t)
    if y_min < 0:
        ticks = [y_min] + ticks
    if ticks:
        ax.set_yticks(sorted(set(float(t) for t in ticks)))


def apply_negative_r2_axis_compression(ax, metric: str, max_negative_fraction: float = 0.25):
    """Compress y-values below 0 so negative range is <= 25% of axis height."""
    if not is_uncalibrated_r2_metric(metric):
        return

    y_min, y_max = ax.get_ylim()
    if y_min >= 0.0:
        return

    pos_span = max(1e-6, y_max - 0.0)
    neg_span = max(1e-6, 0.0 - y_min)

    # Ensure negative visual span ratio <= max_negative_fraction.
    target_ratio = max_negative_fraction / max(1e-6, 1.0 - max_negative_fraction)
    alpha = target_ratio * pos_span / neg_span
    alpha = float(np.clip(alpha, 0.02, 1.0))

    def forward(y):
        y = np.asarray(y, dtype=float)
        out = y.copy()
        mask = y < 0.0
        out[mask] = alpha * y[mask]
        return out

    def inverse(y):
        y = np.asarray(y, dtype=float)
        out = y.copy()
        mask = y < 0.0
        out[mask] = y[mask] / alpha
        return out

    ax.set_yscale('function', functions=(forward, inverse))

    ticks = []
    for t in [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]:
        if y_min <= t <= y_max:
            ticks.append(t)
    if y_max > 1.0:
        ticks.append(min(y_max, 1.2))
    if ticks:
        ax.set_yticks(sorted(set(float(t) for t in ticks)))


def format_model_name(model_name: str) -> str:
    """Transform model names to display format (aligned with plot_dataset_ranking.format_model_name)."""
    name_map = {
        'dataset_mean': 'Dataset Mean',
        'control_mean': 'Control Mean',
        'technical_duplicate': 'Technical Duplicate',
        'interpolated_duplicate': 'Interpolated Duplicate',
        'additive': 'Additive',
        'linear': 'Linear',
        'ground_truth': 'Ground Truth',
        'baselines': 'Baselines',
        'random_perturbation': 'Random Perturbation',
        'presage': 'PRESAGE',
        'sclambda': 'scLambda',
        'scgpt': 'scGPT',
        'fmlp_genept': 'fMLP-GenePT',
        'fmlp_esm2': 'fMLP-ESM2',
        'fmlp_geneformer': 'fMLP-Geneformer',
        'fmlp_scgpt': 'fMLP-scGPT',
        'gears': 'GEARS',
        'epistasis_shared': 'Epistasis (Global)',
        'epistasis_specific': 'Epistasis (Specific)',
        'epistasis_hybrid': 'Epistasis (Hybrid)',
        'linear_regression': 'Linear Regression',
        'onehot_linear': 'Linear Regression',
        'linear_regression_wmse': 'Linear Regression (wMSE)',
        'onehot_linear_wmse': 'Linear Regression (wMSE)',
        'pdae': 'PDAE',
        'target_scaling': 'Target Scaling',
        'synthetic_perturbation': 'Target Scaling',
        'target_scaling_oracle': 'Target Scaling (Oracle)',
        'synthetic_perturbation_oracle': 'Target Scaling (Oracle)',
    }
    return name_map.get(model_name, model_name.replace("_", " ").title())


# Pre-computed formatted names for tick styling (four-way key, same as plot_dataset_ranking legend)
_FORMATTED_ORACLE_NAMES = frozenset(format_model_name(m) for m in ORACLE_MODELS)
_FORMATTED_LEARNED_NAMES = frozenset(format_model_name(m) for m in SIMPLE_MODELS)
_FORMATTED_UNLEARNED_NAMES = frozenset(format_model_name(m) for m in BASELINE_MODELS) | {
    'Linear', 'Baselines', 'Ground Truth', 'Random Perturbation',
}


def apply_model_category_tick_style(label, fontsize: int | None = None) -> None:
    """Colour / font weight / style for a tick label from its display text."""
    text = label.get_text()
    if text in _FORMATTED_ORACLE_NAMES:
        label.set_color(MODEL_CATEGORY_COLORS["oracle"])
        label.set_fontweight("bold")
        label.set_fontstyle("normal")
    elif text in _FORMATTED_LEARNED_NAMES:
        label.set_color(MODEL_CATEGORY_COLORS["learned_baseline"])
        label.set_fontweight("bold")
        label.set_fontstyle("normal")
    elif text in _FORMATTED_UNLEARNED_NAMES:
        label.set_color(MODEL_CATEGORY_COLORS["unlearned_baseline"])
        label.set_fontstyle("italic")
        label.set_fontweight("normal")
    else:
        label.set_color(MODEL_CATEGORY_COLORS["deep_learning"])
        label.set_fontweight("bold")
        label.set_fontstyle("normal")
    if fontsize is not None:
        label.set_fontsize(fontsize)


def _style_yticklabels(ax, fontsize: int = 13) -> None:
    """Style y-tick labels: Oracle / Deep Learning / Learned Baseline / Unlearned Baseline."""
    for label in ax.get_yticklabels():
        apply_model_category_tick_style(label, fontsize)


def read_detailed_metrics_csv(
    csv_path: Path,
    *,
    include_interpolated_duplicate: bool = False,
    exclude_models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load detailed_metrics.csv; optionally drop ``interpolated_duplicate`` rows; optionally exclude models."""
    df = pd.read_csv(csv_path)
    if not include_interpolated_duplicate:
        df = df[df["model"] != "interpolated_duplicate"].copy()
    if exclude_models:
        df = df[~df["model"].isin(exclude_models)].copy()
    return df


def load_and_process_data(
    csv_path: Path,
    exclude_baselines: bool = False,
    *,
    include_interpolated_duplicate: bool = False,
    exclude_models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load detailed metrics CSV and calculate summary statistics."""
    df = read_detailed_metrics_csv(
        csv_path,
        include_interpolated_duplicate=include_interpolated_duplicate,
        exclude_models=exclude_models,
    )
    
    # Optionally exclude baseline models to focus on trained models
    if exclude_baselines:
        baseline_keywords = ['control_mean', 'technical_duplicate',
                           'interpolated_duplicate', 'additive', 'dataset_mean', 
                           'linear', 'baselines', 'ground_truth', 'random_perturbation']
        df = df[~df['model'].isin(baseline_keywords)]
    
    # Calculate mean, SEM, std, and count for each model-metric combination
    summary = df.groupby(['model', 'metric'])['value'].agg(['mean', 'sem', 'std', 'count']).reset_index()
    
    return summary


def create_mean_heatmap(summary_df: pd.DataFrame, output_path: Path, figsize=(16, 10)):
    """Create heatmap of model × metric with annotated mean values."""
    # Pivot to get model × metric matrix
    pivot_df = summary_df.pivot(index='model', columns='metric', values='mean')
    
    # Reorder columns to match METRIC_INFO order
    available_metrics = [m for m in METRIC_INFO.keys() if m in pivot_df.columns]
    pivot_df = pivot_df[available_metrics]
    
    # Sort rows (models) by mean performance across all metrics
    # For each model, calculate average z-score (accounting for direction)
    # Treat NaN as 0 for sorting purposes
    model_scores = []
    for model in pivot_df.index:
        zscores = []
        for metric in available_metrics:
            val = pivot_df.loc[model, metric]
            if pd.isna(val):
                val = 0  # Treat NaN as 0
            metric_mean = pivot_df[metric].fillna(0).mean()  # Treat NaN as 0 in mean calculation
            metric_std = pivot_df[metric].fillna(0).std()
            if metric_std > 0:
                zscore = (val - metric_mean) / metric_std
                # Flip for "lower is better" metrics
                if not METRIC_INFO[metric]['higher_better']:
                    zscore = -zscore
                zscores.append(zscore)
        model_scores.append(np.mean(zscores) if zscores else 0)
    
    # Sort models by average z-score (best to worst)
    model_order = [m for _, m in sorted(zip(model_scores, pivot_df.index), reverse=True)]
    pivot_df = pivot_df.loc[model_order]
    
    # Sort columns (metrics) by mean value (treat NaN as 0)
    metric_means = pivot_df.fillna(0).mean(axis=0)
    sorted_metrics = metric_means.sort_values(ascending=False).index.tolist()
    pivot_df = pivot_df[sorted_metrics]
    
    # Rename row indices (models) to display names
    pivot_df.index = [format_model_name(m) for m in pivot_df.index]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap with annotations
    heatmap = sns.heatmap(
        pivot_df, 
        annot=True, 
        fmt='.3f',
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Metric Value'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        annot_kws={'size': 11, 'weight': 'bold'}
    )
    
    # Set colorbar label size
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Metric Value', fontsize=14, fontweight='bold')
    
    ax.set_title('Multi-Model Benchmark Results: Mean Metric Values', 
                 fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Metrics', fontsize=16, fontweight='bold')
    ax.set_ylabel('Models', fontsize=16, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=13)
    
    _style_yticklabels(ax)
    
    plt.tight_layout()
    _savefig(output_path)
    plt.close()
    
    print(f"✓ Saved mean heatmap to: {output_path}")


def create_zscore_heatmap(summary_df: pd.DataFrame, output_path: Path, figsize=(16, 10)):
    """Create heatmap showing performance relative to dataset_mean baseline."""
    # Pivot to get model × metric matrix
    pivot_df = summary_df.pivot(index='model', columns='metric', values='mean')
    
    # Reorder columns to match METRIC_INFO order
    available_metrics = [m for m in METRIC_INFO.keys() if m in pivot_df.columns]
    pivot_df = pivot_df[available_metrics]
    
    # Determine which baseline to use for each metric
    # deltapert metrics: control_mean is the null baseline
    # other metrics: dataset_mean is the null baseline
    
    # Calculate DRF (Discriminatory Reliability Factor) vs. appropriate baseline
    # DRF = fraction of distance from baseline to perfect performance
    relative_df = pivot_df.copy()
    for col in relative_df.columns:
        # CELLSIMBENCH:
        # Choose baseline based on metric type
        # if 'deltapert' in col:
        #     # For deltapert metrics, use control_mean as null
        #     if 'control_mean' in pivot_df.index:
        #         baseline_val = pivot_df.loc['control_mean', col]
        #     else:
        #         print(f"Warning: control_mean not found for {col}, using dataset_mean")
        #         baseline_val = pivot_df.loc['dataset_mean', col] if 'dataset_mean' in pivot_df.index else pivot_df[col].mean()
        # else:
        #     # For other metrics, use dataset_mean as null
        #     if 'dataset_mean' in pivot_df.index:
        #         baseline_val = pivot_df.loc['dataset_mean', col]
        #     else:
        #         print(f"Warning: dataset_mean not found for {col}, using mean")
        #         baseline_val = pivot_df[col].mean()
        baseline_val = pivot_df.loc['control_mean', col]
        
        if METRIC_INFO[col]['higher_better']:
            # For "higher is better" metrics (perfect = 1.0)
            # DRF = (model - baseline) / (1.0 - baseline)
            perfect_val = 1.0
            denominator = perfect_val - baseline_val
            if abs(denominator) > 1e-6:
                relative_df[col] = (pivot_df[col] - baseline_val) / denominator
            else:
                # Baseline already at perfect, just show difference
                relative_df[col] = pivot_df[col] - baseline_val
        else:
            # For "lower is better" metrics (perfect = 0.0)
            # DRF = (baseline - model) / (baseline - 0.0)
            perfect_val = 0.0
            denominator = baseline_val - perfect_val
            if abs(denominator) > 1e-6:
                relative_df[col] = (baseline_val - pivot_df[col]) / denominator
            else:
                # Baseline already at perfect, just show difference
                relative_df[col] = baseline_val - pivot_df[col]
        
        # Clip to reasonable range: DRF typically in [-1, 2]
        # DRF = 0: same as baseline
        # DRF = 1: perfect performance
        # DRF < 0: worse than baseline
        # DRF > 1: better than "perfect" (rare, usually means overfitting)
        relative_df[col] = relative_df[col].clip(-1, 2)
    
    # Sort rows (models) by mean relative improvement (best to worst)
    # Treat NaN as 0 for sorting
    model_mean_improvement = relative_df.fillna(0).mean(axis=1).sort_values(ascending=False)
    relative_df = relative_df.loc[model_mean_improvement.index]
    
    # Sort columns (metrics) by mean improvement across all models
    # Treat NaN as 0 for sorting
    metric_mean_improvement = relative_df.fillna(0).mean(axis=0).sort_values(ascending=False)
    relative_df = relative_df[metric_mean_improvement.index]
    
    # Rename columns to display names (after sorting)
    column_mapping = {m: METRIC_INFO[m]['display'] for m in available_metrics if m in relative_df.columns}
    relative_df.columns = [column_mapping.get(c, c) for c in relative_df.columns]
    
    # Rename row indices (models) to display names
    relative_df.index = [format_model_name(m) for m in relative_df.index]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create diverging heatmap (positive = better than dataset_mean)
    heatmap = sns.heatmap(
        relative_df,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',  # Red = worse, Blue = better
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'DRF: Distance to Perfect\n0=Baseline | 1=Perfect'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        annot_kws={'size': 11, 'weight': 'bold'}
    )
    
    # Set colorbar label size
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('DRF: Distance to Perfect\n0=Baseline | 1=Perfect', fontsize=14, fontweight='bold')
    
    # CELLSIMBENCH:
    # ax.set_title('Multi-Model Benchmark: DRF vs. Null Baseline\n(dataset_mean for most metrics, control_mean for Δpert metrics)', 
    #              fontsize=20, fontweight='bold', pad=20)
    ax.set_title('Multi-Model Benchmark: DRF vs. Control Mean', 
                 fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Metrics', fontsize=16, fontweight='bold')
    ax.set_ylabel('Models', fontsize=16, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=13)
    
    _style_yticklabels(ax)
    
    plt.tight_layout()
    _savefig(output_path)
    plt.close()
    
    print(f"✓ Saved relative performance heatmap to: {output_path}")


def create_summary_table(summary_df: pd.DataFrame, output_path: Path):
    """Create table with mean ± SEM for all models and metrics."""
    # Pivot to get model × metric matrix for mean
    mean_df = summary_df.pivot(index='model', columns='metric', values='mean')
    sem_df = summary_df.pivot(index='model', columns='metric', values='sem')
    
    # Reorder columns to match METRIC_INFO order
    available_metrics = [m for m in METRIC_INFO.keys() if m in mean_df.columns]
    mean_df = mean_df[available_metrics]
    sem_df = sem_df[available_metrics]
    
    # Create formatted table with mean ± SEM
    table_data = []
    for model in mean_df.index:
        row = {'Model': model}
        for metric in available_metrics:
            mean_val = mean_df.loc[model, metric]
            sem_val = sem_df.loc[model, metric]
            # Format as "mean ± sem"
            row[METRIC_INFO[metric]['display']] = f"{mean_val:.4f} ± {sem_val:.4f}"
        table_data.append(row)
    
    table_df = pd.DataFrame(table_data)
    
    # Save as CSV
    table_df.to_csv(output_path, index=False)
    print(f"✓ Saved summary table to: {output_path}")
    
    # Also create a visual table figure
    fig, ax = plt.subplots(figsize=(20, len(table_df) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.15] + [0.12] * (len(table_df.columns) - 1)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(table_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_df) + 1):
        for j in range(len(table_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F2F2F2')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Multi-Model Benchmark: Summary Statistics (Mean ± SEM)', 
              fontsize=14, fontweight='bold', pad=20)
    
    table_fig_path = output_path.parent / output_path.name.replace('.csv', '_figure.png')
    _savefig(table_fig_path)
    plt.close()
    
    print(f"✓ Saved summary table figure to: {table_fig_path}")


def create_statistical_comparison_heatmap(
    csv_path: Path,
    output_path: Path,
    dataset_name: str,
    figsize=(20, 10),
    *,
    include_interpolated_duplicate: bool = False,
    exclude_models: Optional[List[str]] = None,
):
    """Create heatmap showing statistical test results vs. appropriate baseline."""
    # Load full data (not summary)
    df = read_detailed_metrics_csv(
        csv_path,
        include_interpolated_duplicate=include_interpolated_duplicate,
        exclude_models=exclude_models,
    )
    
    # Get unique models and metrics
    models = df['model'].unique()
    metrics = [m for m in METRIC_INFO.keys() if m in df['metric'].unique()]
    
    # Initialize matrices for test statistics and p-values
    # Initialize with NaN instead of leaving uninitialized
    test_stats = pd.DataFrame(np.nan, index=models, columns=metrics, dtype=float)
    p_values = pd.DataFrame(np.nan, index=models, columns=metrics, dtype=float)
    
    # For each model and metric, perform paired t-test vs. appropriate baseline
    # Store raw p-values for correction
    raw_p_values = p_values.copy()
    
    for metric in metrics:
        # Determine baseline based on metric type
        # CELLSIMBENCH:
        # if 'deltapert' in metric:
        #     baseline_model = 'control_mean'
        # else:
        #     baseline_model = 'dataset_mean'
        baseline_model = 'control_mean'
        # Get baseline data
        baseline_data = df[(df['model'] == baseline_model) & (df['metric'] == metric)]
        
        if len(baseline_data) == 0:
            print(f"Warning: {baseline_model} not found for {metric}, skipping")
            continue
        
        # Store raw p-values for this metric to apply correction
        metric_p_values = []
        metric_models = []
        
        # For each model, test against baseline
        for model in models:
            if model == baseline_model or model == 'ground_truth':
                # Skip baseline vs itself and ground truth
                test_stats.loc[model, metric] = 0.0
                raw_p_values.loc[model, metric] = 1.0
                continue
            
            model_data = df[(df['model'] == model) & (df['metric'] == metric)]
            
            if len(model_data) == 0:
                continue
            
            # Merge on perturbation to ensure pairing
            merged = pd.merge(
                baseline_data[['perturbation', 'value']],
                model_data[['perturbation', 'value']],
                on='perturbation',
                suffixes=('_baseline', '_model')
            )
            
            # Remove rows with NaN values (common for DEG metrics)
            merged = merged.dropna(subset=['value_baseline', 'value_model'])
            
            if len(merged) < 3:
                # Need at least 3 pairs for t-test
                continue
            
            # Perform one-sided paired t-test
            try:
                if METRIC_INFO[metric]['higher_better']:
                    # Test if model > baseline
                    t_stat, pval = ttest_rel(
                        merged['value_model'], 
                        merged['value_baseline'],
                        alternative='greater'
                    )
                else:
                    # Test if model < baseline (better for MSE/WMSE)
                    t_stat, pval = ttest_rel(
                        merged['value_model'],
                        merged['value_baseline'],
                        alternative='less'
                    )
                    # Flip sign so positive = better
                    t_stat = -t_stat
                
                test_stats.loc[model, metric] = t_stat  
                raw_p_values.loc[model, metric] = pval
                metric_p_values.append(pval)
                metric_models.append(model)
            except Exception as e:
                print(f"Warning: t-test failed for {model} on {metric}: {e}")
                continue
        
        # Apply Bonferroni correction within this metric
        n_tests = len(metric_p_values)
        if n_tests > 0:
            for model, raw_p in zip(metric_models, metric_p_values):
                corrected_p = min(raw_p * n_tests, 1.0)  # Cap at 1.0
                p_values.loc[model, metric] = corrected_p
    
    # Convert to numeric
    test_stats = test_stats.apply(pd.to_numeric, errors='coerce')
    p_values = p_values.apply(pd.to_numeric, errors='coerce')
    
    # Sort rows and columns by mean test statistic
    # Treat NaN as 0 for sorting
    row_means = test_stats.fillna(0).mean(axis=1).sort_values(ascending=False)
    col_means = test_stats.fillna(0).mean(axis=0).sort_values(ascending=False)
    
    test_stats = test_stats.loc[row_means.index, col_means.index]
    p_values = p_values.loc[row_means.index, col_means.index]
    
    # Create annotations with test statistic and significance stars
    annotations = test_stats.copy().astype(str)
    for i, model in enumerate(test_stats.index):
        for j, metric in enumerate(test_stats.columns):
            stat_val = test_stats.iloc[i, j]
            p_val = p_values.iloc[i, j]
            
            if pd.isna(stat_val) or pd.isna(p_val):
                # Leave blank for null baselines and missing data
                annotations.iloc[i, j] = ''
            else:
                # Add significance stars (4 levels)
                if p_val < 0.0001:
                    sig = '****'
                elif p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                else:
                    sig = ''
                
                # Format: "t-stat\nstars"
                if sig:
                    annotations.iloc[i, j] = f"{stat_val:.1f}\n{sig}"
                else:
                    annotations.iloc[i, j] = f"{stat_val:.1f}"
    
    # Rename columns to display names
    display_cols = [METRIC_INFO[m]['display'] for m in test_stats.columns]
    test_stats.columns = display_cols
    annotations.columns = display_cols
    
    # Rename row indices (models) to display names
    test_stats.index = [format_model_name(m) for m in test_stats.index]
    annotations.index = [format_model_name(m) for m in annotations.index]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap with diverging colors (clipped at ±30)
    heatmap = sns.heatmap(
        test_stats,
        annot=annotations,
        fmt='',
        cmap='RdBu_r',
        center=0,
        vmin=-30,
        vmax=30,
        cbar_kws={'label': 't-statistic'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        annot_kws={'size': 10, 'weight': 'bold'}
    )
    
    # Set colorbar label size
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('t-statistic vs baseline (↑)', 
                   fontsize=14, fontweight='bold')
    
    ax.set_title(f'Performance over negative control ({dataset_name})', 
                 fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Metrics', fontsize=16, fontweight='bold')
    ax.set_ylabel('Models', fontsize=16, fontweight='bold')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=13)
    
    _style_yticklabels(ax)
    
    plt.tight_layout()
    _savefig(output_path)
    plt.close()
    
    print(f"✓ Saved statistical comparison heatmap to: {output_path}")


def create_stripplot_for_metric(args):
    """Create a strip plot for a single metric (for parallel processing)."""
    df, metric, output_dir, dataset_name, use_log_scale, sort_by = args
    
    metric_df = df[df['metric'] == metric].copy()
    
    if metric_df.empty:
        return None
    
    # For calibrated metrics: clip values to [0, 1] so that means/medians
    # and all statistics are computed on the capped data.
    if is_calibrated_metric(metric):
        metric_df['value'] = metric_df['value'].clip(0, 1)

    metric_df = stripplot_metric_df_for_model(metric_df, metric)
    if metric_df.empty:
        return None
    
    # Order by configured statistic; keep both means and medians for reference lines.
    if sort_by == 'mean':
        model_order_values = metric_df.groupby('model')['value'].mean()
    else:
        model_order_values = metric_df.groupby('model')['value'].median()
    model_means = metric_df.groupby('model')['value'].mean()
    model_medians = metric_df.groupby('model')['value'].median()
    
    # Sort based on metric type
    if is_calibrated_metric(metric) or metric.lower() in ['mse', 'wmse', 'fmse']:
        ordered_models = model_order_values.sort_values(ascending=True).index.tolist()
    else:
        ordered_models = model_order_values.sort_values(ascending=False).index.tolist()
    
    # Reorder models
    metric_df['model'] = pd.Categorical(metric_df['model'], 
                                       categories=ordered_models, 
                                       ordered=True)
    
    # Calculate figure size
    n_models = len(ordered_models)
    fig_width = max(10, min(20, n_models * 1.1))  # More spacing
    fig_height = 7
    
    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
    ax.set_facecolor('white')
    
    mean_color = 'black'
    median_color = 'black'

    palette = [
        MODEL_CATEGORY_COLORS[get_model_category(model_name)]
        for model_name in ordered_models
    ]

    n_perts = int(metric_df["perturbation"].nunique())
    base_alpha = 0.65 if use_log_scale else 0.45
    alpha_val = stripplot_alpha_for_dataset(
        dataset_name, n_perts, base_alpha=base_alpha,
    )

    # Strip plot (higher alpha for log scale version)
    sns.stripplot(data=metric_df, x='model', y='value', 
                 hue='model',
                 palette=palette,
                 size=3,
                 alpha=alpha_val,
                 jitter=True,
                 legend=False,
                 ax=ax)
    
    # Add mean and median lines aligned to ordered model positions.
    for i, model_name in enumerate(ordered_models):
        model_mean = float(model_means.loc[model_name])
        model_median = float(model_medians.loc[model_name])
        ax.hlines(model_mean, i - 0.25, i + 0.25, 
                 colors=mean_color, 
                 linewidth=2.5,
                 zorder=3)
        ax.hlines(model_median, i - 0.25, i + 0.25,
                 colors=median_color,
                 linewidth=2.0,
                 linestyles=':',
                 zorder=3)
        
        if n_models <= 8:
            ax.text(i, model_mean, f'{model_mean:.3f}', 
                   horizontalalignment='center',
                   verticalalignment='bottom',
                   fontsize=13,
                   color=mean_color,
                   fontweight='bold')
    
    # Customize plot
    metric_display = METRIC_INFO.get(metric, {}).get('display', metric)
    
    # Add directional arrow in axis label
    direction_arrow = '↓' if not METRIC_INFO.get(metric, {}).get('higher_better', True) else '↑'
    
    # Apply log scale for MSE/WMSE in log version
    if use_log_scale and metric.lower() in ['mse', 'wmse']:
        ax.set_yscale('log')
        _style_log_axis_nature(ax, axis='y')
        ylabel = f'{metric_display} (log scale, {direction_arrow})'
    else:
        ylabel = f'{metric_display} ({direction_arrow})'
    
    ax.set_title(f'{metric_display} in {dataset_name}', 
                fontsize=20, fontweight='bold', pad=25)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    
    # Style x-axis labels with nice formatting
    ax.set_xticks(range(len(ordered_models)))
    display_labels = [format_model_name(m) for m in ordered_models]
    xticklabels = ax.set_xticklabels(display_labels, rotation=45, ha='right')
    
    for label in xticklabels:
        apply_model_category_tick_style(label, 12)
    
    # Grid and styling
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=12)
    
    # For uncalibrated higher-is-better delta metrics (R² and Pearson delta variants):
    # clip the y-axis at LOWER_CLIP and annotate per-model count of points below it.
    _LOWER_CLIP = -0.25
    _delta_higher_better_metrics = {
        'r2_deltactrl', 'r2_deltactrl_degs', 'r2_deltapert', 'r2_deltapert_degs',
        'fr2_deltactrl', 'fr2_deltapert',
        'weighted_r2_deltactrl', 'weighted_r2_deltapert',
        'pearson_deltapert', 'pearson_deltapert_degs',
        'fpearson_deltactrl', 'fpearson_deltapert',
        'weighted_pearson_deltactrl', 'weighted_pearson_deltapert',
    }
    if metric in _delta_higher_better_metrics:
        for i, model_name in enumerate(ordered_models):
            model_df_subset = metric_df[metric_df['model'] == model_name]
            n_below = int((model_df_subset['value'] < _LOWER_CLIP).sum())
            if n_below > 0:
                ax.text(i + 0.12, _LOWER_CLIP + 0.02, f'↓ {n_below}',
                        ha='left', va='bottom',
                        fontsize=10,
                        color='#555555')
        current_ylim = ax.get_ylim()
        ax.set_ylim(_LOWER_CLIP, current_ylim[1])

    # Add reference line at 0 for certain metrics (0 = chance for PDS; 0 = no correlation for others)
    if any(keyword in metric.lower() for keyword in ['pearson', 'r2', 'delta', 'pds']):
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
    
    # Model-type key + mean/median (text-only category labels: plot_dataset_ranking style)
    add_stripplot_model_category_legend(
        ax,
        mean_line_handle=Line2D([0], [0], color=mean_color, linewidth=3, label='Mean'),
        median_line_handle=Line2D(
            [0], [0], color=median_color, linewidth=2.5, linestyle=':', label='Median'
        ),
        fontsize=12,
    )

    if is_calibrated_metric(metric):
        # Values are already clipped to [0,1]; enforce hard y-axis bounds.
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save with appropriate filename
    if use_log_scale:
        filename = f"stripplot_log_{metric}.png"
    else:
        filename = f"stripplot_{metric}.png"
    
    _savefig(output_dir / filename)
    plt.close()
    
    return filename


def create_forest_plot_for_metric(args):
    """Create a forest plot showing performance difference from baseline with 95% CI."""
    df, metric, output_dir, dataset_name = args
    
    metric_df = df[df['metric'] == metric].copy()
    
    if metric_df.empty:
        return None
    
    # Determine null baseline for this metric
    if 'deltapert' in metric:
        null_baseline = 'control_mean'
    else:
        null_baseline = 'dataset_mean'
    
    # Get baseline data
    baseline_data = metric_df[metric_df['model'] == null_baseline]
    
    if len(baseline_data) == 0:
        return None
    
    # Calculate differences from baseline for each model
    model_differences = {}
    models = metric_df['model'].unique()
    
    for model_name in models:
        if model_name == null_baseline:
            continue
        
        model_data = metric_df[metric_df['model'] == model_name]
        
        # Merge on perturbation
        merged = pd.merge(
            baseline_data[['perturbation', 'value']],
            model_data[['perturbation', 'value']],
            on='perturbation',
            suffixes=('_baseline', '_model')
        )
        
        # Remove NaN values
        merged = merged.dropna(subset=['value_baseline', 'value_model'])
        
        if len(merged) >= 3:
            # Calculate difference for each perturbation
            differences = merged['value_model'] - merged['value_baseline']
            
            # Calculate mean and 95% CI using bootstrap
            mean_diff = differences.mean()
            
            # Bootstrap CI (non-parametric)
            result = bootstrap(
                (differences.values,),
                statistic=np.mean,
                n_resamples=10000,
                confidence_level=0.95,
                method='percentile',
                random_state=42
            )
            
            model_differences[model_name] = {
                'mean': mean_diff,
                'ci_lower': result.confidence_interval.low,
                'ci_upper': result.confidence_interval.high,
                'n': len(differences)
            }
    
    # Calculate t-tests (same as stripplot)
    test_results = {}
    
    for model_name in models:
        if model_name == null_baseline:
            continue
        
        model_data = metric_df[metric_df['model'] == model_name]
        
        merged = pd.merge(
            baseline_data[['perturbation', 'value']],
            model_data[['perturbation', 'value']],
            on='perturbation',
            suffixes=('_baseline', '_model')
        )
        
        merged = merged.dropna(subset=['value_baseline', 'value_model'])
        
        if len(merged) >= 3:
            try:
                if METRIC_INFO.get(metric, {}).get('higher_better', True):
                    t_stat, p_val = ttest_rel(
                        merged['value_model'],
                        merged['value_baseline'],
                        alternative='greater'
                    )
                else:
                    t_stat, p_val = ttest_rel(
                        merged['value_model'],
                        merged['value_baseline'],
                        alternative='less'
                    )
                
                test_results[model_name] = (t_stat, p_val)
            except:
                test_results[model_name] = (np.nan, np.nan)
    
    if not model_differences:
        return None
    
    # Sort models by mean difference
    # For "higher is better": positive diff is better (sort descending)
    # For "lower is better": negative diff is better (sort ascending)
    if METRIC_INFO.get(metric, {}).get('higher_better', True):
        sorted_models = sorted(model_differences.items(), 
                              key=lambda x: x[1]['mean'], 
                              reverse=True)
    else:
        # For MSE/WMSE: more negative difference = better
        sorted_models = sorted(model_differences.items(), 
                              key=lambda x: x[1]['mean'], 
                              reverse=False)
    
    # Create forest plot (narrower)
    fig, ax = plt.subplots(figsize=(8, max(7, len(sorted_models) * 0.3)))
    
    # Plot each model
    y_positions = []
    for i, (model_name, stats) in enumerate(sorted_models):
        y_pos = len(sorted_models) - i - 1
        y_positions.append(y_pos)

        color = MODEL_CATEGORY_COLORS[get_model_category(model_name)]
        
        # Plot point estimate
        ax.plot(stats['mean'], y_pos, 'o', color=color, markersize=7.5, zorder=3)
        
        # Plot 95% CI as error bars
        ax.plot([stats['ci_lower'], stats['ci_upper']], [y_pos, y_pos],
               color=color, linewidth=2, zorder=2)
        
        # Add caps to error bars
        cap_height = 0.15
        ax.plot([stats['ci_lower'], stats['ci_lower']], 
               [y_pos - cap_height, y_pos + cap_height],
               color=color, linewidth=2, zorder=2)
        ax.plot([stats['ci_upper'], stats['ci_upper']], 
               [y_pos - cap_height, y_pos + cap_height],
               color=color, linewidth=2, zorder=2)
    
    # Add vertical line at 0 (no difference from baseline)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    
    # Set y-axis labels with styled text and nice formatting
    ax.set_yticks(y_positions)
    model_labels = [format_model_name(model_name) for model_name, _ in sorted_models]
    ax.set_yticklabels(model_labels)
    
    # Style y-axis labels (Oracle / Deep Learning / Learned / Unlearned)
    for label in ax.get_yticklabels():
        apply_model_category_tick_style(label, 13)
    
    # Add statistical annotations (same as stripplot)
    if test_results:
        # Find max CI upper bound to position annotations
        max_ci_upper = max(stats['ci_upper'] for _, stats in sorted_models)
        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min
        
        # Position annotations slightly beyond max CI (5% of range)
        annot_x = max_ci_upper + x_range * 0.05
        
        for i, (model_name, _) in enumerate(sorted_models):
            y_pos = len(sorted_models) - i - 1
            
            if model_name in test_results:
                t_stat, p_val = test_results[model_name]
                
                if not np.isnan(t_stat) and not np.isnan(p_val):
                    # Format p-value
                    if p_val < 0.01:
                        p_text = f"p={p_val:.1e}"
                    elif p_val < 1.0:
                        p_text = f"p={p_val:.3f}"
                    else:
                        p_text = f"p=1"
                    
                    annot_text = f"{p_text}\nt={t_stat:.1f}"
                    
                    # Position to the right of all CIs, left-aligned
                    ax.text(annot_x, y_pos, annot_text,
                           ha='left', va='center',
                           fontsize=10)
    
    # Labels and title
    metric_display = METRIC_INFO.get(metric, {}).get('display', metric)
    direction_arrow = '↓' if not METRIC_INFO.get(metric, {}).get('higher_better', True) else '↑'
    baseline_display = format_model_name(null_baseline).lower()
    
    ax.set_xlabel(f'Δ metric (prediction - {baseline_display}) ({direction_arrow})', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Model', fontsize=14, fontweight='bold')
    ax.set_title(f'{metric_display} in {dataset_name}', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save
    filename = f"forest_{metric}.png"
    _savefig(output_dir / filename)
    plt.close()
    
    return filename


def create_all_stripplots(
    csv_path: Path,
    output_dir: Path,
    dataset_name: str,
    sort_by: str = 'mean',
    *,
    include_interpolated_duplicate: bool = False,
    exclude_models: Optional[List[str]] = None,
):
    """Create strip plots for all metrics in parallel."""
    df = read_detailed_metrics_csv(
        csv_path,
        include_interpolated_duplicate=include_interpolated_duplicate,
        exclude_models=exclude_models,
    )
    
    # Get unique metrics
    metrics = [m for m in METRIC_INFO.keys() if m in df['metric'].unique()]
    
    print(f"Creating {len(metrics)} strip plots in parallel...")
    
    # Prepare arguments for parallel processing
    args_list = [(df, metric, output_dir, dataset_name, False, sort_by) for metric in metrics]
    
    # Use all available CPUs
    n_workers = min(multiprocessing.cpu_count(), len(metrics))
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        filenames = list(executor.map(create_stripplot_for_metric, args_list))
    
    # Filter out None results
    filenames = [f for f in filenames if f is not None]
    
    print(f"✓ Created {len(filenames)} strip plots")
    
    # Also create log-scale versions
    print(f"Creating {len(metrics)} log-scale strip plots in parallel...")
    args_list_log = [(df, metric, output_dir, dataset_name, True, sort_by) for metric in metrics]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        filenames_log = list(executor.map(create_stripplot_for_metric, args_list_log))
    
    filenames_log = [f for f in filenames_log if f is not None]
    print(f"✓ Created {len(filenames_log)} log-scale strip plots")
    
    # Uncapped annotated stripplots for calibrated metrics
    calibrated_metrics = [m for m in metrics if is_calibrated_metric(m)]
    if calibrated_metrics:
        print(f"Creating {len(calibrated_metrics)} uncapped stripplots for calibrated metrics...")
        uncapped_args = [(df, m, output_dir, dataset_name, sort_by) for m in calibrated_metrics]
        uncapped_workers = min(multiprocessing.cpu_count(), len(calibrated_metrics))
        with ProcessPoolExecutor(max_workers=uncapped_workers) as executor:
            uncapped_fnames = list(executor.map(create_uncapped_stripplot_for_metric, uncapped_args))
        uncapped_fnames = [f for f in uncapped_fnames if f is not None]
        print(f"✓ Created {len(uncapped_fnames)} uncapped stripplots")


def create_uncapped_stripplot_for_metric(args):
    """Create an uncapped stripplot for a calibrated metric with out-of-range annotations.

    The y-axis is clipped to [0, UPPER_CLIP]. Per-model text annotations show
    the count of points below 0 (better than oracle) and above UPPER_CLIP
    (much worse than negative baseline), mirroring the approach used for
    R² metrics in the standard stripplot.
    """
    UPPER_CLIP = 1.5

    df, metric, output_dir, dataset_name, sort_by = args

    metric_df = df[df['metric'] == metric].copy()
    metric_df = stripplot_metric_df_for_model(metric_df, metric)
    if metric_df.empty:
        return None

    # Sort order on raw (unclipped) values
    if sort_by == 'mean':
        order_vals = metric_df.groupby('model')['value'].mean()
    else:
        order_vals = metric_df.groupby('model')['value'].median()

    if is_calibrated_metric(metric):
        ordered_models = order_vals.sort_values(ascending=True).index.tolist()
    else:
        ordered_models = order_vals.sort_values(ascending=False).index.tolist()

    metric_df['model'] = pd.Categorical(metric_df['model'],
                                        categories=ordered_models, ordered=True)
    model_means   = metric_df.groupby('model')['value'].mean()
    model_medians = metric_df.groupby('model')['value'].median()

    n_models = len(ordered_models)
    fig_width = max(10, min(20, n_models * 1.1))
    fig_height = 7

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
    ax.set_facecolor('white')

    mean_color = 'black'
    median_color = 'black'

    palette = [MODEL_CATEGORY_COLORS[get_model_category(m)] for m in ordered_models]
    n_perts = int(metric_df["perturbation"].nunique())
    base_alpha = 0.65 if n_models <= 5 else 0.45
    alpha_val = stripplot_alpha_for_dataset(dataset_name, n_perts, base_alpha=base_alpha)

    sns.stripplot(
        data=metric_df, x='model', y='value',
        hue='model',
        palette=palette,
        size=4,
        alpha=alpha_val,
        jitter=True,
        legend=False,
        ax=ax,
    )

    # Mean and median lines
    for i, model_name in enumerate(ordered_models):
        if model_name in model_means.index:
            mean_val = model_means[model_name]
            ax.hlines(mean_val, i - 0.3, i + 0.3, colors=mean_color,
                      linewidth=2.5, zorder=3)
        if model_name in model_medians.index:
            median_val = model_medians[model_name]
            ax.hlines(median_val, i - 0.3, i + 0.3, colors=median_color,
                      linewidth=2.0, linestyle=':', zorder=3)

    # Reference lines
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
    ax.axhline(y=1, color='#999999', linestyle='--', linewidth=1, alpha=0.5, zorder=1)

    # Annotate counts outside display range
    any_clipped = False
    for i, model_name in enumerate(ordered_models):
        subset = metric_df[metric_df['model'] == model_name]
        n_below = int((subset['value'] < 0).sum())
        n_above = int((subset['value'] > UPPER_CLIP).sum())
        if n_below > 0:
            any_clipped = True
            ax.text(i + 0.12, 0.04, f'↓ {n_below}',
                    ha='left', va='bottom', fontsize=10, color='#555555')
        if n_above > 0:
            any_clipped = True
            ax.text(i + 0.12, UPPER_CLIP - 0.08, f'↑ {n_above}',
                    ha='left', va='top', fontsize=10, color='#555555')

    if any_clipped:
        ax.set_ylim(0, UPPER_CLIP)
    else:
        current_ylim = ax.get_ylim()
        ax.set_ylim(0, min(current_ylim[1], UPPER_CLIP))

    metric_display  = METRIC_INFO.get(metric, {}).get('display', metric)
    direction_arrow = '↓' if not METRIC_INFO.get(metric, {}).get('higher_better', True) else '↑'
    ax.set_title(f'{metric_display} in {dataset_name}',
                 fontsize=20, fontweight='bold', pad=25)
    ax.set_xlabel('')
    ax.set_ylabel(f'{metric_display} ({direction_arrow}, uncapped)',
                  fontsize=16, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)

    ax.set_xticks(range(n_models))
    display_labels = [format_model_name(m) for m in ordered_models]
    xticklabels = ax.set_xticklabels(display_labels, rotation=45, ha='right')
    for label in xticklabels:
        apply_model_category_tick_style(label, 12)

    add_stripplot_model_category_legend(
        ax,
        mean_line_handle=Line2D([0], [0], color=mean_color, linewidth=3, label='Mean'),
        median_line_handle=Line2D(
            [0], [0], color=median_color, linewidth=2.5, linestyle=':', label='Median'
        ),
        fontsize=12,
    )

    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    filename = f"stripplot_uncapped_{metric}.png"
    _savefig(output_dir / filename)
    plt.close()
    return filename


# ---------------------------------------------------------------------------
# Calibrated-only heatmap
# ---------------------------------------------------------------------------

# Calibrated metrics shown in the calibrated-only heatmaps (in display order).
# We intentionally exclude calibrated R² delta metrics because they are
# algebraically redundant with calibrated MSE metrics:
#   cr2_deltapert == cmse
#   cweighted_r2_deltapert == cwmse
# cpds_pearson_deltapert is omitted (not meaningful for dataset-mean-style baselines in the same grid).
# Column order: all unweighted calibrated metrics, then all weighted (pairwise groups).
_CAL_HEATMAP_METRICS = [
    'cmse', 'cpearson_deltapert', 'cpds',
    'cwmse', 'cweighted_pearson_deltapert', 'cpds_wmse',
]

# Column labels for the heatmap only (short names without leading "c").
_CAL_HEATMAP_DISPLAY_NAMES = {
    'cmse': 'MSE',
    'cpearson_deltapert': 'Pearson Δ Pert',
    'cwmse': 'wMSE',
    'cweighted_pearson_deltapert': 'wPearson Δ Pert',
    'cpds': 'PDS',
    'cpds_wmse': 'wPDS',
}


def _finalize_calibrated_pivots(
    pivot_mean: pd.DataFrame,
    pivot_std: pd.DataFrame,
    cal_metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort rows (best first) and apply heatmap column/row display names."""
    pivot_mean = pivot_mean.reindex(columns=cal_metrics)
    pivot_std = pivot_std.reindex(columns=cal_metrics)
    row_means = pivot_mean.mean(axis=1)
    sort_order = row_means.sort_values(ascending=True).index
    pivot_mean = pivot_mean.loc[sort_order]
    pivot_std = pivot_std.loc[sort_order]
    col_names = [_CAL_HEATMAP_DISPLAY_NAMES[m] for m in cal_metrics]
    pivot_mean.columns = col_names
    pivot_std.columns = col_names
    pivot_mean.index = [format_model_name(m) for m in pivot_mean.index]
    pivot_std.index = pivot_mean.index
    return pivot_mean, pivot_std


def _finalize_calibrated_pivots_median(
    pivot_median: pd.DataFrame,
    pivot_q1: pd.DataFrame,
    pivot_q3: pd.DataFrame,
    cal_metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort rows (best first) and apply heatmap column/row display names (median + quartiles)."""
    pivot_median = pivot_median.reindex(columns=cal_metrics)
    pivot_q1 = pivot_q1.reindex(columns=cal_metrics)
    pivot_q3 = pivot_q3.reindex(columns=cal_metrics)
    row_means = pivot_median.mean(axis=1)
    sort_order = row_means.sort_values(ascending=True).index
    pivot_median = pivot_median.loc[sort_order]
    pivot_q1 = pivot_q1.loc[sort_order]
    pivot_q3 = pivot_q3.loc[sort_order]
    col_names = [_CAL_HEATMAP_DISPLAY_NAMES[m] for m in cal_metrics]
    pivot_median.columns = col_names
    pivot_q1.columns = col_names
    pivot_q3.columns = col_names
    pivot_median.index = [format_model_name(m) for m in pivot_median.index]
    pivot_q1.index = pivot_median.index
    pivot_q3.index = pivot_median.index
    return pivot_median, pivot_q1, pivot_q3


def _build_calibrated_pivot_summary(summary_df: pd.DataFrame) -> "tuple[pd.DataFrame, pd.DataFrame] | None":
    """Uncapped heatmap: mean/std from aggregated detailed_metrics (same as summary table)."""
    available = set(summary_df['metric'].unique())
    cal_metrics = [m for m in _CAL_HEATMAP_METRICS if m in available]
    if not cal_metrics:
        return None
    sub = summary_df[summary_df['metric'].isin(cal_metrics)]
    pivot_mean = sub.pivot(index='model', columns='metric', values='mean')
    pivot_std = sub.pivot(index='model', columns='metric', values='std')
    return _finalize_calibrated_pivots(pivot_mean, pivot_std, cal_metrics)


def _build_calibrated_pivot_capped_from_detailed(
    detailed_df: pd.DataFrame,
) -> "tuple[pd.DataFrame, pd.DataFrame] | None":
    """Capped heatmap: clip each per-perturbation value to [0, 1], then mean and std (matches stripplots)."""
    available = set(detailed_df['metric'].unique())
    cal_metrics = [m for m in _CAL_HEATMAP_METRICS if m in available]
    if not cal_metrics:
        return None
    sub = detailed_df[detailed_df['metric'].isin(cal_metrics)].copy()
    sub['value'] = sub['value'].clip(lower=0, upper=1)
    stats = sub.groupby(['model', 'metric'])['value'].agg(['mean', 'std']).reset_index()
    pivot_mean = stats.pivot(index='model', columns='metric', values='mean')
    pivot_std = stats.pivot(index='model', columns='metric', values='std')
    return _finalize_calibrated_pivots(pivot_mean, pivot_std, cal_metrics)


def _build_calibrated_pivot_median_from_detailed(
    detailed_df: pd.DataFrame,
    *,
    cap: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Median and Q1/Q3 per model×metric; optional clip to [0, 1] (capped)."""
    available = set(detailed_df['metric'].unique())
    cal_metrics = [m for m in _CAL_HEATMAP_METRICS if m in available]
    if not cal_metrics:
        return None
    sub = detailed_df[detailed_df['metric'].isin(cal_metrics)].copy()
    if cap:
        sub['value'] = sub['value'].clip(lower=0, upper=1)

    def q1(s: pd.Series) -> float:
        return float(s.quantile(0.25))

    def q3(s: pd.Series) -> float:
        return float(s.quantile(0.75))

    stats = sub.groupby(['model', 'metric'])['value'].agg(
        median='median', q1=q1, q3=q3
    ).reset_index()
    pivot_m = stats.pivot(index='model', columns='metric', values='median')
    pivot_q1 = stats.pivot(index='model', columns='metric', values='q1')
    pivot_q3 = stats.pivot(index='model', columns='metric', values='q3')
    return _finalize_calibrated_pivots_median(pivot_m, pivot_q1, pivot_q3, cal_metrics)


def create_calibrated_heatmap(
    summary_df: pd.DataFrame,
    output_path: Path,
    figsize=(14, 8),
    cap: bool = False,
    *,
    detailed_csv_path: Path | None = None,
    include_interpolated_duplicate: bool = False,
    statistic: str = 'mean',
    exclude_models: Optional[List[str]] = None,
) -> None:
    """Heatmap of calibrated metrics with a shared raw-value colour scale.

    ``cap=True``: mean and std are computed from per-perturbation values clipped to [0, 1]
    (same convention as calibrated stripplots). Requires ``detailed_csv_path``.

    Uncapped: mean ± std from summary aggregates; colours use [0, 1] so 1 is always red.

    ``statistic='median'``: per-perturbation medians with Q1–Q3 in annotations; requires
    ``detailed_csv_path`` for both capped and uncapped (same filters as the mean paths).
    """
    if statistic not in ('mean', 'median'):
        raise ValueError("statistic must be 'mean' or 'median'")

    if statistic == 'median':
        if detailed_csv_path is None:
            raise ValueError("detailed_csv_path is required when statistic='median'")
        df = read_detailed_metrics_csv(
            detailed_csv_path,
            include_interpolated_duplicate=include_interpolated_duplicate,
            exclude_models=exclude_models,
        )
        built = _build_calibrated_pivot_median_from_detailed(df, cap=cap)
        if built is None:
            print("  No calibrated metrics found; skipping calibrated heatmap.")
            return
        pivot_df, pivot_q1, pivot_q3 = built
        annot = np.empty(pivot_df.shape, dtype=object)
        for i in range(pivot_df.shape[0]):
            for j in range(pivot_df.shape[1]):
                med = pivot_df.iloc[i, j]
                q1 = pivot_q1.iloc[i, j]
                q3 = pivot_q3.iloc[i, j]
                if pd.isna(med):
                    annot[i, j] = ""
                elif pd.notna(q1) and pd.notna(q3):
                    annot[i, j] = (
                        f"{float(med):.2f}\n[{float(q1):.2f}, {float(q3):.2f}]"
                    )
                else:
                    annot[i, j] = f"{float(med):.2f}"
        annot_kws = {'size': 9, 'weight': 'bold'}
        title = 'Calibrated Performance (median)'
        cbar_label = 'Calibrated Metric Value (median)'
    else:
        if cap:
            if detailed_csv_path is None:
                raise ValueError("detailed_csv_path is required when cap=True for calibrated heatmap")
            df = read_detailed_metrics_csv(
                detailed_csv_path,
                include_interpolated_duplicate=include_interpolated_duplicate,
                exclude_models=exclude_models,
            )
            built = _build_calibrated_pivot_capped_from_detailed(df)
        else:
            built = _build_calibrated_pivot_summary(summary_df)
        if built is None:
            print("  No calibrated metrics found; skipping calibrated heatmap.")
            return
        pivot_df, pivot_std = built

        annot = np.empty(pivot_df.shape, dtype=object)
        for i in range(pivot_df.shape[0]):
            for j in range(pivot_df.shape[1]):
                m = pivot_df.iloc[i, j]
                s = pivot_std.iloc[i, j]
                if pd.isna(m):
                    annot[i, j] = ""
                elif pd.notna(s):
                    annot[i, j] = f"{float(m):.2f} ± {float(s):.2f}"
                else:
                    annot[i, j] = f"{float(m):.2f}"
        annot_kws = {'size': 11, 'weight': 'bold'}
        title = 'Calibrated Performance'
        cbar_label = 'Calibrated Metric Value'

    # Fixed [0, 1] colour scale: green = 0 (best), red = 1 (worst).
    vmin, vmax = 0.0, 1.0
    if cap:
        heatmap_data = pivot_df  # means capped to [0,1] (mean path) or medians of capped
    else:
        heatmap_data = pivot_df.clip(lower=0, upper=1)  # colour only; annot uses raw stats

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        heatmap_data,
        annot=annot,
        fmt='',  # pre-formatted strings (mean ± std); default 'g' would error on str
        mask=pivot_df.isna(),
        cmap='RdYlGn_r',        # red = bad (high), green = good (low)
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': cbar_label},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        annot_kws=annot_kws,
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label(cbar_label, fontsize=13, fontweight='bold')

    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(rotation=45, ha='right', fontsize=12)

    _style_yticklabels(ax)

    plt.tight_layout()
    _savefig(output_path)
    plt.close()
    print(f"✓ Saved calibrated heatmap to: {output_path}")


# ---------------------------------------------------------------------------
# Calibration usefulness: mse/cmse vs effect-size scatter
# ---------------------------------------------------------------------------

def _style_log_axis_nature(ax, *, axis: str = "x") -> None:
    """Style a log-scaled axis with 1-2-5 tick convention and clean labels."""
    axis_obj = getattr(ax, f"{axis}axis")
    axis_obj.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))
    axis_obj.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 1.0, numticks=50))
    axis_obj.set_minor_formatter(NullFormatter())

    def _fmt(x: float, _pos: int) -> str:
        if x <= 0:
            return ""
        return f"{x:.2g}"

    axis_obj.set_major_formatter(FuncFormatter(_fmt))


def create_calibration_demonstration(
    csv_path: Path,
    output_dir: Path,
    dataset_name: str,
    *,
    include_interpolated_duplicate: bool = False,
    exclude_models: Optional[List[str]] = None,
) -> None:
    """3×2 scatter grid: technical_duplicate uncalibrated metrics vs effect size.

    Effect size proxy = per-perturbation ``control_mean`` MSE.
    Each panel shows one uncalibrated metric for technical_duplicate against
    effect size, with a log-x regression line and Spearman ρ annotation.
    Calibrated metrics are intentionally excluded: calibration removes the
    effect-size bias so those panels always show ρ ≈ 0 by design.

    Layout (row × col):
        Row 0: MSE, wMSE
        Row 1: R² Δ Pert, wR² Δ Pert
        Row 2: Pearson Δ Pert, wPearson Δ Pert
    """
    df = read_detailed_metrics_csv(
        csv_path,
        include_interpolated_duplicate=include_interpolated_duplicate,
        exclude_models=exclude_models,
    )

    # Metrics to display: key → (display_name, higher_is_better)
    demo_metrics = [
        ('mse',                      'MSE',           False),
        ('wmse',                     'wMSE',          False),
        ('r2_deltapert',             'R² Δ Pert',     True),
        ('weighted_r2_deltapert',    'wR² Δ Pert',    True),
        ('pearson_deltapert',        'Pearson Δ Pert',True),
        ('weighted_pearson_deltapert','wPearson Δ Pert',True),
    ]

    available = set(df['metric'].unique())
    demo_metrics = [(k, d, h) for k, d, h in demo_metrics if k in available]

    if not demo_metrics:
        print("  Skipping calibration demonstration: no required metrics found.")
        return

    # Effect size proxy: control_mean MSE per perturbation
    effect_size_df = df[(df['model'] == 'control_mean') & (df['metric'] == 'mse')][
        ['perturbation', 'value']
    ].copy().rename(columns={'value': 'effect_size'})

    if effect_size_df.empty:
        print("  Skipping calibration demonstration: no control_mean/mse data.")
        return

    # Prediction model: technical_duplicate (oracle baseline)
    focus_model = 'technical_duplicate'
    if focus_model not in df['model'].unique():
        oracles = [m for m in df['model'].unique() if get_model_category(m) == 'oracle']
        if not oracles:
            print("  Skipping calibration demonstration: technical_duplicate not found.")
            return
        focus_model = oracles[0]

    dot_color = MODEL_CATEGORY_COLORS['oracle']

    n_metrics = len(demo_metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5.5, n_rows * 4.0),
        facecolor='white',
        squeeze=False,
    )

    for idx, (metric, metric_display, higher_is_better) in enumerate(demo_metrics):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        model_data = df[(df['model'] == focus_model) & (df['metric'] == metric)][
            ['perturbation', 'value']
        ].copy()
        merged = pd.merge(effect_size_df, model_data, on='perturbation').dropna()

        if len(merged) < 5:
            ax.set_visible(False)
            continue

        ax.scatter(
            merged['effect_size'], merged['value'],
            color=dot_color, alpha=0.45, s=8, linewidths=0, rasterized=True,
        )

        log_x = np.log10(merged['effect_size'].clip(lower=1e-12))
        slope, intercept = np.polyfit(log_x, merged['value'], 1)
        x_range = np.linspace(log_x.min(), log_x.max(), 200)
        ax.plot(10 ** x_range, slope * x_range + intercept,
                color='#333333', linewidth=1.2, linestyle='--', alpha=0.7)

        rho, pval = spearmanr(merged['effect_size'], merged['value'])
        if pval == 0:
            p_text = r'$p < 10^{-300}$'
        elif pval < 0.001:
            p_text = f'$p$ = {pval:.1e}'
        else:
            p_text = f'$p$ = {pval:.3f}'
        place_bottom_right = ('r2' in metric) or ('pearson' in metric)
        ax.text(
            0.97,
            0.04 if place_bottom_right else 0.96,
            f'Spearman $\\rho$ = {rho:.2f}\n{p_text}',
            transform=ax.transAxes,
            ha='right',
            va='bottom' if place_bottom_right else 'top',
            fontsize=8,
            bbox=dict(
                boxstyle='round,pad=0.25',
                facecolor='white',
                edgecolor='#999999',
                alpha=0.85,
                linewidth=0.5,
            ),
        )

        direction = '↓' if not higher_is_better else '↑'
        ax.set_ylabel(f'{metric_display} ({direction})', fontsize=10)
        ax.set_title(f'{metric_display}', fontsize=11, fontweight='bold', pad=6)
        ax.set_xscale('log')
        _style_log_axis_nature(ax, axis='x')
        ax.tick_params(axis='both', which='major', labelsize=9, length=3, width=0.6)
        ax.tick_params(axis='both', which='minor', length=1.5, width=0.4)
        ax.grid(True, alpha=0.20, linestyle='-', linewidth=0.4)
        ax.set_axisbelow(True)

    # Hide any unused axes
    for idx in range(len(demo_metrics), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    plt.suptitle(
        f'Uncalibrated metrics correlate with effect size  ({format_model_name(focus_model)}, {dataset_name})',
        fontsize=12, fontweight='bold', y=1.01,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    fig.supxlabel(
        'Effect size — MSE (control, perturbation)',
        fontsize=11,
        y=0.01,
    )

    out_png = output_dir / 'calibration_demonstration.png'
    _savefig(out_png)
    plt.close()
    print(f"✓ Saved calibration demonstration to: {out_png}")


def create_all_forest_plots(
    csv_path: Path,
    output_dir: Path,
    dataset_name: str,
    *,
    include_interpolated_duplicate: bool = False,
    exclude_models: Optional[List[str]] = None,
):
    """Create forest plots for all metrics in parallel."""
    df = read_detailed_metrics_csv(
        csv_path,
        include_interpolated_duplicate=include_interpolated_duplicate,
        exclude_models=exclude_models,
    )
    
    # Get unique metrics
    metrics = [m for m in METRIC_INFO.keys() if m in df['metric'].unique()]
    
    print(f"Creating {len(metrics)} forest plots in parallel...")
    
    # Prepare arguments for parallel processing
    args_list = [(df, metric, output_dir, dataset_name) for metric in metrics]
    
    # Use all available CPUs
    n_workers = min(multiprocessing.cpu_count(), len(metrics))
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        filenames = list(executor.map(create_forest_plot_for_metric, args_list))
    
    # Filter out None results
    filenames = [f for f in filenames if f is not None]
    
    print(f"✓ Created {len(filenames)} forest plots")


def create_auxiliary_plots(
    csv_path: Path,
    output_dir: Path,
    dataset_name: str,
    *,
    include_interpolated_duplicate: bool = False,
    exclude_models: Optional[List[str]] = None,
):
    """Create focused auxiliary plots for paper figures."""
    df = read_detailed_metrics_csv(
        csv_path,
        include_interpolated_duplicate=include_interpolated_duplicate,
        exclude_models=exclude_models,
    )
    df_full = pd.read_csv(csv_path)
    if exclude_models:
        df_full = df_full[~df_full["model"].isin(exclude_models)].copy()

    # Create aux_plots subdirectory
    aux_dir = output_dir / 'aux_plots'
    aux_dir.mkdir(exist_ok=True, parents=True)
    
    print("Creating auxiliary plots for paper...")
    
    # Simple 2-model plots
    create_aux_plot(df, 'mse', 'dataset_mean', 'technical_duplicate', 
                   aux_dir, dataset_name)
    create_aux_plot(df, 'pearson_deltactrl', 'dataset_mean', 'technical_duplicate',
                   aux_dir, dataset_name)
    
    # Expanded plots: 3 models when interpolated_duplicate is included in pipeline data
    if include_interpolated_duplicate:
        expanded_models = ['dataset_mean', 'interpolated_duplicate', 'technical_duplicate']
    else:
        expanded_models = ['dataset_mean', 'technical_duplicate']
    create_aux_plot_expanded(df, 'mse', expanded_models, aux_dir, dataset_name)
    create_aux_plot_expanded(df, 'pearson_deltactrl', expanded_models, aux_dir, dataset_name)
    
    # Scatter plots need interpolated_duplicate rows whenever present in the CSV
    create_duplicate_comparison_scatter(df_full, aux_dir, dataset_name)
    create_baseline_comparison_scatter(df_full, aux_dir, dataset_name)
    
    print(f"✓ Created auxiliary plots in {aux_dir}")


def create_aux_plot(df: pd.DataFrame, metric: str, baseline_model: str, 
                   comparison_model: str, output_dir: Path, dataset_name: str):
    """Create a simplified comparison plot for paper figure."""
    
    metric_df = df[df['metric'] == metric].copy()
    
    if metric_df.empty:
        return
    
    # Filter to only the two models of interest
    models_to_plot = [baseline_model, comparison_model]
    plot_df = metric_df[metric_df['model'].isin(models_to_plot)]
    
    if plot_df.empty:
        return
    
    # Order by median; keep means for reference lines.
    model_order_values = plot_df.groupby('model')['value'].median()
    model_means = plot_df.groupby('model')['value'].mean()
    
    # Order: baseline first, then comparison
    ordered_models = [baseline_model, comparison_model]
    plot_df['model'] = pd.Categorical(plot_df['model'], 
                                      categories=ordered_models, 
                                      ordered=True)
    
    # Create figure (narrower)
    fig, ax = plt.subplots(figsize=(5, 7), facecolor='white')
    ax.set_facecolor('white')
    
    # Color by plotting group (oracle / baseline / model)
    colors = [MODEL_CATEGORY_COLORS[get_model_category(m)] for m in ordered_models]

    n_perts = int(plot_df["perturbation"].nunique())
    strip_alpha = stripplot_alpha_for_dataset(dataset_name, n_perts, base_alpha=0.5)

    # Strip plot
    sns.stripplot(data=plot_df, x='model', y='value',
                 hue='model',
                 palette=colors,
                 size=4,
                 alpha=strip_alpha,
                 jitter=True,
                 legend=False,
                 ax=ax)
    
    # Add mean lines (black, narrower)
    mean_color = 'black'
    for i, model in enumerate(ordered_models):
        if model in model_means.index:
            mean_val = model_means[model]
            ax.hlines(mean_val, i - 0.25, i + 0.25,
                     colors=mean_color,
                     linewidth=3,
                     zorder=3)
            
            # Add mean value text
            ax.text(i, mean_val, f'{mean_val:.4f}',
                   ha='center', va='bottom',
                   fontsize=14,
                   color=mean_color,
                   fontweight='bold')
    
    # Styling
    metric_display = METRIC_INFO.get(metric, {}).get('display', metric)
    direction_arrow = '↓' if not METRIC_INFO.get(metric, {}).get('higher_better', True) else '↑'
    ax.set_title(f'{metric_display} in {dataset_name}',
                fontsize=20, fontweight='bold', pad=25)
    ax.set_xlabel('', fontsize=16)
    ax.set_ylabel(f'{metric_display} ({direction_arrow})', fontsize=16, fontweight='bold')
    
    # Style x-axis labels with nice formatting
    display_labels = [format_model_name(baseline_model), format_model_name(comparison_model)]
    ax.set_xticklabels(display_labels, fontsize=14, fontweight='normal')
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Cap y-axis at -1 for R² metrics and annotate per-model counts
    r2_metrics = ['r2_deltactrl', 'r2_deltactrl_degs', 'r2_deltapert', 'r2_deltapert_degs',
                  'weighted_r2_deltactrl', 'weighted_r2_deltapert',
                  'cweighted_r2_deltactrl', 'cweighted_r2_deltapert']
    if metric in r2_metrics:
        any_clipped = False
        for i, model_name in enumerate(ordered_models):
            model_df_subset = plot_df[plot_df['model'] == model_name]
            n_below = (model_df_subset['value'] < -1).sum()
            if n_below > 0:
                any_clipped = True
                # Add clean text annotation on plot at y=-0.97, offset to the right
                ax.text(i + 0.3, -0.97, f'↓ n={n_below}',
                       ha='left', va='center',
                       fontsize=9,
                       color='#555555')  # Dark grey
        
        if any_clipped:
            current_ylim = ax.get_ylim()
            ax.set_ylim(-1, current_ylim[1])
    
    # Set log scale for MSE and WMSE
    if metric.lower() in ['mse', 'wmse']:
        ax.set_yscale('log')
    
    # Add reference line at 0 for certain metrics
    if any(keyword in metric.lower() for keyword in ['pearson', 'r2', 'delta']):
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
    
    plt.tight_layout()
    
    # Save
    filename = f"aux_{metric}.png"
    _savefig(output_dir / filename)
    plt.close()
    
    print(f"  ✓ Saved {filename}")


def create_aux_plot_expanded(df: pd.DataFrame, metric: str, models_to_plot: list,
                             output_dir: Path, dataset_name: str):
    """Create expanded auxiliary plot with multiple baselines."""
    
    metric_df = df[df['metric'] == metric].copy()
    
    if metric_df.empty:
        return
    
    # Filter to specified models
    plot_df = metric_df[metric_df['model'].isin(models_to_plot)]
    
    if plot_df.empty:
        return
    
    # Calculate means for ordering
    model_order_values = plot_df.groupby('model')['value'].median()
    model_means = plot_df.groupby('model')['value'].mean()
    
    # Order by median (best to worst)
    if metric.lower() in ['mse', 'wmse', 'cmse', 'cwmse', 'fmse', 'fcmse']:
        ordered_models = model_order_values.sort_values(ascending=True).index.tolist()
    else:
        ordered_models = model_order_values.sort_values(ascending=False).index.tolist()
    
    plot_df['model'] = pd.Categorical(plot_df['model'], 
                                      categories=ordered_models, 
                                      ordered=True)
    
    # Create figure (narrower for 3 models)
    fig, ax = plt.subplots(figsize=(7.3, 5), facecolor='white')
    ax.set_facecolor('white')
    
    # Color by plotting group (oracle / baseline / model)
    colors = [MODEL_CATEGORY_COLORS[get_model_category(m)] for m in ordered_models]

    n_perts = int(plot_df["perturbation"].nunique())
    strip_alpha = stripplot_alpha_for_dataset(dataset_name, n_perts, base_alpha=0.5)

    # Strip plot
    sns.stripplot(data=plot_df, x='model', y='value',
                 hue='model',
                 palette=colors,
                 size=4,
                 alpha=strip_alpha,
                 jitter=True,
                 legend=False,
                 ax=ax)
    
    # Add mean lines (black)
    mean_color = 'black'
    for i, model in enumerate(ordered_models):
        if model in model_means.index:
            mean_val = model_means[model]
            ax.hlines(mean_val, i - 0.25, i + 0.25,
                     colors=mean_color,
                     linewidth=3,
                     zorder=3)
            
            # Add mean value text
            ax.text(i, mean_val, f'{mean_val:.4f}',
                   ha='center', va='bottom',
                   fontsize=14,
                   color=mean_color,
                   fontweight='bold')
    
    # Styling
    metric_display = METRIC_INFO.get(metric, {}).get('display', metric)
    direction_arrow = '↓' if not METRIC_INFO.get(metric, {}).get('higher_better', True) else '↑'
    ax.set_title(f'{metric_display} in {dataset_name}',
                fontsize=20, fontweight='bold', pad=25)
    ax.set_xlabel('', fontsize=16)
    ax.set_ylabel(f'{metric_display} ({direction_arrow})', fontsize=16, fontweight='bold')
    
    # Style x-axis labels with nice formatting (tilted to avoid overlap)
    display_labels = [format_model_name(m) for m in ordered_models]
    ax.set_xticklabels(display_labels, fontsize=14, fontweight='normal')
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add reference line at 0 for certain metrics
    if any(keyword in metric.lower() for keyword in ['pearson', 'r2', 'delta']):
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
    
    plt.tight_layout()
    
    # Save
    filename = f"aux_expanded_{metric}.png"
    _savefig(output_dir / filename)
    plt.close()
    
    print(f"  ✓ Saved {filename}")


def create_duplicate_comparison_scatter(df: pd.DataFrame, output_dir: Path, dataset_name: str):
    """Create scatter plot comparing technical vs interpolated duplicate MSE per perturbation."""
    
    # Filter to MSE metric only
    mse_df = df[df['metric'] == 'mse'].copy()
    
    if mse_df.empty:
        return
    
    # Get data for each baseline
    tech_dup = mse_df[mse_df['model'] == 'technical_duplicate'][['perturbation', 'value']]
    interp_dup = mse_df[mse_df['model'] == 'interpolated_duplicate'][['perturbation', 'value']]
    dataset_mean = mse_df[mse_df['model'] == 'dataset_mean'][['perturbation', 'value']]
    
    if tech_dup.empty or interp_dup.empty or dataset_mean.empty:
        print("  Warning: Missing baseline data for duplicate comparison scatter")
        return
    
    # Merge on perturbation
    merged = pd.merge(tech_dup, interp_dup, on='perturbation', suffixes=('_tech', '_interp'))
    merged = pd.merge(merged, dataset_mean, on='perturbation')
    merged.columns = ['perturbation', 'tech_dup_mse', 'interp_dup_mse', 'mean_mse']
    
    # Remove any NaN values
    merged = merged.dropna()
    
    if len(merged) == 0:
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    
    # Log-transform mean_mse for coloring
    merged['log_mean_mse'] = np.log10(merged['mean_mse'])
    
    # Create scatter plot
    scatter = ax.scatter(
        merged['tech_dup_mse'],
        merged['interp_dup_mse'],
        c=merged['log_mean_mse'],
        cmap='viridis',
        s=50,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log₁₀(Dataset Mean MSE)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add identity line (x=y)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.7, linewidth=2, zorder=1, label='Identity (x=y)')
    
    # Set log scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Labels and title
    ax.set_xlabel('Technical Duplicate MSE', fontsize=14, fontweight='bold')
    ax.set_ylabel('Interpolated Duplicate MSE', fontsize=14, fontweight='bold')
    ax.set_title('Interpolated Duplicate Vs Technical Duplicate Error',
                fontsize=16, fontweight='bold', pad=20)
    
    # Add legend for identity line
    ax.legend(loc='upper left', fontsize=11)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    filename = "aux_duplicate_comparison_scatter.png"
    _savefig(output_dir / filename)
    plt.close()
    
    print(f"  ✓ Saved {filename}")


def create_baseline_comparison_scatter(df: pd.DataFrame, output_dir: Path, dataset_name: str):
    """Create scatter plot comparing interpolated duplicate vs dataset mean, colored by tech dup."""
    
    # Filter to MSE metric only
    mse_df = df[df['metric'] == 'mse'].copy()
    
    if mse_df.empty:
        return
    
    # Get data for each baseline
    tech_dup = mse_df[mse_df['model'] == 'technical_duplicate'][['perturbation', 'value']]
    interp_dup = mse_df[mse_df['model'] == 'interpolated_duplicate'][['perturbation', 'value']]
    dataset_mean = mse_df[mse_df['model'] == 'dataset_mean'][['perturbation', 'value']]
    
    if tech_dup.empty or interp_dup.empty or dataset_mean.empty:
        print("  Warning: Missing baseline data for baseline comparison scatter")
        return
    
    # Merge on perturbation
    merged = pd.merge(interp_dup, dataset_mean, on='perturbation', suffixes=('_interp', '_mean'))
    merged = pd.merge(merged, tech_dup, on='perturbation')
    merged.columns = ['perturbation', 'interp_dup_mse', 'mean_mse', 'tech_dup_mse']
    
    # Remove any NaN values
    merged = merged.dropna()
    
    if len(merged) == 0:
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    
    # Log-transform tech_dup_mse for coloring
    merged['log_tech_dup_mse'] = np.log10(merged['tech_dup_mse'])
    
    # Create scatter plot
    scatter = ax.scatter(
        merged['interp_dup_mse'],
        merged['mean_mse'],
        c=merged['log_tech_dup_mse'],
        cmap='viridis',
        s=50,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log₁₀(Technical Duplicate MSE)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add identity line (x=y)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.7, linewidth=2, zorder=1, label='Identity (x=y)')
    
    # Set log scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Labels and title
    ax.set_xlabel('Interpolated Duplicate MSE', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dataset Mean MSE', fontsize=14, fontweight='bold')
    ax.set_title('Interpolated Duplicate Vs Dataset Mean Error',
                fontsize=16, fontweight='bold', pad=20)
    
    # Add legend for identity line
    ax.legend(loc='upper left', fontsize=11)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    filename = "aux_baseline_comparison_scatter.png"
    _savefig(output_dir / filename)
    plt.close()
    
    print(f"  ✓ Saved {filename}")


def create_latex_tables(
    csv_path: Path,
    mean_all_path: Path,
    mean_nodeg_path: Path,
    ttest_all_path: Path,
    ttest_nodeg_path: Path,
    dataset_name: str,
    *,
    include_interpolated_duplicate: bool = False,
    exclude_models: Optional[List[str]] = None,
):
    """Create LaTeX tables: two versions each for means and t-tests (with/without DEG metrics)."""
    df = read_detailed_metrics_csv(
        csv_path,
        include_interpolated_duplicate=include_interpolated_duplicate,
        exclude_models=exclude_models,
    )
    
    # Get models and metrics
    all_models = [m for m in df['model'].unique() if m != 'ground_truth']
    all_metrics = [m for m in METRIC_INFO.keys() if m in df['metric'].unique()]
    
    # Classify models as baselines or trained models
    baseline_keywords = ['control_mean', 'technical_duplicate', 
                        'interpolated_duplicate', 'additive', 'dataset_mean', 
                        'linear', 'baselines', 'random_perturbation']
    
    baselines = [m for m in all_models if m in baseline_keywords]
    trained_models = [m for m in all_models if m not in baseline_keywords]
    
    # Sort within each group
    baselines.sort()
    trained_models.sort()
    
    # Calculate mean and SEM
    summary = df.groupby(['model', 'metric'])['value'].agg(['mean', 'sem']).reset_index()
    
    # Calculate t-statistics for each model vs appropriate baseline
    t_stats = {}
    p_values = {}
    
    for metric in all_metrics:
        # Determine baseline
        if 'deltapert' in metric:
            baseline_model = 'control_mean'
        else:
            baseline_model = 'dataset_mean'
        
        baseline_data = df[(df['model'] == baseline_model) & (df['metric'] == metric)]
        
        if len(baseline_data) == 0:
            continue
        
        for model in all_models:
            if model == baseline_model:
                continue
            
            model_data = df[(df['model'] == model) & (df['metric'] == metric)]
            
            if len(model_data) == 0:
                continue
            
            # Merge on perturbation
            merged = pd.merge(
                baseline_data[['perturbation', 'value']],
                model_data[['perturbation', 'value']],
                on='perturbation',
                suffixes=('_baseline', '_model')
            )
            
            merged = merged.dropna()
            
            if len(merged) >= 3:
                try:
                    if METRIC_INFO[metric]['higher_better']:
                        t_stat, p_val = ttest_rel(
                            merged['value_model'],
                            merged['value_baseline'],
                            alternative='greater'
                        )
                    else:
                        t_stat, p_val = ttest_rel(
                            merged['value_model'],
                            merged['value_baseline'],
                            alternative='less'
                        )
                    
                    t_stats[(model, metric)] = t_stat
                    p_values[(model, metric)] = p_val
                except:
                    pass
    
    # Positive controls (don't bold if they win)
    positive_controls = ['technical_duplicate', 'interpolated_duplicate']
    
    # Create all four table versions
    # Version 1: Means with all metrics
    metrics_all = all_metrics
    create_single_latex_table(mean_all_path, summary, trained_models, baselines, metrics_all,
                             positive_controls, dataset_name, "mean", "all")
    
    # Version 2: Means without DEG metrics
    metrics_nodeg = [m for m in all_metrics if '_degs' not in m]
    create_single_latex_table(mean_nodeg_path, summary, trained_models, baselines, metrics_nodeg,
                             positive_controls, dataset_name, "mean", "nodeg")
    
    # Version 3: T-tests with all metrics  
    create_single_latex_table(ttest_all_path, summary, trained_models, baselines, metrics_all,
                             positive_controls, dataset_name, "ttest", "all", t_stats, p_values)
    
    # Version 4: T-tests without DEG metrics
    create_single_latex_table(ttest_nodeg_path, summary, trained_models, baselines, metrics_nodeg,
                             positive_controls, dataset_name, "ttest", "nodeg", t_stats, p_values)
    
    print(f"✓ Saved LaTeX tables to: {mean_all_path.parent}")


def create_single_latex_table(output_path, summary, trained_models, baselines, metrics,
                              positive_controls, dataset_name, table_type, version,
                              t_stats=None, p_values=None):
    """Helper to create a single LaTeX table."""
    
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    
    table_name = "Mean performance ± SEM" if table_type == "mean" else "Statistical comparison vs baseline"
    deg_suffix = " (all metrics)" if version == "all" else " (excluding DEG metrics)"
    lines.append(f"\\caption{{{table_name} for {dataset_name}{deg_suffix}.}}")
    lines.append(f"\\label{{tab:multimodel_{table_type}_{version}_{dataset_name}}}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    
    # Build header with Type column
    metric_headers = [METRIC_INFO[m]['display'] for m in metrics]
    header_line = "\\textbf{Type} & \\textbf{Model} & " + " & ".join([f"\\textbf{{{h}}}" for h in metric_headers]) + " \\\\"
    lines.append("\\begin{tabular}{ll" + "c" * len(metrics) + "}")
    lines.append("\\toprule")
    lines.append(header_line)
    lines.append("\\midrule")
    
    if table_type == "mean":
        # Mean ± SEM table
        for i, model in enumerate(trained_models):
            model_display = format_model_name(model)
            type_cell = "\\multirow{" + str(len(trained_models)) + "}{*}{Model}" if i == 0 else ""
            row_vals = []
            
            for metric in metrics:
                model_summary = summary[(summary['model'] == model) & (summary['metric'] == metric)]
                
                if len(model_summary) > 0:
                    mean_val = model_summary['mean'].iloc[0]
                    sem_val = model_summary['sem'].iloc[0]
                    
                    # Check if best among non-positive-control models
                    metric_summary = summary[summary['metric'] == metric]
                    # Filter to exclude positive controls
                    non_control_summary = metric_summary[~metric_summary['model'].isin(positive_controls)]
                    
                    if len(non_control_summary) > 0:
                        if METRIC_INFO[metric]['higher_better']:
                            best_val = non_control_summary['mean'].max()
                        else:
                            best_val = non_control_summary['mean'].min()
                        
                        is_best = abs(mean_val - best_val) < 1e-6
                        should_bold = is_best and model not in positive_controls
                    else:
                        should_bold = False
                    
                    if should_bold:
                        row_vals.append(f"\\textbf{{{mean_val:.3f} ± {sem_val:.3f}}}")
                    else:
                        row_vals.append(f"{mean_val:.3f} ± {sem_val:.3f}")
                else:
                    row_vals.append("-")
            
            lines.append(f"{type_cell} & {model_display} & " + " & ".join(row_vals) + " \\\\")
        
        lines.append("\\midrule")
        
        # Add baselines section
        for i, model in enumerate(baselines):
            model_display = format_model_name(model)
            type_cell = "\\multirow{" + str(len(baselines)) + "}{*}{Baseline}" if i == 0 else ""
            row_vals = []
            
            for metric in metrics:
                model_summary = summary[(summary['model'] == model) & (summary['metric'] == metric)]
                
                if len(model_summary) > 0:
                    mean_val = model_summary['mean'].iloc[0]
                    sem_val = model_summary['sem'].iloc[0]
                    
                    # Check if best among non-positive-control models (same logic as trained models)
                    metric_summary_all = summary[summary['metric'] == metric]
                    non_control_summary = metric_summary_all[~metric_summary_all['model'].isin(positive_controls)]
                    
                    if len(non_control_summary) > 0:
                        if METRIC_INFO[metric]['higher_better']:
                            best_val = non_control_summary['mean'].max()
                        else:
                            best_val = non_control_summary['mean'].min()
                        
                        is_best = abs(mean_val - best_val) < 1e-6
                        should_bold = is_best and model not in positive_controls
                    else:
                        should_bold = False
                    
                    if should_bold:
                        row_vals.append(f"\\textbf{{{mean_val:.3f} ± {sem_val:.3f}}}")
                    else:
                        row_vals.append(f"{mean_val:.3f} ± {sem_val:.3f}")
                else:
                    row_vals.append("-")
            
            lines.append(f"{type_cell} & {model_display} & " + " & ".join(row_vals) + " \\\\")
    
    else:  # t-test table
        if t_stats is None or p_values is None:
            return
        
        # Add trained models section
        for i, model in enumerate(trained_models):
            model_display = format_model_name(model)
            type_cell = "\\multirow{" + str(len(trained_models)) + "}{*}{Model}" if i == 0 else ""
            row_vals = []
            
            for metric in metrics:
                if (model, metric) in t_stats:
                    t_val = t_stats[(model, metric)]
                    p_val = p_values[(model, metric)]
                    
                    # Stars
                    if p_val < 0.0001:
                        stars = '****'
                    elif p_val < 0.001:
                        stars = '***'
                    elif p_val < 0.01:
                        stars = '**'
                    elif p_val < 0.05:
                        stars = '*'
                    else:
                        stars = ''
                    
                    # Check if best among non-positive-control models
                    metric_pvals = {k: v for k, v in p_values.items() 
                                   if k[1] == metric and k[0] not in positive_controls}
                    if len(metric_pvals) > 0:
                        best_pval = min(metric_pvals.values())
                        is_best_pval = abs(p_val - best_pval) < 1e-10
                    else:
                        is_best_pval = False
                    
                    should_bold = is_best_pval and stars and model not in positive_controls
                    
                    if should_bold:
                        row_vals.append(f"\\textbf{{{t_val:.1f}({stars})}}")
                    else:
                        row_vals.append(f"{t_val:.1f}({stars})" if stars else f"{t_val:.1f}")
                else:
                    row_vals.append("-")
            
            lines.append(f"{type_cell} & {model_display} & " + " & ".join(row_vals) + " \\\\")
        
        lines.append("\\midrule")
        
        # Add baselines section
        for i, model in enumerate(baselines):
            model_display = format_model_name(model)
            type_cell = "\\multirow{" + str(len(baselines)) + "}{*}{Baseline}" if i == 0 else ""
            row_vals = []
            
            for metric in metrics:
                if (model, metric) in t_stats:
                    t_val = t_stats[(model, metric)]
                    p_val = p_values[(model, metric)]
                    
                    # Stars
                    if p_val < 0.0001:
                        stars = '****'
                    elif p_val < 0.001:
                        stars = '***'
                    elif p_val < 0.01:
                        stars = '**'
                    elif p_val < 0.05:
                        stars = '*'
                    else:
                        stars = ''
                    
                    row_vals.append(f"{t_val:.1f}({stars})" if stars else f"{t_val:.1f}")
                else:
                    row_vals.append("-")
            
            lines.append(f"{type_cell} & {model_display} & " + " & ".join(row_vals) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    
    # Save
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description='Generate beautiful summary plots for multi-model benchmarking results'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to detailed_metrics.csv file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: same as input CSV)'
    )
    parser.add_argument(
        '--exclude-baselines',
        action='store_true',
        help='Exclude baseline models from plots (default: include them as controls)'
    )
    parser.add_argument(
        '--stripplot-sort-by',
        type=str,
        choices=['median', 'mean'],
        default='mean',
        help='Statistic used to order models in stripplots (default: mean)'
    )
    parser.add_argument(
        '--include-forest-plots',
        action='store_true',
        help='Generate forest plots (default: disabled)'
    )
    parser.add_argument(
        '--include-interpolated-duplicate',
        action='store_true',
        help=(
            'Include interpolated_duplicate in heatmaps, stripplots, forest plots, '
            'LaTeX tables, and expanded aux plots (default: exclude that baseline)'
        ),
    )
    parser.add_argument(
        '--exclude-models',
        nargs='+',
        default=[],
        metavar='MODEL',
        help=(
            'Exclude these model names (exact match on the model column), e.g. '
            'scgpt to omit scGPT from all summary plots and tables.'
        ),
    )

    args = parser.parse_args()
    
    # Load CSV
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    print(f"\nProcessing: {csv_path}")
    print("=" * 80)
    
    # Determine output directory - create additional_results subfolder
    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        base_dir = csv_path.parent
    
    output_dir = base_dir / 'additional_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load and process data
    print("Loading data...")
    exclude_baselines = args.exclude_baselines
    include_interp_dup = args.include_interpolated_duplicate
    exclude_models = args.exclude_models if args.exclude_models else None
    if not include_interp_dup:
        print(
            "Excluding model interpolated_duplicate from summary plots and tables "
            "(use --include-interpolated-duplicate to include)."
        )
    if exclude_models:
        print(f"Excluding models from all plots/tables: {', '.join(exclude_models)}")
    summary_df = load_and_process_data(
        csv_path,
        exclude_baselines=exclude_baselines,
        include_interpolated_duplicate=include_interp_dup,
        exclude_models=exclude_models,
    )
    
    if exclude_baselines:
        print("(Excluding baseline models, showing only trained models)")
    else:
        print("(Including baseline models as controls)")
    
    n_models = summary_df['model'].nunique()
    n_metrics = summary_df['metric'].nunique()
    print(f"Found {n_models} models × {n_metrics} metrics")
    
    # Extract dataset name from CSV path
    # Expected format: .../benchmark_{models}_{dataset}/...
    try:
        parent_dir_name = csv_path.parent.parent.name
        dataset_name = parent_dir_name.split('_')[-1]  # Get last part
    except:
        dataset_name = "dataset"
    
    print(f"Dataset: {dataset_name}")
    
    # Apply Nature-journal plotting conventions globally (font, DPI, PDF, spines)
    _apply_nature_rc()
    
    # Generate plots
    print("\nGenerating plots...")
    print("-" * 80)
    
    # 1. Mean heatmap
    mean_heatmap_path = output_dir / 'multimodel_summary_heatmap.png'
    create_mean_heatmap(summary_df, mean_heatmap_path)
    
    # 2. Relative performance heatmap (vs. dataset_mean baseline)
    relative_heatmap_path = output_dir / 'multimodel_relative_performance.png'
    create_zscore_heatmap(summary_df, relative_heatmap_path)
    
    # 3. Summary table
    table_path = output_dir / 'multimodel_summary_table.csv'
    create_summary_table(summary_df, table_path)
    
    # 4. Statistical comparison heatmap
    stats_heatmap_path = output_dir / 'multimodel_statistical_comparison.png'
    create_statistical_comparison_heatmap(
        csv_path,
        stats_heatmap_path,
        dataset_name,
        include_interpolated_duplicate=include_interp_dup,
        exclude_models=exclude_models,
    )
    
    # 5. Calibrated-only heatmap — shared raw-value colour scale (uncapped)
    cal_heatmap_path = output_dir / 'multimodel_calibrated_heatmap.png'
    create_calibrated_heatmap(summary_df, cal_heatmap_path, cap=False)

    # 5b. Same heatmap with values capped to [0,1] (mean/std from capped per-perturbation values)
    cal_heatmap_capped_path = output_dir / 'multimodel_calibrated_heatmap_capped.png'
    create_calibrated_heatmap(
        summary_df,
        cal_heatmap_capped_path,
        cap=True,
        detailed_csv_path=csv_path,
        include_interpolated_duplicate=include_interp_dup,
        exclude_models=exclude_models,
    )

    # 5c/5d. Same heatmaps with median (Q1–Q3) instead of mean ± std
    cal_heatmap_median_path = output_dir / 'multimodel_calibrated_heatmap_median.png'
    create_calibrated_heatmap(
        summary_df,
        cal_heatmap_median_path,
        cap=False,
        detailed_csv_path=csv_path,
        include_interpolated_duplicate=include_interp_dup,
        statistic='median',
        exclude_models=exclude_models,
    )
    cal_heatmap_capped_median_path = output_dir / 'multimodel_calibrated_heatmap_capped_median.png'
    create_calibrated_heatmap(
        summary_df,
        cal_heatmap_capped_median_path,
        cap=True,
        detailed_csv_path=csv_path,
        include_interpolated_duplicate=include_interp_dup,
        statistic='median',
        exclude_models=exclude_models,
    )
    
    # 6. Strip plots for all metrics (parallelized)
    create_all_stripplots(
        csv_path,
        output_dir,
        dataset_name,
        sort_by=args.stripplot_sort_by,
        include_interpolated_duplicate=include_interp_dup,
        exclude_models=exclude_models,
    )
    
    # 7. Forest plots for all metrics (parallelized, opt-in)
    if args.include_forest_plots:
        create_all_forest_plots(
            csv_path,
            output_dir,
            dataset_name,
            include_interpolated_duplicate=include_interp_dup,
            exclude_models=exclude_models,
        )
    else:
        print("Skipping forest plots (use --include-forest-plots to enable).")
    
    # 8. Auxiliary plots for paper
    create_auxiliary_plots(
        csv_path,
        output_dir,
        dataset_name,
        include_interpolated_duplicate=include_interp_dup,
        exclude_models=exclude_models,
    )
    
    # 9. Calibration demonstration (mse/cmse vs effect size)
    create_calibration_demonstration(
        csv_path,
        output_dir,
        dataset_name,
        include_interpolated_duplicate=include_interp_dup,
        exclude_models=exclude_models,
    )
    
    # 10. LaTeX tables (four versions: means/ttests × with_deg/without_deg)
    latex_mean_all_path = output_dir / 'multimodel_latex_table_means_all.tex'
    latex_mean_nodeg_path = output_dir / 'multimodel_latex_table_means_nodeg.tex'
    latex_ttest_all_path = output_dir / 'multimodel_latex_table_ttests_all.tex'
    latex_ttest_nodeg_path = output_dir / 'multimodel_latex_table_ttests_nodeg.tex'
    create_latex_tables(
        csv_path,
        latex_mean_all_path,
        latex_mean_nodeg_path,
        latex_ttest_all_path,
        latex_ttest_nodeg_path,
        dataset_name,
        include_interpolated_duplicate=include_interp_dup,
        exclude_models=exclude_models,
    )
    
    print("\n" + "=" * 80)
    print("✅ All plots generated successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()


