"""
Visualization utilities for perturbation discrimination score (PDS) results.

Reads precomputed analysis outputs and generates paper-ready plots.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colormaps as mpl_colormaps
from matplotlib import colors as mpl_colors
from plotnine import (
    aes,
    geom_errorbar,
    geom_col,
    facet_wrap,
    geom_errorbarh,
    geom_point,
    geom_text,
    geom_tile,
    geom_violin,
    ggplot,
    labs,
    position_dodge,
    scale_fill_cmap,
    scale_fill_manual,
    scale_color_manual,
    scale_y_continuous,
    scale_y_log10,
    element_blank,
    element_line,
    element_text,
    guide_legend,
    guides,
    theme,
    theme_minimal,
)
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analyses.perturbation_discrimination import metric_utils as base

METRIC_LABELS = {
    "MAE_Mean": "MAE (Mean)",
    "MAE_Median": "MAE (Median)",
    "Energy_Distance": "Energy Distance",
    "Wasserstein_1D": "Wasserstein Distance",
}
FILTER_LABELS = {
    "highly_variable": "HVG",
    "lowest_cv": "CV",
    "lowest_cv_per_perturbation": "Low CV (per Pert.)",
    "deg_control": "DEG",
    "deg_synthetic": "DEG (synthetic)",
    "deg_per_perturbation_control": "DEG (per Pert.)",
    "deg_per_perturbation_synthetic": "DEG (synthetic, per Pert.)",
    "top_discriminating": "PDS",
    "top_discriminating_per_perturbation": "PDS (per Pert.)",
}

NATURE_BASE_SIZE = 7
PLOT_DPI = 600
NATURE_AXIS_LABEL_SIZE = 7
NATURE_TICK_SIZE = 6
NATURE_LEGEND_SIZE = 6
# Nature two-column width is 183 mm.
# For two panels side-by-side, target ~89 mm each (single-column width),
# leaving ~5 mm gutter between panels.
FIG_SIZE_TWO_UP = (3.5, 2.35)  # ~89 mm x 60 mm
FIG_SIZE_SINGLE_COLUMN = (3.5, 2.6)  # ~89 mm x 66 mm
BAR_DODGE_WIDTH = 0.8
BAR_WIDTH = 0.72


def _clean_metric_name(metric: str) -> str:
    """Map metric names to display labels."""
    return METRIC_LABELS.get(metric, metric)


def _clean_filter_name(filtering: str) -> str:
    """Map filter names to display labels."""
    return FILTER_LABELS.get(filtering, filtering)


def _dataset_palette(datasets: Sequence[str]) -> Dict[str, str]:
    """Build a stable color palette for datasets."""
    colors = sns.color_palette("tab10", n_colors=len(datasets))
    return {dataset: mpl_colors.to_hex(color) for dataset, color in zip(datasets, colors)}


def _dataset_palette_viridis(datasets: Sequence[str]) -> Dict[str, str]:
    """Build a viridis color palette for datasets."""
    cmap = mpl_colormaps["viridis"]
    if len(datasets) <= 1:
        colors = [cmap(0.5)]
    else:
        colors = [cmap(i / (len(datasets) - 1)) for i in range(len(datasets))]
    return {dataset: mpl_colors.to_hex(color) for dataset, color in zip(datasets, colors)}


def set_publication_style(style: str) -> None:
    """Apply publication-friendly plotting style.

    Args:
        style: Style preset name (e.g., "nature" or "existing").
    """
    if style == "existing":
        sns.set_theme(style="whitegrid", context="paper")
        return
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": NATURE_BASE_SIZE,
            "axes.titlesize": NATURE_BASE_SIZE,
            "legend.fontsize": NATURE_BASE_SIZE - 1,
            "xtick.labelsize": NATURE_BASE_SIZE - 1,
            "ytick.labelsize": NATURE_BASE_SIZE - 1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.dpi": PLOT_DPI,
        }
    )


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int = 300,
) -> None:
    """Save a figure to multiple formats.

    Args:
        fig: Matplotlib figure.
        output_dir: Output directory.
        stem: Filename stem (without extension).
        formats: File extensions to write.
        dpi: Resolution for raster outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=dpi, bbox_inches="tight")


def _dataset_legend() -> guides:
    """Use a compact, publication-friendly legend style for datasets."""
    return guides(fill=guide_legend(title=""))


def load_dataset_path(dataset_name: str, root: Path) -> Path:
    """Resolve dataset path from config YAML.

    Args:
        dataset_name: Dataset name (matching YAML file stem).
        root: Project root path.

    Returns:
        Absolute path to dataset h5ad file.
    """
    config_path = root / "cellsimbench" / "configs" / "dataset" / f"{dataset_name}.yaml"
    with config_path.open("r") as handle:
        config = yaml.safe_load(handle)
    return (root / config["data_path"]).resolve()


def read_summary(
    results_root: Path,
    dataset: str,
    filtering: str,
    allow_missing: bool = False,
) -> Optional[pd.DataFrame]:
    """Read summary CSV for a dataset/filter combination.

    Args:
        results_root: Results root directory.
        dataset: Dataset name.
        filtering: Filtering method.
        allow_missing: If True, return None when missing instead of raising.

    Returns:
        Summary DataFrame, or None if missing and allow_missing=True.
    """
    summary_path = results_root / dataset / "aggregates" / filtering / "summary.csv"
    if not summary_path.exists():
        if allow_missing:
            print(f"WARNING: Missing summary file: {summary_path}")
            return None
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    return pd.read_csv(summary_path)


def list_available_datasets(results_root: Path) -> List[str]:
    """List datasets that have perturbation-discrimination aggregates under ``results_root``."""
    datasets: List[str] = []
    if not results_root.exists():
        return datasets
    skip = {
        "metric_comparison",
        "metric_comparison_latest",
        "metric_comparison_old",
        "sensitivity",
    }
    for child in sorted(results_root.iterdir()):
        if not child.is_dir() or child.name in skip:
            continue
        if (child / "aggregates").is_dir():
            datasets.append(child.name)
    return datasets


def plot_metric_heatmap(
    results_root: Path,
    datasets: Sequence[str],
    filtering: str,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    """Plot heatmap of metrics by dataset for a fixed filtering method.

    Args:
        results_root: Root directory of analysis results.
        datasets: Dataset names.
        filtering: Filtering method to visualize.
        output_dir: Directory to save plots.
    """
    rows: List[pd.DataFrame] = []
    for dataset in datasets:
        summary = read_summary(results_root, dataset, filtering, allow_missing=True)
        if summary is None:
            continue
        summary["Dataset"] = dataset
        rows.append(summary)

    if not rows:
        print(f"WARNING: No summaries found for filter '{filtering}'. Skipping heatmap.")
        return
    df = pd.concat(rows, ignore_index=True)
    df["Metric"] = df["Metric"].map(_clean_metric_name)
    df["Filter"] = _clean_filter_name(filtering)
    if "Std_Score_Trials" in df.columns:
        df["Std_Score_Display"] = df["Std_Score_Trials"].astype(float)
    elif "Std_Score_Perts" in df.columns:
        df["Std_Score_Display"] = df["Std_Score_Perts"].astype(float)
    else:
        df["Std_Score_Display"] = np.nan
    min_score = float(df["Mean_Score"].min())
    max_score = float(df["Mean_Score"].max())
    mid_score = min_score + 0.5 * (max_score - min_score)
    df["LabelColor"] = np.where(df["Mean_Score"] >= mid_score, "light", "dark")
    df["Label"] = df.apply(
        lambda row: f"{row['Mean_Score']:.3f} ({row['Std_Score_Display']:.3f})"
        if np.isfinite(row["Std_Score_Display"])
        else f"{row['Mean_Score']:.3f} (n/a)",
        axis=1,
    )
    pivot = df.pivot(index="Dataset", columns="Metric", values="Mean_Score")
    metric_order = pivot.mean(axis=0).sort_values(ascending=True).index.tolist()
    dataset_order = pivot.mean(axis=1).sort_values(ascending=True).index.tolist()
    df["Dataset"] = pd.Categorical(df["Dataset"], categories=dataset_order, ordered=True)
    df["Metric"] = pd.Categorical(df["Metric"], categories=metric_order, ordered=True)
    plot = (
        ggplot(df, aes("Metric", "Dataset", fill="Mean_Score"))
        + geom_tile()
        + geom_text(aes(label="Label", color="LabelColor"), size=9, fontweight="bold")
        + scale_fill_cmap("YlGnBu")
        + scale_color_manual(values={"dark": "black", "light": "white"})
        + theme_minimal(base_size=11)
        + theme(figure_size=(11, 6))
        + labs(title=f"PDS Heatmap ({_clean_filter_name(filtering)})", x="Metric", y="Dataset")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        plot.save(
            filename=str(output_dir / f"heatmap_{filtering}.{fmt}"),
            dpi=300,
            verbose=False,
        )


def plot_energy_filter_comparison(
    results_root: Path,
    datasets: Sequence[str],
    filtering_methods: Sequence[str],
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    """Plot Energy Distance comparison across filtering methods.

    Args:
        results_root: Root directory of analysis results.
        datasets: Dataset names.
        filtering_methods: Filtering methods to compare.
        output_dir: Directory to save plots.
    """
    rows: List[Dict[str, object]] = []
    for dataset in datasets:
        for filtering in filtering_methods:
            summary = read_summary(results_root, dataset, filtering, allow_missing=True)
            if summary is None:
                continue
            energy_row = summary[summary["Metric"] == "Energy_Distance"]
            if energy_row.empty:
                print(
                    f"WARNING: Missing Energy_Distance in summary for {dataset}/{filtering}."
                )
                continue
            energy_row = energy_row.iloc[0]
            std_trials = float(energy_row.get("Std_Score_Trials", np.nan))
            if np.isnan(std_trials):
                std_trials = float(energy_row.get("Std_Score_Perts", 0.0))
            rows.append(
                {
                    "Dataset": dataset,
                    "Filtering": filtering,
                    "Mean_Score": float(energy_row["Mean_Score"]),
                    "Std_Score_Trials": std_trials,
                }
            )

    if not rows:
        print("WARNING: No Energy Distance summaries found. Skipping filter comparison.")
        return
    df = pd.DataFrame(rows)
    df["Filtering"] = df["Filtering"].map(_clean_filter_name)
    dataset_order = (
        df.groupby("Dataset")["Mean_Score"].mean().sort_values(ascending=True).index.tolist()
    )
    df["Dataset"] = pd.Categorical(df["Dataset"], categories=dataset_order, ordered=True)
    df["Filtering"] = pd.Categorical(
        df["Filtering"],
        categories=[_clean_filter_name(f) for f in filtering_methods],
        ordered=True,
    )
    dodge_width = BAR_DODGE_WIDTH
    plot = (
        ggplot(df, aes(x="Filtering", y="Mean_Score", fill="Dataset"))
        + geom_col(position=position_dodge(width=dodge_width), width=BAR_WIDTH)
        + geom_errorbar(
            aes(
                ymin="Mean_Score-Std_Score_Trials",
                ymax="Mean_Score+Std_Score_Trials",
            ),
            position=position_dodge(width=dodge_width),
            width=0.4,
        )
        + scale_y_continuous(limits=(0, None), expand=(0, 0))
        + scale_fill_manual(values=_dataset_palette_viridis(list(df["Dataset"].cat.categories)))
        + theme_minimal(base_size=NATURE_BASE_SIZE)
        + theme(
            figure_size=FIG_SIZE_TWO_UP,
            axis_text_x=element_text(rotation=30, ha="right"),
            axis_title=element_text(size=NATURE_AXIS_LABEL_SIZE),
            axis_text=element_text(size=NATURE_TICK_SIZE),
            axis_line=element_line(size=0.4, color="black"),
            axis_ticks=element_line(size=0.4, color="black"),
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            legend_position="none",
            legend_title=element_text(size=NATURE_LEGEND_SIZE),
            legend_text=element_text(size=NATURE_LEGEND_SIZE),
        )
        + labs(x="", y="Mean PDS (+/- trial s.d.)")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        plot.save(
            filename=str(output_dir / f"energy_filter_comparison.{fmt}"),
            dpi=PLOT_DPI,
            verbose=False,
        )


def plot_threshold_comparison(
    results_root: Path,
    datasets: Sequence[str],
    output_dir: Path,
    metric: str = "Energy_Distance",
    formats: Sequence[str] = ("png",),
) -> None:
    """Plot mean max-gene count reaching 0.8 PDS threshold.

    Args:
        results_root: Root directory of analysis results.
        datasets: Dataset names.
        output_dir: Directory to save plots.
        metric: Metric name to plot.
    """
    rows: List[Dict[str, object]] = []
    for dataset in datasets:
        per_pert_path = (
            results_root / dataset / "aggregates" / "thresholds_per_perturbation.csv"
        )
        if not per_pert_path.exists():
            print(f"WARNING: Missing threshold per-pert file: {per_pert_path}")
            continue
        per_pert = pd.read_csv(per_pert_path)
        metric_rows = per_pert[per_pert["Metric"] == metric]
        if metric_rows.empty:
            print(f"WARNING: Metric '{metric}' missing from {per_pert_path}")
            continue
        metric_rows = metric_rows.copy()
        metric_rows["Dataset"] = dataset
        rows.append(metric_rows)

    if not rows:
        print(f"WARNING: No threshold summaries found for metric '{metric}'. Skipping.")
        return
    df = pd.concat(rows, ignore_index=True)
    positive = df.loc[df["Max_N_Above_0p8"] > 0, "Max_N_Above_0p8"].astype(float)
    if positive.empty:
        print(
            f"WARNING: No positive threshold values for metric '{metric}'. "
            "Skipping log-scale threshold plot."
        )
        return
    min_pow = int(np.floor(np.log10(float(positive.min()))))
    max_pow = int(np.ceil(np.log10(float(positive.max()))))
    log_breaks = [10 ** p for p in range(min_pow, max_pow + 1)]
    dataset_order = (
        df.groupby("Dataset")["Max_N_Above_0p8"].mean().sort_values(ascending=True).index
    )
    df["Dataset"] = pd.Categorical(df["Dataset"], categories=dataset_order, ordered=True)
    plot = (
        ggplot(df, aes(x="Dataset", y="Max_N_Above_0p8", fill="Dataset"))
        + geom_violin(trim=True)
        + scale_fill_manual(values=_dataset_palette_viridis(list(df["Dataset"].cat.categories)))
        + geom_point(
            data=df.groupby("Dataset", as_index=False)["Max_N_Above_0p8"].mean(),
            mapping=aes(x="Dataset", y="Max_N_Above_0p8"),
            size=1.8,
            color="black",
        )
        + scale_y_log10(
            breaks=log_breaks,
            labels=lambda vals: [f"{int(v):d}" if np.isfinite(v) else "" for v in vals],
        )
        + theme_minimal(base_size=11)
        + theme(figure_size=(8, 4))
        + labs(
            title=f"Max Genes with Mean PDS ≥ 0.8 ({_clean_metric_name(metric)})",
            x="Dataset",
            y="Max N (per perturbation)",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        plot.save(
            filename=str(output_dir / f"threshold_{metric}.{fmt}"),
            dpi=300,
            verbose=False,
        )


def plot_mean_pds_per_gene(
    results_root: Path,
    datasets: Sequence[str],
    output_dir: Path,
    metric: str = "Energy_Distance",
    formats: Sequence[str] = ("png",),
) -> None:
    """Plot mean PDS (averaged over perturbations) for each gene."""
    rows: List[Dict[str, object]] = []
    for dataset in datasets:
        pds_path = results_root / dataset / "pds_full.h5ad"
        if not pds_path.exists():
            print(f"WARNING: Missing PDS file: {pds_path}")
            continue
        payload = base.load_pds_full(pds_path)
        metrics = [str(item) for item in payload["metrics"]]
        if metric not in metrics:
            print(f"WARNING: Metric '{metric}' missing from {pds_path}")
            continue
        metric_idx = metrics.index(metric)
        scores = np.asarray(payload["scores_mean"], dtype=float)[metric_idx]
        genes = [str(item) for item in payload["genes"]]
        if scores.shape[1] != len(genes):
            print(f"WARNING: Unexpected tensor shape in {pds_path}; skipping.")
            continue
        per_gene_mean = np.mean(scores, axis=0)
        for gene, mean_val in zip(genes, per_gene_mean):
            rows.append(
                {
                    "Dataset": dataset,
                    "Gene": gene,
                    "Mean_PDS_Per_Gene": float(mean_val),
                }
            )

    if not rows:
        print(f"WARNING: No per-gene mean PDS values for metric '{metric}'. Skipping.")
        return

    df = pd.DataFrame(rows)
    dataset_order = (
        df.groupby("Dataset")["Mean_PDS_Per_Gene"].mean().sort_values(ascending=True).index
    )
    df["Dataset"] = pd.Categorical(df["Dataset"], categories=dataset_order, ordered=True)
    plot = (
        ggplot(df, aes(x="Dataset", y="Mean_PDS_Per_Gene", fill="Dataset"))
        + geom_violin(trim=True)
        + scale_fill_manual(values=_dataset_palette_viridis(list(df["Dataset"].cat.categories)))
        + _dataset_legend()
        + theme_minimal(base_size=NATURE_BASE_SIZE)
        + theme(
            figure_size=FIG_SIZE_SINGLE_COLUMN,
            axis_title=element_text(size=NATURE_AXIS_LABEL_SIZE),
            axis_text=element_text(size=NATURE_TICK_SIZE),
            axis_text_x=element_text(rotation=35, ha="right"),
            axis_line=element_line(size=0.4, color="black"),
            axis_ticks=element_line(size=0.4, color="black"),
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            legend_position="right",
            legend_title=element_text(size=NATURE_LEGEND_SIZE),
            legend_text=element_text(size=NATURE_LEGEND_SIZE),
        )
        + labs(x="", y="Mean PDS (per gene)")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        plot.save(
            filename=str(output_dir / f"mean_pds_per_gene_{metric}.{fmt}"),
            dpi=PLOT_DPI,
            verbose=False,
        )


def plot_thresholds_across_metrics(
    results_root: Path,
    datasets: Sequence[str],
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    """Plot threshold gene counts across datasets and metrics.

    Args:
        results_root: Root directory of analysis results.
        datasets: Dataset names.
        output_dir: Directory to save plots.
        formats: Output formats.
    """
    rows: List[Dict[str, object]] = []
    for dataset in datasets:
        summary_path = results_root / dataset / "aggregates" / "thresholds_summary.json"
        if not summary_path.exists():
            print(f"WARNING: Missing threshold summary: {summary_path}")
            continue
        with summary_path.open("r") as handle:
            summary = json.load(handle)
        thresholds = summary.get("Thresholds", {})
        if not thresholds:
            print(f"WARNING: Empty threshold data in {summary_path}")
            continue
        for metric, stats in thresholds.items():
            rows.append(
                {
                    "Dataset": dataset,
                    "Metric": metric,
                    "Mean_Max_N_Above_0p8": stats.get("Mean_Max_N_Above_0p8", np.nan),
                }
            )
    if not rows:
        print("WARNING: No threshold summaries found. Skipping thresholds plot.")
        return
    df = pd.DataFrame(rows)
    df["Metric"] = df["Metric"].map(_clean_metric_name)
    pivot = df.pivot(index="Dataset", columns="Metric", values="Mean_Max_N_Above_0p8")
    metric_order = pivot.mean(axis=0).sort_values(ascending=True).index.tolist()
    dataset_order = pivot.mean(axis=1).sort_values(ascending=True).index.tolist()
    df["Dataset"] = pd.Categorical(df["Dataset"], categories=dataset_order, ordered=True)
    df["Metric"] = pd.Categorical(df["Metric"], categories=metric_order, ordered=True)
    plot = (
        ggplot(df, aes(x="Metric", y="Mean_Max_N_Above_0p8", fill="Dataset"))
        + geom_col(position=position_dodge(width=BAR_DODGE_WIDTH), width=BAR_WIDTH)
        + scale_y_continuous(limits=(0, None), expand=(0, 0))
        + scale_fill_manual(values=_dataset_palette_viridis(list(df["Dataset"].cat.categories)))
        + theme_minimal(base_size=11)
        + theme(figure_size=(10, 4))
        + labs(
            title="Mean Max Genes with Mean PDS ≥ 0.8",
            x="Metric",
            y="Mean max N",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        plot.save(
            filename=str(output_dir / f"thresholds_across_metrics.{fmt}"),
            dpi=300,
            verbose=False,
        )


def plot_overall_metric_comparison(
    results_root: Path,
    datasets: Sequence[str],
    filtering_methods: Sequence[str],
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    """Plot overall mean PDS across filters, grouped by metric and dataset.

    Excludes ``MAE_Median`` (MAE on median expression) so the panel shows the
    other summary metrics only (e.g. MAE mean, Energy Distance, Wasserstein).
    """
    rows: List[Dict[str, object]] = []
    for dataset in datasets:
        for filtering in filtering_methods:
            summary = read_summary(results_root, dataset, filtering, allow_missing=True)
            if summary is None:
                continue
            for _, row in summary.iterrows():
                if "Std_Score_Trials" in summary.columns:
                    std_score = row["Std_Score_Trials"]
                elif "Std_Score_Perts" in summary.columns:
                    std_score = row["Std_Score_Perts"]
                else:
                    std_score = np.nan
                rows.append(
                    {
                        "Dataset": dataset,
                        "Metric": row["Metric"],
                        "Mean_Score": float(row["Mean_Score"]),
                        "Std_Score_Trials": float(std_score) if pd.notnull(std_score) else np.nan,
                    }
                )
    if not rows:
        print("WARNING: No summaries found for overall metric comparison.")
        return
    df = pd.DataFrame(rows)
    df = df.loc[df["Metric"] != "MAE_Median"].copy()
    if df.empty:
        print("WARNING: No rows left after excluding MAE_Median. Skipping overall metric comparison.")
        return
    df["Metric"] = df["Metric"].map(_clean_metric_name)
    df = (
        df.groupby(["Dataset", "Metric"], as_index=False)
        .agg({"Mean_Score": "mean", "Std_Score_Trials": "mean"})
        .assign(
            ymin=lambda frame: frame["Mean_Score"] - frame["Std_Score_Trials"],
            ymax=lambda frame: frame["Mean_Score"] + frame["Std_Score_Trials"],
        )
    )
    metric_order = (
        df.groupby("Metric")["Mean_Score"].mean().sort_values(ascending=True).index.tolist()
    )
    dataset_order = (
        df.groupby("Dataset")["Mean_Score"].mean().sort_values(ascending=True).index.tolist()
    )
    df["Metric"] = pd.Categorical(df["Metric"], categories=metric_order, ordered=True)
    df["Dataset"] = pd.Categorical(df["Dataset"], categories=dataset_order, ordered=True)
    dodge = position_dodge(width=BAR_DODGE_WIDTH)
    plot = (
        ggplot(df, aes(x="Metric", y="Mean_Score", fill="Dataset"))
        + geom_col(position=dodge, width=BAR_WIDTH)
        + geom_errorbar(
            aes(ymin="ymin", ymax="ymax"),
            position=dodge,
            width=0.2,
        )
        + scale_y_continuous(limits=(0, None), expand=(0, 0))
        + scale_fill_manual(values=_dataset_palette_viridis(list(df["Dataset"].cat.categories)))
        + _dataset_legend()
        + theme_minimal(base_size=NATURE_BASE_SIZE)
        + theme(
            figure_size=FIG_SIZE_TWO_UP,
            axis_title=element_text(size=NATURE_AXIS_LABEL_SIZE),
            axis_text=element_text(size=NATURE_TICK_SIZE),
            axis_text_x=element_text(rotation=30, ha="right"),
            axis_line=element_line(size=0.4, color="black"),
            axis_ticks=element_line(size=0.4, color="black"),
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            legend_position="right",
            legend_title=element_text(size=NATURE_LEGEND_SIZE),
            legend_text=element_text(size=NATURE_LEGEND_SIZE),
        )
        + labs(x="", y="Mean PDS")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        plot.save(
            filename=str(output_dir / f"overall_metric_comparison.{fmt}"),
            dpi=PLOT_DPI,
            verbose=False,
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Visualize PDS analysis results.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to visualize.",
    )
    parser.add_argument(
        "--output-formats",
        nargs="+",
        default=["pdf", "png"],
        help="Output formats (e.g., pdf png).",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="nature",
        help="Plot style preset (nature or existing).",
    )
    parser.add_argument(
        "--filtering",
        type=str,
        default="highly_variable",
        help="Filtering method for heatmap/discrepancy plots.",
    )
    parser.add_argument(
        "--compare-filtering",
        nargs="+",
        default=[
            "highly_variable",
            "lowest_cv",
            # "lowest_cv_per_perturbation",
            "deg_control",
            # "deg_synthetic",
            "top_discriminating",
            "deg_per_perturbation_control",
            # "deg_per_perturbation_synthetic",
            "top_discriminating_per_perturbation",
        ],
        help="Filtering methods for Energy Distance comparison.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for visualization script."""
    args = parse_args()
    results_root = ROOT / "analyses" / "perturbation_discrimination" / "results"
    output_dir = ROOT / "analyses" / "perturbation_discrimination" / "figures" / "publication"
    set_publication_style(args.style)
    if not args.datasets:
        args.datasets = list_available_datasets(results_root)

    plot_metric_heatmap(results_root, args.datasets, args.filtering, output_dir, args.output_formats)
    plot_metric_heatmap(
        results_root,
        args.datasets,
        "top_discriminating_per_perturbation",
        output_dir,
        args.output_formats,
    )
    plot_energy_filter_comparison(
        results_root, args.datasets, args.compare_filtering, output_dir, args.output_formats
    )
    plot_overall_metric_comparison(
        results_root, args.datasets, args.compare_filtering, output_dir, args.output_formats
    )
    plot_thresholds_across_metrics(results_root, args.datasets, output_dir, args.output_formats)
    plot_threshold_comparison(
        results_root,
        args.datasets,
        output_dir,
        metric="Energy_Distance",
        formats=args.output_formats,
    )
    plot_mean_pds_per_gene(
        results_root,
        args.datasets,
        output_dir,
        metric="Energy_Distance",
        formats=args.output_formats,
    )
if __name__ == "__main__":
    main()
