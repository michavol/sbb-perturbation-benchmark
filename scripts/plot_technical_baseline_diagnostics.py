#!/usr/bin/env python3
"""Nature-style diagnostic plots for technical-duplicate baselines across datasets.

Creates three figures:
1. Heatmap of technical-duplicate PDS scores (datasets x PDS metrics)
2. Heatmap of calibration direction fidelity (1 - unexpected_denominator_sign_fraction)
3. Bar chart of effective gene statistics (mean +/- std per dataset)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Nature-style plotting conventions (matching plot_multimodel_summary.py)
# ---------------------------------------------------------------------------
_NATURE_FONT = "Arial"
_NATURE_AXIS_LABEL_SIZE = 7
_NATURE_TICK_SIZE = 6
_NATURE_LEGEND_SIZE = 6
_NATURE_TITLE_SIZE = 7
_NATURE_DPI = 450

ALL_DATASETS = [
    "wessels23",
    "norman19",
    "replogle20",
    "adamson16",
    "frangieh21",
    "replogle22k562",
    "nadig25hepg2",
]

DEFAULT_PDS_METRICS = [
    "pds", "pds_wmse", "pds_pearson_deltapert",
    "pds_weighted_pearson_deltapert", "pds_r2_deltapert", "pds_weighted_r2_deltapert",
]

DEFAULT_CALIBRATED_METRICS = [
    "cwmse",
    "cmse",
    "cpearson_deltapert",
    "cweighted_pearson_deltapert",
    "cr2_deltapert",
    "cweighted_r2_deltapert",
    "cpds",
    "cpds_wmse",
    "cpds_pearson_deltapert",
    "cpds_weighted_pearson_deltapert",
    "cpds_r2_deltapert",
    "cpds_weighted_r2_deltapert",
]

METRIC_DISPLAY = {
    "mse": "MSE",
    "wmse": "wMSE",
    "cmse": "cMSE",
    "cwmse": "cwMSE",
    "pds": "PDS (MSE)",
    "pds_wmse": "PDS (wMSE)",
    "pds_pearson_deltapert": "PDS (Pearson\u0394Pert)",
    "cpds": "cPDS",
    "cpds_wmse": "cPDS (wMSE)",
    "cpds_pearson_deltapert": "cPDS (Pearson\u0394Pert)",
    "cpds_weighted_pearson_deltapert": "cPDS (wPearson\u0394Pert)",
    "cpds_r2_deltapert": "cPDS (R\u00b2\u0394Pert)",
    "cpds_weighted_r2_deltapert": "cPDS (wR\u00b2\u0394Pert)",
    "pds_weighted_pearson_deltapert": "PDS (wPearson\u0394Pert)",
    "pds_r2_deltapert": "PDS (R\u00b2\u0394Pert)",
    "pds_weighted_r2_deltapert": "PDS (wR\u00b2\u0394Pert)",
    "pearson_deltapert": "Pearson\u0394Pert",
    "weighted_pearson_deltapert": "wPearson\u0394Pert",
    "cpearson_deltapert": "cPearson\u0394Pert",
    "cweighted_pearson_deltapert": "cwPearson\u0394Pert",
    "r2_deltapert": "R\u00b2\u0394Pert",
    "weighted_r2_deltapert": "wR\u00b2\u0394Pert",
    "cr2_deltapert": "cR\u00b2\u0394Pert",
    "cweighted_r2_deltapert": "cwR\u00b2\u0394Pert",
}

PDS_AXIS_LABELS = {
    "pds": "MSE",
    "pds_wmse": "wMSE",
    "pds_pearson_deltapert": "Pearson\u0394Pert",
    "pds_weighted_pearson_deltapert": "wPearson\u0394Pert",
    "pds_r2_deltapert": "R\u00b2\u0394Pert",
    "pds_weighted_r2_deltapert": "wR\u00b2\u0394Pert",
}

DATASET_COLORS = {
    "wessels23": "#1b5e20",
    "norman19": "#ff7f0e",
    "replogle20": "#9467bd",
    "adamson16": "#17becf",
    "frangieh21": "#d62728",
    "replogle22k562": "#e377c2",
    "nadig25hepg2": "#8c564b",
}

SINGLE_PERT_DATASETS = frozenset(
    {"adamson16", "frangieh21", "replogle22k562", "nadig25hepg2", "nadig25hpeg2"}
)

DATASET_ALIASES: Dict[str, str] = {
    "nadig25hpeg2": "nadig25hepg2",
    "repogle22k562": "replogle22k562",
}


def _apply_nature_rc() -> None:
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


def _savefig(fig: plt.Figure, path: Path, formats: Tuple[str, ...] = ("png", "pdf")) -> None:
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=_NATURE_DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved {out}")


def _metric_display(metric: str) -> str:
    if metric in METRIC_DISPLAY:
        label = METRIC_DISPLAY[metric]
    else:
        label = metric.replace("_", " ").title()
    # For calibrated metrics, remove the leading "c" from display labels.
    if metric.startswith("c") and len(metric) > 1:
        base_metric = metric[1:]
        if base_metric in METRIC_DISPLAY:
            label = METRIC_DISPLAY[base_metric]
        elif label.startswith("c"):
            label = label[1:]
    return label


def _darken(hex_color: str, factor: float = 0.7) -> str:
    """Darken a hex colour by *factor* (0 = black, 1 = unchanged)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(
        int(r * factor), int(g * factor), int(b * factor)
    )


# ---------------------------------------------------------------------------
# Dataset / directory helpers (from plot_dataset_ranking.py)
# ---------------------------------------------------------------------------

def canonical_dataset(ds: str) -> str:
    return DATASET_ALIASES.get(ds, ds)


def ordered_unique_canonical(datasets: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for d in datasets:
        c = canonical_dataset(d)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _bench_matches_dataset(bench_name: str, ds: str) -> bool:
    ds = canonical_dataset(ds)
    if ds == "nadig25hepg2":
        return "nadig25hepg2" in bench_name or "nadig25hpeg2" in bench_name
    return ds in bench_name


def _find_dirs(base: Path, datasets: List[str]) -> Dict[str, Path | None]:
    """Return the latest benchmark directory for each dataset (or None)."""
    results: Dict[str, Path | None] = {}
    for ds in datasets:
        found: Path | None = None
        for pattern in ["gears_*", "benchmark_*", "_benchmark_*"]:
            for bench in (base / "outputs").glob(pattern):
                if not _bench_matches_dataset(bench.name, ds):
                    continue
                for sub in sorted(bench.iterdir(), reverse=True):
                    if sub.is_dir() and (sub / "detailed_metrics.csv").exists():
                        found = sub
                        break
                if found:
                    break
            if found:
                break
        results[canonical_dataset(ds)] = found
    return results


def _is_combo(p: str) -> bool:
    return any(s in str(p) for s in ("+", "__", "|", ";")) or "_" in str(p)


def _is_single_pert_row(p: str) -> bool:
    s = str(p)
    return not any(x in s for x in ("+", "__", "|", ";"))


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_technical_duplicate_scores(
    bench_dirs: Dict[str, Path | None],
    metrics: List[str],
) -> pd.DataFrame:
    """Load mean technical-duplicate score per (dataset, metric)."""
    rows: List[Dict] = []
    for ds, bench_dir in bench_dirs.items():
        if bench_dir is None:
            continue
        csv_path = bench_dir / "detailed_metrics.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df = df[df["model"] == "technical_duplicate"]
        if ds in SINGLE_PERT_DATASETS:
            df = df[df["perturbation"].apply(_is_single_pert_row)]
        else:
            df = df[df["perturbation"].apply(_is_combo)]
        for metric in metrics:
            vals = df.loc[df["metric"] == metric, "value"].dropna()
            if vals.empty:
                continue
            rows.append({"dataset": ds, "metric": metric, "mean_value": vals.mean()})
    return pd.DataFrame(rows)


def load_calibration_fidelity(
    bench_dirs: Dict[str, Path | None],
    calibrated_metrics: List[str],
) -> pd.DataFrame:
    """Load 1 - unexpected_denominator_sign_fraction per (dataset, metric)."""
    rows: List[Dict] = []
    for ds, bench_dir in bench_dirs.items():
        if bench_dir is None:
            continue
        json_path = bench_dir / "calibration_aggregated_denominators.json"
        if not json_path.exists():
            print(f"  WARNING: {json_path} not found for {ds}")
            continue
        with open(json_path) as f:
            data = json.load(f)
        per_metric = data.get("per_metric", {})
        for metric in calibrated_metrics:
            if metric not in per_metric:
                continue
            frac = per_metric[metric].get("unexpected_denominator_sign_fraction", 0.0)
            rows.append({
                "dataset": ds,
                "metric": metric,
                "fidelity": 1.0 - frac,
            })
    return pd.DataFrame(rows)


def load_effective_gene_stats(
    bench_dirs: Dict[str, Path | None],
) -> pd.DataFrame:
    """Load effective gene summary stats per dataset."""
    rows: List[Dict] = []
    for ds, bench_dir in bench_dirs.items():
        if bench_dir is None:
            continue
        json_path = bench_dir / "effective_gene_stats.json"
        if not json_path.exists():
            print(f"  WARNING: {json_path} not found for {ds}")
            continue
        with open(json_path) as f:
            data = json.load(f)
        rows.append({
            "dataset": ds,
            "mean": data.get("mean", np.nan),
            "std": data.get("std", np.nan),
            "median": data.get("median", np.nan),
            "min": data.get("min", np.nan),
            "max": data.get("max", np.nan),
            "n_perts": data.get("n_perts_with_weights", 0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot 1: Technical Duplicate PDS Heatmap
# ---------------------------------------------------------------------------

def plot_technical_duplicate_heatmap(
    df: pd.DataFrame,
    datasets: List[str],
    metrics: List[str],
    output_path: Path,
    formats: Tuple[str, ...],
) -> None:
    if df.empty:
        print("  No technical-duplicate data; skipping heatmap.")
        return

    # Datasets on y-axis, metrics on x-axis
    pivot = df.pivot(index="dataset", columns="metric", values="mean_value")
    metric_order = [m for m in metrics if m in pivot.columns]
    ds_order = [d for d in datasets if d in pivot.index]
    pivot = pivot.loc[ds_order, metric_order]

    # Sort both axes by mean value (descending for datasets, descending for metrics)
    ds_sorted = pivot.mean(axis=1).sort_values(ascending=False).index.tolist()
    metric_sorted = pivot.mean(axis=0).sort_values(ascending=False).index.tolist()
    pivot = pivot.loc[ds_sorted, metric_sorted]

    annot = np.empty(pivot.shape, dtype=object)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            annot[i, j] = "" if pd.isna(v) else f"{v:.3f}"

    fig_w = max(3.0, 0.65 * len(metric_sorted) + 1.5)
    fig_h = max(2.0, 0.35 * len(ds_sorted) + 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        pivot,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="#cccccc",
        ax=ax,
        annot_kws={"size": _NATURE_TICK_SIZE},
        cbar_kws={"label": "PDS Score on Technical Duplicate", "shrink": 0.8},
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=_NATURE_TICK_SIZE)
    cbar.set_label("PDS Score on Technical Duplicate", fontsize=_NATURE_AXIS_LABEL_SIZE)

    ax.set_xticklabels([PDS_AXIS_LABELS.get(m, _metric_display(m)) for m in metric_sorted],
                       rotation=45, ha="right", fontsize=_NATURE_TICK_SIZE)
    ax.set_yticklabels(ds_sorted, rotation=0, fontsize=_NATURE_TICK_SIZE)
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.tight_layout()
    _savefig(fig, output_path, formats)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Calibration Direction Fidelity Heatmap
# ---------------------------------------------------------------------------

def plot_calibration_fidelity_heatmap(
    df: pd.DataFrame,
    datasets: List[str],
    calibrated_metrics: List[str],
    output_path: Path,
    formats: Tuple[str, ...],
) -> None:
    if df.empty:
        print("  No calibration data; skipping heatmap.")
        return

    # Datasets on y-axis, metrics on x-axis
    pivot = df.pivot(index="dataset", columns="metric", values="fidelity")
    metric_order = [m for m in calibrated_metrics if m in pivot.columns]
    ds_order = [d for d in datasets if d in pivot.index]
    pivot = pivot.loc[ds_order, metric_order]

    # Sort both axes by mean fidelity (descending)
    ds_sorted = pivot.mean(axis=1).sort_values(ascending=False).index.tolist()
    metric_sorted = pivot.mean(axis=0).sort_values(ascending=False).index.tolist()
    pivot = pivot.loc[ds_sorted, metric_sorted]

    annot = np.empty(pivot.shape, dtype=object)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if pd.isna(v):
                annot[i, j] = ""
            else:
                annot[i, j] = f"{v:.1%}" if v < 0.995 else "100%"

    fig_w = max(4.0, 0.55 * len(metric_sorted) + 1.5)
    fig_h = max(2.0, 0.35 * len(ds_sorted) + 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        pivot,
        annot=annot,
        fmt="",
        cmap="YlGn",
        vmin=0.5,
        vmax=1.0,
        linewidths=0.5,
        linecolor="#cccccc",
        ax=ax,
        annot_kws={"size": _NATURE_TICK_SIZE},
        cbar_kws={"label": "BDS", "shrink": 0.8},
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=_NATURE_TICK_SIZE)
    cbar.set_label("BDS (Fraction Tech. Dup. > Control)", fontsize=_NATURE_AXIS_LABEL_SIZE)

    ax.set_xticklabels([_metric_display(m) for m in metric_sorted],
                       rotation=45, ha="right", fontsize=_NATURE_TICK_SIZE)
    ax.set_yticklabels(ds_sorted, rotation=0, fontsize=_NATURE_TICK_SIZE)
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.tight_layout()
    _savefig(fig, output_path, formats)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Effective Gene Stats Bar Chart
# ---------------------------------------------------------------------------

def plot_effective_gene_stats(
    df: pd.DataFrame,
    datasets: List[str],
    output_path: Path,
    formats: Tuple[str, ...],
) -> None:
    if df.empty:
        print("  No effective gene data; skipping bar chart.")
        return

    ds_order = [d for d in datasets if d in df["dataset"].values]
    df = df.set_index("dataset").loc[ds_order].reset_index()
    df = df.sort_values("mean", ascending=False).reset_index(drop=True)

    colors = [DATASET_COLORS.get(d, "#666666") for d in df["dataset"]]

    fig, ax = plt.subplots(figsize=(3.5, 2.35))

    x = np.arange(len(df))
    bars = ax.bar(
        x,
        df["mean"],
        yerr=df["std"],
        capsize=2.5,
        color=colors,
        edgecolor=[_darken(c, 0.7) for c in colors],
        linewidth=0.6,
        width=0.62,
        error_kw={"linewidth": 0.7, "capthick": 0.7, "color": "#333333"},
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"], rotation=45, ha="right", fontsize=_NATURE_TICK_SIZE)
    ax.set_ylabel("Effective Gene Count", fontsize=_NATURE_AXIS_LABEL_SIZE)
    ax.set_xlabel("")

    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="both", which="major", length=2.5, width=0.6, pad=2)

    ax.set_xlim(-0.55, len(df) - 0.45)
    ax.set_ylim(0, None)

    fig.tight_layout()
    _savefig(fig, output_path, formats)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nature-style diagnostic plots for technical-duplicate baselines."
    )
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    parser.add_argument("--pds-metrics", nargs="+", default=DEFAULT_PDS_METRICS)
    parser.add_argument("--calibrated-metrics", nargs="+", default=DEFAULT_CALIBRATED_METRICS)
    parser.add_argument("--base-path", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/_benchmark_dataset"),
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()

    _apply_nature_rc()

    datasets = ordered_unique_canonical(args.datasets)
    formats = tuple(args.formats)

    print("Searching for benchmark directories...")
    bench_dirs = _find_dirs(args.base_path, datasets)
    for ds, d in bench_dirs.items():
        status = str(d) if d else "NOT FOUND"
        print(f"  {ds}: {status}")

    # Plot 1 — Technical Duplicate PDS Heatmap
    print("\nPlot 1: Technical Duplicate PDS scores")
    td_df = load_technical_duplicate_scores(bench_dirs, args.pds_metrics)
    if not td_df.empty:
        print(f"  Loaded {len(td_df)} (dataset, metric) entries")
    plot_technical_duplicate_heatmap(
        td_df, datasets, args.pds_metrics,
        args.output_dir / "technical_duplicate_pds",
        formats,
    )

    # Plot 2 — Calibration Direction Fidelity
    print("\nPlot 2: Calibration Direction Fidelity")
    cal_df = load_calibration_fidelity(bench_dirs, args.calibrated_metrics)
    if not cal_df.empty:
        print(f"  Loaded {len(cal_df)} (dataset, metric) entries")
    plot_calibration_fidelity_heatmap(
        cal_df, datasets, args.calibrated_metrics,
        args.output_dir / "calibration_direction_fidelity",
        formats,
    )

    # Plot 3 — Effective Gene Stats
    print("\nPlot 3: Effective Gene Stats")
    eff_df = load_effective_gene_stats(bench_dirs)
    if not eff_df.empty:
        print(f"  Loaded stats for {len(eff_df)} datasets")
    plot_effective_gene_stats(
        eff_df, datasets,
        args.output_dir / "effective_gene_stats",
        formats,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
