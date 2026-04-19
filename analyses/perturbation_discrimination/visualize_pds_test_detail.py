#!/usr/bin/env python3
"""Visualize per-perturbation PDS or BDS permutation-test results from *detail.csv.

Assumes num_trials=1 (one row per dataset / metric / perturbation). If multiple
trial IDs appear, rows are restricted to trial == 0 with a warning.

Outputs (under --out-dir):
  - Heatmaps: fraction passing FDR (adj_pval < α); −log10(median p); mean raw p with mean±std
    annotations (colors still −log10(mean p)) — **metrics on y-axis, datasets on x-axis**;
    median and mean R_obs each mapped to PDS score in
    [−1, 1] via rank_to_score (1 = best retrieval); separate heatmaps ``heatmap_Robs_median`` and
    ``heatmap_Robs_mean``.
    Fraction heatmap order: --heatmap-order-key, --heatmap-row-agg, --heatmap-col-agg (defaults: fraction,
    mean, mean). Median and mean p heatmaps use marginal −log10(median p) for axes order. R_obs
    heatmaps use marginal PDS from median R_obs and from mean R_obs respectively (higher = better).
  - Pairwise Spearman correlation of p-values or observed ranks R_obs across metrics per dataset.

Why p-value correlations are often modest: metrics use different distances (MSE vs correlation vs
energy), so the same perturbation can get very different p-values under different nulls; p-values
are also a nonlinear transform of the test statistic. R_obs correlation asks a different question:
whether metrics agree on *relative* difficulty (rank of self-match among cross-perturbation
distances), which is often more stable than agreement on calibrated p-values.

Set-overlap panels (--plots jaccard, szymkiewicz-simpson): pairwise overlap of
perturbations called significant (or not) under each metric at the same FDR threshold
as the fraction heatmap. Jaccard uses |∩|/|∪|; Szymkiewicz–Simpson uses |∩|/min(|A|,|B|).
Cell annotations (three lines): |S_i ∩ S_j| / N, |S_i ∪ S_j| / N, then the overlap index
(Jaccard or Szymkiewicz–Simpson, matching the heatmap color). N = perturbations with
complete rows for that dataset panel.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

# Undefined Spearman: blank cells (match figure background).
_SPEARMAN_BAD_COLOR = "#ffffff"

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

_NATURE_FONT = "Arial"
_NATURE_AXIS_LABEL_SIZE = 7
_NATURE_TICK_SIZE = 6
_NATURE_LEGEND_SIZE = 6
_NATURE_TITLE_SIZE = 7
_NATURE_DPI = 450
# Overall scale for heatmap / grid figure inches (width × height).
_FIG_SIZE_SCALE = 0.82
# Shrink canvas vs. prior layout; pair with larger axis tick fonts below.
_HEATMAP_FIG_SCALE = 0.63
_GRID_FIG_SCALE = 0.66
# Dataset (rows) and metric (columns) tick labels on heatmaps / grids.
_HEATMAP_AXIS_TICK_SIZE = 8.5
_GRID_AXIS_TICK_SIZE = 8.0
_GRID_PANEL_TITLE_SIZE = 9.0
# Cell numeric annotations (smaller than axis labels when figure is compact).
_HEATMAP_CELL_ANNOT_SIZE = 7.5
_HEATMAP_CELL_ANNOT_MEAN_SIZE = 6.5
# Per-column / per-row inches for multi-panel grids (Spearman, Jaccard, Szymkiewicz–Simpson).
_GRID_PANEL_W_PER_COL = 6.85
_GRID_PANEL_H_PER_ROW = 5.85
# Three-line labels (∩%, ∪%, index) in overlap heatmaps.
_OVERLAP_ANNOT_FONT_PT = 3.6
# Heatmap figure title (figure coordinates, bold only).
_HEATMAP_TITLE_BOLD_PT = 9.0

ALL_KNOWN_METRICS: List[str] = [
    "mse",
    "wmse",
    "wmse_vc",
    "pearson",
    "wpearson",
    "pearson_dp",
    "wpearson_dp",
    "wpearson_dp_vc",
    "pearson_dc",
    "wpearson_dc",
    "r2_dp",
    "wr2_dp",
    "top20_mse",
    "top20_pearson_dp",
    "top100_mse",
    "top100_energy",
    "energy",
    "wenergy",
]

METRIC_DISPLAY: Dict[str, str] = {
    "mse": "MSE",
    "wmse": "wMSE (syn)",
    "wmse_vc": "wMSE (vc)",
    "pearson": "Pearson",
    "wpearson": "wPearson (syn)",
    "pearson_dp": "Pearson Δpert",
    "wpearson_dp": "wPearson Δpert (syn)",
    "wpearson_dp_vc": "wPearson Δpert (vc)",
    "pearson_dc": "Pearson Δctrl",
    "wpearson_dc": "wPearson Δctrl (syn)",
    "r2_dp": "R² Δpert",
    "wr2_dp": "wR² Δpert (syn)",
    "top20_mse": "top20 MSE (syn)",
    "top20_pearson_dp": "top20 Pearson Δpert (syn)",
    "top100_mse": "top100 MSE (syn)",
    "top100_energy": "top100 Energy (syn)",
    "energy": "Energy dist.",
    "wenergy": "wEnergy dist. (syn)",
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


def _infer_panel_prefix(input_path: Path) -> str:
    """BDS vs PDS label for figure titles (path-based default)."""
    parts = {p.lower() for p in input_path.parts}
    if "bds_test" in parts or "bds_test_detail.csv" == input_path.name.lower():
        return "BDS Testing"
    if "pds_test" in parts or "pds_test_detail.csv" == input_path.name.lower():
        return "PDS Testing"
    return "PDS Testing"


def _p_value_kind_label(p_col: str) -> str:
    """Human-readable name for the p-value column used in colorbars and titles."""
    if p_col == "adj_pval":
        return "adjusted p-value"
    if p_col == "raw_pval":
        return "raw p-value"
    return str(p_col).replace("_", " ")


def _figure_bold_title(
    fig: plt.Figure,
    *,
    bold_line: str,
    top_frac: float = 0.94,
) -> None:
    """Single bold title line; reserves a small top margin."""
    fig.subplots_adjust(top=top_frac)
    fig.text(
        0.5,
        0.99,
        bold_line,
        ha="center",
        va="top",
        fontweight="bold",
        fontsize=_HEATMAP_TITLE_BOLD_PT,
        transform=fig.transFigure,
    )


def _style_heatmap_axis_labels(ax: plt.Axes, *, tick_size: float) -> None:
    """Larger fonts for dataset (y) and metric (x) labels on value heatmaps."""
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=tick_size)
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=tick_size)


def _style_seaborn_cbar(fig: plt.Figure, data_ax: plt.Axes, *, tick_size: float) -> None:
    """Colorbar created by seaborn heatmap lives on another axes."""
    for a in fig.axes:
        if a is data_ax:
            continue
        a.tick_params(labelsize=tick_size)
        yl = a.get_ylabel()
        if yl:
            a.yaxis.label.set_fontsize(tick_size)
        xl = a.get_xlabel()
        if xl:
            a.xaxis.label.set_fontsize(tick_size)


def _savefig(fig: plt.Figure, path: Path, formats: Tuple[str, ...]) -> None:
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=_NATURE_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"  Saved {out}")


def _fallback_metric_order(present: Iterable[str]) -> List[str]:
    present_set = set(present)
    ordered: List[str] = []
    for m in ALL_KNOWN_METRICS:
        d = METRIC_DISPLAY.get(m, m)
        if d in present_set and d not in ordered:
            ordered.append(d)
    rest = sorted(present_set - set(ordered))
    ordered.extend(rest)
    return ordered


def _strip_axis_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clear index/column names so seaborn does not show 'dataset' / 'metric_display'."""
    out = df.copy()
    out.index.name = None
    out.columns.name = None
    return out


def _agg_add_neg_log(agg: pd.DataFrame) -> pd.DataFrame:
    out = agg.copy()
    out["neg_log_median_p"] = -np.log10(np.maximum(out["median_p"].astype(float), 1e-300))
    return out


def _add_r_obs_pds_columns(agg: pd.DataFrame, n_per_ds: Dict[str, int]) -> pd.DataFrame:
    """Map 1-based average rank R_obs to PDS score in [-1, 1] (see pds_core.rank_to_score).

    Adds ``median_R_obs_pds`` from ``median_R_obs`` and ``mean_R_obs_pds`` from ``mean_R_obs`` when
    those columns exist.
    """
    out = agg.copy()
    K_arr = out["dataset"].map(lambda d: int(n_per_ds.get(d, 1))).to_numpy()

    def _map_col(rank_col: str, out_col: str) -> None:
        r = out[rank_col].astype(float).to_numpy()
        pds = np.ones_like(r, dtype=float)
        mask = (K_arr > 1) & np.isfinite(r)
        pds[mask] = 1.0 - 2.0 * (r[mask] - 1.0) / (K_arr[mask].astype(float) - 1.0)
        out[out_col] = pds

    if "median_R_obs" in out.columns:
        _map_col("median_R_obs", "median_R_obs_pds")
    if "mean_R_obs" in out.columns:
        _map_col("mean_R_obs", "mean_R_obs_pds")
    return out


def _heatmap_axes_orders(
    agg: pd.DataFrame,
    *,
    order_key: str,
    row_agg: str,
    col_agg: str,
) -> Tuple[List[str], List[str]]:
    """Rows = datasets (best first / top), cols = metrics (best first / left).

    Row default ``min`` = pessimistic (weakest metric per dataset); weak datasets sink.
    """
    if order_key == "logp":
        agg = _agg_add_neg_log(agg)
        score_col = "neg_log_median_p"
    elif order_key == "fraction":
        score_col = "fraction_significant"
    elif order_key == "rob":
        if "median_R_obs_pds" not in agg.columns:
            raise ValueError("median_R_obs_pds column required for rob ordering")
        score_col = "median_R_obs_pds"
    elif order_key == "rob_mean":
        if "mean_R_obs_pds" not in agg.columns:
            raise ValueError("mean_R_obs_pds column required for rob_mean ordering")
        score_col = "mean_R_obs_pds"
    else:
        raise ValueError(order_key)

    def _agg_one(gb, how: str) -> pd.Series:
        s = gb[score_col]
        if how == "mean":
            return s.mean()
        if how == "median":
            return s.median()
        if how == "min":
            return s.min()
        raise ValueError(how)

    ds_scores = _agg_one(agg.groupby("dataset"), row_agg).sort_values(ascending=False)
    m_scores = _agg_one(agg.groupby("metric_display"), col_agg).sort_values(ascending=False)

    row_order = list(ds_scores.index)
    col_order = [c for c in m_scores.index if c in set(agg["metric_display"].unique())]
    for c in _fallback_metric_order(agg["metric_display"].unique()):
        if c not in col_order:
            col_order.append(c)

    return row_order, col_order


def _load_and_prepare(
    path: Path,
    *,
    datasets: Optional[Sequence[str]],
    metrics: Optional[Sequence[str]],
    bag_mode: Optional[str],
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "trial" in df.columns:
        trials = df["trial"].dropna().unique()
        if len(trials) > 1:
            print(
                f"WARNING: Multiple trials in {path}; using trial==0 only.",
                file=sys.stderr,
            )
            df = df[df["trial"] == 0].copy()
    if datasets is not None:
        df = df[df["dataset"].isin(datasets)].copy()
    if metrics is not None:
        df = df[df["metric"].isin(metrics)].copy()
    if bag_mode is not None:
        if "bag_mode" not in df.columns:
            raise SystemExit("CSV has no bag_mode column; remove --bag-mode.")
        df = df[df["bag_mode"] == bag_mode].copy()
    elif "bag_mode" in df.columns:
        modes = df["bag_mode"].dropna().unique()
        if len(modes) > 1:
            print(
                f"WARNING: Multiple bag_mode values {list(modes)}; pass --bag-mode to filter.",
                file=sys.stderr,
            )
    if df.empty:
        raise SystemExit("No rows left after filtering.")
    df["significant"] = df["significant"].astype(bool)
    if "metric" in df.columns and "metric_display" in df.columns:
        df = df.copy()
        df["metric_display"] = [
            METRIC_DISPLAY.get(str(m), md) for m, md in zip(df["metric"], df["metric_display"])
        ]
    return df


def _aggregate_per_dataset_metric(df: pd.DataFrame, p_col: str) -> pd.DataFrame:
    agg_kw: Dict[str, Tuple[str, str]] = {
        "fraction_significant": ("significant", "mean"),
        "median_p": (p_col, "median"),
        "mean_p": (p_col, "mean"),
        "std_p": (p_col, "std"),
    }
    if "R_obs" in df.columns:
        agg_kw["median_R_obs"] = ("R_obs", "median")
        agg_kw["mean_R_obs"] = ("R_obs", "mean")
        agg_kw["std_R_obs"] = ("R_obs", "std")
    return df.groupby(["dataset", "metric", "metric_display"], as_index=False).agg(
        **agg_kw
    )


def _marginal_order_label(row_agg: str, col_agg: str, key_lab: str) -> str:
    return f"rows by {row_agg}({key_lab}), cols by {col_agg}({key_lab})"


def _mean_std_annotation(mean_df: pd.DataFrame, std_df: pd.DataFrame) -> pd.DataFrame:
    """String matrix for heatmap cells: mean on first line, ±std on second."""
    out = pd.DataFrame(index=mean_df.index, columns=mean_df.columns, dtype=object)
    for i in mean_df.index:
        for c in mean_df.columns:
            m = mean_df.loc[i, c]
            s = std_df.loc[i, c]
            if pd.isna(m):
                out.loc[i, c] = ""
            elif pd.isna(s):
                out.loc[i, c] = f"{float(m):.3f}"
            else:
                out.loc[i, c] = f"{float(m):.3f}\n±{float(s):.3f}"
    return out


def _style_mean_p_value_colorbar(
    fig: plt.Figure,
    main_ax: plt.Axes,
    *,
    tick_label_size: float,
    ylabel: str,
) -> None:
    """Keep heatmap colors in −log10(mean p) space; show raw p on colorbar tick labels."""
    others = [a for a in fig.axes if a is not main_ax]
    if not others:
        return
    cax = others[0]
    if not main_ax.collections:
        return
    vmin, vmax = main_ax.collections[0].get_clim()
    candidates = [
        0.0,
        0.30103,
        0.5,
        1.0,
        1.30103,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        5.0,
        6.0,
    ]
    ticks = [t for t in candidates if vmin - 1e-12 <= t <= vmax + 1e-12]
    if len(ticks) < 2:
        ticks = np.linspace(vmin, vmax, num=min(6, max(2, int(round(vmax - vmin)) + 1)))
    cax.set_yticks(ticks)

    def _raw_p_label(y: float, _pos: int) -> str:
        if y <= 0:
            return "1"
        p = float(10 ** (-y))
        if p >= 0.01:
            return f"{p:.3g}"
        return f"{p:.1e}"

    cax.yaxis.set_major_formatter(mticker.FuncFormatter(_raw_p_label))
    cax.set_ylabel(ylabel)
    cax.tick_params(labelsize=tick_label_size)
    cax.yaxis.label.set_fontsize(tick_label_size)


def plot_value_heatmaps(
    agg: pd.DataFrame,
    *,
    out_base: Path,
    formats: Tuple[str, ...],
    p_col: str,
    order_key: str,
    row_agg: str,
    col_agg: str,
    alpha: float,
    panel_prefix: str,
    n_per_ds: Optional[Dict[str, int]] = None,
    skip_r_obs_heatmaps: bool = False,
) -> None:
    row_order_frac, col_order_frac = _heatmap_axes_orders(
        agg, order_key=order_key, row_agg=row_agg, col_agg=col_agg
    )

    row_order_p, col_order_p = _heatmap_axes_orders(
        agg, order_key="logp", row_agg=row_agg, col_agg=col_agg
    )

    def _pivot(
        col: str,
        row_order: List[str],
        col_order: List[str],
        *,
        data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        a = agg if data is None else data
        p = a.pivot(index="dataset", columns="metric_display", values=col)
        p = p.reindex(index=row_order, columns=[c for c in col_order if c in p.columns])
        return p

    def _pivot_metrics_on_y(
        col: str,
        dataset_order: List[str],
        metric_order: List[str],
        *,
        data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Rows = metric_display, cols = dataset (transpose of _pivot)."""
        a = agg if data is None else data
        p = a.pivot(index="metric_display", columns="dataset", values=col)
        p = p.reindex(
            index=[m for m in metric_order if m in p.index],
            columns=[d for d in dataset_order if d in p.columns],
        )
        return p

    frac = _strip_axis_names(
        _pivot_metrics_on_y("fraction_significant", row_order_frac, col_order_frac)
    )
    med = _pivot_metrics_on_y("median_p", row_order_p, col_order_p)
    med_log = _strip_axis_names(-np.log10(np.maximum(med.astype(float), 1e-300)))

    mean_p = _pivot_metrics_on_y("mean_p", row_order_p, col_order_p)
    std_p = _pivot_metrics_on_y("std_p", row_order_p, col_order_p)
    mean_log = _strip_axis_names(-np.log10(np.maximum(mean_p.astype(float), 1e-300)))
    mean_annot = _mean_std_annotation(mean_p, std_p)

    w = max(12.0, 0.65 * len(frac.columns)) * _FIG_SIZE_SCALE * _HEATMAP_FIG_SCALE
    h = max(6.0, 0.62 * len(frac.index)) * _FIG_SIZE_SCALE * _HEATMAP_FIG_SCALE
    annot_kw = {"size": _HEATMAP_CELL_ANNOT_SIZE}
    annot_kw_mean = {"size": _HEATMAP_CELL_ANNOT_MEAN_SIZE}
    ax_tick = _HEATMAP_AXIS_TICK_SIZE

    pk = _p_value_kind_label(p_col)
    alpha_disp = f"{float(alpha):g}"
    frac_cbar_label = f"Fraction of significant perturbations (α={alpha_disp})"
    median_cbar_label = f"−log10(median {pk})"
    if p_col == "adj_pval":
        mean_bold = f"{panel_prefix} (Mean adjusted p-value)"
        mean_cbar_label = "Mean adjusted p-value"
    else:
        mean_bold = f"{panel_prefix} (Mean {_p_value_kind_label(p_col)})"
        mean_cbar_label = f"Mean {_p_value_kind_label(p_col)}"

    fig1, ax1 = plt.subplots(figsize=(w, h))
    sns.heatmap(
        frac,
        ax=ax1,
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws=annot_kw,
        cbar_kws={"label": frac_cbar_label},
    )
    _style_heatmap_axis_labels(ax1, tick_size=ax_tick)
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    _style_seaborn_cbar(fig1, ax1, tick_size=ax_tick)
    fig1.tight_layout()
    _figure_bold_title(fig1, bold_line=f"{panel_prefix} (Significant perturbation fraction)")
    _savefig(fig1, out_base.parent / f"{out_base.name}_heatmap_frac_sig", formats)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(w, h))
    sns.heatmap(
        med_log,
        ax=ax2,
        cmap="viridis",
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws=annot_kw,
        cbar_kws={"label": median_cbar_label},
    )
    _style_heatmap_axis_labels(ax2, tick_size=ax_tick)
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    _style_seaborn_cbar(fig2, ax2, tick_size=ax_tick)
    fig2.tight_layout()
    med_title_scope = "adjusted p-values" if p_col == "adj_pval" else "raw p-values"
    _figure_bold_title(fig2, bold_line=f"{panel_prefix} ({med_title_scope})")
    _savefig(fig2, out_base.parent / f"{out_base.name}_heatmap_median_p", formats)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(w, h))
    sns.heatmap(
        mean_log,
        ax=ax3,
        cmap="viridis",
        linewidths=0.5,
        linecolor="white",
        annot=mean_annot,
        fmt="",
        annot_kws=annot_kw_mean,
        cbar_kws={"label": ""},
    )
    _style_mean_p_value_colorbar(
        fig3, ax3, tick_label_size=ax_tick, ylabel=mean_cbar_label
    )
    _style_heatmap_axis_labels(ax3, tick_size=ax_tick)
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    fig3.tight_layout()
    _figure_bold_title(fig3, bold_line=mean_bold)
    _savefig(fig3, out_base.parent / f"{out_base.name}_heatmap_mean_p", formats)
    plt.close(fig3)

    if skip_r_obs_heatmaps:
        return
    if "median_R_obs" not in agg.columns or not n_per_ds:
        return

    agg_rob = _add_r_obs_pds_columns(agg, n_per_ds)
    row_order_r, col_order_r = _heatmap_axes_orders(
        agg_rob, order_key="rob", row_agg=row_agg, col_agg=col_agg
    )
    med_pds = _pivot("median_R_obs_pds", row_order_r, col_order_r, data=agg_rob)
    med_pds = _strip_axis_names(med_pds.astype(float))

    fig4, ax4 = plt.subplots(figsize=(w, h))
    cmap_r = sns.color_palette("coolwarm", as_cmap=True).copy()
    cmap_r.set_bad("#ffffff")
    sns.heatmap(
        med_pds,
        ax=ax4,
        cmap=cmap_r,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws=annot_kw,
        cbar_kws={"label": "Discrimination score from median R_obs [−1, 1]"},
    )
    _style_heatmap_axis_labels(ax4, tick_size=ax_tick)
    ax4.set_xlabel("")
    ax4.set_ylabel("")
    _style_seaborn_cbar(fig4, ax4, tick_size=ax_tick)
    fig4.tight_layout()
    _figure_bold_title(fig4, bold_line=f"{panel_prefix} (Median R_obs score)")
    _savefig(fig4, out_base.parent / f"{out_base.name}_heatmap_Robs_median", formats)
    plt.close(fig4)

    if "mean_R_obs_pds" not in agg_rob.columns:
        return

    row_order_rm, col_order_rm = _heatmap_axes_orders(
        agg_rob, order_key="rob_mean", row_agg=row_agg, col_agg=col_agg
    )
    mean_pds = _pivot("mean_R_obs_pds", row_order_rm, col_order_rm, data=agg_rob)
    mean_pds = _strip_axis_names(mean_pds.astype(float))

    fig5, ax5 = plt.subplots(figsize=(w, h))
    cmap_rm = sns.color_palette("coolwarm", as_cmap=True).copy()
    cmap_rm.set_bad("#ffffff")
    sns.heatmap(
        mean_pds,
        ax=ax5,
        cmap=cmap_rm,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws=annot_kw,
        cbar_kws={"label": "Discrimination score from mean R_obs [−1, 1]"},
    )
    _style_heatmap_axis_labels(ax5, tick_size=ax_tick)
    ax5.set_xlabel("")
    ax5.set_ylabel("")
    _style_seaborn_cbar(fig5, ax5, tick_size=ax_tick)
    fig5.tight_layout()
    _figure_bold_title(fig5, bold_line=f"{panel_prefix} (Mean R_obs score)")
    _savefig(fig5, out_base.parent / f"{out_base.name}_heatmap_Robs_mean", formats)
    plt.close(fig5)


def _spearman_vec_pair(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman ρ for two aligned vectors (e.g. p-values); NaN if undefined.

    Identical vectors → 1.0. If either is constant and not equal to the other → NaN.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape == y.shape and np.allclose(x, y, rtol=1e-9, atol=0.0):
        return 1.0
    if (np.nanstd(x) == 0) or (np.nanstd(y) == 0):
        return float("nan")
    rho, _ = spearmanr(x, y)
    return float(rho)


def _pairwise_column_spearman(df_ds: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pairwise Spearman between metric columns (aligned per perturbation)."""
    wide = df_ds.pivot_table(
        index="perturbation",
        columns="metric_display",
        values=value_col,
        aggfunc="first",
    )
    wide = wide.dropna(axis=0, how="any")
    cols = list(wide.columns)
    n = len(cols)
    if n < 2:
        return pd.DataFrame(np.nan, index=cols, columns=cols)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a = wide.iloc[:, i].to_numpy()
            b = wide.iloc[:, j].to_numpy()
            rho = _spearman_vec_pair(a, b)
            mat[i, j] = mat[j, i] = rho
    return pd.DataFrame(mat, index=cols, columns=cols)


def plot_correlation_panels(
    df: pd.DataFrame,
    agg: pd.DataFrame,
    *,
    out_base: Path,
    formats: Tuple[str, ...],
    value_col: str,
    order_key: str,
    row_agg: str,
    col_agg: str,
    suptitle: str,
    outfile_suffix: str,
) -> None:
    ds_order, metric_order = _heatmap_axes_orders(
        agg, order_key=order_key, row_agg=row_agg, col_agg=col_agg
    )

    n = len(ds_order)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            _GRID_PANEL_W_PER_COL * _FIG_SIZE_SCALE * _GRID_FIG_SCALE * ncols,
            _GRID_PANEL_H_PER_ROW * _FIG_SIZE_SCALE * _GRID_FIG_SCALE * nrows,
        ),
        squeeze=False,
    )
    g_tick = _GRID_AXIS_TICK_SIZE
    for idx, ds in enumerate(ds_order):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sub = df[df["dataset"] == ds]
        corr = _pairwise_column_spearman(sub, value_col)
        order = [m for m in metric_order if m in corr.index and m in corr.columns]
        if not order:
            ax.set_visible(False)
            continue
        corr = _strip_axis_names(corr.reindex(index=order, columns=order))
        cmap = sns.color_palette("coolwarm", as_cmap=True).copy()
        cmap.set_bad(_SPEARMAN_BAD_COLOR)
        sns.heatmap(
            corr,
            ax=ax,
            vmin=-1,
            vmax=1,
            cmap=cmap,
            center=0,
            square=True,
            linewidths=0.3,
            linecolor="white",
            cbar=idx == 0,
            cbar_kws={"shrink": 0.8, "label": "Spearman ρ"},
        )
        ax.set_title(ds, fontsize=_GRID_PANEL_TITLE_SIZE)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=g_tick)
        plt.setp(ax.get_yticklabels(), fontsize=g_tick)
        if idx == 0 and len(fig.axes) > 1:
            cb = fig.axes[-1]
            if cb is not ax:
                cb.tick_params(labelsize=g_tick)
                if cb.get_ylabel():
                    cb.yaxis.label.set_fontsize(g_tick)
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle(suptitle, y=1.02, fontsize=_NATURE_TITLE_SIZE)
    fig.tight_layout()
    _savefig(fig, out_base.parent / f"{out_base.name}_heatmap_{outfile_suffix}", formats)
    plt.close(fig)


def _pairwise_set_overlap(
    df_ds: pd.DataFrame,
    *,
    significant_sets: bool,
    mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pairwise overlap coefficient, intersection fraction, and union fraction.

    For each metric, S_i = perturbations with FDR significance (or its complement).
    N is the number of perturbations retained after dropping rows with any missing metric.

    Returns:
        coeff: overlap coefficient matrix (color).
            - jaccard: J(i,j) = |S_i ∩ S_j| / |S_i ∪ S_j|. Empty union → NaN.
            - szymkiewicz_simpson: SS(i,j) = |S_i ∩ S_j| / min(|S_i|, |S_j|). min size 0 → NaN.
            Diagonal 1 when the corresponding set is nonempty; else NaN.
        inter_frac: |S_i ∩ S_j| / N for every cell (diagonal = |S_i| / N).
        union_frac: |S_i ∪ S_j| / N for every cell (diagonal equals inter_frac).
    """
    if mode not in ("jaccard", "szymkiewicz_simpson"):
        raise ValueError(f"mode must be jaccard or szymkiewicz_simpson, got {mode!r}")
    wide = df_ds.pivot_table(
        index="perturbation",
        columns="metric_display",
        values="significant",
        aggfunc="first",
    )
    wide = wide.dropna(axis=0, how="any")
    cols = list(wide.columns)
    n = len(cols)
    if n < 2:
        empty = pd.DataFrame(np.nan, index=cols, columns=cols)
        return empty, empty, empty
    idx = wide.index
    n_pert = int(len(idx))
    sets: List[Set[object]] = []
    for i in range(n):
        si = wide.iloc[:, i].astype(bool).to_numpy()
        if significant_sets:
            sets.append(set(idx[si]))
        else:
            sets.append(set(idx[~si]))

    mat = np.full((n, n), np.nan)
    inter_frac = np.full((n, n), np.nan)
    union_frac = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(i, n):
            inter_sz = len(sets[i] & sets[j])
            union_sz = len(sets[i] | sets[j])
            if n_pert > 0:
                inter_frac[i, j] = inter_frac[j, i] = float(inter_sz) / float(n_pert)
                union_frac[i, j] = union_frac[j, i] = float(union_sz) / float(n_pert)
            if i == j:
                ni = len(sets[i])
                mat[i, i] = float("nan") if ni == 0 else 1.0
                continue
            if mode == "jaccard":
                uni = sets[i] | sets[j]
                coeff = float("nan") if not uni else float(inter_sz) / float(len(uni))
            else:
                denom = min(len(sets[i]), len(sets[j]))
                coeff = float("nan") if denom == 0 else float(inter_sz) / float(denom)
            mat[i, j] = mat[j, i] = coeff

    coeff_df = pd.DataFrame(mat, index=cols, columns=cols)
    inter_df = pd.DataFrame(inter_frac, index=cols, columns=cols)
    union_df = pd.DataFrame(union_frac, index=cols, columns=cols)
    return coeff_df, inter_df, union_df


def _overlap_triple_annotations(
    inter: pd.DataFrame,
    union: pd.DataFrame,
    coeff: pd.DataFrame,
) -> pd.DataFrame:
    """Three lines: ∩% of N, ∪% of N, overlap index (Jaccard or SS, same as color scale)."""
    out = pd.DataFrame(index=inter.index, columns=inter.columns, dtype=object)
    for ri in inter.index:
        for ci in inter.columns:
            a = inter.loc[ri, ci]
            b = union.loc[ri, ci]
            if (
                pd.isna(a)
                or not np.isfinite(float(a))
                or pd.isna(b)
                or not np.isfinite(float(b))
            ):
                out.loc[ri, ci] = ""
                continue
            line1 = f"{100.0 * float(a):.1f}%"
            line2 = f"{100.0 * float(b):.1f}%"
            k = coeff.loc[ri, ci]
            if pd.isna(k) or not np.isfinite(float(k)):
                line3 = "—"
            else:
                line3 = f"{float(k):.2f}"
            out.loc[ri, ci] = f"{line1}\n{line2}\n{line3}"
    return out


def plot_set_overlap_panels(
    df: pd.DataFrame,
    agg: pd.DataFrame,
    *,
    out_base: Path,
    formats: Tuple[str, ...],
    order_key: str,
    row_agg: str,
    col_agg: str,
    significant_sets: bool,
    alpha: float,
    outfile_suffix: str,
    suptitle: str,
    overlap_mode: str,
    cbar_label: str,
) -> None:
    ds_order, metric_order = _heatmap_axes_orders(
        agg, order_key=order_key, row_agg=row_agg, col_agg=col_agg
    )

    n = len(ds_order)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            _GRID_PANEL_W_PER_COL * _FIG_SIZE_SCALE * _GRID_FIG_SCALE * ncols,
            _GRID_PANEL_H_PER_ROW * _FIG_SIZE_SCALE * _GRID_FIG_SCALE * nrows,
        ),
        squeeze=False,
    )
    g_tick = _GRID_AXIS_TICK_SIZE
    for idx, ds in enumerate(ds_order):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sub = df[df["dataset"] == ds]
        ov, inter_frac, union_frac = _pairwise_set_overlap(
            sub, significant_sets=significant_sets, mode=overlap_mode
        )
        order = [m for m in metric_order if m in ov.index and m in ov.columns]
        if not order:
            ax.set_visible(False)
            continue
        ov = _strip_axis_names(ov.reindex(index=order, columns=order))
        inter_frac = _strip_axis_names(inter_frac.reindex(index=order, columns=order))
        union_frac = _strip_axis_names(union_frac.reindex(index=order, columns=order))
        annot = _overlap_triple_annotations(inter_frac, union_frac, ov)
        cmap = sns.color_palette("viridis", as_cmap=True).copy()
        cmap.set_bad("#ffffff")
        sns.heatmap(
            ov,
            ax=ax,
            vmin=0,
            vmax=1,
            cmap=cmap,
            square=True,
            linewidths=0.3,
            linecolor="white",
            annot=annot,
            fmt="",
            annot_kws={"size": _OVERLAP_ANNOT_FONT_PT},
            cbar=idx == 0,
            cbar_kws={"shrink": 0.8, "label": cbar_label},
        )
        ax.set_title(ds, fontsize=_GRID_PANEL_TITLE_SIZE)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=g_tick)
        plt.setp(ax.get_yticklabels(), fontsize=g_tick)
        if idx == 0 and len(fig.axes) > 1:
            cb = fig.axes[-1]
            if cb is not ax:
                cb.tick_params(labelsize=g_tick)
                if cb.get_ylabel():
                    cb.yaxis.label.set_fontsize(g_tick)
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle(suptitle, y=1.02, fontsize=_NATURE_TITLE_SIZE)
    fig.tight_layout()
    _savefig(fig, out_base.parent / f"{out_base.name}_heatmap_{outfile_suffix}", formats)
    plt.close(fig)


def parse_plots(s: str) -> List[str]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    allowed = {
        "heatmaps",
        "pvalue-correlation",
        "rank-correlation",
        "jaccard",
        "szymkiewicz-simpson",
        "sig-correlation",
    }
    bad = set(parts) - allowed
    if bad:
        raise SystemExit(f"Unknown --plots entries: {bad}; allowed: {allowed}")
    out: List[str] = []
    for p in parts:
        if p == "sig-correlation":
            out.append("pvalue-correlation")
        else:
            out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("analyses/perturbation_discrimination/results/pds_test/pds_test_detail.csv"),
        help="Path to pds_test_detail.csv or bds_test_detail.csv (BDS: corrected_pval → adj_pval)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input>/../figures)",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="pds_detail",
        help="Basename prefix for output files",
    )
    parser.add_argument(
        "--panel-title-prefix",
        type=str,
        default=None,
        help=(
            "Bold title prefix (e.g. 'PDS Testing'). Default: infer from input path "
            "(pds_test → PDS Testing, bds_test → BDS Testing)."
        ),
    )
    parser.add_argument(
        "--plots",
        type=str,
        default="heatmaps,pvalue-correlation,rank-correlation",
        help=(
            "Comma-separated: heatmaps, pvalue-correlation, rank-correlation, jaccard, "
            "szymkiewicz-simpson (alias: sig-correlation)"
        ),
    )
    parser.add_argument("--datasets", nargs="*", default=None, help="Filter datasets")
    parser.add_argument("--metrics", nargs="*", default=None, help="Filter metric names (internal)")
    parser.add_argument("--bag-mode", default=None, help="Filter bag_mode (e.g. half)")
    parser.add_argument(
        "--p-value",
        choices=("adj_pval", "raw_pval"),
        default="adj_pval",
        help="P-value column for median heatmap and p-value Spearman panels",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png,pdf",
        help="Comma-separated: png, pdf, ...",
    )
    parser.add_argument(
        "--heatmap-order-key",
        choices=("fraction", "logp"),
        default="fraction",
        help="Cell score for ordering fraction heatmap and correlation/overlap panels; median & mean p heatmaps always use −log10(median p) marginal order",
    )
    parser.add_argument(
        "--heatmap-row-agg",
        choices=("mean", "median", "min"),
        default="mean",
        help="How to aggregate scores across metrics per dataset (min = pessimistic; weak datasets at bottom)",
    )
    parser.add_argument(
        "--heatmap-col-agg",
        choices=("mean", "median", "min"),
        default="mean",
        help="How to aggregate scores across datasets per metric",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="FDR threshold: fraction heatmap uses (adj_pval < α); recomputed from CSV",
    )
    parser.add_argument(
        "--skip-r-obs-heatmaps",
        action="store_true",
        help="Do not write median/mean R_obs score heatmaps (still allows rank-correlation if in --plots).",
    )
    args = parser.parse_args()

    p_col = args.p_value
    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    plots = parse_plots(args.plots)

    inp = args.input.resolve()
    if not inp.is_file():
        raise SystemExit(f"Missing input file: {inp}")

    out_dir = args.out_dir.resolve() if args.out_dir else (inp.parent / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / args.out_prefix

    _apply_nature_rc()

    df = _load_and_prepare(inp, datasets=args.datasets, metrics=args.metrics, bag_mode=args.bag_mode)
    if "adj_pval" not in df.columns:
        if "corrected_pval" in df.columns:
            df = df.copy()
            df["adj_pval"] = df["corrected_pval"]
        else:
            raise SystemExit(
                "Input CSV must contain adj_pval (or corrected_pval for BDS) for --alpha significance."
            )
    df["significant"] = df["adj_pval"].astype(float) < float(args.alpha)
    agg = _aggregate_per_dataset_metric(df, p_col)
    n_per_ds: Dict[str, int] = (
        df.groupby("dataset")["perturbation"].nunique().astype(int).to_dict()
    )

    print(f"Loaded {len(df)} rows from {inp} (significant := adj_pval < {args.alpha})")
    print(f"Writing to {out_dir} (prefix={args.out_prefix})")

    panel_prefix = args.panel_title_prefix or _infer_panel_prefix(inp)

    if "heatmaps" in plots:
        plot_value_heatmaps(
            agg,
            out_base=out_base,
            formats=formats,
            p_col=p_col,
            order_key=args.heatmap_order_key,
            row_agg=args.heatmap_row_agg,
            col_agg=args.heatmap_col_agg,
            alpha=args.alpha,
            panel_prefix=panel_prefix,
            n_per_ds=n_per_ds,
            skip_r_obs_heatmaps=args.skip_r_obs_heatmaps,
        )
    if "pvalue-correlation" in plots:
        plot_correlation_panels(
            df,
            agg,
            out_base=out_base,
            formats=formats,
            value_col=p_col,
            order_key=args.heatmap_order_key,
            row_agg=args.heatmap_row_agg,
            col_agg=args.heatmap_col_agg,
            suptitle=(
                f"Pairwise Spearman correlation of per-perturbation {p_col} across metrics\n"
                "(blank = undefined ρ when a column has no variation)"
            ),
            outfile_suffix="pvalue_spearman",
        )
    if "rank-correlation" in plots:
        plot_correlation_panels(
            df,
            agg,
            out_base=out_base,
            formats=formats,
            value_col="R_obs",
            order_key=args.heatmap_order_key,
            row_agg=args.heatmap_row_agg,
            col_agg=args.heatmap_col_agg,
            suptitle=(
                "Pairwise Spearman correlation of observed rank R_obs across metrics\n"
                "(rank of d(Q_p,R_p) among {d(Q_p,R_q)}; lower = sharper; blank = no variation)"
            ),
            outfile_suffix="Robs_spearman",
        )
    if "jaccard" in plots:
        plot_set_overlap_panels(
            df,
            agg,
            out_base=out_base,
            formats=formats,
            order_key=args.heatmap_order_key,
            row_agg=args.heatmap_row_agg,
            col_agg=args.heatmap_col_agg,
            significant_sets=True,
            alpha=args.alpha,
            outfile_suffix="jaccard_significant",
            suptitle=(
                f"Jaccard overlap of FDR-significant perturbations (adj_pval < {args.alpha})\n"
                "J(i,j) = |S_i ∩ S_j| / |S_i ∪ S_j| where S_i = perturbations significant under metric i"
            ),
            overlap_mode="jaccard",
            cbar_label="Jaccard",
        )
        plot_set_overlap_panels(
            df,
            agg,
            out_base=out_base,
            formats=formats,
            order_key=args.heatmap_order_key,
            row_agg=args.heatmap_row_agg,
            col_agg=args.heatmap_col_agg,
            significant_sets=False,
            alpha=args.alpha,
            outfile_suffix="jaccard_insignificant",
            suptitle=(
                f"Jaccard overlap of non-significant perturbations (adj_pval ≥ {args.alpha})\n"
                "J(i,j) = |U_i ∩ U_j| / |U_i ∪ U_j| where U_i = perturbations not significant under metric i"
            ),
            overlap_mode="jaccard",
            cbar_label="Jaccard",
        )
    if "szymkiewicz-simpson" in plots:
        plot_set_overlap_panels(
            df,
            agg,
            out_base=out_base,
            formats=formats,
            order_key=args.heatmap_order_key,
            row_agg=args.heatmap_row_agg,
            col_agg=args.heatmap_col_agg,
            significant_sets=True,
            alpha=args.alpha,
            outfile_suffix="szymkiewicz_simpson_significant",
            suptitle=(
                f"Szymkiewicz–Simpson overlap of FDR-significant perturbations (adj_pval < {args.alpha})\n"
                "SS(i,j) = |S_i ∩ S_j| / min(|S_i|,|S_j|) where S_i = perturbations significant under metric i"
            ),
            overlap_mode="szymkiewicz_simpson",
            cbar_label="Szymkiewicz–Simpson",
        )
        plot_set_overlap_panels(
            df,
            agg,
            out_base=out_base,
            formats=formats,
            order_key=args.heatmap_order_key,
            row_agg=args.heatmap_row_agg,
            col_agg=args.heatmap_col_agg,
            significant_sets=False,
            alpha=args.alpha,
            outfile_suffix="szymkiewicz_simpson_insignificant",
            suptitle=(
                f"Szymkiewicz–Simpson overlap of non-significant perturbations (adj_pval ≥ {args.alpha})\n"
                "SS(i,j) = |U_i ∩ U_j| / min(|U_i|,|U_j|) where U_i = perturbations not significant under metric i"
            ),
            overlap_mode="szymkiewicz_simpson",
            cbar_label="Szymkiewicz–Simpson",
        )


if __name__ == "__main__":
    main()
