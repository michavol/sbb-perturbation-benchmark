#!/usr/bin/env python3
"""Signal dilution curves for inter-perturbation discrimination.

Modes:
- single_cell_pds_rank_curve: one cell per half, PDS score vs top-k ranked genes (--gene-ranking: pds or deg)
- bag_size_curve: bag-vs-bag PDS score vs bag size
- effective_genes_curve: single-cell PDS vs target effective gene number (weight sweep)
- bag_rank_curve: bag PDS score vs top-k ranked genes, one plot per bag size (--bag-sizes)
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mpl_colors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyses.perturbation_discrimination import metric_utils as base

METRIC_DISPLAY_NAMES: Dict[str, str] = {
    "mse": "MSE",
    "wmse": "Weighted MSE",
    "r2_delta": "R² Delta",
    "w_r2_delta": "Weighted R² Delta",
    "pearson_delta": "Pearson Delta",
    "w_pearson_delta": "Weighted Pearson Delta",
    "cosine_sim": "Cosine Distance",
    "w_cosine_sim": "Weighted Cosine Distance",
    "top_n_deg_mse": "TopN DEG MSE",
    "top_n_deg_r2": "TopN DEG R² Delta",
    "top_n_deg_pearson": "TopN DEG Pearson Delta",
    "top20_mse": "TopN DEG MSE (per pert.)",
    "top20_pearson_dp": "TopN DEG Pearson Delta (per pert.)",
    "energy_distance": "Energy Distance",
    "weighted_energy_distance": "Weighted Energy Distance",
    "std_energy_distance": "Standardized Energy Distance",
    "w_std_energy_distance": "Weighted Std. Energy Distance",
}

NATURE_FONT = "Arial"
NATURE_AXIS_LABEL_SIZE = 7
NATURE_TICK_SIZE = 6
NATURE_LEGEND_SIZE = 6
NATURE_TITLE_SIZE = 7
NATURE_FIG_SIZE = (3.5, 2.35)  # ~89 mm wide
# Effective-genes curve: wider canvas for legend + global vs per-pert series
EFFECTIVE_GENES_CURVE_FIG_SIZE = (6.8, 2.35)
NATURE_HEATMAP_FIG_SIZE = (7.2, 3.8)  # two-up style for 2x4 grid
NATURE_DPI = 450

METRIC_PAIR_BASE = {
    "mse": "mse",
    "wmse": "mse",
    "top_n_deg_mse": "top_n_deg_mse",
    "top20_mse": "top20_mse",
    "top20_pearson_dp": "top20_pearson_dp",
    "r2_delta": "r2_delta",
    "w_r2_delta": "r2_delta",
    "top_n_deg_r2": "top_n_deg_r2",
    "pearson_delta": "pearson_delta",
    "w_pearson_delta": "pearson_delta",
    "top_n_deg_pearson": "top_n_deg_pearson",
    "energy_distance": "energy_distance",
    "weighted_energy_distance": "energy_distance",
    "std_energy_distance": "std_energy_distance",
    "w_std_energy_distance": "std_energy_distance",
    "cosine_sim": "cosine_sim",
    "w_cosine_sim": "cosine_sim",
}
BASE_PAIR_COLORS = {
    "mse": "#1f77b4",
    "r2_delta": "#2ca02c",
    "pearson_delta": "#ff7f0e",
    "energy_distance": "#9467bd",
    "std_energy_distance": "#8c564b",
    "cosine_sim": "#17becf",
    # top-N DEG variants: same hue family, dotted line
    "top_n_deg_mse": "#1f77b4",
    "top_n_deg_r2": "#2ca02c",
    "top_n_deg_pearson": "#ff7f0e",
    "top20_mse": "#42a5f5",
    "top20_pearson_dp": "#ffb74d",
}
PAIR_COLORS_EXPLICIT = {
    "mse": {"unweighted": "#8ec5ff", "weighted": "#0b3d91"},
    "r2_delta": {"unweighted": "#8fd694", "weighted": "#1b7f2a"},
    "pearson_delta": {"unweighted": "#ffc48a", "weighted": "#c45a00"},
    # Global TopN DEG vs per-pert top-N: paired blue / paired orange hues
    "top_n_deg_mse": {"deg": "#1565c0"},
    "top_n_deg_r2": {"deg": "#00b050"},
    "top_n_deg_pearson": {"deg": "#ef6c00"},
    "top20_mse": {"deg": "#42a5f5"},
    "top20_pearson_dp": {"deg": "#ffb74d"},
}


def _metric_label(metric: str) -> str:
    """Return a human-readable display name for a metric."""
    return METRIC_DISPLAY_NAMES.get(metric, metric)


def _apply_nature_rc() -> None:
    """Set Matplotlib defaults aligned with Nature plotting conventions."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [NATURE_FONT, "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": NATURE_AXIS_LABEL_SIZE,
            "axes.titlesize": NATURE_TITLE_SIZE,
            "xtick.labelsize": NATURE_TICK_SIZE,
            "ytick.labelsize": NATURE_TICK_SIZE,
            "legend.fontsize": NATURE_LEGEND_SIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _metric_plot_style(metric: str) -> Tuple[str, str]:
    """Return (color, linestyle) with related hues for weighted variants."""
    base = METRIC_PAIR_BASE.get(metric, metric)
    is_top_deg = metric.startswith("top_n_deg_") or metric.startswith("top20_")
    is_weighted = metric.startswith("w") or metric.startswith("weighted_")
    if is_top_deg and base in PAIR_COLORS_EXPLICIT:
        return PAIR_COLORS_EXPLICIT[base]["deg"], "-"
    if base in PAIR_COLORS_EXPLICIT:
        tone = "weighted" if is_weighted else "unweighted"
        return PAIR_COLORS_EXPLICIT[base][tone], "-"
    base_color = BASE_PAIR_COLORS.get(base, "#4c4c4c")
    rgb = np.asarray(mpl_colors.to_rgb(base_color), dtype=float)
    if is_weighted:
        darker = np.clip(0.28 * rgb, 0.0, 1.0)
        return mpl_colors.to_hex(darker), "-"
    lighter = np.clip(0.6 * rgb + 0.4 * np.ones_like(rgb), 0.0, 1.0)
    return mpl_colors.to_hex(lighter), "-"


def _metric_marker(metric: str) -> str:
    """Use distinct markers to reduce visual overlap between curves."""
    return "o"


def _pretty_pert_label(label: str) -> str:
    """Make perturbation labels cleaner for compact heatmap axes."""
    text = str(label).strip()
    text = text.replace("__", "+").replace("|", "+").replace(";", "+")
    text = text.replace("_", "+")
    text = re.sub(r"\++", "+", text)
    text = re.sub(r"\s*\+\s*", " + ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _stacked_pert_label(label: str) -> str:
    """Render combos on multiple lines without separator symbols."""
    pretty = _pretty_pert_label(label)
    parts = [part.strip() for part in pretty.split("+") if part.strip()]
    if len(parts) <= 1:
        return pretty
    return "\n".join(parts)


def _ordered_metrics(metrics: Sequence[str]) -> List[str]:
    """Order unweighted/weighted metric pairs adjacently for readability."""
    preferred = [
        "mse", "wmse", "top_n_deg_mse", "top20_mse",
        "energy_distance", "weighted_energy_distance",
        "r2_delta", "w_r2_delta", "top_n_deg_r2",
        "pearson_delta", "w_pearson_delta", "top_n_deg_pearson", "top20_pearson_dp",
        "std_energy_distance", "w_std_energy_distance",
        "cosine_sim", "w_cosine_sim",
    ]
    metric_set = set(metrics)
    ordered = [m for m in preferred if m in metric_set]
    ordered.extend([m for m in metrics if m not in set(ordered)])
    return ordered


@dataclass
class TrialResult:
    metric: str
    x_value: int
    trial: int
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run signal dilution curve analyses.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("single_cell_pds_rank_curve", "bag_size_curve", "effective_genes_curve", "bag_rank_curve", "snr_curve", "snr_rank_curve", "snr_bag_rank_curve"),
        required=True,
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument(
        "--metrics",
        type=str,
        default="mse,wmse",
        help=(
            "Supported metrics: mse, wmse, r2_delta, w_r2_delta, pearson_delta, w_pearson_delta, "
            "cosine_sim, w_cosine_sim, top_n_deg_mse, top_n_deg_r2, top_n_deg_pearson, "
            "top20_mse, top20_pearson_dp, "
            "energy_distance, weighted_energy_distance, "
            "std_energy_distance, w_std_energy_distance"
        ),
    )
    parser.add_argument("--max-perturbations", type=int, default=None)
    parser.add_argument(
        "--perturbation-sort",
        type=str,
        choices=("top", "bottom", "pds_bottom", "pds_top"),
        default="top",
        help=(
            "When --max-perturbations is set, how to rank perturbations: "
            "'top' selects the N with the MOST cells (default), "
            "'bottom' selects the N with the LEAST cells, "
            "'pds_bottom' selects the N hardest-to-discriminate (lowest mean per-gene PDS from pds_full), "
            "'pds_top' selects the N easiest-to-discriminate (highest mean per-gene PDS)."
        ),
    )
    parser.add_argument("--cells-per-perturbation", type=int, default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--control-label", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument(
        "--trial-cell-budget",
        type=int,
        default=100,
        help=(
            "Per-trial cell budget per perturbation used to average multiple "
            "single-cell or bag comparisons."
        ),
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=("budget", "all_cells"),
        default="budget",
        help=(
            "Cell splitting mode for single-cell comparisons. 'budget' uses "
            "--trial-cell-budget cells per perturbation. 'all_cells' uses ALL "
            "available cells split into two halves (imbalance allowed if odd count)."
        ),
    )
    parser.add_argument(
        "--perturbation-filter",
        type=str,
        choices=("all", "single", "double"),
        default="all",
        help="Filter perturbations: 'all'=all, 'single'=single-gene only, 'double'=double combos only.",
    )

    parser.add_argument("--pds-result-file", type=str, default="pds_full.h5ad")
    parser.add_argument("--pds-metric", type=str, default="Energy_Distance")
    parser.add_argument(
        "--weight-source",
        type=str,
        choices=("deg", "pds"),
        default="deg",
        help="Weight source for weighted metrics.",
    )
    parser.add_argument(
        "--weight-scheme",
        type=str,
        choices=("normal", "rank"),
        default="normal",
    )
    parser.add_argument("--base-weight-exponent", type=float, default=2.0)
    parser.add_argument(
        "--target-effective-genes",
        type=float,
        default=None,
        help="Target effective gene number. If omitted, use raw weights without calibration.",
    )
    parser.add_argument(
        "--target-effective-genes-mode",
        type=str,
        choices=("mean", "per_pert"),
        default="mean",
        help="Calibration mode for DEG weights.",
    )
    parser.add_argument(
        "--pds-target-effective-genes-mode",
        type=str,
        choices=("mean", "per_pert"),
        default="mean",
        help="Calibration mode for PDS weights.",
    )

    parser.add_argument(
        "--gene-ranking",
        type=str,
        choices=("pds", "deg"),
        default="pds",
        help="Gene ranking source for single_cell_pds_rank_curve top-k selection.",
    )
    parser.add_argument(
        "--topk-values",
        type=str,
        default="10,20,50,70,100,150,200,400,800,1600,3200,6400",
        help="Top-k values for single_cell_pds_rank_curve (10,20,50,100,200,...).",
    )
    parser.add_argument(
        "--bag-sizes",
        type=str,
        default="2,5,10,15,20,30,40,50,70,90",
        help="Bag sizes for bag_size_curve and snr_curve (when --snr-x-axis bag_size).",
    )
    parser.add_argument(
        "--snr-x-axis",
        type=str,
        choices=("effective_genes", "bag_size"),
        default="effective_genes",
        help=(
            "X-axis for snr_curve mode: 'effective_genes' sweeps the target effective "
            "gene count with fixed all-cells split (default); 'bag_size' sweeps the "
            "number of cells per bag with fixed gene weights."
        ),
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=20,
        help=(
            "Number of independent bag-pair resamples per (trial, x_value) task in "
            "snr_curve mode. More resamples give better variance estimates for the "
            "pairwise Cohen's d. Default: 20."
        ),
    )
    parser.add_argument(
        "--effective-genes-values",
        type=str,
        default="auto_log10",
        help=(
            "Target effective gene numbers for effective_genes_curve. "
            "Use 'auto_log10' (default) for log-spaced values, 'auto_linear' for "
            "linearly-spaced values, or a comma-separated list (e.g. 10,50,100,500)."
        ),
    )
    parser.add_argument(
        "--max-effective-genes",
        type=int,
        default=None,
        help="Cap the effective_genes_curve / snr_curve sweep at this maximum value "
             "(inclusive). Values larger than this are dropped. If omitted and "
             "metrics include top20_mse or top20_pearson_dp, the default cap is 2000.",
    )
    parser.add_argument(
        "--no-log-x",
        action="store_true",
        default=False,
        help=(
            "Use a linear x-axis instead of logarithmic. Also switches the default "
            "x-value sampling to 'auto_linear' unless --effective-genes-values is "
            "explicitly provided."
        ),
    )
    parser.add_argument(
        "--bag-normalize-repeats",
        action="store_true",
        default=True,
        help=(
            "For bag-size mode, additionally scale repeat averaging by "
            "max_bag_size/current_bag_size on top of trial-cell-budget repeats."
        ),
    )
    parser.add_argument(
        "--no-bag-normalize-repeats",
        action="store_false",
        dest="bag_normalize_repeats",
        help="Disable repeat averaging for bag-size variance normalization.",
    )
    parser.add_argument(
        "--r2-delta-mode",
        type=str,
        choices=("pert", "ctrl"),
        default="pert",
    )

    parser.add_argument("--run-distance-heatmaps", action="store_true")
    parser.add_argument(
        "--heatmap-x-values",
        type=str,
        default="1",
        help="Comma-separated x values to export heatmaps for.",
    )
    parser.add_argument(
        "--heatmap-metrics",
        type=str,
        default=None,
        help=(
            "Optional metric subset for heatmaps only. "
            "If omitted, uses --metrics."
        ),
    )
    parser.add_argument(
        "--heatmap-max-perturbations",
        type=int,
        default=10,
        help="Maximum number of perturbations included in heatmaps.",
    )
    parser.add_argument(
        "--heatmap-max-bags-per-perturbation",
        type=int,
        default=8,
        help="Maximum number of bags (or items) sampled per perturbation for heatmaps.",
    )
    # Backward-compatibility aliases.
    parser.add_argument(
        "--heatmap-max-cell-per-pert",
        dest="heatmap_max_bags_per_perturbation",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--heatmap-max-cells",
        dest="heatmap_max_bags_per_perturbation",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--heatmap-quantile-bins", type=int, default=100)
    parser.add_argument(
        "--heatmap-quantile-scope",
        type=str,
        choices=("global", "per_perturbation"),
        default="per_perturbation",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Regenerate PNG plots from saved CSV/NPZ artifacts without recomputing "
            "distance matrices or trial scores."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=24,
        help="Number of parallel worker processes (default: 24). Uses fork-based multiprocessing.",
    )
    return parser.parse_args()


def _parse_csv_ints(raw: str, arg_name: str) -> List[int]:
    vals: List[int] = []
    for token in raw.split(","):
        text = token.strip()
        if not text:
            continue
        try:
            val = int(text)
        except ValueError as exc:
            raise ValueError(f"Invalid integer in --{arg_name}: {text}") from exc
        if val <= 0:
            raise ValueError(f"--{arg_name} requires positive integers.")
        vals.append(val)
    if not vals:
        raise ValueError(f"--{arg_name} is empty.")
    return sorted(set(vals))


def _default_log10_topk(n_genes: int) -> List[int]:
    """Generate default logarithmic (base-10 friendly) top-k values starting from 1.

    Produces the sequence 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, ...
    until reaching n_genes, then appends n_genes.
    """
    if n_genes <= 0:
        return []
    # Small values starting from 1
    small_vals = [1, 2, 3, 5, 7]
    multipliers = [1.0, 1.5, 2.0, 3.0, 5.0, 7.0]
    out: List[int] = [v for v in small_vals if v <= n_genes]
    decade = 10
    while True:
        for m in multipliers:
            k = int(round(m * decade))
            if k > n_genes:
                break
            out.append(k)
        decade *= 10
        if decade > n_genes * 10:
            break
    if not out or out[-1] != n_genes:
        out.append(int(n_genes))
    return sorted(set(out))


def _default_linear_topk(n_genes: int, n_steps: int = 20) -> List[int]:
    """Generate default linearly-spaced top-k values starting from 1.

    Produces ~n_steps evenly-spaced integers from 1 to n_genes (inclusive),
    always including 1 and n_genes.
    """
    if n_genes <= 0:
        return []
    if n_genes == 1:
        return [1]
    step = max(1, (n_genes - 1) // (n_steps - 1))  # -1 because we include both endpoints
    out = [1] + list(range(1 + step, n_genes, step))
    if not out or out[-1] != n_genes:
        out.append(int(n_genes))
    return sorted(set(out))


def _resolve_topk_values(raw: str, n_genes: int) -> List[int]:
    """Resolve top-k values from CLI string or default auto-log behavior."""
    if raw.strip().lower() in {"auto", "auto_log2", "auto_log10", "default"}:
        return _default_log10_topk(n_genes=n_genes)
    if raw.strip().lower() == "auto_linear":
        return _default_linear_topk(n_genes=n_genes)
    return [k for k in _parse_csv_ints(raw, "topk-values") if k <= n_genes]


def _resolve_effective_genes_values(raw: str, n_genes: int) -> List[int]:
    """Resolve effective-gene sweep values from CLI string or auto-log10/auto-linear."""
    if raw.strip().lower() in {"auto", "auto_log10", "default"}:
        return _default_log10_topk(n_genes=n_genes)
    if raw.strip().lower() == "auto_linear":
        return _default_linear_topk(n_genes=n_genes)
    return [k for k in _parse_csv_ints(raw, "effective-genes-values") if k <= n_genes]


def _apply_max_effective_genes_cap(
    eff_values: List[int],
    args: argparse.Namespace,
    metrics: Sequence[str],
    *,
    silent: bool = False,
) -> List[int]:
    """Cap the x-axis sweep; per-pert top-N metrics default to 2000 unless overridden."""
    max_eg = getattr(args, "max_effective_genes", None)
    if max_eg is None and any(m in metrics for m in ("top20_mse", "top20_pearson_dp")):
        max_eg = 2000
        if not silent:
            print(
                "  Per-pert top-N metrics (top20_*): capping sweep at target eff. genes ≤ 2000 "
                "(set --max-effective-genes to override)."
            )
    if max_eg is not None and max_eg > 0:
        return [v for v in eff_values if v <= int(max_eg)]
    return eff_values


def _resolved_max_effective_genes(
    args: argparse.Namespace,
    metrics: Sequence[str],
) -> Optional[int]:
    """Effective cap after defaulting to 2000 for top20_* metrics."""
    max_eg = getattr(args, "max_effective_genes", None)
    if max_eg is None and any(m in metrics for m in ("top20_mse", "top20_pearson_dp")):
        return 2000
    return max_eg


def _parse_metrics(raw: str) -> List[str]:
    allowed = {
        "mse",
        "wmse",
        "r2_delta",
        "w_r2_delta",
        "pearson_delta",
        "w_pearson_delta",
        "cosine_sim",
        "w_cosine_sim",
        "top_n_deg_mse",
        "top_n_deg_r2",
        "top_n_deg_pearson",
        "top20_mse",
        "top20_pearson_dp",
        "energy_distance",
        "weighted_energy_distance",
        "std_energy_distance",
        "w_std_energy_distance",
    }
    metrics = [m.strip() for m in raw.split(",") if m.strip()]
    if not metrics:
        raise ValueError("No metrics parsed from --metrics.")
    unknown = [m for m in metrics if m not in allowed]
    if unknown:
        raise ValueError(f"Unsupported metrics: {unknown}. Allowed: {sorted(allowed)}")
    out: List[str] = []
    for metric in metrics:
        if metric not in out:
            out.append(metric)
    return out


def _resolve_weight_transforms(weight_scheme: str) -> Tuple[str, str]:
    if weight_scheme == "rank":
        return "rank01", "rank01"
    return "minmax", "clip01"


def _filter_perturbations_by_type(
    perturbations: Sequence[str],
    filter_mode: str,
) -> List[str]:
    """Filter perturbations by type: all, single (no separator), or double (with separator).

    Detects combo perturbations by common separators: single underscore _, double
    underscore __, pipe |, semicolon ;, or plus +. Single-gene perturbations should
    not contain these separators.

    Args:
        perturbations: List of perturbation labels.
        filter_mode: 'all', 'single', or 'double'.

    Returns:
        Filtered list of perturbation labels.
    """
    if filter_mode == "all":
        return list(perturbations)

    out: List[str] = []
    for pert in perturbations:
        # Check if it's a combo perturbation (contains _, __, |, ;, or +)
        # Single underscore _ is included (e.g., wessels23: "DOT1L_INTS1")
        is_combo = any(sep in str(pert) for sep in ("__", "|", ";", "+"))
        # Also check for single underscore as separator (common in many datasets)
        if not is_combo:
            is_combo = "_" in str(pert)
        if filter_mode == "double" and is_combo:
            out.append(pert)
        elif filter_mode == "single" and not is_combo:
            out.append(pert)
    return out


def _drop_control_perturbation(data: base.PreparedData) -> base.PreparedData:
    """Restrict analysis to non-control perturbations."""
    y = np.asarray(data.y, dtype=str)
    mask = y != str(data.control_label)
    if not np.any(mask):
        raise ValueError("No non-control perturbations remain after filtering control label.")
    X_new = np.asarray(data.X, dtype=float)[mask, :]
    y_new = y[mask]
    perts = sorted(np.unique(y_new).tolist())
    return base.PreparedData(
        X=X_new,
        y=y_new,
        genes=list(data.genes),
        perturbations=perts,
        control_label=str(data.control_label),
    )


def _load_weight_families(
    args: argparse.Namespace,
    dataset_name: str,
    data: base.PreparedData,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    deg_transform, pds_transform = _resolve_weight_transforms(args.weight_scheme)
    deg_weights = base._load_deg_weights(
        dataset_name=dataset_name,
        genes=data.genes,
        perturbations=data.perturbations,
        deg_weight_transform=deg_transform,
        base_weight_exponent=float(args.base_weight_exponent),
    )
    if args.target_effective_genes is not None:
        deg_weights = base._calibrate_weights_per_perturbation(
            weights_per_pert=deg_weights,
            target_effective_genes=float(args.target_effective_genes),
            mode=str(args.target_effective_genes_mode),
        )
    pds_weights = base._load_pds_weights(
        dataset_name=dataset_name,
        genes=data.genes,
        perturbations=data.perturbations,
        pds_result_file=args.pds_result_file,
        pds_metric=args.pds_metric,
        pds_weight_transform=pds_transform,
        base_weight_exponent=float(args.base_weight_exponent),
    )
    if args.target_effective_genes is not None:
        pds_weights = base._calibrate_weights_per_perturbation(
            weights_per_pert=pds_weights,
            target_effective_genes=float(args.target_effective_genes),
            mode=str(args.pds_target_effective_genes_mode),
        )
    active = deg_weights if args.weight_source == "deg" else pds_weights
    return deg_weights, pds_weights, active


def _resolve_ranked_pds_gene_indices(
    dataset_name: str,
    genes: Sequence[str],
    perturbations: Sequence[str],
    pds_result_file: str,
    pds_metric: str,
) -> np.ndarray:
    path = base.ROOT / "analyses" / "perturbation_discrimination" / "results" / dataset_name / pds_result_file
    payload = base.load_pds_full(path)
    metrics = [str(item) for item in payload["metrics"]]
    if pds_metric not in metrics:
        raise ValueError(f"PDS metric '{pds_metric}' not in {metrics}.")
    metric_idx = metrics.index(pds_metric)
    scores = np.asarray(payload["scores_mean"], dtype=float)[metric_idx]
    pds_perts = [str(item) for item in payload["perturbations"]]
    pds_genes = [str(item) for item in payload["genes"]]
    pert_to_idx = {pert: i for i, pert in enumerate(pds_perts)}
    active = [pert_to_idx[pert] for pert in perturbations if pert in pert_to_idx]
    if not active:
        raise ValueError("No overlap between selected perturbations and PDS perturbations.")
    mean_scores = np.mean(scores[np.asarray(active, dtype=int)], axis=0)
    gene_to_idx = {str(g): i for i, g in enumerate(genes)}
    aligned_idx: List[int] = []
    aligned_score: List[float] = []
    for i, gene in enumerate(pds_genes):
        idx = gene_to_idx.get(gene)
        if idx is None:
            continue
        aligned_idx.append(idx)
        aligned_score.append(float(mean_scores[i]))
    if not aligned_idx:
        raise ValueError("No overlapping genes between data and PDS results.")
    aligned_idx_arr = np.asarray(aligned_idx, dtype=int)
    aligned_score_arr = np.asarray(aligned_score, dtype=float)
    ranked = aligned_idx_arr[np.argsort(aligned_score_arr)[::-1]]
    remaining = np.setdiff1d(np.arange(len(genes), dtype=int), ranked, assume_unique=False)
    return np.concatenate([ranked, remaining])


def _resolve_ranked_deg_gene_indices(
    deg_weights: Dict[str, np.ndarray],
    n_genes: int,
) -> np.ndarray:
    """Return gene indices sorted by descending mean DEG weight (full ranking)."""
    mat = np.stack([np.asarray(v, dtype=float) for v in deg_weights.values()], axis=0)
    mean_w = np.mean(mat, axis=0)
    ranked = np.argsort(mean_w)[::-1]
    return ranked


def _top_deg_indices(weights_per_pert: Dict[str, np.ndarray], n_keep: int) -> np.ndarray:
    mat = np.stack([np.asarray(v, dtype=float) for v in weights_per_pert.values()], axis=0)
    mean_w = np.mean(mat, axis=0)
    order = np.argsort(mean_w)[::-1]
    return order[: min(n_keep, order.size)]


def _build_topk_indices(
    weights_per_pert: Dict[str, np.ndarray],
    k: int,
) -> Dict[str, np.ndarray]:
    """Per-perturbation top-k gene indices by weight (same idea as run_pds_test._build_top20_indices)."""
    out: Dict[str, np.ndarray] = {}
    for pert, w in weights_per_pert.items():
        w = np.asarray(w, dtype=float)
        kk = min(int(k), int(w.size))
        if kk < 1:
            continue
        out[str(pert)] = np.argsort(w)[-kk:]
    return out


def _pearson_vec_top20(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r matching run_pds_test._pearson_vec."""
    ca = a - a.mean()
    cb = b - b.mean()
    denom = np.sqrt(np.sum(ca**2) * np.sum(cb**2))
    if denom < 1e-12:
        return 0.0
    return float(np.clip(np.sum(ca * cb) / denom, -1.0, 1.0))


def _query_weighted_sqeuclidean(Xq: np.ndarray, Xt: np.ndarray, q_labels: np.ndarray, weights: Dict[str, np.ndarray]) -> np.ndarray:
    out = np.zeros((Xq.shape[0], Xt.shape[0]), dtype=float)
    for i in range(Xq.shape[0]):
        w = np.asarray(weights.get(str(q_labels[i]), np.zeros(Xq.shape[1], dtype=float)), dtype=float)
        w_sum = float(np.sum(w))
        if w_sum < 1e-12:
            w = np.full(Xq.shape[1], 1.0 / float(max(Xq.shape[1], 1)))
        else:
            w = w / w_sum
        diff = Xt - Xq[i : i + 1, :]
        out[i, :] = np.sum(np.square(diff) * w[None, :], axis=1)
    return np.maximum(out, 0.0)


def _query_r2_distance(
    Xq_delta: np.ndarray,
    Xt_delta: np.ndarray,
    q_labels: np.ndarray,
    weights: Dict[str, np.ndarray],
    eps: float = 1e-12,
) -> np.ndarray:
    """R² distance: SSE / SST per (query, truth) pair.

    Both Xq_delta and Xt_delta should already be delta-centered (e.g. X - baseline).
    Weights are per-query-perturbation and normalized internally.
    For unweighted R² pass uniform weights.
    """
    n_q, n_g = Xq_delta.shape
    n_t = Xt_delta.shape[0]
    out = np.zeros((n_q, n_t), dtype=float)
    for i in range(n_q):
        w = np.asarray(weights.get(str(q_labels[i]), np.zeros(n_g, dtype=float)), dtype=float)
        w_sum = float(np.sum(w))
        if w_sum < eps:
            w = np.full(n_g, 1.0 / float(max(n_g, 1)))
        else:
            w = w / w_sum
        diff = Xt_delta - Xq_delta[i : i + 1, :]
        sse = np.sum(np.square(diff) * w[None, :], axis=1)
        truth_wmean = np.sum(Xt_delta * w[None, :], axis=1)
        centered = Xt_delta - truth_wmean[:, None]
        sst = np.sum(np.square(centered) * w[None, :], axis=1)
        non_const = sst > eps
        row = np.ones(n_t, dtype=float)
        if np.any(non_const):
            row[non_const] = sse[non_const] / sst[non_const]
        zero_err = sse[~non_const] <= eps
        row[~non_const] = np.where(zero_err, 0.0, 1.0)
        out[i, :] = np.nan_to_num(row, nan=1.0, posinf=1.0, neginf=1.0)
    return np.maximum(out, 0.0)


def _query_weighted_corrdist(Xq: np.ndarray, Xt: np.ndarray, q_labels: np.ndarray, weights: Dict[str, np.ndarray]) -> np.ndarray:
    out = np.zeros((Xq.shape[0], Xt.shape[0]), dtype=float)
    for i in range(Xq.shape[0]):
        w = np.asarray(weights.get(str(q_labels[i]), np.zeros(Xq.shape[1], dtype=float)), dtype=float)
        w_sum = float(np.sum(w))
        if w_sum < 1e-12:
            w = np.full(Xq.shape[1], 1.0 / float(max(Xq.shape[1], 1)))
        else:
            w = w / w_sum
        sqrt_w = np.sqrt(w)
        q = Xq[i] * sqrt_w
        t = Xt * sqrt_w[None, :]
        q_center = q - np.mean(q)
        t_center = t - np.mean(t, axis=1, keepdims=True)
        num = np.sum(t_center * q_center[None, :], axis=1)
        den = np.linalg.norm(q_center) * np.linalg.norm(t_center, axis=1)
        corr = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
        out[i, :] = np.nan_to_num(1.0 - np.clip(corr, -1.0, 1.0), nan=1.0, posinf=1.0, neginf=1.0)
    return out


def _weighted_pairwise_rowdist(A: np.ndarray, B: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pairwise weighted Euclidean distance between rows of A and B."""
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0.0, None)
    w_sum = float(np.sum(w))
    if w_sum < 1e-12:
        w = np.full(A.shape[1], 1.0 / float(max(A.shape[1], 1)))
    else:
        w = w / w_sum
    sq = np.sum(((A[:, None, :] - B[None, :, :]) ** 2) * w[None, None, :], axis=2)
    return np.sqrt(np.maximum(sq, 0.0))


def _energy_distance_from_samples(Xa: np.ndarray, Xb: np.ndarray, weights: np.ndarray) -> float:
    """Energy distance between two sample clouds with weighted Euclidean norm."""
    d_ab = _weighted_pairwise_rowdist(Xa, Xb, weights=weights)
    d_aa = _weighted_pairwise_rowdist(Xa, Xa, weights=weights)
    d_bb = _weighted_pairwise_rowdist(Xb, Xb, weights=weights)
    return float(2.0 * np.mean(d_ab) - np.mean(d_aa) - np.mean(d_bb))


def _std_energy_distance_from_samples(Xa: np.ndarray, Xb: np.ndarray, weights: np.ndarray) -> float:
    """Standardized energy distance H = D^2(F_X, F_Y) / (2 * E||X - Y||).

    Returns 0 when the denominator is near-zero.
    """
    d_ab = _weighted_pairwise_rowdist(Xa, Xb, weights=weights)
    d_aa = _weighted_pairwise_rowdist(Xa, Xa, weights=weights)
    d_bb = _weighted_pairwise_rowdist(Xb, Xb, weights=weights)
    numerator = 2.0 * np.mean(d_ab) - np.mean(d_aa) - np.mean(d_bb)
    denominator = 2.0 * np.mean(d_ab)
    if denominator < 1e-12:
        return 0.0
    return float(numerator / denominator)


def _query_cosine_distance(
    Xq: np.ndarray,
    Xt: np.ndarray,
    q_labels: np.ndarray,
    weights: Dict[str, np.ndarray],
) -> np.ndarray:
    """Weighted cosine distance: 1 - cos(sqrt(w)*q, sqrt(w)*t)."""
    out = np.zeros((Xq.shape[0], Xt.shape[0]), dtype=float)
    for i in range(Xq.shape[0]):
        w = np.asarray(weights.get(str(q_labels[i]), np.zeros(Xq.shape[1], dtype=float)), dtype=float)
        w_sum = float(np.sum(w))
        if w_sum < 1e-12:
            w = np.full(Xq.shape[1], 1.0 / float(max(Xq.shape[1], 1)))
        else:
            w = w / w_sum
        sqrt_w = np.sqrt(w)
        q = Xq[i] * sqrt_w
        t = Xt * sqrt_w[None, :]
        q_norm = np.linalg.norm(q)
        t_norms = np.linalg.norm(t, axis=1)
        den = q_norm * t_norms
        num = np.sum(t * q[None, :], axis=1)
        cos = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
        out[i, :] = np.nan_to_num(1.0 - np.clip(cos, -1.0, 1.0), nan=1.0, posinf=1.0, neginf=1.0)
    return out


def _query_truth_scores(dist: np.ndarray) -> np.ndarray:
    scores = np.zeros(dist.shape[0], dtype=float)
    for i in range(dist.shape[0]):
        order = np.argsort(dist[i])
        rank = int(np.where(order == i)[0][0])
        scores[i] = base._rank_to_score(rank=rank, n_labels=dist.shape[0])
    return scores


def _sample_one_cell_per_half(X: np.ndarray, y: np.ndarray, perturbations: Sequence[str], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    q_idx: List[int] = []
    t_idx: List[int] = []
    for pert in perturbations:
        idx = np.where(y == pert)[0]
        if idx.size < 2:
            raise ValueError(f"Perturbation '{pert}' needs >=2 cells for single-cell half-splits.")
        chosen = rng.choice(idx, size=2, replace=False)
        q_idx.append(int(chosen[0]))
        t_idx.append(int(chosen[1]))
    return np.asarray(q_idx, dtype=int), np.asarray(t_idx, dtype=int)


def _build_single_cell_pairs_for_trial(
    y: np.ndarray,
    perturbations: Sequence[str],
    trial_cell_budget: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build non-overlapping single-cell query/truth index pairs per trial.

    Uses up to `trial_cell_budget` cells per perturbation and creates
    floor(usable_cells/2) repeats, consistent across perturbations.
    """
    if trial_cell_budget <= 1:
        raise ValueError("--trial-cell-budget must be >=2 for single-cell mode.")
    rng = np.random.default_rng(seed)
    per_pert_cells: Dict[str, np.ndarray] = {}
    n_repeats_per_pert: List[int] = []
    for pert in perturbations:
        idx = np.where(y == pert)[0]
        usable = min(int(idx.size), int(trial_cell_budget))
        if usable < 2:
            raise ValueError(f"Perturbation '{pert}' has only {idx.size} cells; needs at least 2.")
        chosen = rng.choice(idx, size=usable, replace=False)
        rng.shuffle(chosen)
        per_pert_cells[str(pert)] = np.asarray(chosen, dtype=int)
        n_repeats_per_pert.append(usable // 2)
    n_repeats = int(min(n_repeats_per_pert))
    if n_repeats <= 0:
        raise ValueError("No single-cell repeats can be formed from trial budget.")
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for rep in range(n_repeats):
        q: List[int] = []
        t: List[int] = []
        for pert in perturbations:
            arr = per_pert_cells[str(pert)]
            q.append(int(arr[2 * rep]))
            t.append(int(arr[2 * rep + 1]))
        out.append((np.asarray(q, dtype=int), np.asarray(t, dtype=int)))
    return out


def _build_bag_mean_pairs_all_cells(
    X: np.ndarray,
    y: np.ndarray,
    perturbations: Sequence[str],
    seed: int,
    n_resamples: int = 1,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build bag-mean query/truth pairs using ALL available cells per perturbation.

    For each perturbation, all available cells are randomly split into two halves
    (bag size = n_cells // 2). Returns `n_resamples` independent random splits,
    each as a `(Xq_means, Xt_means)` tuple with shape `(n_perts, n_genes)`.

    With n_resamples > 1 the caller can pool the resulting distance matrices to
    obtain proper within-distance variance estimates for pairwise Cohen's d.
    """
    rng = np.random.default_rng(seed)
    n_genes = int(X.shape[1])
    n_perts = len(perturbations)
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    for _ in range(n_resamples):
        Xq_means = np.zeros((n_perts, n_genes), dtype=float)
        Xt_means = np.zeros((n_perts, n_genes), dtype=float)
        for i, pert in enumerate(perturbations):
            idx = np.where(y == pert)[0]
            if idx.size < 2:
                raise ValueError(f"Perturbation '{pert}' has only {idx.size} cells; needs at least 2.")
            shuffled = idx.copy()
            rng.shuffle(shuffled)
            mid = len(shuffled) // 2
            Xq_means[i] = np.mean(X[shuffled[:mid]], axis=0)
            Xt_means[i] = np.mean(X[shuffled[mid:]], axis=0)
        pairs.append((Xq_means, Xt_means))

    return pairs


def _sample_one_bag_per_half(
    X: np.ndarray,
    y: np.ndarray,
    perturbations: Sequence[str],
    bag_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    rng = np.random.default_rng(seed)
    q_means: List[np.ndarray] = []
    t_means: List[np.ndarray] = []
    q_members: List[np.ndarray] = []
    t_members: List[np.ndarray] = []
    skipped: List[str] = []
    need = 2 * bag_size
    for pert in perturbations:
        idx = np.where(y == pert)[0]
        if idx.size < need:
            skipped.append(pert)
            continue
        chosen = rng.choice(idx, size=need, replace=False)
        q = np.sort(chosen[:bag_size].astype(int))
        t = np.sort(chosen[bag_size:].astype(int))
        q_members.append(q)
        t_members.append(t)
        q_means.append(np.mean(X[q, :], axis=0))
        t_means.append(np.mean(X[t, :], axis=0))
    if skipped:
        import warnings
        warnings.warn(
            f"Skipped {len(skipped)} perturbations with <{need} cells for bag_size={bag_size}: {skipped[:5]}"
            f"{'...' if len(skipped) > 5 else ''}",
            stacklevel=2,
        )
    if not q_means:
        raise ValueError(
            f"No perturbations with enough cells (need {need}) for bag_size={bag_size}. "
            f"All {len(perturbations)} perturbations were skipped."
        )
    return np.asarray(q_means, dtype=float), np.asarray(t_means, dtype=float), q_members, t_members


def _build_bag_pairs_for_trial(
    y: np.ndarray,
    perturbations: Sequence[str],
    bag_size: int,
    trial_cell_budget: int,
    seed: int,
    normalize_repeats: bool,
    max_bag_size: int,
) -> List[Tuple[List[np.ndarray], List[np.ndarray]]]:
    """Build non-overlapping bag query/truth index pairs per trial.

    Number of repeats is controlled by available budget and optionally scaled by
    `max_bag_size / bag_size` for variance comparability across bag sizes.
    """
    if bag_size <= 0:
        raise ValueError("bag_size must be positive.")
    if trial_cell_budget <= 0:
        raise ValueError("--trial-cell-budget must be positive.")
    rng = np.random.default_rng(seed)
    per_pert_cells: Dict[str, np.ndarray] = {}
    repeat_caps: List[int] = []
    for pert in perturbations:
        idx = np.where(y == pert)[0]
        usable = min(int(idx.size), int(trial_cell_budget))
        need = 2 * int(bag_size)
        if usable < need:
            raise ValueError(
                f"Perturbation '{pert}' has usable={usable} cells (<{need}) for bag-size={bag_size}."
            )
        chosen = rng.choice(idx, size=usable, replace=False)
        rng.shuffle(chosen)
        per_pert_cells[str(pert)] = np.asarray(chosen, dtype=int)
        repeat_caps.append(usable // need)
    n_repeats = int(min(repeat_caps))
    if normalize_repeats and max_bag_size > 0:
        n_repeats *= max(1, int(max_bag_size // bag_size))
    n_repeats = max(1, n_repeats)
    out: List[Tuple[List[np.ndarray], List[np.ndarray]]] = []
    for rep in range(n_repeats):
        q_members: List[np.ndarray] = []
        t_members: List[np.ndarray] = []
        for pert in perturbations:
            arr = per_pert_cells[str(pert)]
            cap = int(arr.size // (2 * bag_size))
            local_rep = rep % max(1, cap)
            start = local_rep * (2 * bag_size)
            block = arr[start : start + (2 * bag_size)]
            q_members.append(np.sort(block[:bag_size].astype(int)))
            t_members.append(np.sort(block[bag_size:].astype(int)))
        out.append((q_members, t_members))
    return out


def _save_curve(trials: Sequence[TrialResult], out_csv: Path, x_col: str) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped: Dict[Tuple[str, int], List[float]] = {}
    for row in trials:
        grouped.setdefault((row.metric, row.x_value), []).append(float(row.score))
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "metric",
                x_col,
                "score_mean",
                "score_std",
                "n_trials",
                "effective_genes_mean",
                "effective_genes_median",
                "effective_genes_std",
            ]
        )
        for (metric, xval), vals in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
            arr = np.asarray(vals, dtype=float)
            eff_mean = float("nan")
            eff_med = float("nan")
            eff_std = float("nan")
            if hasattr(_save_curve, "_effective_gene_stats"):
                lookup = getattr(_save_curve, "_effective_gene_stats")
                entry = lookup.get((metric, int(xval)))
                if entry is not None:
                    eff_mean, eff_med, eff_std = entry
            writer.writerow(
                [
                    metric,
                    xval,
                    f"{float(np.mean(arr)):.8f}",
                    f"{float(np.std(arr, ddof=1)):.8f}" if arr.size > 1 else "0.00000000",
                    int(arr.size),
                    f"{eff_mean:.8f}" if np.isfinite(eff_mean) else "",
                    f"{eff_med:.8f}" if np.isfinite(eff_med) else "",
                    f"{eff_std:.8f}" if np.isfinite(eff_std) else "",
                ]
            )


def _plot_curve_from_summary_csv(
    csv_path: Path,
    out_png: Path,
    x_col: str,
    x_label: str,
    title: str,
    *,
    log_x: bool = False,
    y_label: str = "PDS",
    show_legend: bool = True,
    emphasize_high_scores: bool = False,
    exponential_y: bool = False,
    figure_size: Tuple[float, float] = NATURE_FIG_SIZE,
    legend_outside_right: bool = False,
) -> None:
    """Render a curve directly from saved summary CSV (no recomputation)."""
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in summary CSV: {csv_path}")

    grouped: Dict[str, List[Tuple[int, float, float]]] = {}
    for row in rows:
        metric = str(row["metric"])
        x_val = int(float(row[x_col]))
        mean = float(row["score_mean"])
        std = float(row["score_std"])
        grouped.setdefault(metric, []).append((x_val, mean, std))

    _apply_nature_rc()
    plt.figure(figsize=figure_size)
    y_mins: List[float] = []
    y_maxs: List[float] = []
    for metric in _ordered_metrics(list(grouped.keys())):
        pts = sorted(grouped[metric], key=lambda x: x[0])
        x_arr = np.asarray([p[0] for p in pts], dtype=float)
        means = np.asarray([p[1] for p in pts], dtype=float)
        stds = np.asarray([p[2] for p in pts], dtype=float)
        color, linestyle = _metric_plot_style(metric)
        marker = _metric_marker(metric)
        plt.plot(
            x_arr,
            means,
            marker=marker,
            linewidth=1.3,
            markersize=2.7,
            linestyle=linestyle,
            color=color,
            label=_metric_label(metric),
            zorder=3 if metric.startswith("top_n_deg_") or metric.startswith("top20_") else 2,
        )
        plt.fill_between(x_arr, means - stds, means + stds, alpha=0.14, color=color)
        y_mins.append(float(np.min(means - stds)))
        y_maxs.append(float(np.max(means + stds)))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    if log_x:
        plt.xscale("log", base=10)
    ax = plt.gca()
    ax.grid(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    if emphasize_high_scores and y_mins and y_maxs:
        y_min = max(0.85, min(y_mins) - 0.01)
        y_max = min(1.0, max(y_maxs) + 0.004)
        if y_max > y_min + 0.02:
            plt.ylim(y_min, y_max)
    if exponential_y:
        eps = 1e-4
        forward = lambda y: -np.log10(np.clip(1.0 - np.asarray(y), eps, 1.0))
        inverse = lambda z: 1.0 - np.power(10.0, -np.asarray(z))
        ax.set_yscale("function", functions=(forward, inverse))
        y_ticks = [0.80, 0.90, 0.95, 0.98, 0.99]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{t:.2f}" for t in y_ticks])
    if show_legend:
        if legend_outside_right:
            plt.legend(
                frameon=False,
                fontsize=NATURE_LEGEND_SIZE,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
            )
        else:
            plt.legend(frameon=False, fontsize=NATURE_LEGEND_SIZE, loc="lower right")
    if legend_outside_right:
        plt.tight_layout(rect=(0.0, 0.0, 0.88, 1.0))
    else:
        plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=NATURE_DPI, bbox_inches="tight")
    plt.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def _plot_curve(
    trials: Sequence[TrialResult],
    out_png: Path,
    x_label: str,
    title: str,
    log_x: bool = False,
    effective_gene_stats: Optional[Dict[Tuple[str, int], Tuple[float, float, float]]] = None,
    show_eff_gene_box: bool = True,
    annotation_box: Optional[str] = None,
    y_label: str = "PDS",
    show_legend: bool = True,
    emphasize_high_scores: bool = False,
    exponential_y: bool = False,
    y_limits: Optional[Tuple[float, float]] = None,
    figure_size: Tuple[float, float] = NATURE_FIG_SIZE,
    legend_outside_right: bool = False,
) -> None:
    _apply_nature_rc()
    grouped: Dict[str, Dict[int, List[float]]] = {}
    for row in trials:
        grouped.setdefault(row.metric, {}).setdefault(row.x_value, []).append(float(row.score))
    plt.figure(figsize=figure_size)
    summary_lines: List[str] = []
    y_mins: List[float] = []
    y_maxs: List[float] = []
    for metric in _ordered_metrics(list(grouped.keys())):
        x_sorted = sorted(grouped[metric])
        means = np.asarray([np.mean(grouped[metric][x]) for x in x_sorted], dtype=float)
        stds = np.asarray(
            [np.std(np.asarray(grouped[metric][x], dtype=float), ddof=1) if len(grouped[metric][x]) > 1 else 0.0 for x in x_sorted],
            dtype=float,
        )
        x_arr = np.asarray(x_sorted, dtype=float)
        color, linestyle = _metric_plot_style(metric)
        marker = _metric_marker(metric)
        plt.plot(
            x_arr,
            means,
            marker=marker,
            linewidth=1.3,
            markersize=2.7,
            linestyle=linestyle,
            color=color,
            label=_metric_label(metric),
            zorder=3 if metric.startswith("top_n_deg_") or metric.startswith("top20_") else 2,
        )
        plt.fill_between(x_arr, means - stds, means + stds, alpha=0.14, color=color)
        y_mins.append(float(np.min(means - stds)))
        y_maxs.append(float(np.max(means + stds)))
        if show_eff_gene_box and effective_gene_stats is not None:
            x_last = int(x_sorted[-1])
            stats = effective_gene_stats.get((metric, x_last))
            if stats is not None:
                em, ed, es = stats
                summary_lines.append(f"{_metric_label(metric)}: {em:.1f}/{ed:.1f}/{es:.1f}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    if log_x:
        plt.xscale("log", base=10)
    ax = plt.gca()
    ax.grid(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    if emphasize_high_scores and y_mins and y_maxs:
        y_min = max(0.85, min(y_mins) - 0.01)
        y_max = min(1.0, max(y_maxs) + 0.004)
        if y_max > y_min + 0.02:
            plt.ylim(y_min, y_max)
    if exponential_y:
        eps = 1e-4
        forward = lambda y: -np.log10(np.clip(1.0 - np.asarray(y), eps, 1.0))
        inverse = lambda z: 1.0 - np.power(10.0, -np.asarray(z))
        ax.set_yscale("function", functions=(forward, inverse))
        y_ticks = [0.80, 0.90, 0.95, 0.98, 0.99]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{t:.2f}" for t in y_ticks])
    if y_limits is not None:
        plt.ylim(float(y_limits[0]), float(y_limits[1]))
    if show_legend:
        if legend_outside_right:
            plt.legend(
                frameon=False,
                fontsize=NATURE_LEGEND_SIZE,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
            )
        else:
            plt.legend(frameon=False, fontsize=NATURE_LEGEND_SIZE, loc="lower right")
    box_text = annotation_box
    if box_text is None and summary_lines:
        box_text = "Eff. genes mean/med/std (@max x):\n" + "\n".join(summary_lines)
    if box_text:
        plt.gca().text(
            1.02,
            0.98,
            box_text,
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            fontsize=NATURE_LEGEND_SIZE,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
    if legend_outside_right:
        plt.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    else:
        plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=NATURE_DPI, bbox_inches="tight")
    plt.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def _resolve_heatmap_x(all_x: Sequence[int], raw: str) -> List[int]:
    uniq = sorted(set(int(v) for v in all_x))
    if not uniq:
        return []
    if raw.strip():
        wanted = _parse_csv_ints(raw, "heatmap-x-values")
        wanted_set = set(wanted)
        return [x for x in uniq if x in wanted_set]
    mid = uniq[len(uniq) // 2]
    return sorted(set([uniq[0], mid, uniq[-1]]))


def _quantile_bin_block(block: np.ndarray, n_bins: int) -> np.ndarray:
    finite = np.isfinite(block)
    out = np.zeros_like(block, dtype=float)
    if not np.any(finite):
        return out
    vals = np.asarray(block[finite], dtype=float)
    edges = np.quantile(vals, np.linspace(0.0, 1.0, n_bins + 1))
    if np.allclose(edges, edges[0]):
        out[finite] = 0.5
        return out
    bin_ids = np.digitize(block, bins=edges[1:-1], right=True).astype(float)
    bin_ids = np.clip(bin_ids, 0.0, float(n_bins - 1))
    out = bin_ids / float(n_bins - 1)
    out[~finite] = 0.0
    return out


def _quantile_bin_matrix_with_scope(matrix: np.ndarray, labels: np.ndarray, n_bins: int, scope: str) -> np.ndarray:
    if scope == "global":
        return _quantile_bin_block(matrix, n_bins=n_bins)
    if scope != "per_perturbation":
        raise ValueError(f"Unsupported quantile scope: {scope}")
    y = np.asarray(labels, dtype=str)
    out = np.zeros_like(matrix, dtype=float)
    for pert in np.unique(y):
        rows = np.where(y == pert)[0]
        out[rows, :] = _quantile_bin_block(matrix[rows, :], n_bins=n_bins)
    return out


def _plot_metric_heatmaps_with_ranges(
    matrices: Dict[str, np.ndarray],
    labels: np.ndarray,
    metrics: Sequence[str],
    output_path: Path,
    quantile_bins: int,
    quantile_scope: str,
) -> None:
    """Plot heatmaps with perturbation range labels on both axes."""
    if quantile_bins <= 1:
        raise ValueError("--heatmap-quantile-bins must be >1.")
    y = np.asarray(labels, dtype=str)
    order = np.argsort(y, kind="stable")
    y_ord = y[order]
    uniq = np.unique(y_ord)
    centers: List[float] = []
    bounds: List[int] = []
    start = 0
    for pert in uniq:
        idx = np.where(y_ord == pert)[0]
        left = int(idx[0])
        right = int(idx[-1])
        centers.append(0.5 * (left + right))
        bounds.append(right + 1)
        start = right + 1

    _apply_nature_rc()
    ordered = _ordered_metrics(metrics)
    pair_grid8 = [
        "mse", "energy_distance", "r2_delta", "pearson_delta",
        "wmse", "weighted_energy_distance", "w_r2_delta", "w_pearson_delta",
    ]
    pair_grid6 = [
        "mse", "r2_delta", "pearson_delta",
        "wmse", "w_r2_delta", "w_pearson_delta",
    ]
    if all(m in ordered for m in pair_grid8):
        ordered = pair_grid8
    elif all(m in ordered for m in pair_grid6):
        ordered = pair_grid6
    n_metrics = len(ordered)
    if n_metrics == 8:
        n_rows, n_cols = 2, 4
    elif n_metrics == 6:
        n_rows, n_cols = 2, 3
    else:
        n_rows, n_cols = int(np.ceil(n_metrics / 4)), min(4, n_metrics)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8.6, 6.0) if (n_rows, n_cols) == (2, 3) else (
            NATURE_HEATMAP_FIG_SIZE if (n_rows, n_cols) == (2, 4) else (2.2 * n_cols, 2.2 * n_rows)
        ),
        sharex=True,
        sharey=True,
    )
    axes_arr = np.atleast_1d(axes).ravel()
    for ax in axes_arr[n_metrics:]:
        ax.axis("off")

    img_ref = None
    for i, metric in enumerate(ordered):
        ax = axes_arr[i]
        dist = np.asarray(matrices[metric], dtype=float)
        dist = dist[np.ix_(order, order)]
        if "r2" in metric.lower():
            dist = np.maximum(dist, 0.0)
        vis = _quantile_bin_matrix_with_scope(
            matrix=dist,
            labels=y_ord,
            n_bins=int(quantile_bins),
            scope=str(quantile_scope),
        )
        img = ax.imshow(vis, aspect="equal", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
        if img_ref is None:
            img_ref = img
        for b in bounds[:-1]:
            ax.axhline(float(b) - 0.5, color="white", linewidth=0.4, alpha=0.6)
            ax.axvline(float(b) - 0.5, color="white", linewidth=0.4, alpha=0.6)
        ax.set_title(_metric_label(metric), fontsize=NATURE_TITLE_SIZE + 1)
        pretty_labels = [_stacked_pert_label(u) for u in uniq]
        ax.set_xticks(centers)
        ax.set_yticks(centers)
        ax.set_xticklabels(pretty_labels, rotation=35, ha="right", fontsize=5.5)
        ax.set_yticklabels(pretty_labels, fontsize=5.5)
        ax.set_xlabel("")
        ax.set_ylabel("")

    if img_ref is not None:
        cbar = fig.colorbar(img_ref, ax=axes_arr[:n_metrics].tolist(), fraction=0.02, pad=0.06, location="right")
        cbar.set_label("Quantile rank", fontsize=NATURE_AXIS_LABEL_SIZE)
        cbar.ax.tick_params(labelsize=NATURE_TICK_SIZE)

    fig.suptitle("Pairwise distance heatmaps (bag size = 1)", fontsize=NATURE_TITLE_SIZE)
    fig.subplots_adjust(left=0.11, right=0.84, bottom=0.2, top=0.9, wspace=0.2, hspace=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=NATURE_DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _build_all_bags_for_heatmap(
    X: np.ndarray,
    y: np.ndarray,
    perturbations: Sequence[str],
    bag_size: int,
    seed: int,
    max_perturbations: int,
    max_bags_per_perturbation: int,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Build capped non-overlapping bags per perturbation for pairwise heatmaps."""
    rng = np.random.default_rng(seed)
    bag_means: List[np.ndarray] = []
    bag_members: List[np.ndarray] = []
    bag_labels: List[str] = []
    # Limit perturbations by available cell counts (largest first).
    perts = [str(p) for p in perturbations]
    if max_perturbations > 0 and len(perts) > max_perturbations:
        counts = np.asarray([np.sum(y == pert) for pert in perts], dtype=int)
        order = np.argsort(counts)[::-1]
        perts = [perts[i] for i in order[:max_perturbations]]
    for pert in perts:
        idx = np.where(y == pert)[0]
        if idx.size < bag_size:
            continue
        shuffled = np.asarray(idx, dtype=int).copy()
        rng.shuffle(shuffled)
        n_bags = int(shuffled.size // bag_size)
        if n_bags <= 0:
            continue
        if max_bags_per_perturbation > 0:
            n_bags = min(n_bags, int(max_bags_per_perturbation))
        blocks = shuffled[: n_bags * bag_size].reshape(n_bags, bag_size)
        for block in blocks:
            members = np.sort(block.astype(int))
            bag_members.append(members)
            bag_labels.append(str(pert))
            bag_means.append(np.mean(X[members, :], axis=0))
    if not bag_means:
        raise ValueError("No bags were created for heatmap export.")
    return np.asarray(bag_means, dtype=float), bag_members, np.asarray(bag_labels, dtype=str)


def _save_heatmaps(
    args: argparse.Namespace,
    out_dir: Path,
    matrices: Dict[Tuple[str, int], np.ndarray],
    labels: np.ndarray,
    x_values: Sequence[int],
) -> None:
    if not args.run_distance_heatmaps:
        return

    def _subset_indices_for_heatmap(labels_arr: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(labels_arr, dtype=str)
        rng = np.random.default_rng(int(args.seed))
        perts = np.unique(y_arr)
        if int(args.heatmap_max_perturbations) > 0 and perts.size > int(args.heatmap_max_perturbations):
            counts = np.asarray([np.sum(y_arr == p) for p in perts], dtype=int)
            order = np.argsort(counts)[::-1]
            perts = perts[order[: int(args.heatmap_max_perturbations)]]
        keep: List[np.ndarray] = []
        for pert in perts:
            idx = np.where(y_arr == pert)[0]
            if int(args.heatmap_max_bags_per_perturbation) > 0 and idx.size > int(args.heatmap_max_bags_per_perturbation):
                idx = np.sort(rng.choice(idx, size=int(args.heatmap_max_bags_per_perturbation), replace=False))
            keep.append(np.asarray(idx, dtype=int))
        if not keep:
            return np.array([], dtype=int)
        return np.sort(np.concatenate(keep).astype(int))

    keep_idx = _subset_indices_for_heatmap(labels)
    if keep_idx.size == 0:
        return

    selected_x = _resolve_heatmap_x(x_values, args.heatmap_x_values)
    for x in selected_x:
        mats: Dict[str, np.ndarray] = {}
        metrics: List[str] = []
        for (metric, x_key), dist in matrices.items():
            if x_key == x:
                mat = np.asarray(dist, dtype=float)
                mats[metric] = mat[np.ix_(keep_idx, keep_idx)]
                metrics.append(metric)
        if not mats:
            continue
        labels_sub = np.asarray(labels, dtype=str)[keep_idx]
        out = out_dir / f"pairwise_distance_heatmaps_x{x}.png"
        npz_path = out_dir / f"pairwise_distance_matrices_x{x}.npz"
        np.savez_compressed(npz_path, labels=labels_sub, **mats)
        base._plot_metric_heatmaps(
            matrices=mats,
            labels=labels_sub,
            metrics=metrics,
            output_path=out,
            max_perturbations=0,
            max_cells=0,
            quantile_bins=int(args.heatmap_quantile_bins),
            quantile_scope=str(args.heatmap_quantile_scope),
            seed=int(args.seed),
        )


def _singlecell_metric_distance(
    metric: str,
    Xq: np.ndarray,
    Xt: np.ndarray,
    labels: np.ndarray,
    control_mean: np.ndarray,
    active_weights: Dict[str, np.ndarray],
    top_deg_local_idx: np.ndarray,
    *,
    dataset_mean: Optional[np.ndarray] = None,
    perpert_topn: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    if metric == "mse":
        return np.mean(np.square(Xq[:, None, :] - Xt[None, :, :]), axis=2)
    if metric == "wmse":
        return _query_weighted_sqeuclidean(Xq, Xt, labels, active_weights)
    if metric == "r2_delta":
        uniform_w = {k: np.ones(Xq.shape[1], dtype=float) for k in active_weights}
        return _query_r2_distance(
            Xq - control_mean[None, :],
            Xt - control_mean[None, :],
            labels,
            uniform_w,
        )
    if metric == "w_r2_delta":
        return _query_r2_distance(
            Xq - control_mean[None, :],
            Xt - control_mean[None, :],
            labels,
            active_weights,
        )
    if metric == "pearson_delta":
        Xq_d = Xq - control_mean[None, :]
        Xt_d = Xt - control_mean[None, :]
        return _query_weighted_corrdist(Xq_d, Xt_d, labels, {k: np.ones(Xq.shape[1], dtype=float) for k in active_weights})
    if metric == "pearson_deltapert":
        if dataset_mean is None:
            raise ValueError("dataset_mean is required for pearson_deltapert")
        Xq_d = Xq - dataset_mean[None, :]
        Xt_d = Xt - dataset_mean[None, :]
        return _query_weighted_corrdist(
            Xq_d,
            Xt_d,
            labels,
            {k: np.ones(Xq.shape[1], dtype=float) for k in active_weights},
        )
    if metric == "w_pearson_delta":
        Xq_d = Xq - control_mean[None, :]
        Xt_d = Xt - control_mean[None, :]
        return _query_weighted_corrdist(Xq_d, Xt_d, labels, active_weights)
    if metric == "cosine_sim":
        uniform_w = {k: np.ones(Xq.shape[1], dtype=float) for k in active_weights}
        return _query_cosine_distance(Xq, Xt, labels, uniform_w)
    if metric == "w_cosine_sim":
        return _query_cosine_distance(Xq, Xt, labels, active_weights)
    if metric == "top_n_deg_mse":
        if top_deg_local_idx.size == 0:
            raise ValueError("No overlap between top DEG indices and selected genes.")
        q = Xq[:, top_deg_local_idx]
        t = Xt[:, top_deg_local_idx]
        return np.mean(np.square(q[:, None, :] - t[None, :, :]), axis=2)
    if metric == "top_n_deg_r2":
        if top_deg_local_idx.size == 0:
            raise ValueError("No overlap between top DEG indices and selected genes.")
        n_deg = top_deg_local_idx.size
        q = Xq[:, top_deg_local_idx] - control_mean[top_deg_local_idx][None, :]
        t = Xt[:, top_deg_local_idx] - control_mean[top_deg_local_idx][None, :]
        uniform_deg = {k: np.ones(n_deg, dtype=float) for k in active_weights}
        return _query_r2_distance(q, t, labels, uniform_deg)
    if metric == "top_n_deg_pearson":
        if top_deg_local_idx.size == 0:
            raise ValueError("No overlap between top DEG indices and selected genes.")
        n_deg = top_deg_local_idx.size
        q = Xq[:, top_deg_local_idx] - control_mean[top_deg_local_idx][None, :]
        t = Xt[:, top_deg_local_idx] - control_mean[top_deg_local_idx][None, :]
        uniform_deg = {k: np.ones(n_deg, dtype=float) for k in active_weights}
        return _query_weighted_corrdist(q, t, labels, uniform_deg)
    if metric == "energy_distance":
        out = np.zeros((Xq.shape[0], Xt.shape[0]), dtype=float)
        for i in range(Xq.shape[0]):
            for j in range(Xt.shape[0]):
                out[i, j] = _energy_distance_from_samples(
                    Xq[i : i + 1, :],
                    Xt[j : j + 1, :],
                    np.ones(Xq.shape[1], dtype=float),
                )
        return out
    if metric == "weighted_energy_distance":
        out = np.zeros((Xq.shape[0], Xt.shape[0]), dtype=float)
        for i in range(Xq.shape[0]):
            w = np.asarray(active_weights.get(str(labels[i]), np.zeros(Xq.shape[1], dtype=float)), dtype=float)
            if float(np.sum(w)) < 1e-12:
                w = np.ones(Xq.shape[1], dtype=float)
            for j in range(Xt.shape[0]):
                out[i, j] = _energy_distance_from_samples(
                    Xq[i : i + 1, :],
                    Xt[j : j + 1, :],
                    w,
                )
        return out
    if metric == "pds":
        # PDS metric: compute MSE distances, then convert to PDS retrieval scores
        # Returns a "distance" matrix where lower = worse retrieval (higher raw distance)
        # We use negative PDS as the distance so that standard distance logic applies
        dist_mse = np.mean(np.square(Xq[:, None, :] - Xt[None, :, :]), axis=2)
        # Convert each row's distances to PDS scores
        pds_scores = np.zeros_like(dist_mse)
        for i in range(dist_mse.shape[0]):
            order = np.argsort(dist_mse[i])
            for j in range(dist_mse.shape[1]):
                rank = int(np.where(order == j)[0][0])
                pds_scores[i, j] = base._rank_to_score(rank=rank, n_labels=dist_mse.shape[1])
        # Return negative PDS as distance (so lower = worse retrieval)
        return -pds_scores
    if metric == "wpds":
        # wPDS metric: compute WMSE distances, then convert to PDS retrieval scores
        dist_wmse = _query_weighted_sqeuclidean(Xq, Xt, labels, active_weights)
        # Convert each row's distances to PDS scores
        pds_scores = np.zeros_like(dist_wmse)
        for i in range(dist_wmse.shape[0]):
            order = np.argsort(dist_wmse[i])
            for j in range(dist_wmse.shape[1]):
                rank = int(np.where(order == j)[0][0])
                pds_scores[i, j] = base._rank_to_score(rank=rank, n_labels=dist_wmse.shape[1])
        # Return negative PDS as distance (so lower = worse retrieval)
        return -pds_scores
    if metric == "pds_pearson_deltapert":
        if dataset_mean is None:
            raise ValueError("dataset_mean is required for pds_pearson_deltapert")
        Xq_d = Xq - dataset_mean[None, :]
        Xt_d = Xt - dataset_mean[None, :]
        dist_pearson = _query_weighted_corrdist(
            Xq_d,
            Xt_d,
            labels,
            {k: np.ones(Xq.shape[1], dtype=float) for k in active_weights},
        )
        pds_scores = np.zeros_like(dist_pearson)
        for i in range(dist_pearson.shape[0]):
            order = np.argsort(dist_pearson[i])
            for j in range(dist_pearson.shape[1]):
                rank = int(np.where(order == j)[0][0])
                pds_scores[i, j] = base._rank_to_score(
                    rank=rank,
                    n_labels=dist_pearson.shape[1],
                )
        return -pds_scores
    if metric == "top20_mse":
        if perpert_topn is None:
            raise ValueError("perpert_topn required for top20_mse")
        n = int(Xq.shape[0])
        d = np.zeros((n, n), dtype=float)
        for i in range(n):
            idx_n = perpert_topn.get(str(labels[i]))
            if idx_n is None:
                d[i, :] = np.nan
                continue
            mi = Xq[i, idx_n]
            d[i, :] = np.mean(np.square(mi[None, :] - Xt[:, idx_n]), axis=1)
        return d
    if metric == "top20_pearson_dp":
        if dataset_mean is None:
            raise ValueError("dataset_mean required for top20_pearson_dp")
        if perpert_topn is None:
            raise ValueError("perpert_topn required for top20_pearson_dp")
        n = int(Xq.shape[0])
        d = np.zeros((n, n), dtype=float)
        for i in range(n):
            idx_n = perpert_topn.get(str(labels[i]))
            if idx_n is None:
                d[i, :] = np.nan
                continue
            di = Xq[i, idx_n] - dataset_mean[idx_n]
            for j in range(n):
                dj = Xt[j, idx_n] - dataset_mean[idx_n]
                d[i, j] = 1.0 - _pearson_vec_top20(di, dj)
        return d
    raise ValueError(f"Unsupported single-cell metric: {metric}")


def _effective_gene_stats_for_metric(
    metric: str,
    n_features: int,
    labels: np.ndarray,
    weights_per_pert: Dict[str, np.ndarray],
    top_deg_local_idx: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Compute effective gene stats (mean/median/std) for one metric context."""
    unweighted = {
        "mse",
        "r2_delta",
        "pearson_delta",
        "pearson_deltapert",
        "cosine_sim",
        "energy_distance",
        "std_energy_distance",
        "pds",
        "pds_pearson_deltapert",
    }
    if metric in unweighted:
        val = float(n_features)
        return val, val, 0.0
    if metric in ("top_n_deg_mse", "top_n_deg_r2", "top_n_deg_pearson"):
        n = float(0 if top_deg_local_idx is None else int(top_deg_local_idx.size))
        return n, n, 0.0
    if metric in ("top20_mse", "top20_pearson_dp"):
        n = float(0 if top_deg_local_idx is None else int(top_deg_local_idx.size))
        return n, n, 0.0
    weighted = {"wmse", "wmse_dataset", "w_r2_delta", "w_pearson_delta", "w_cosine_sim", "weighted_energy_distance", "w_std_energy_distance", "wpds"}
    if metric in weighted:
        vals: List[float] = []
        for label in labels:
            w = np.asarray(weights_per_pert.get(str(label), np.zeros(n_features, dtype=float)), dtype=float)
            vals.append(float(base._effective_genes_single(w)))
        arr = np.asarray(vals, dtype=float)
        return float(np.mean(arr)), float(np.median(arr)), float(np.std(arr, ddof=0))
    val = float(n_features)
    return val, val, 0.0


# ── Fork-based multiprocessing worker infrastructure ──────────────────

_W: Dict[str, Any] = {}


def _set_worker_state(**kwargs: Any) -> None:
    """Set module-level worker state (inherited by forked children)."""
    _W.update(kwargs)


def _merge_worker_result(
    result: Dict[str, Any],
    trial_rows: List[TrialResult],
    eff_stats: Dict[Tuple[str, int], Tuple[float, float, float]],
    heatmap_mats: Optional[Dict[Tuple[str, int], np.ndarray]] = None,
) -> None:
    """Merge a worker's result dict into the main accumulators."""
    trial_rows.extend(result["rows"])
    for key, val in result["eff"].items():
        if key not in eff_stats:
            eff_stats[key] = val
    if heatmap_mats is not None and "hmats" in result:
        heatmap_mats.update(result["hmats"])


def _resolve_n_jobs(args: argparse.Namespace) -> int:
    """Resolve effective n_jobs, falling back to 1 if fork is unavailable."""
    n_jobs = max(1, int(getattr(args, "n_jobs", 1)))
    if n_jobs > 1:
        start_method = mp.get_start_method(allow_none=True)
        if start_method is not None and start_method != "fork":
            print(
                f"WARNING: multiprocessing start method is '{start_method}', not 'fork'; "
                "falling back to n_jobs=1."
            )
            n_jobs = 1
    return n_jobs


def _sc_worker(task: Tuple[int, int]) -> Dict[str, Any]:
    """Worker for single-cell mode: process one (trial, k) pair."""
    trial, k = task
    X: np.ndarray = _W["X"]
    labels: np.ndarray = _W["labels"]
    control_mean: np.ndarray = _W["control_mean"]
    active_weights: Dict[str, np.ndarray] = _W["active_weights"]
    gene_ranked: np.ndarray = _W["gene_ranked"]
    deg_ranked: np.ndarray = _W.get("deg_ranked", np.array([], dtype=int))
    n_top_deg: int = int(_W.get("n_top_deg", 0))
    metrics: List[str] = _W["metrics"]
    pair_repeats = _W["pair_repeats_by_trial"][trial]

    selected = np.asarray(gene_ranked[:k], dtype=int)
    w_subset = {pert: np.asarray(w, dtype=float)[selected] for pert, w in active_weights.items()}
    # Resolve top-N DEG indices *within the selected gene subset*.
    #
    # Previously we intersected `selected` with the global top-N DEG indices, which can be
    # empty for small-k subsets (causing top-N metrics to fail). Instead, we choose the
    # top-N DEG genes among the currently selected genes, falling back to min(N, k).
    if deg_ranked.size > 0 and n_top_deg > 0:
        selected_set = set(int(x) for x in selected.tolist())
        top_deg_in_selected: List[int] = []
        for g in deg_ranked.tolist():
            gi = int(g)
            if gi in selected_set:
                top_deg_in_selected.append(gi)
                if len(top_deg_in_selected) >= min(int(n_top_deg), int(selected.size)):
                    break
        if top_deg_in_selected:
            local_top_deg = np.asarray([int(np.where(selected == gi)[0][0]) for gi in top_deg_in_selected], dtype=int)
        else:
            local_top_deg = np.array([], dtype=int)
    else:
        local_top_deg = np.array([], dtype=int)

    rows: List[TrialResult] = []
    eff: Dict[Tuple[str, int], Tuple[float, float, float]] = {}
    hmats: Dict[Tuple[str, int], np.ndarray] = {}

    for metric in metrics:
        rep_scores: List[float] = []
        first_dist: Optional[np.ndarray] = None
        for q_idx, t_idx in pair_repeats:
            Xq = X[q_idx][:, selected]
            Xt = X[t_idx][:, selected]
            dist = _singlecell_metric_distance(
                metric=metric,
                Xq=Xq,
                Xt=Xt,
                labels=labels,
                control_mean=control_mean[selected],
                active_weights=w_subset,
                top_deg_local_idx=local_top_deg,
            )
            scores = _query_truth_scores(dist)
            rep_scores.append(float(np.mean(scores)))
            if first_dist is None:
                first_dist = np.asarray(dist, dtype=float)
        rows.append(TrialResult(
            metric=metric,
            x_value=int(k),
            trial=trial,
            score=float(np.mean(np.asarray(rep_scores, dtype=float))),
        ))
        eff[(metric, int(k))] = _effective_gene_stats_for_metric(
            metric=metric,
            n_features=int(selected.size),
            labels=labels,
            weights_per_pert=w_subset,
            top_deg_local_idx=local_top_deg,
        )
        if trial == 0 and first_dist is not None:
            hmats[(metric, int(k))] = first_dist

    return {"rows": rows, "eff": eff, "hmats": hmats}


def _eg_worker(task: Tuple[int, int]) -> Dict[str, Any]:
    """Worker for effective-genes mode: process one (trial, target_eff) pair."""
    trial, target_eff = task
    X: np.ndarray = _W["X"]
    labels: np.ndarray = _W["labels"]
    control_mean: np.ndarray = _W["control_mean"]
    dataset_mean: np.ndarray = _W["dataset_mean"]
    raw_active: Dict[str, np.ndarray] = _W["raw_active"]
    raw_deg_weights: Dict[str, np.ndarray] = _W["raw_deg_weights"]
    calib_mode: str = _W["calib_mode"]
    metrics: List[str] = _W["metrics"]
    pair_repeats = _W["pair_repeats_by_trial"][trial]
    n_genes = int(X.shape[1])

    calibrated = base._calibrate_weights_per_perturbation(
        weights_per_pert=raw_active,
        target_effective_genes=float(target_eff),
        mode=calib_mode,
    )
    # In effective-genes mode, top-N metrics use N equal to current x-axis target.
    n_top = int(max(1, round(float(target_eff))))
    top_deg_local = _top_deg_indices(raw_deg_weights, n_keep=n_top)
    perpert_topn = _build_topk_indices(calibrated, k=n_top)

    rows: List[TrialResult] = []
    eff: Dict[Tuple[str, int], Tuple[float, float, float]] = {}

    for metric in metrics:
        rep_scores: List[float] = []
        for q_arr, t_arr in pair_repeats:
            # q_arr/t_arr are either:
            #   int indices -> look up raw cells from X  (budget mode)
            #   float arrays of shape (n_perts, n_genes) -> pre-computed bag means (all_cells mode)
            if np.issubdtype(np.asarray(q_arr).dtype, np.floating):
                Xq = np.asarray(q_arr, dtype=float)
                Xt = np.asarray(t_arr, dtype=float)
            else:
                Xq = X[q_arr]
                Xt = X[t_arr]
            dist = _singlecell_metric_distance(
                metric=metric,
                Xq=Xq,
                Xt=Xt,
                labels=labels,
                control_mean=control_mean,
                active_weights=calibrated,
                top_deg_local_idx=top_deg_local,
                dataset_mean=dataset_mean,
                perpert_topn=perpert_topn,
            )
            scores = _query_truth_scores(dist)
            rep_scores.append(float(np.mean(scores)))
        rows.append(TrialResult(
            metric=metric,
            x_value=int(target_eff),
            trial=trial,
            score=float(np.mean(np.asarray(rep_scores, dtype=float))),
        ))
        eff[(metric, int(target_eff))] = _effective_gene_stats_for_metric(
            metric=metric,
            n_features=n_genes,
            labels=labels,
            weights_per_pert=calibrated,
            top_deg_local_idx=top_deg_local,
        )

    return {"rows": rows, "eff": eff}


@dataclass
class SNRResult:
    """Per-(metric, x_value, trial) discrimination result.

    All quantities are derived from the full (n_perts × n_perts) distance matrix D
    where D[i,i] is the within-perturbation distance (same perturbation, two random
    cell-halves) and D[i,j] (i≠j) is the between-perturbation distance.

    Attributes:
        snr:     mean_between / mean_within  (ratio of distribution means, scale-free
                 within a metric but not comparable across metrics with different scales)
        dprime:  Cohen's d = (mean_between - mean_within) / sigma_pooled  — equivalent
                 to d' from signal detection theory; comparable across metrics because the
                 pooled SD cancels the metric's absolute scale
        auc:     Φ(d' / √2) — probability that a random between-distance exceeds a random
                 within-distance; a non-parametric lower-bound estimate is also available
                 but the Gaussian approximation suffices here
        within:  mean of the within-distance distribution (n_perts values)
        between: mean of the between-distance distribution (n_perts*(n_perts-1) values)
        _dist:   Optional storage of the full distance matrix for intermediate aggregation
    """
    metric: str
    x_value: int
    trial: int
    snr: float
    dprime: float
    auc: float
    within: float
    between: float
    _dist: Optional[np.ndarray] = None  # For internal aggregation use only


def _dprime_auc(within_vals: np.ndarray, between_vals: np.ndarray) -> Tuple[float, float]:
    """Global Cohen's d from pooled within/between distance distributions.

    d' = (μ_B - μ_W) / σ_pooled  where σ_pooled = sqrt((σ_B² + σ_W²) / 2)
    AUC = Φ(d' / √2)

    NOTE: sensitive to cross-perturbation variance in between distances; prefer
    _dprime_per_pert_avg for the effective-gene sweep.
    """
    from scipy.special import ndtr
    mu_w = float(np.mean(within_vals))
    mu_b = float(np.mean(between_vals))
    var_w = float(np.var(within_vals, ddof=1)) if within_vals.size > 1 else 0.0
    var_b = float(np.var(between_vals, ddof=1)) if between_vals.size > 1 else 0.0
    sigma_pooled = float(np.sqrt((var_w + var_b) / 2.0))
    if sigma_pooled < 1e-15:
        dprime = float("inf") if mu_b > mu_w else 0.0
    else:
        dprime = (mu_b - mu_w) / sigma_pooled
    auc = float(ndtr(dprime / np.sqrt(2.0)))
    return dprime, auc


def _dprime_per_pert_avg(dist: np.ndarray) -> Tuple[float, float]:
    """Per-perturbation Cohen's d from a single distance matrix (single resample).

    For each perturbation i:
        within_i  = dist[i, i]
        between_i = dist[i, j]  for j != i  (n-1 values)
        d'_i      = (mean(between_i) - within_i) / std(between_i)

    NOTE: within_i is a single point so σ_within is not estimated.
    Use _dprime_per_pert_resampled when multiple distance matrices are available.
    """
    from scipy.special import ndtr
    n = dist.shape[0]
    dprime_per: List[float] = []
    for i in range(n):
        within_i = float(dist[i, i])
        between_i = np.concatenate([dist[i, :i], dist[i, i + 1:]])
        mu_b = float(np.mean(between_i))
        sigma_b = float(np.std(between_i, ddof=1)) if between_i.size > 1 else 0.0
        if sigma_b < 1e-15:
            d_i = float("inf") if mu_b > within_i else 0.0
        else:
            d_i = (mu_b - within_i) / sigma_b
        dprime_per.append(d_i)
    mean_d = float(np.mean(dprime_per))
    auc = float(ndtr(mean_d / np.sqrt(2.0)))
    return mean_d, auc


def _dprime_pairwise_avg(dist_list: List[np.ndarray]) -> Tuple[float, float]:
    """Directional pairwise Cohen's d, averaged over all ordered pairs (i→j).

    For each ordered pair (i, j) with i ≠ j, uses only dist values that share the
    same weight reference frame (both computed with pert i's weights):
        within_i   = [dist_b[i, i]  for b in B]   → B values
        between_ij = [dist_b[i, j]  for b in B]   → B values

        d'_{i→j} = (mean(between_ij) - mean(within_i)) / sqrt((var_W + var_B) / 2)

    The final d' = mean(d'_{i→j}) over all n*(n-1) ordered pairs.
    AUC = Φ(d' / √2).

    Key properties:
      - Handles asymmetric metrics (e.g. wmse) correctly: pooling dist[i,j] with
        dist[j,i] would mix two different absolute scales; using only the i→j
        direction keeps both groups in pert i's reference frame.
      - Symmetric metrics (mse, pearson_delta) give identical i→j and j→i d' values
        so the average equals either direction alone.
      - No cross-pair variance inflation: σ is estimated per ordered pair only.
      - With B > 1 resamples, variance estimates improve for both within and between.
    """
    from scipy.special import ndtr
    B = len(dist_list)
    n = dist_list[0].shape[0]
    # Stack all B matrices for fast extraction: shape (B, n, n)
    D = np.stack(dist_list, axis=0)   # (B, n, n)
    diag_vals = D[:, np.arange(n), np.arange(n)]   # (B, n)  — within[b, i]

    dprime_pairs: List[float] = []
    for i in range(n):
        within_i = diag_vals[:, i]          # B values, all use pert i's frame
        for j in range(n):
            if j == i:
                continue
            between_ij = D[:, i, j]         # B values, all use pert i's frame
            mu_w = float(np.mean(within_i))
            mu_b = float(np.mean(between_ij))
            var_w = float(np.var(within_i,    ddof=1)) if B > 1 else 0.0
            var_b = float(np.var(between_ij,  ddof=1)) if B > 1 else 0.0
            sigma_pooled = float(np.sqrt((var_w + var_b) / 2.0))
            if sigma_pooled < 1e-15:
                d_ij = float("inf") if mu_b > mu_w else 0.0
            else:
                d_ij = (mu_b - mu_w) / sigma_pooled
            dprime_pairs.append(d_ij)
    mean_d = float(np.mean(dprime_pairs))
    auc = float(ndtr(mean_d / np.sqrt(2.0)))
    return mean_d, auc


def _dprime_per_pert_resampled(dist_list: List[np.ndarray]) -> Tuple[float, float]:
    """Kept for backward compatibility; delegates to _dprime_pairwise_avg."""
    return _dprime_pairwise_avg(dist_list)


def _dprime_per_pert_pooled(dist_list: List[np.ndarray]) -> Tuple[float, float]:
    """Per-perturbation Cohen's d with within/between distributions pooled across
    resamples AND target perturbations.

    For each perturbation i (all distances use pert i's weights = ground-truth frame):

        within_i  = {dist_b[i, i]  for b = 1..B}         → B values
        between_i = {dist_b[i, j]  for b = 1..B, j ≠ i}  → B*(n-1) values

        d'_i = (mean(between_i) - mean(within_i))
               / sqrt((var(within_i) + var(between_i)) / 2)

    d' = mean(d'_i) over all n perturbations.
    AUC = Φ(d' / √2).

    The σ_between includes biological diversity (different targets j) *and* cell-split
    noise (different resamples b). For large bags the cell-split component is negligible
    and σ is dominated by biological spread, which is exactly the normaliser needed to
    make d' comparable across metrics: each metric's effect size is expressed in units of
    its own natural cross-perturbation variability.
    """
    from scipy.special import ndtr
    B = len(dist_list)
    n = dist_list[0].shape[0]
    D = np.stack(dist_list, axis=0)   # (B, n, n)

    dprime_per: List[float] = []
    for i in range(n):
        within_i = D[:, i, i]                                  # (B,)
        between_i = D[:, i, :].reshape(B, n)
        between_i = np.delete(between_i, i, axis=1).ravel()    # (B*(n-1),)

        mu_w = float(np.mean(within_i))
        mu_b = float(np.mean(between_i))
        var_w = float(np.var(within_i,  ddof=1)) if within_i.size > 1 else 0.0
        var_b = float(np.var(between_i, ddof=1)) if between_i.size > 1 else 0.0
        sigma_i = float(np.sqrt((var_w + var_b) / 2.0))

        if sigma_i < 1e-15:
            d_i = float("inf") if mu_b > mu_w else 0.0
        else:
            d_i = (mu_b - mu_w) / sigma_i
        dprime_per.append(d_i)

    mean_d = float(np.mean(dprime_per))
    auc = float(ndtr(mean_d / np.sqrt(2.0)))
    return mean_d, auc


def _snr_worker(task: Tuple[int, int]) -> Dict[str, Any]:
    """Worker for SNR/d' mode: process one (trial, target_eff) pair.

    Computes the full (n_perts × n_perts) distance matrix for each metric, then
    extracts:
      within  = diagonal entries D[i,i]  (same perturbation, different half-splits)
      between = off-diagonal entries D[i,j] i≠j  (different perturbations)

    From these two distributions it derives:
      SNR     = mean(between) / mean(within)          [ratio-of-means; not cross-metric comparable]
      d'      = Cohen's d between the two distributions  [cross-metric comparable]
      AUC     = Φ(d'/√2)                              [probability between > within; interpretable]
    """
    trial, target_eff = task
    X: np.ndarray = _W["X"]
    labels: np.ndarray = _W["labels"]
    control_mean: np.ndarray = _W["control_mean"]
    raw_active: Dict[str, np.ndarray] = _W["raw_active"]
    raw_deg_weights: Dict[str, np.ndarray] = _W["raw_deg_weights"]
    calib_mode: str = _W["calib_mode"]
    metrics: List[str] = _W["metrics"]
    pair_repeats = _W["pair_repeats_by_trial"][trial]
    n_genes = int(X.shape[1])

    calibrated = base._calibrate_weights_per_perturbation(
        weights_per_pert=raw_active,
        target_effective_genes=float(target_eff),
        mode=calib_mode,
    )
    n_top = int(max(1, round(float(target_eff))))
    top_deg_local = _top_deg_indices(raw_deg_weights, n_keep=n_top)

    rows: List[SNRResult] = []
    eff: Dict[Tuple[str, int], Tuple[float, float, float]] = {}

    # Pre-compute one dist matrix per pair_repeat for each metric, then pool
    # resamples for the proper unequal-variance per-perturbation Cohen's d.
    dists_by_metric: Dict[str, List[np.ndarray]] = {m: [] for m in metrics}
    for q_arr, t_arr in pair_repeats:
        Xq = np.asarray(q_arr, dtype=float) if np.issubdtype(np.asarray(q_arr).dtype, np.floating) else X[q_arr]
        Xt = np.asarray(t_arr, dtype=float) if np.issubdtype(np.asarray(t_arr).dtype, np.floating) else X[t_arr]
        for metric in metrics:
            dists_by_metric[metric].append(_singlecell_metric_distance(
                metric=metric, Xq=Xq, Xt=Xt, labels=labels,
                control_mean=control_mean, active_weights=calibrated,
                top_deg_local_idx=top_deg_local,
            ))

    for metric in metrics:
        dist_list = dists_by_metric[metric]
        # SNR / within / between from the first dist (summary stats only)
        d0 = dist_list[0]
        n = d0.shape[0]
        mask = ~np.eye(n, dtype=bool)
        within_all = np.concatenate([np.diag(d) for d in dist_list])
        between_all = np.concatenate([d[mask] for d in dist_list])
        mu_w = float(np.mean(within_all))
        mu_b = float(np.mean(between_all))
        snr = mu_b / mu_w if mu_w > 1e-15 else float("inf")
        dprime, auc = _dprime_per_pert_pooled(dist_list)

        rows.append(SNRResult(
            metric=metric,
            x_value=int(target_eff),
            trial=trial,
            snr=snr,
            dprime=dprime,
            auc=auc,
            within=mu_w,
            between=mu_b,
        ))
        eff[(metric, int(target_eff))] = _effective_gene_stats_for_metric(
            metric=metric,
            n_features=n_genes,
            labels=labels,
            weights_per_pert=calibrated,
            top_deg_local_idx=top_deg_local,
        )

    return {"rows": rows, "eff": eff}


def _snr_bs_worker(task: Tuple[int, int]) -> Dict[str, Any]:
    """Worker for SNR bag-size mode: process one (trial, bag_size) pair.

    Uses fixed calibrated weights and sweeps bag size (cells per bag) on the x-axis.
    Builds one random bag per perturbation per half, computes the (n_perts × n_perts)
    distance matrix, then derives d', AUC, and SNR exactly as in _snr_worker.
    """
    trial, bag_size = task
    X: np.ndarray = _W["X"]
    y: np.ndarray = _W["y"]
    labels: np.ndarray = _W["labels"]
    control_mean: np.ndarray = _W["control_mean"]
    active_weights: Dict[str, np.ndarray] = _W["active_weights"]
    metrics: List[str] = _W["metrics"]
    base_seed: int = _W["base_seed"]
    n_resamples: int = int(_W.get("n_resamples", 20))
    n_genes = int(X.shape[1])
    top_deg_local = _W.get("top_deg_global", np.array([], dtype=int))

    # Sample n_resamples independent bag pairs at this bag_size
    dist_lists: Dict[str, List[np.ndarray]] = {m: [] for m in metrics}
    for r in range(n_resamples):
        seed = base_seed + trial * 10007 + int(bag_size) * 97 + r
        try:
            Xq_means, Xt_means, _, _ = _sample_one_bag_per_half(
                X=X, y=y, perturbations=list(labels), bag_size=int(bag_size), seed=seed,
            )
        except ValueError:
            return {"rows": [], "eff": {}}
        for metric in metrics:
            dist_lists[metric].append(_singlecell_metric_distance(
                metric=metric, Xq=Xq_means, Xt=Xt_means, labels=labels,
                control_mean=control_mean, active_weights=active_weights,
                top_deg_local_idx=top_deg_local,
            ))

    rows: List[SNRResult] = []
    eff: Dict[Tuple[str, int], Tuple[float, float, float]] = {}

    for metric in metrics:
        dist_list = dist_lists[metric]
        n = dist_list[0].shape[0]
        mask = ~np.eye(n, dtype=bool)
        within_all = np.concatenate([np.diag(d) for d in dist_list])
        between_all = np.concatenate([d[mask] for d in dist_list])
        mu_w = float(np.mean(within_all))
        mu_b = float(np.mean(between_all))
        snr = mu_b / mu_w if mu_w > 1e-15 else float("inf")
        dprime, auc = _dprime_per_pert_pooled(dist_list)

        rows.append(SNRResult(
            metric=metric,
            x_value=int(bag_size),
            trial=trial,
            snr=snr,
            dprime=dprime,
            auc=auc,
            within=mu_w,
            between=mu_b,
        ))
        eff[(metric, int(bag_size))] = _effective_gene_stats_for_metric(
            metric=metric,
            n_features=n_genes,
            labels=labels,
            weights_per_pert=active_weights,
            top_deg_local_idx=top_deg_local,
        )

    return {"rows": rows, "eff": eff}


def _snr_rank_worker(task: Tuple[int, int]) -> Dict[str, Any]:
    """Worker for SNR rank mode: process one (trial, k) pair.

    Sweeps top-k genes from a ranking (PDS or DEG), uses all-cell splits,
    and computes Cohen's d' and AUC for each metric.
    """
    trial, k = task
    X: np.ndarray = _W["X"]
    y: np.ndarray = _W["y"]
    labels: np.ndarray = _W["labels"]
    control_mean: np.ndarray = _W["control_mean"]
    active_weights: Dict[str, np.ndarray] = _W["active_weights"]
    gene_ranked: np.ndarray = _W["gene_ranked"]
    deg_ranked: np.ndarray = _W.get("deg_ranked", np.array([], dtype=int))
    n_top_deg: int = int(_W.get("n_top_deg", 0))
    metrics: List[str] = _W["metrics"]
    pair_repeats = _W["pair_repeats_by_trial"][trial]
    n_genes = int(X.shape[1])

    # Select top-k genes by ranking
    selected = np.asarray(gene_ranked[:k], dtype=int)
    X_sel = X[:, selected]
    ctrl_sel = control_mean[selected]
    w_subset = {pert: np.asarray(w, dtype=float)[selected] for pert, w in active_weights.items()}

    # Resolve top-N DEG indices within the selected gene subset
    if deg_ranked.size > 0 and n_top_deg > 0:
        selected_set = set(int(x) for x in selected.tolist())
        top_deg_in_selected: List[int] = []
        for g in deg_ranked.tolist():
            gi = int(g)
            if gi in selected_set:
                top_deg_in_selected.append(gi)
                if len(top_deg_in_selected) >= min(int(n_top_deg), int(selected.size)):
                    break
        if top_deg_in_selected:
            top_deg_local = np.asarray(
                [int(np.where(selected == gi)[0][0]) for gi in top_deg_in_selected], dtype=int
            )
        else:
            top_deg_local = np.array([], dtype=int)
    else:
        top_deg_local = np.array([], dtype=int)

    rows: List[SNRResult] = []
    eff: Dict[Tuple[str, int], Tuple[float, float, float]] = {}

    # Pre-compute distance matrices for each pair_repeat
    dists_by_metric: Dict[str, List[np.ndarray]] = {m: [] for m in metrics}
    for q_arr, t_arr in pair_repeats:
        # q_arr/t_arr can be bag means (float) or cell indices (int)
        if np.issubdtype(np.asarray(q_arr).dtype, np.floating):
            # Already bag means - just subset genes (columns)
            Xq = np.asarray(q_arr, dtype=float)[:, selected]
            Xt = np.asarray(t_arr, dtype=float)[:, selected]
        else:
            # Cell indices - subset cells and genes
            Xq = X_sel[np.asarray(q_arr, dtype=int), :]
            Xt = X_sel[np.asarray(t_arr, dtype=int), :]
        for metric in metrics:
            dists_by_metric[metric].append(_singlecell_metric_distance(
                metric=metric, Xq=Xq, Xt=Xt, labels=labels,
                control_mean=ctrl_sel, active_weights=w_subset,
                top_deg_local_idx=top_deg_local,
            ))

    for metric in metrics:
        dist_list = dists_by_metric[metric]
        d0 = dist_list[0]
        n = d0.shape[0]
        mask = ~np.eye(n, dtype=bool)
        within_all = np.concatenate([np.diag(d) for d in dist_list])
        between_all = np.concatenate([d[mask] for d in dist_list])
        mu_w = float(np.mean(within_all))
        mu_b = float(np.mean(between_all))
        snr = mu_b / mu_w if mu_w > 1e-15 else float("inf")
        dprime, auc = _dprime_per_pert_pooled(dist_list)

        rows.append(SNRResult(
            metric=metric,
            x_value=int(k),
            trial=trial,
            snr=snr,
            dprime=dprime,
            auc=auc,
            within=mu_w,
            between=mu_b,
        ))
        eff[(metric, int(k))] = _effective_gene_stats_for_metric(
            metric=metric,
            n_features=n_genes,
            labels=labels,
            weights_per_pert=active_weights,
            top_deg_local_idx=top_deg_local,
        )

    return {"rows": rows, "eff": eff}


def _bs_worker(task: Tuple[int, int]) -> Dict[str, Any]:
    """Worker for bag-size mode: process one (trial, bag_size) pair."""
    trial, bag_size = task
    X: np.ndarray = _W["X"]
    y: np.ndarray = _W["y"]
    labels: np.ndarray = _W["labels"]
    control_mean: np.ndarray = _W["control_mean"]
    active_weights: Dict[str, np.ndarray] = _W["active_weights"]
    top_deg_global: np.ndarray = _W.get("top_deg_global", np.array([], dtype=int))
    metrics: List[str] = _W["metrics"]
    perturbations: List[str] = _W["perturbations"]
    trial_cell_budget: int = _W["trial_cell_budget"]
    base_seed: int = _W["base_seed"]
    bag_normalize: bool = _W["bag_normalize"]
    max_bag_size: int = _W["max_bag_size"]

    bag_pairs = _build_bag_pairs_for_trial(
        y=y,
        perturbations=perturbations,
        bag_size=int(bag_size),
        trial_cell_budget=trial_cell_budget,
        seed=base_seed + trial * 1009 + int(bag_size),
        normalize_repeats=bag_normalize,
        max_bag_size=max_bag_size,
    )

    rows: List[TrialResult] = []
    eff: Dict[Tuple[str, int], Tuple[float, float, float]] = {}
    hmats: Dict[Tuple[str, int], np.ndarray] = {}
    per_metric_vals: Dict[str, List[float]] = {metric: [] for metric in metrics}

    for metric in metrics:
        eff[(metric, int(bag_size))] = _effective_gene_stats_for_metric(
            metric=metric,
            n_features=int(X.shape[1]),
            labels=labels,
            weights_per_pert=active_weights,
            top_deg_local_idx=top_deg_global,
        )

    for rep, (q_members, t_members) in enumerate(bag_pairs):
        Xq_mean = np.asarray([np.mean(X[idx, :], axis=0) for idx in q_members], dtype=float)
        Xt_mean = np.asarray([np.mean(X[idx, :], axis=0) for idx in t_members], dtype=float)
        for metric in metrics:
            dist = _bag_metric_distance(
                metric=metric,
                Xq_mean=Xq_mean,
                Xt_mean=Xt_mean,
                X_cells=X,
                q_members=q_members,
                t_members=t_members,
                labels=labels,
                control_mean=control_mean,
                active_weights=active_weights,
                top_deg_local_idx=top_deg_global,
            )
            scores = _query_truth_scores(dist)
            per_metric_vals[metric].append(float(np.mean(scores)))
            if trial == 0 and rep == 0:
                hmats[(metric, int(bag_size))] = np.asarray(dist, dtype=float)

    for metric in metrics:
        rows.append(TrialResult(
            metric=metric,
            x_value=int(bag_size),
            trial=trial,
            score=float(np.mean(np.asarray(per_metric_vals[metric], dtype=float))),
        ))

    return {"rows": rows, "eff": eff, "hmats": hmats}


def _bag_rank_worker(task: Tuple[int, int, int]) -> Dict[str, Any]:
    """Worker for bag_rank_curve: process one (trial, k, bag_size) triplet."""
    trial, k, bag_size = task
    X: np.ndarray = _W["X"]
    labels: np.ndarray = _W["labels"]
    control_mean: np.ndarray = _W["control_mean"]
    active_weights: Dict[str, np.ndarray] = _W["active_weights"]
    gene_ranked: np.ndarray = _W["gene_ranked"]
    deg_ranked: np.ndarray = _W.get("deg_ranked", np.array([], dtype=int))
    n_top_deg: int = int(_W.get("n_top_deg", 0))
    metrics: List[str] = _W["metrics"]
    bag_pairs_by_trial_size: Dict[Tuple[int, int], Any] = _W["bag_pairs_by_trial_size"]

    selected = np.asarray(gene_ranked[:k], dtype=int)
    X_sel = X[:, selected]
    ctrl_sel = control_mean[selected]
    w_subset = {pert: np.asarray(w, dtype=float)[selected] for pert, w in active_weights.items()}

    # Resolve top-N DEG indices within the selected gene subset.
    if deg_ranked.size > 0 and n_top_deg > 0:
        selected_set = set(int(x) for x in selected.tolist())
        top_deg_in_selected: List[int] = []
        for g in deg_ranked.tolist():
            gi = int(g)
            if gi in selected_set:
                top_deg_in_selected.append(gi)
                if len(top_deg_in_selected) >= min(int(n_top_deg), int(selected.size)):
                    break
        if top_deg_in_selected:
            top_deg_local = np.asarray(
                [int(np.where(selected == gi)[0][0]) for gi in top_deg_in_selected], dtype=int
            )
        else:
            top_deg_local = np.array([], dtype=int)
    else:
        top_deg_local = np.array([], dtype=int)

    bag_pairs = bag_pairs_by_trial_size[(trial, bag_size)]

    rows: List[TrialResult] = []
    for metric in metrics:
        rep_scores: List[float] = []
        for q_members, t_members in bag_pairs:
            Xq_mean = np.asarray([np.mean(X_sel[idx, :], axis=0) for idx in q_members], dtype=float)
            Xt_mean = np.asarray([np.mean(X_sel[idx, :], axis=0) for idx in t_members], dtype=float)
            dist = _bag_metric_distance(
                metric=metric,
                Xq_mean=Xq_mean,
                Xt_mean=Xt_mean,
                X_cells=X_sel,
                q_members=q_members,
                t_members=t_members,
                labels=labels,
                control_mean=ctrl_sel,
                active_weights=w_subset,
                top_deg_local_idx=top_deg_local,
            )
            scores = _query_truth_scores(dist)
            rep_scores.append(float(np.mean(scores)))
        rows.append(TrialResult(
            metric=metric,
            x_value=int(k),
            trial=trial,
            score=float(np.mean(np.asarray(rep_scores, dtype=float))),
        ))

    return {"rows": rows, "bag_size": bag_size}


def _snr_bag_rank_worker(task: Tuple[int, int, int, int]) -> Dict[str, Any]:
    """Worker for snr_bag_rank_curve: process one (trial, k, bag_size, resample) quadruplet.

    Similar to _snr_bs_worker but with gene subset selection. Each task generates
    one distance matrix for a specific bag pair resample.
    """
    trial, k, bag_size, resample_idx = task
    X: np.ndarray = _W["X"]
    y: np.ndarray = _W["y"]
    labels: np.ndarray = _W["labels"]
    control_mean: np.ndarray = _W["control_mean"]
    active_weights: Dict[str, np.ndarray] = _W["active_weights"]
    gene_ranked: np.ndarray = _W["gene_ranked"]
    deg_ranked: np.ndarray = _W.get("deg_ranked", np.array([], dtype=int))
    n_top_deg: int = int(_W.get("n_top_deg", 0))
    metrics: List[str] = _W["metrics"]
    n_genes = int(X.shape[1])
    base_seed: int = _W["base_seed"]

    # Select top-k genes by ranking
    selected = np.asarray(gene_ranked[:k], dtype=int)
    X_sel = X[:, selected]
    ctrl_sel = control_mean[selected]
    w_subset = {pert: np.asarray(w, dtype=float)[selected] for pert, w in active_weights.items()}

    # Resolve top-N DEG indices within the selected gene subset
    if deg_ranked.size > 0 and n_top_deg > 0:
        selected_set = set(int(x) for x in selected.tolist())
        top_deg_in_selected: List[int] = []
        for g in deg_ranked.tolist():
            gi = int(g)
            if gi in selected_set:
                top_deg_in_selected.append(gi)
                if len(top_deg_in_selected) >= min(int(n_top_deg), int(selected.size)):
                    break
        if top_deg_in_selected:
            top_deg_local = np.asarray(
                [int(np.where(selected == gi)[0][0]) for gi in top_deg_in_selected], dtype=int
            )
        else:
            top_deg_local = np.array([], dtype=int)
    else:
        top_deg_local = np.array([], dtype=int)

    # Sample one bag pair with unique seed per resample
    seed = base_seed + trial * 10007 + k * 97 + bag_size * 13 + resample_idx
    try:
        Xq_mean, Xt_mean, _, _ = _sample_one_bag_per_half(
            X=X_sel, y=y, perturbations=list(labels), bag_size=int(bag_size), seed=seed,
        )
    except ValueError:
        return {"rows": [], "eff": {}, "bag_size": bag_size, "k": k, "trial": trial}

    rows: List[SNRResult] = []
    eff: Dict[Tuple[str, int], Tuple[float, float, float]] = {}

    # Compute distance matrix for each metric (one dist matrix per resample)
    for metric in metrics:
        dist = _singlecell_metric_distance(
            metric=metric,
            Xq=Xq_mean,
            Xt=Xt_mean,
            labels=labels,
            control_mean=ctrl_sel,
            active_weights=w_subset,
            top_deg_local_idx=top_deg_local,
        )

        rows.append(SNRResult(
            metric=metric,
            x_value=int(k),
            trial=trial,
            snr=0.0,  # Will be computed after aggregating resamples
            dprime=0.0,  # Will be computed after aggregating resamples
            auc=0.0,  # Will be computed after aggregating resamples
            within=float(np.mean(np.diag(dist))),
            between=float(np.mean(dist[~np.eye(dist.shape[0], dtype=bool)])),
            _dist=dist,  # Store distance matrix for aggregation
        ))
        eff[(metric, int(k))] = _effective_gene_stats_for_metric(
            metric=metric,
            n_features=n_genes,
            labels=labels,
            weights_per_pert=active_weights,
            top_deg_local_idx=top_deg_local,
        )

    return {"rows": rows, "eff": eff, "bag_size": bag_size, "k": k, "trial": trial, "resample": resample_idx}


# ── Mode entry points ────────────────────────────────────────────────


def run_single_cell_mode(args: argparse.Namespace, dataset_name: str, data: base.PreparedData, out_dir: Path) -> None:
    metrics = _parse_metrics(args.metrics)
    gene_ranking_source: str = getattr(args, "gene_ranking", "pds")
    ranking_label = "DEG" if gene_ranking_source == "deg" else "PDS"
    suffix = f"_{ranking_label.lower()}" if ranking_label != "PDS" else ""
    if bool(getattr(args, "plot_only", False)):
        csv_path = out_dir / f"single_cell_pds_rank_curve{suffix}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing summary CSV for --plot-only: {csv_path}")
        _plot_curve_from_summary_csv(
            csv_path=csv_path,
            out_png=out_dir / f"single_cell_pds_rank_curve{suffix}.png",
            x_col="n_genes",
            x_label="Top-k genes from per-gene PDS ranking",
            title=f"Single-cell signal dilution ({dataset_name})",
            log_x=True,
            y_label="Mean PDS",
            figure_size=(4.8, 2.8),
            legend_outside_right=True,
        )
        return
    bag_only = {"energy_distance", "weighted_energy_distance", "std_energy_distance", "w_std_energy_distance"}
    if any(metric in bag_only for metric in metrics):
        raise ValueError(f"Bag-only metrics ({bag_only & set(metrics)}) are not supported in single_cell_pds_rank_curve mode.")
    n_genes = int(data.X.shape[1])
    topk_values = [k for k in _resolve_topk_values(args.topk_values, n_genes=n_genes) if k <= n_genes]
    if not topk_values or topk_values[-1] != n_genes:
        topk_values.append(n_genes)
    topk_values = sorted(set(topk_values))
    if not topk_values:
        raise ValueError("No top-k value is <= n_genes.")

    deg_weights, _pds_weights, active_weights = _load_weight_families(args, dataset_name, data)

    gene_ranking_source = getattr(args, "gene_ranking", "pds")
    if gene_ranking_source == "deg":
        gene_ranked = _resolve_ranked_deg_gene_indices(deg_weights, n_genes=int(data.X.shape[1]))
        ranking_label = "DEG"
    else:
        gene_ranked = _resolve_ranked_pds_gene_indices(
            dataset_name=dataset_name,
            genes=data.genes,
            perturbations=data.perturbations,
            pds_result_file=args.pds_result_file,
            pds_metric=args.pds_metric,
        )
        ranking_label = "PDS"
    n_top_deg = int(max(1, round(float(args.target_effective_genes)))) if args.target_effective_genes is not None else 100
    # Full DEG ranking used to pick top-N within each selected gene subset.
    deg_ranked = _resolve_ranked_deg_gene_indices(deg_weights, n_genes=int(data.X.shape[1]))

    X = np.asarray(data.X, dtype=float)
    y = np.asarray(data.y, dtype=str)
    labels = np.asarray(data.perturbations, dtype=str)
    ctrl_idx = np.where(y == data.control_label)[0]
    control_mean = np.mean(X[ctrl_idx, :], axis=0) if ctrl_idx.size > 0 else np.mean(X, axis=0)

    trial_rows: List[TrialResult] = []
    eff_stats: Dict[Tuple[str, int], Tuple[float, float, float]] = {}
    heatmap_mats: Dict[Tuple[str, int], np.ndarray] = {}
    n_trials = int(args.n_trials)
    n_k = len(topk_values)
    n_jobs = _resolve_n_jobs(args)

    pair_repeats_by_trial = {
        trial: _build_single_cell_pairs_for_trial(
            y=y,
            perturbations=data.perturbations,
            trial_cell_budget=int(args.trial_cell_budget),
            seed=int(args.seed) + trial,
        )
        for trial in range(n_trials)
    }

    _set_worker_state(
        X=X, labels=labels, control_mean=control_mean,
        active_weights=active_weights, gene_ranked=gene_ranked,
        deg_ranked=deg_ranked, n_top_deg=n_top_deg, metrics=metrics,
        pair_repeats_by_trial=pair_repeats_by_trial,
    )

    tasks = [(trial, k) for trial in range(n_trials) for k in topk_values]
    n_tasks = len(tasks)
    print(f"  single-cell mode: {n_trials} trials x {n_k} k-values x {len(metrics)} metrics ({n_tasks} tasks, {n_jobs} jobs)")

    if n_jobs > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            for i, result in enumerate(pool.imap_unordered(_sc_worker, tasks)):
                _merge_worker_result(result, trial_rows, eff_stats, heatmap_mats)
                if (i + 1) % max(1, n_tasks // 20) == 0 or (i + 1) == n_tasks:
                    print(f"  completed {i + 1}/{n_tasks} tasks", end="\r", flush=True)
    else:
        for i, task in enumerate(tasks):
            t, k = task
            print(f"  trial {t + 1}/{n_trials}  k={k} ({i + 1}/{n_tasks})", end="\r", flush=True)
            result = _sc_worker(task)
            _merge_worker_result(result, trial_rows, eff_stats, heatmap_mats)

    print()  # finish carriage-return progress line
    setattr(_save_curve, "_effective_gene_stats", eff_stats)
    _save_curve(trial_rows, out_csv=out_dir / f"single_cell_pds_rank_curve{suffix}.csv", x_col="n_genes")
    weight_mode = (
        str(args.pds_target_effective_genes_mode)
        if str(args.weight_source) == "pds"
        else str(args.target_effective_genes_mode)
    )
    annotation_lines = [
        f"Weighting: {str(args.weight_source).upper()} ({str(args.weight_scheme)})",
        f"Calibration: {weight_mode}",
        f"Delta baseline mode: {str(args.r2_delta_mode)}",
        "Bag size: 1 cell per side",
    ]
    _plot_curve(
        trial_rows,
        out_png=out_dir / f"single_cell_pds_rank_curve{suffix}.png",
        x_label="Top-k genes from per-gene PDS ranking",
        title=f"Single-cell signal dilution ({dataset_name})",
        log_x=True,
        effective_gene_stats=eff_stats,
        y_label="Mean PDS",
        show_eff_gene_box=False,
        annotation_box=None,
        figure_size=(4.8, 2.8),
        legend_outside_right=True,
    )
    _save_heatmaps(args, out_dir / f"heatmaps_single_cell{suffix}", heatmap_mats, labels=labels, x_values=topk_values)


def run_effective_genes_mode(
    args: argparse.Namespace,
    dataset_name: str,
    data: base.PreparedData,
    out_dir: Path,
) -> None:
    """Sweep target effective gene number on x-axis, single-cell PDS on y-axis.

    Uses all genes but recalibrates weights for each target effective gene count.
    Top-N DEG metrics (if selected) are evaluated on a fixed DEG subset and
    therefore provide a constant-reference line against weight-calibrated metrics.
    """
    metrics = _parse_metrics(args.metrics)
    log_x = not bool(getattr(args, "no_log_x", False))

    # When --no-log-x is set and user hasn't overridden sampling, switch to linear
    eff_genes_values_raw = str(args.effective_genes_values)
    if not log_x and eff_genes_values_raw.strip().lower() in {"auto", "auto_log10", "default"}:
        eff_genes_values_raw = "auto_linear"

    if bool(getattr(args, "plot_only", False)):
        csv_path = out_dir / "effective_genes_curve.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing summary CSV for --plot-only: {csv_path}")
        _plot_curve_from_summary_csv(
            csv_path=csv_path,
            out_png=out_dir / "effective_genes_curve.png",
            x_col="target_effective_genes",
            x_label="Target effective gene number",
            title=f"Effective genes vs. single-cell PDS ({dataset_name})",
            log_x=log_x,
            y_label="PDS",
            show_legend=True,
            legend_outside_right=True,
            figure_size=EFFECTIVE_GENES_CURVE_FIG_SIZE,
        )
        return
    bag_only = {"std_energy_distance", "w_std_energy_distance"}
    if any(m in bag_only for m in metrics):
        raise ValueError(f"effective_genes_curve does not support {bag_only & set(metrics)}.")

    # Filter perturbations by type (all/single/double)
    filter_mode = str(getattr(args, "perturbation_filter", "all"))
    filtered_perts = _filter_perturbations_by_type(data.perturbations, filter_mode)
    if not filtered_perts:
        raise ValueError(f"No perturbations remain after applying filter='{filter_mode}'.")
    print(f"  Filter mode '{filter_mode}': using {len(filtered_perts)} perturbations (from {len(data.perturbations)} total)")

    # Optionally cap to top/bottom N perturbations by cell count
    max_perts = int(args.max_perturbations) if args.max_perturbations is not None else 0
    pert_sort = str(getattr(args, "perturbation_sort", "top"))
    if max_perts > 0 and len(filtered_perts) > max_perts:
        y_tmp = np.asarray(data.y, dtype=str)
        counts = np.asarray([int(np.sum(y_tmp == p)) for p in filtered_perts], dtype=int)
        if pert_sort == "bottom":
            order = np.argsort(counts)  # ascending: fewest cells first
        else:
            order = np.argsort(counts)[::-1]  # descending: most cells first
        filtered_perts = [filtered_perts[i] for i in order[:max_perts]]
        print(f"  Perturbation sort '{pert_sort}': capped to {max_perts} perturbations")

    # Filter data to only include cells from filtered perturbations
    y_all = np.asarray(data.y, dtype=str)
    mask = np.isin(y_all, filtered_perts)
    if not np.any(mask):
        raise ValueError("No cells remain after perturbation filtering.")
    X_filtered = np.asarray(data.X, dtype=float)[mask, :]
    y_filtered = y_all[mask]
    labels = np.asarray(filtered_perts, dtype=str)

    n_genes = int(X_filtered.shape[1])
    eff_values = _resolve_effective_genes_values(eff_genes_values_raw, n_genes)
    eff_values = _apply_max_effective_genes_cap(eff_values, args, metrics)
    if not eff_values:
        raise ValueError("No effective-gene values are <= n_genes.")

    deg_transform, pds_transform = _resolve_weight_transforms(args.weight_scheme)
    raw_deg_weights = base._load_deg_weights(
        dataset_name=dataset_name,
        genes=data.genes,
        perturbations=filtered_perts,
        deg_weight_transform=deg_transform,
        base_weight_exponent=float(args.base_weight_exponent),
    )
    raw_pds_weights = base._load_pds_weights(
        dataset_name=dataset_name,
        genes=data.genes,
        perturbations=filtered_perts,
        pds_result_file=args.pds_result_file,
        pds_metric=args.pds_metric,
        pds_weight_transform=pds_transform,
        base_weight_exponent=float(args.base_weight_exponent),
    )
    raw_active = raw_deg_weights if args.weight_source == "deg" else raw_pds_weights

    X = X_filtered
    y = y_filtered
    ctrl_idx = np.where(y == data.control_label)[0]
    control_mean = np.mean(X[ctrl_idx, :], axis=0) if ctrl_idx.size > 0 else np.mean(X, axis=0)
    dataset_mean = np.mean(X, axis=0)

    trial_rows: List[TrialResult] = []
    eff_stats: Dict[Tuple[str, int], Tuple[float, float, float]] = {}
    n_trials = int(args.n_trials)
    n_ev = len(eff_values)
    n_jobs = _resolve_n_jobs(args)

    calib_mode = str(args.target_effective_genes_mode) if args.weight_source == "deg" else str(args.pds_target_effective_genes_mode)

    # Choose pair building function based on split mode
    split_mode = str(getattr(args, "split_mode", "budget"))
    if split_mode == "all_cells":
        print(f"  Split mode 'all_cells': bag size = n_cells//2 per perturbation, comparing bag means (imbalance allowed)")
        pair_repeats_by_trial = {
            trial: _build_bag_mean_pairs_all_cells(
                X=X,
                y=y,
                perturbations=filtered_perts,
                seed=int(args.seed) + trial,
            )
            for trial in range(n_trials)
        }
    else:
        print(f"  Split mode 'budget': using up to {args.trial_cell_budget} cells per perturbation")
        pair_repeats_by_trial = {
            trial: _build_single_cell_pairs_for_trial(
                y=y,
                perturbations=filtered_perts,
                trial_cell_budget=int(args.trial_cell_budget),
                seed=int(args.seed) + trial,
            )
            for trial in range(n_trials)
        }

    _set_worker_state(
        X=X,
        labels=labels,
        control_mean=control_mean,
        dataset_mean=dataset_mean,
        raw_deg_weights=raw_deg_weights,
        raw_active=raw_active,
        calib_mode=calib_mode,
        metrics=metrics,
        pair_repeats_by_trial=pair_repeats_by_trial,
    )

    tasks = [(trial, ev) for trial in range(n_trials) for ev in eff_values]
    n_tasks = len(tasks)
    print(f"  effective-genes mode: {n_trials} trials x {n_ev} eff-gene values x {len(metrics)} metrics ({n_tasks} tasks, {n_jobs} jobs)")

    if n_jobs > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            for i, result in enumerate(pool.imap_unordered(_eg_worker, tasks)):
                _merge_worker_result(result, trial_rows, eff_stats)
                if (i + 1) % max(1, n_tasks // 20) == 0 or (i + 1) == n_tasks:
                    print(f"  completed {i + 1}/{n_tasks} tasks", end="\r", flush=True)
    else:
        for i, task in enumerate(tasks):
            t, ev = task
            print(f"  trial {t + 1}/{n_trials}  eff_genes={ev} ({i + 1}/{n_tasks})", end="\r", flush=True)
            result = _eg_worker(task)
            _merge_worker_result(result, trial_rows, eff_stats)

    print()  # finish carriage-return progress line
    setattr(_save_curve, "_effective_gene_stats", eff_stats)
    _save_curve(trial_rows, out_csv=out_dir / "effective_genes_curve.csv", x_col="target_effective_genes")
    weight_mode = (
        str(args.pds_target_effective_genes_mode)
        if str(args.weight_source) == "pds"
        else str(args.target_effective_genes_mode)
    )
    _plot_curve(
        trial_rows,
        out_png=out_dir / "effective_genes_curve.png",
        x_label="Target effective gene number",
        title=f"Effective genes vs. single-cell PDS ({dataset_name})",
        log_x=log_x,
        effective_gene_stats=eff_stats,
        show_eff_gene_box=False,
        annotation_box=None,
        y_label="PDS",
        show_legend=True,
        legend_outside_right=True,
        figure_size=EFFECTIVE_GENES_CURVE_FIG_SIZE,
    )


def _bag_metric_distance(
    metric: str,
    Xq_mean: np.ndarray,
    Xt_mean: np.ndarray,
    X_cells: np.ndarray,
    q_members: Sequence[np.ndarray],
    t_members: Sequence[np.ndarray],
    labels: np.ndarray,
    control_mean: np.ndarray,
    active_weights: Dict[str, np.ndarray],
    top_deg_local_idx: Optional[np.ndarray] = None,
) -> np.ndarray:
    n_g = Xq_mean.shape[1]
    uniform = np.ones(n_g, dtype=float)

    def _dist_mat(dist_fn, use_w):
        out = np.zeros((len(q_members), len(t_members)), dtype=float)
        for i in range(len(q_members)):
            w = (
                np.asarray(active_weights.get(str(labels[i]), np.zeros(n_g, dtype=float)), dtype=float)
                if use_w else uniform
            )
            for j in range(len(t_members)):
                out[i, j] = dist_fn(X_cells[q_members[i], :], X_cells[t_members[j], :], w)
        return out

    if metric == "energy_distance":
        return _dist_mat(_energy_distance_from_samples, False)
    if metric == "weighted_energy_distance":
        return _dist_mat(_energy_distance_from_samples, True)
    if metric == "std_energy_distance":
        return _dist_mat(_std_energy_distance_from_samples, False)
    if metric == "w_std_energy_distance":
        return _dist_mat(_std_energy_distance_from_samples, True)
    if metric == "mse":
        return np.mean(np.square(Xq_mean[:, None, :] - Xt_mean[None, :, :]), axis=2)
    if metric == "top_n_deg_mse":
        if top_deg_local_idx is None or top_deg_local_idx.size == 0:
            raise ValueError("No overlap between top DEG indices and selected genes.")
        q = Xq_mean[:, top_deg_local_idx]
        t = Xt_mean[:, top_deg_local_idx]
        return np.mean(np.square(q[:, None, :] - t[None, :, :]), axis=2)
    if metric == "wmse":
        return _query_weighted_sqeuclidean(Xq_mean, Xt_mean, labels, active_weights)
    if metric == "top_n_deg_r2":
        if top_deg_local_idx is None or top_deg_local_idx.size == 0:
            raise ValueError("No overlap between top DEG indices and selected genes.")
        idx = np.asarray(top_deg_local_idx, dtype=int)
        q = Xq_mean[:, idx] - control_mean[idx][None, :]
        t = Xt_mean[:, idx] - control_mean[idx][None, :]
        uniform_deg = {k: np.ones(idx.size, dtype=float) for k in active_weights}
        return _query_r2_distance(q, t, labels, uniform_deg)
    if metric == "r2_delta":
        uniform_w = {k: uniform for k in active_weights}
        return _query_r2_distance(
            Xq_mean - control_mean[None, :],
            Xt_mean - control_mean[None, :],
            labels,
            uniform_w,
        )
    if metric == "w_r2_delta":
        return _query_r2_distance(
            Xq_mean - control_mean[None, :],
            Xt_mean - control_mean[None, :],
            labels,
            active_weights,
        )
    if metric == "pearson_delta":
        return _query_weighted_corrdist(
            Xq_mean - control_mean[None, :], Xt_mean - control_mean[None, :],
            labels, {k: uniform for k in active_weights},
        )
    if metric == "w_pearson_delta":
        return _query_weighted_corrdist(
            Xq_mean - control_mean[None, :], Xt_mean - control_mean[None, :],
            labels, active_weights,
        )
    if metric == "top_n_deg_pearson":
        if top_deg_local_idx is None or top_deg_local_idx.size == 0:
            raise ValueError("No overlap between top DEG indices and selected genes.")
        idx = np.asarray(top_deg_local_idx, dtype=int)
        q = Xq_mean[:, idx] - control_mean[idx][None, :]
        t = Xt_mean[:, idx] - control_mean[idx][None, :]
        uniform_deg = {k: np.ones(idx.size, dtype=float) for k in active_weights}
        return _query_weighted_corrdist(q, t, labels, uniform_deg)
    if metric == "cosine_sim":
        return _query_cosine_distance(Xq_mean, Xt_mean, labels, {k: uniform for k in active_weights})
    if metric == "w_cosine_sim":
        return _query_cosine_distance(Xq_mean, Xt_mean, labels, active_weights)
    raise ValueError(f"Unsupported bag metric: {metric}")


def _save_snr_csv(results: List[SNRResult], out_csv: Path, x_col: str = "x_value") -> None:
    """Write SNR/d'/AUC results to a tidy CSV aggregated over trials."""
    import csv
    from collections import defaultdict
    accum: Dict[Tuple[str, int], List[SNRResult]] = defaultdict(list)
    for r in results:
        accum[(r.metric, r.x_value)].append(r)

    rows_out = []
    for (metric, x_val), rlist in sorted(accum.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rows_out.append({
            "metric": metric,
            x_col: x_val,
            "dprime_mean": float(np.mean([r.dprime for r in rlist])),
            "dprime_std": float(np.std([r.dprime for r in rlist])),
            "auc_mean": float(np.mean([r.auc for r in rlist])),
            "auc_std": float(np.std([r.auc for r in rlist])),
            "snr_mean": float(np.mean([r.snr for r in rlist])),
            "snr_std": float(np.std([r.snr for r in rlist])),
            "within_mean": float(np.mean([r.within for r in rlist])),
            "within_std": float(np.std([r.within for r in rlist])),
            "between_mean": float(np.mean([r.between for r in rlist])),
            "between_std": float(np.std([r.between for r in rlist])),
            "n_trials": len(rlist),
        })
    if not rows_out:
        raise ValueError(f"No results to save to {out_csv}. Check that workers produced valid results.")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)


def _load_snr_csv(csv_path: Path) -> List[SNRResult]:
    """Load SNR/d'/AUC results from a CSV file (for --plot-only mode)."""
    import csv
    results: List[SNRResult] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        x_col = None
        for row in reader:
            if x_col is None:
                # Determine x-axis column name (could be target_effective_genes, bag_size, or n_genes)
                for col in ["target_effective_genes", "bag_size", "n_genes", "x_value"]:
                    if col in row:
                        x_col = col
                        break
            results.append(SNRResult(
                metric=row["metric"],
                x_value=int(float(row[x_col])) if x_col else 0,
                trial=0,  # Aggregated data - trial info lost
                dprime=float(row["dprime_mean"]),
                auc=float(row["auc_mean"]),
                snr=float(row["snr_mean"]),
                within=float(row["within_mean"]),
                between=float(row["between_mean"]),
            ))
    return results


def _plot_snr_curve(
    results: List[SNRResult],
    out_png: Path,
    dataset_name: str,
    log_x: bool = True,
    x_label: str = "Target effective gene number",
) -> None:
    """Plot d' (Cohen's d) and AUC as two side-by-side panels.

    Uses the same Nature-style formatting, colors, and labels as other discrimination plots.
    Only shows d' and AUC as these are cross-metric comparable. Raw distance plots are
    omitted because different metrics have vastly different scales (e.g., MSE vs Pearson).

    Panel layout:
      1. d' (Cohen's d) — cross-metric comparable discriminability index
      2. AUC = Φ(d'/√2) — probability that between-dist > within-dist
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from collections import defaultdict
    dprime_by: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    auc_by: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        dprime_by[r.metric][r.x_value].append(r.dprime)
        auc_by[r.metric][r.x_value].append(r.auc)

    metric_names = _ordered_metrics(list(dprime_by.keys()))
    _apply_nature_rc()

    panel_w, panel_h = NATURE_FIG_SIZE
    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2.5, panel_h))
    panels = [
        (axes[0], dprime_by,  "d′ (Cohen's d)",           "d′",       False),
        (axes[1], auc_by,     "AUC = Φ(d′/√2)",          "AUC",      False),
    ]

    legend_handles: List[Any] = []
    legend_labels: List[str] = []
    for ax, data_by, title, ylabel, log_y in panels:
        for metric in metric_names:
            if metric not in data_by:
                continue
            xs = sorted(data_by[metric].keys())
            ys_mean = np.asarray([float(np.mean(data_by[metric][x])) for x in xs])
            ys_std  = np.asarray([float(np.std(data_by[metric][x]))  for x in xs])
            color, linestyle = _metric_plot_style(metric)
            marker = _metric_marker(metric)
            line, = ax.plot(xs, ys_mean,
                            marker=marker, markersize=2.7, linewidth=1.3,
                            linestyle=linestyle, color=color,
                            label=_metric_label(metric))
            ax.fill_between(xs, ys_mean - ys_std, ys_mean + ys_std, alpha=0.14, color=color)
            if ax is axes[0]:
                legend_handles.append(line)
                legend_labels.append(_metric_label(metric))
        if log_x:
            ax.set_xscale("log", base=10)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n({dataset_name})", fontsize=NATURE_TITLE_SIZE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)

    fig.legend(
        legend_handles, legend_labels,
        frameon=False, fontsize=NATURE_LEGEND_SIZE,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.85, 1.0))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=NATURE_DPI, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run_snr_curve(
    args: argparse.Namespace,
    dataset_name: str,
    data: base.PreparedData,
    out_dir: Path,
) -> None:
    """Sweep target effective gene number; report SNR = between / within distance.

    For each metric M and target effective gene count G:
      - Calibrate per-perturbation weights to G effective genes.
      - For each perturbation P split all cells into two halves (all_cells mode).
      - Compute the (n_perts x n_perts) pairwise distance matrix using M.
      - within_dist(P)  = dist[P, P]  (same perturbation, different halves)
      - between_dist(P) = mean dist[P, Q] for Q != P
      - SNR(P) = between_dist(P) / within_dist(P)
      - Report mean SNR, mean within, and mean between over all perturbations.

    SNR is dimensionless and directly comparable across metrics because the same
    distance function is used in both numerator and denominator.
    """
    metrics = _parse_metrics(args.metrics)
    log_x = not bool(getattr(args, "no_log_x", False))

    eff_genes_values_raw = str(args.effective_genes_values)
    if not log_x and eff_genes_values_raw.strip().lower() in {"auto", "auto_log10", "default"}:
        eff_genes_values_raw = "auto_linear"

    # Filter perturbations by type
    filter_mode = str(getattr(args, "perturbation_filter", "all"))
    filtered_perts = _filter_perturbations_by_type(data.perturbations, filter_mode)
    if not filtered_perts:
        raise ValueError(f"No perturbations remain after filter='{filter_mode}'.")
    print(f"  Filter mode '{filter_mode}': using {len(filtered_perts)} perturbations (from {len(data.perturbations)} total)")

    # Optional perturbation cap (already handled by _prepare_data, but guard here too)
    max_perts = int(args.max_perturbations) if args.max_perturbations is not None else 0
    pert_sort = str(getattr(args, "perturbation_sort", "top"))
    if max_perts > 0 and len(filtered_perts) > max_perts:
        y_tmp = np.asarray(data.y, dtype=str)
        counts = np.asarray([int(np.sum(y_tmp == p)) for p in filtered_perts], dtype=int)
        order = np.argsort(counts) if pert_sort == "bottom" else np.argsort(counts)[::-1]
        filtered_perts = [filtered_perts[i] for i in order[:max_perts]]

    y_all = np.asarray(data.y, dtype=str)
    mask = np.isin(y_all, filtered_perts)
    X_filtered = np.asarray(data.X, dtype=float)[mask, :]
    y_filtered = y_all[mask]
    labels = np.asarray(filtered_perts, dtype=str)

    n_genes = int(X_filtered.shape[1])
    eff_values = _resolve_effective_genes_values(eff_genes_values_raw, n_genes)
    eff_values = _apply_max_effective_genes_cap(eff_values, args, metrics)
    if not eff_values:
        raise ValueError("No effective-gene values are <= n_genes.")

    deg_transform, pds_transform = _resolve_weight_transforms(args.weight_scheme)
    raw_deg_weights = base._load_deg_weights(
        dataset_name=dataset_name,
        genes=data.genes,
        perturbations=filtered_perts,
        deg_weight_transform=deg_transform,
        base_weight_exponent=float(args.base_weight_exponent),
    )
    raw_pds_weights = base._load_pds_weights(
        dataset_name=dataset_name,
        genes=data.genes,
        perturbations=filtered_perts,
        pds_result_file=args.pds_result_file,
        pds_metric=args.pds_metric,
        pds_weight_transform=pds_transform,
        base_weight_exponent=float(args.base_weight_exponent),
    )
    raw_active = raw_deg_weights if args.weight_source == "deg" else raw_pds_weights

    X = X_filtered
    y = y_filtered
    ctrl_idx = np.where(y == data.control_label)[0]
    control_mean = np.mean(X[ctrl_idx, :], axis=0) if ctrl_idx.size > 0 else np.mean(X, axis=0)

    snr_x_axis = str(getattr(args, "snr_x_axis", "effective_genes"))
    calib_mode = str(args.target_effective_genes_mode) if args.weight_source == "deg" else str(args.pds_target_effective_genes_mode)
    n_trials = int(args.n_trials)
    n_jobs = _resolve_n_jobs(args)
    all_results: List[SNRResult] = []

    if snr_x_axis == "bag_size":
        # ── bag_size x-axis: fixed weights calibrated to target_effective_genes ──
        if args.target_effective_genes is not None:
            target_eg = float(args.target_effective_genes)
            calibrated = base._calibrate_weights_per_perturbation(
                weights_per_pert=raw_active,
                target_effective_genes=target_eg,
                mode=calib_mode,
            )
            print(f"  bag_size mode: weights calibrated to {int(target_eg)} effective genes.")
        else:
            calibrated = raw_active
            target_eg = None
            print("  bag_size mode: no --target-effective-genes given; using raw weights directly.")
        # Determine valid bag sizes: need 2*bag_size cells per perturbation
        y_for_count = y
        min_cells = int(min(np.sum(y_for_count == p) for p in filtered_perts))
        max_valid_bs = min_cells // 2
        raw_bag_sizes = [int(b) for b in _parse_csv_ints(args.bag_sizes, "bag-sizes")]
        bag_sizes = [b for b in raw_bag_sizes if b <= max_valid_bs]
        if not bag_sizes:
            raise ValueError(f"No bag sizes ≤ {max_valid_bs} (min_cells={min_cells}); reduce --bag-sizes.")
        if len(bag_sizes) < len(raw_bag_sizes):
            dropped = [b for b in raw_bag_sizes if b > max_valid_bs]
            print(f"  WARNING: dropped bag sizes {dropped} (need 2×bag_size ≤ {min_cells} min cells).")

        n_resamples = int(getattr(args, "n_resamples", 20))
        _set_worker_state(
            X=X, y=y, labels=labels, control_mean=control_mean,
            active_weights=calibrated, metrics=metrics,
            base_seed=int(args.seed),
            top_deg_global=np.array([], dtype=int),
            n_resamples=n_resamples,
        )
        tasks = [(trial, bs) for trial in range(n_trials) for bs in bag_sizes]
        n_tasks = len(tasks)
        print(f"  SNR bag-size mode: {n_trials} trials x {len(bag_sizes)} bag sizes x {len(metrics)} metrics ({n_tasks} tasks, {n_jobs} jobs, {n_resamples} resamples/task)")
        eg_info = f"calibrated to {int(target_eg)} effective genes" if target_eg is not None else "raw (uncalibrated)"
        print(f"  Fixed weights: {eg_info}")

        worker_fn = _snr_bs_worker
        x_label = "Cells per bag"
        csv_name = "snr_bag_size_curve.csv"
        png_name = "snr_bag_size_curve.png"
        log_x_plot = False
    else:
        # ── effective_genes x-axis: all_cells split, sweep target eff genes ──
        n_resamples_eg = int(getattr(args, "n_resamples", 20))
        pair_repeats_by_trial = {
            trial: _build_bag_mean_pairs_all_cells(
                X=X, y=y, perturbations=filtered_perts,
                seed=int(args.seed) + trial, n_resamples=n_resamples_eg,
            )
            for trial in range(n_trials)
        }
        _set_worker_state(
            X=X, labels=labels, control_mean=control_mean,
            raw_deg_weights=raw_deg_weights,
            raw_active=raw_active, calib_mode=calib_mode, metrics=metrics,
            pair_repeats_by_trial=pair_repeats_by_trial,
        )
        tasks = [(trial, ev) for trial in range(n_trials) for ev in eff_values]
        n_tasks = len(tasks)
        n_ev = len(eff_values)
        print(f"  SNR mode: {n_trials} trials x {n_ev} eff-gene values x {len(metrics)} metrics ({n_tasks} tasks, {n_jobs} jobs)")
        print(f"  Split mode 'all_cells': bag size = n_cells//2 per perturbation")

        worker_fn = _snr_worker
        x_label = "Target effective gene number"
        csv_name = "snr_curve.csv"
        png_name = "snr_curve.png"
        log_x_plot = log_x

    eff_stats: Dict[Tuple[str, int], Tuple[float, float, float]] = {}
    if n_jobs > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            for i, result in enumerate(pool.imap_unordered(worker_fn, tasks)):
                all_results.extend(result["rows"])
                eff_stats.update(result.get("eff", {}))
                if (i + 1) % max(1, n_tasks // 20) == 0 or (i + 1) == n_tasks:
                    print(f"  completed {i + 1}/{n_tasks} tasks", end="\r", flush=True)
    else:
        for i, task in enumerate(tasks):
            result = worker_fn(task)
            all_results.extend(result["rows"])
            eff_stats.update(result.get("eff", {}))
            print(f"  completed {i + 1}/{n_tasks} tasks", end="\r", flush=True)

    print()
    x_col = "bag_size" if snr_x_axis == "bag_size" else "target_effective_genes"
    _save_snr_csv(all_results, out_csv=out_dir / csv_name, x_col=x_col)
    _plot_snr_curve(
        all_results,
        out_png=out_dir / png_name,
        dataset_name=dataset_name,
        log_x=log_x_plot,
        x_label=x_label,
    )
    print(f"Saved SNR curve outputs in: {out_dir}")


def run_snr_rank_curve(
    args: argparse.Namespace,
    dataset_name: str,
    data: base.PreparedData,
    out_dir: Path,
) -> None:
    """Sweep top-k genes from ranking; report Cohen's d' and AUC.

    Similar to single_cell_pds_rank_curve but uses Cohen's d' quantification
    instead of PDS. Uses all-cell splits (no bag sampling) and sweeps the
    number of top-ranked genes included in the analysis.
    """
    metrics = _parse_metrics(args.metrics)
    gene_ranking_source: str = getattr(args, "gene_ranking", "pds")
    ranking_label = "DEG" if gene_ranking_source == "deg" else "PDS"
    log_x = not bool(getattr(args, "no_log_x", False))

    if bool(getattr(args, "plot_only", False)):
        csv_path = out_dir / f"snr_rank_curve_{ranking_label.lower()}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing summary CSV for --plot-only: {csv_path}")
        _plot_snr_curve(
            _load_snr_csv(csv_path),
            out_png=out_dir / f"snr_rank_curve_{ranking_label.lower()}.png",
            dataset_name=dataset_name,
            log_x=log_x,
            x_label=f"Top-k genes from {ranking_label} ranking",
        )
        return

    bag_only = {"energy_distance", "weighted_energy_distance", "std_energy_distance", "w_std_energy_distance"}
    if any(metric in bag_only for metric in metrics):
        raise ValueError(f"Bag-only metrics ({bag_only & set(metrics)}) are not supported in snr_rank_curve mode.")

    n_genes = int(data.X.shape[1])
    topk_values = [k for k in _resolve_topk_values(args.topk_values, n_genes=n_genes) if k <= n_genes]
    if not topk_values or topk_values[-1] != n_genes:
        topk_values.append(n_genes)
    topk_values = sorted(set(topk_values))
    if not topk_values:
        raise ValueError("No top-k value is <= n_genes.")

    deg_weights, _pds_weights, active_weights = _load_weight_families(args, dataset_name, data)

    if gene_ranking_source == "deg":
        gene_ranked = _resolve_ranked_deg_gene_indices(deg_weights, n_genes=n_genes)
    else:
        gene_ranked = _resolve_ranked_pds_gene_indices(
            dataset_name=dataset_name,
            genes=data.genes,
            perturbations=data.perturbations,
            pds_result_file=args.pds_result_file,
            pds_metric=args.pds_metric,
        )

    n_top_deg = int(max(1, round(float(args.target_effective_genes)))) if args.target_effective_genes is not None else 100
    deg_ranked = _resolve_ranked_deg_gene_indices(deg_weights, n_genes=n_genes)

    X = np.asarray(data.X, dtype=float)
    y = np.asarray(data.y, dtype=str)
    labels = np.asarray(data.perturbations, dtype=str)
    ctrl_idx = np.where(y == data.control_label)[0]
    control_mean = np.mean(X[ctrl_idx, :], axis=0) if ctrl_idx.size > 0 else np.mean(X, axis=0)

    n_trials = int(args.n_trials)
    n_jobs = _resolve_n_jobs(args)

    # Build all-cell pairs for each trial (like snr_curve effective_genes mode)
    n_resamples = int(getattr(args, "n_resamples", 20))
    pair_repeats_by_trial = {
        trial: _build_bag_mean_pairs_all_cells(
            X=X, y=y, perturbations=labels,
            seed=int(args.seed) + trial, n_resamples=n_resamples,
        )
        for trial in range(n_trials)
    }

    _set_worker_state(
        X=X, y=y, labels=labels, control_mean=control_mean,
        active_weights=active_weights, gene_ranked=gene_ranked,
        deg_ranked=deg_ranked, n_top_deg=n_top_deg, metrics=metrics,
        pair_repeats_by_trial=pair_repeats_by_trial,
    )

    tasks = [(trial, k) for trial in range(n_trials) for k in topk_values]
    n_tasks = len(tasks)
    print(f"  SNR rank mode: {n_trials} trials x {len(topk_values)} k-values x {len(metrics)} metrics ({n_tasks} tasks, {n_jobs} jobs)")
    print(f"  Gene ranking: {ranking_label}, Resamples: {n_resamples}")

    all_results: List[SNRResult] = []
    eff_stats: Dict[Tuple[str, int], Tuple[float, float, float]] = {}

    if n_jobs > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            for i, result in enumerate(pool.imap_unordered(_snr_rank_worker, tasks)):
                all_results.extend(result["rows"])
                eff_stats.update(result.get("eff", {}))
                if (i + 1) % max(1, n_tasks // 20) == 0 or (i + 1) == n_tasks:
                    print(f"  completed {i + 1}/{n_tasks} tasks", end="\r", flush=True)
    else:
        for i, task in enumerate(tasks):
            result = _snr_rank_worker(task)
            all_results.extend(result["rows"])
            eff_stats.update(result.get("eff", {}))
            print(f"  completed {i + 1}/{n_tasks} tasks", end="\r", flush=True)

    print()
    csv_name = f"snr_rank_curve_{ranking_label.lower()}.csv"
    png_name = f"snr_rank_curve_{ranking_label.lower()}.png"
    _save_snr_csv(all_results, out_csv=out_dir / csv_name, x_col="n_genes")
    _plot_snr_curve(
        all_results,
        out_png=out_dir / png_name,
        dataset_name=dataset_name,
        log_x=log_x,
        x_label=f"Top-k genes from {ranking_label} ranking",
    )
    print(f"Saved SNR rank curve outputs in: {out_dir}")


def run_snr_bag_rank_curve(
    args: argparse.Namespace,
    dataset_name: str,
    data: base.PreparedData,
    out_dir: Path,
) -> None:
    """Sweep top-k genes from ranking with different bag sizes; report Cohen's d' and AUC.

    Combines bag_rank_curve (bag size sweep) with snr_rank_curve (Cohen's d' quantification).
    For each bag size and top-k gene count, computes d' using bootstrapped resamples.
    """
    metrics = _parse_metrics(args.metrics)
    gene_ranking_source: str = getattr(args, "gene_ranking", "pds")
    ranking_label = "DEG" if gene_ranking_source == "deg" else "PDS"
    log_x = not bool(getattr(args, "no_log_x", False))

    # Bag sizes to test (similar to bag_rank_curve)
    bag_sizes = [int(b) for b in _parse_csv_ints(args.bag_sizes, "bag-sizes")]
    if not bag_sizes:
        raise ValueError("No valid bag sizes specified.")

    if bool(getattr(args, "plot_only", False)):
        # Plot for each bag size
        for bag_size in bag_sizes:
            csv_path = out_dir / f"snr_bag_rank_curve_{ranking_label.lower()}_bs{bag_size}.csv"
            if not csv_path.exists():
                print(f"WARNING: Missing CSV for bag_size={bag_size}: {csv_path}")
                continue
            _plot_snr_curve(
                _load_snr_csv(csv_path),
                out_png=out_dir / f"snr_bag_rank_curve_{ranking_label.lower()}_bs{bag_size}.png",
                dataset_name=dataset_name,
                log_x=log_x,
                x_label=f"Top-k genes from {ranking_label} ranking (bag_size={bag_size})",
            )
        return

    bag_only = {"energy_distance", "weighted_energy_distance", "std_energy_distance", "w_std_energy_distance"}
    if any(metric in bag_only for metric in metrics):
        raise ValueError(f"Bag-only metrics ({bag_only & set(metrics)}) are not supported in snr_bag_rank_curve mode.")

    n_genes = int(data.X.shape[1])
    topk_values = [k for k in _resolve_topk_values(args.topk_values, n_genes=n_genes) if k <= n_genes]
    if not topk_values or topk_values[-1] != n_genes:
        topk_values.append(n_genes)
    topk_values = sorted(set(topk_values))
    if not topk_values:
        raise ValueError("No top-k value is <= n_genes.")

    deg_weights, _pds_weights, active_weights = _load_weight_families(args, dataset_name, data)

    # Filter to only perturbations with valid DEG weights (some may have been skipped due to no DEG data)
    valid_perts = list(active_weights.keys())
    if len(valid_perts) < len(data.perturbations):
        print(f"NOTE: Using {len(valid_perts)}/{len(data.perturbations)} perturbations with valid DEG weights")

    if gene_ranking_source == "deg":
        gene_ranked = _resolve_ranked_deg_gene_indices(deg_weights, n_genes=n_genes)
    else:
        gene_ranked = _resolve_ranked_pds_gene_indices(
            dataset_name=dataset_name,
            genes=data.genes,
            perturbations=valid_perts,  # Use filtered perturbation list
            pds_result_file=args.pds_result_file,
            pds_metric=args.pds_metric,
        )

    n_top_deg = int(max(1, round(float(args.target_effective_genes)))) if args.target_effective_genes is not None else 100
    deg_ranked = _resolve_ranked_deg_gene_indices(deg_weights, n_genes=n_genes)

    X = np.asarray(data.X, dtype=float)
    y = np.asarray(data.y, dtype=str)
    # Filter to only include perturbations with valid weights
    labels = np.asarray(valid_perts, dtype=str)
    valid_mask = np.isin(y, labels)
    X = X[valid_mask, :]
    y = y[valid_mask]
    print(f"  Filtered data: {X.shape[0]} cells from {len(labels)} valid perturbations")

    ctrl_idx = np.where(y == data.control_label)[0]
    control_mean = np.mean(X[ctrl_idx, :], axis=0) if ctrl_idx.size > 0 else np.mean(X, axis=0)

    n_trials = int(args.n_trials)
    n_jobs = _resolve_n_jobs(args)
    n_resamples = int(getattr(args, "n_resamples", 20))
    base_seed = int(args.seed)

    _set_worker_state(
        X=X, y=y, labels=labels, control_mean=control_mean,
        active_weights=active_weights, gene_ranked=gene_ranked,
        deg_ranked=deg_ranked, n_top_deg=n_top_deg, metrics=metrics,
        base_seed=base_seed,
    )

    # Run for each bag size separately (different CSV per bag size)
    for bag_size in bag_sizes:
        print(f"\n  Processing bag_size={bag_size}...")
        # Generate tasks with resample indices: (trial, k, bag_size, resample_idx)
        tasks = [(trial, k, bag_size, r) for trial in range(n_trials) for k in topk_values for r in range(n_resamples)]
        n_tasks = len(tasks)
        print(f"  SNR bag-rank mode: {n_trials} trials x {len(topk_values)} k-values x {n_resamples} resamples x {len(metrics)} metrics ({n_tasks} tasks, {n_jobs} jobs)")

        raw_results: List[SNRResult] = []
        eff_stats: Dict[Tuple[str, int], Tuple[float, float, float]] = {}

        if n_jobs > 1:
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=n_jobs) as pool:
                for i, result in enumerate(pool.imap_unordered(_snr_bag_rank_worker, tasks)):
                    raw_results.extend(result["rows"])
                    # Only store eff stats once per (metric, k) - they don't change with resample
                    if result.get("resample", 0) == 0:
                        eff_stats.update(result.get("eff", {}))
                    if (i + 1) % max(1, n_tasks // 20) == 0 or (i + 1) == n_tasks:
                        print(f"  completed {i + 1}/{n_tasks} tasks", end="\r", flush=True)
        else:
            for i, task in enumerate(tasks):
                result = _snr_bag_rank_worker(task)
                raw_results.extend(result["rows"])
                if result.get("resample", 0) == 0:
                    eff_stats.update(result.get("eff", {}))
                print(f"  completed {i + 1}/{n_tasks} tasks", end="\r", flush=True)

        print()
        
        # Aggregate resamples: group by (metric, trial, k) and compute d' from pooled distances
        from collections import defaultdict
        grouped: Dict[Tuple[str, int, int], List[np.ndarray]] = defaultdict(list)
        for r in raw_results:
            key = (r.metric, r.trial, r.x_value)
            grouped[key].append(r._dist)  # type: ignore[attr-defined]
        
        all_results: List[SNRResult] = []
        for (metric, trial, k), dist_list in grouped.items():
            # Filter to only resamples with a consistent (largest) n, in case some skipped perturbations
            max_n = max(d.shape[0] for d in dist_list)
            dist_list = [d for d in dist_list if d.shape[0] == max_n]
            if not dist_list:
                continue
            d0 = dist_list[0]
            n = d0.shape[0]
            mask = ~np.eye(n, dtype=bool)
            within_all = np.concatenate([np.diag(d) for d in dist_list])
            between_all = np.concatenate([d[mask] for d in dist_list])
            mu_w = float(np.mean(within_all))
            mu_b = float(np.mean(between_all))
            snr = mu_b / mu_w if mu_w > 1e-15 else float("inf")
            dprime, auc = _dprime_per_pert_pooled(dist_list)
            
            all_results.append(SNRResult(
                metric=metric,
                x_value=k,
                trial=trial,
                snr=snr,
                dprime=dprime,
                auc=auc,
                within=mu_w,
                between=mu_b,
            ))
        
        csv_name = f"snr_bag_rank_curve_{ranking_label.lower()}_bs{bag_size}.csv"
        png_name = f"snr_bag_rank_curve_{ranking_label.lower()}_bs{bag_size}.png"
        _save_snr_csv(all_results, out_csv=out_dir / csv_name, x_col="n_genes")
        _plot_snr_curve(
            all_results,
            out_png=out_dir / png_name,
            dataset_name=dataset_name,
            log_x=log_x,
            x_label=f"Top-k genes from {ranking_label} ranking (bag_size={bag_size})",
        )
        print(f"  Saved bag_size={bag_size} outputs")

    print(f"\nSaved SNR bag-rank curve outputs in: {out_dir}")


def run_bag_size_mode(args: argparse.Namespace, dataset_name: str, data: base.PreparedData, out_dir: Path) -> None:
    metrics = _parse_metrics(args.metrics)
    heatmap_metrics = _parse_metrics(args.heatmap_metrics) if args.heatmap_metrics else list(metrics)
    if bool(getattr(args, "plot_only", False)):
        csv_path = out_dir / "bag_size_curve.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing summary CSV for --plot-only: {csv_path}")
        _plot_curve_from_summary_csv(
            csv_path=csv_path,
            out_png=out_dir / "bag_size_curve.png",
            x_col="bag_size",
            x_label="Cells per bag",
            title=f"Bag-size signal dilution ({dataset_name})",
            y_label="PDS",
            exponential_y=False,
            legend_outside_right=True,
            figure_size=(4.8, 2.8),
        )
        if args.run_distance_heatmaps:
            selected = _parse_csv_ints(args.heatmap_x_values, "heatmap-x-values")
            for heat_bag_size in selected:
                npz_path = out_dir / f"pairwise_distance_matrices_bagsize{int(heat_bag_size)}.npz"
                if not npz_path.exists():
                    print(f"WARNING: Missing saved distances for heatmap x={heat_bag_size}: {npz_path}")
                    continue
                payload = np.load(npz_path, allow_pickle=False)
                if "labels" in payload.files:
                    labels_saved = np.asarray(payload["labels"], dtype=str)
                else:
                    # Backward compatibility: reconstruct deterministic bag labels
                    # from saved run parameters without recomputing distances.
                    _, _, labels_saved = _build_all_bags_for_heatmap(
                        X=np.asarray(data.X, dtype=float),
                        y=np.asarray(data.y, dtype=str),
                        perturbations=data.perturbations,
                        bag_size=int(heat_bag_size),
                        seed=int(args.seed),
                        max_perturbations=int(args.heatmap_max_perturbations),
                        max_bags_per_perturbation=int(args.heatmap_max_bags_per_perturbation),
                    )
                mats_saved: Dict[str, np.ndarray] = {}
                for metric in heatmap_metrics:
                    if metric in payload.files:
                        mats_saved[metric] = np.asarray(payload[metric], dtype=float)
                if not mats_saved:
                    print(f"WARNING: No requested heatmap metrics found in {npz_path}")
                    continue
                # Apply the same perturbation/bag capping in plot-only mode.
                y_arr = np.asarray(labels_saved, dtype=str)
                rng = np.random.default_rng(int(args.seed))
                perts = np.unique(y_arr)
                if int(args.heatmap_max_perturbations) > 0 and perts.size > int(args.heatmap_max_perturbations):
                    counts = np.asarray([np.sum(y_arr == p) for p in perts], dtype=int)
                    order = np.argsort(counts)[::-1]
                    perts = perts[order[: int(args.heatmap_max_perturbations)]]
                keep_blocks: List[np.ndarray] = []
                for pert in perts:
                    idx = np.where(y_arr == pert)[0]
                    if int(args.heatmap_max_bags_per_perturbation) > 0 and idx.size > int(args.heatmap_max_bags_per_perturbation):
                        idx = np.sort(
                            rng.choice(idx, size=int(args.heatmap_max_bags_per_perturbation), replace=False)
                        )
                    keep_blocks.append(np.asarray(idx, dtype=int))
                if keep_blocks:
                    keep_idx = np.sort(np.concatenate(keep_blocks).astype(int))
                    labels_saved = y_arr[keep_idx]
                    mats_saved = {
                        metric: np.asarray(mat, dtype=float)[np.ix_(keep_idx, keep_idx)]
                        for metric, mat in mats_saved.items()
                    }
                _plot_metric_heatmaps_with_ranges(
                    matrices=mats_saved,
                    labels=labels_saved,
                    metrics=heatmap_metrics,
                    output_path=out_dir / f"pairwise_distance_heatmaps_bagsize{int(heat_bag_size)}.png",
                    quantile_bins=int(args.heatmap_quantile_bins),
                    quantile_scope=str(args.heatmap_quantile_scope),
                )
        return
    bag_sizes_raw = _parse_csv_ints(args.bag_sizes, "bag-sizes")
    y = np.asarray(data.y, dtype=str)
    counts = np.asarray([np.sum(y == pert) for pert in data.perturbations], dtype=int)
    budget = int(args.trial_cell_budget)
    usable_per_pert = np.minimum(counts, budget)
    max_valid = int(np.min(usable_per_pert) // 2)
    bag_sizes = [b for b in bag_sizes_raw if b <= max_valid]
    removed = [b for b in bag_sizes_raw if b > max_valid]
    if removed:
        print(
            "WARNING: Reducing bag sizes due to limited cells in at least one perturbation.",
            f"Requested={bag_sizes_raw}, max_valid={max_valid}, using={bag_sizes}",
        )
    if not bag_sizes:
        raise ValueError(f"No bag size is feasible; smallest perturbation supports at most bag-size={max_valid}.")

    deg_weights, _pds_weights, active_weights = _load_weight_families(args, dataset_name, data)
    n_top_deg = int(max(1, round(float(args.target_effective_genes)))) if args.target_effective_genes is not None else 100
    top_deg_global = _top_deg_indices(deg_weights, n_keep=n_top_deg)
    X = np.asarray(data.X, dtype=float)
    labels = np.asarray(data.perturbations, dtype=str)
    ctrl_idx = np.where(y == data.control_label)[0]
    control_mean = np.mean(X[ctrl_idx, :], axis=0) if ctrl_idx.size > 0 else np.mean(X, axis=0)

    trial_rows: List[TrialResult] = []
    eff_stats: Dict[Tuple[str, int], Tuple[float, float, float]] = {}
    heatmap_mats: Dict[Tuple[str, int], np.ndarray] = {}
    max_b = int(max(bag_sizes))
    n_trials = int(args.n_trials)
    n_bs = len(bag_sizes)
    n_jobs = _resolve_n_jobs(args)

    _set_worker_state(
        X=X, y=y, labels=labels, control_mean=control_mean,
        active_weights=active_weights, metrics=metrics,
        perturbations=list(data.perturbations),
        trial_cell_budget=int(args.trial_cell_budget),
        base_seed=int(args.seed),
        bag_normalize=bool(args.bag_normalize_repeats),
        max_bag_size=max_b,
        top_deg_global=top_deg_global,
    )

    tasks = [(trial, bs) for trial in range(n_trials) for bs in bag_sizes]
    n_tasks = len(tasks)
    print(f"  bag-size mode: {n_trials} trials x {n_bs} bag sizes x {len(metrics)} metrics ({n_tasks} tasks, {n_jobs} jobs)")

    if n_jobs > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            for i, result in enumerate(pool.imap_unordered(_bs_worker, tasks)):
                _merge_worker_result(result, trial_rows, eff_stats, heatmap_mats)
                if (i + 1) % max(1, n_tasks // 20) == 0 or (i + 1) == n_tasks:
                    print(f"  completed {i + 1}/{n_tasks} tasks", end="\r", flush=True)
    else:
        for i, task in enumerate(tasks):
            t, bs = task
            print(f"  trial {t + 1}/{n_trials}  bag_size={bs} ({i + 1}/{n_tasks})", end="\r", flush=True)
            result = _bs_worker(task)
            _merge_worker_result(result, trial_rows, eff_stats, heatmap_mats)

    print()  # finish carriage-return progress line
    setattr(_save_curve, "_effective_gene_stats", eff_stats)
    _save_curve(trial_rows, out_csv=out_dir / "bag_size_curve.csv", x_col="bag_size")
    _plot_curve(
        trial_rows,
        out_png=out_dir / "bag_size_curve.png",
        x_label="Cells per bag",
        title=f"Bag-size signal dilution ({dataset_name})",
        log_x=False,
        effective_gene_stats=eff_stats,
        show_eff_gene_box=False,
        y_label="PDS",
        emphasize_high_scores=False,
        exponential_y=False,
        legend_outside_right=True,
        figure_size=(4.8, 2.8),
    )
    # Save one or more multi-metric pairwise heatmaps (all genes) with perturbation range labels.
    if args.run_distance_heatmaps:
        requested_x = _parse_csv_ints(args.heatmap_x_values, "heatmap-x-values")
        selected = [int(x) for x in requested_x if int(x) <= max_valid]
        skipped = [int(x) for x in requested_x if int(x) > max_valid]
        if skipped:
            print(
                "WARNING: Some --heatmap-x-values are not feasible and will be skipped.",
                f"Requested={requested_x}, max_valid={max_valid}, skipped={skipped}",
            )
        if not selected:
            fallback = int(bag_sizes[-1])
            print(
                "WARNING: No feasible --heatmap-x-values; using largest feasible bag size from curve.",
                f"fallback={fallback}",
            )
            selected = [fallback]

        control_idx = np.where(y == data.control_label)[0]
        control_mean_full = np.mean(X[control_idx, :], axis=0) if control_idx.size > 0 else np.mean(X, axis=0)

        for heat_bag_size in selected:
            bag_means, bag_members, bag_labels = _build_all_bags_for_heatmap(
                X=X,
                y=y,
                perturbations=data.perturbations,
                bag_size=int(heat_bag_size),
                seed=int(args.seed),
                max_perturbations=int(args.heatmap_max_perturbations),
                max_bags_per_perturbation=int(args.heatmap_max_bags_per_perturbation),
            )
            mats: Dict[str, np.ndarray] = {}
            for metric in heatmap_metrics:
                mats[metric] = _bag_metric_distance(
                    metric=metric,
                    Xq_mean=bag_means,
                    Xt_mean=bag_means,
                    X_cells=X,
                    q_members=bag_members,
                    t_members=bag_members,
                    labels=bag_labels,
                    control_mean=control_mean_full,
                    active_weights=active_weights,
                )
                np.fill_diagonal(mats[metric], 0.0)
            np.savez_compressed(
                out_dir / f"pairwise_distance_matrices_bagsize{int(heat_bag_size)}.npz",
                labels=bag_labels,
                **mats,
            )
            _plot_metric_heatmaps_with_ranges(
                matrices=mats,
                labels=bag_labels,
                metrics=heatmap_metrics,
                output_path=out_dir / f"pairwise_distance_heatmaps_bagsize{int(heat_bag_size)}.png",
                quantile_bins=int(args.heatmap_quantile_bins),
                quantile_scope=str(args.heatmap_quantile_scope),
            )


def run_bag_rank_mode(
    args: argparse.Namespace,
    dataset_name: str,
    data: base.PreparedData,
    out_dir: Path,
) -> None:
    """Sweep top-k genes on x-axis using bag means; one plot per bag size.

    Like single_cell_pds_rank_curve but averages over bags of cells instead of
    using individual cells. The title of each output plot is just the bag size.
    """
    metrics = _parse_metrics(args.metrics)
    gene_ranking_source: str = getattr(args, "gene_ranking", "pds")

    n_genes = int(data.X.shape[1])
    topk_values = [k for k in _resolve_topk_values(args.topk_values, n_genes=n_genes) if k <= n_genes]
    if not topk_values or topk_values[-1] != n_genes:
        topk_values.append(n_genes)
    topk_values = sorted(set(topk_values))
    if not topk_values:
        raise ValueError("No top-k value is <= n_genes.")

    bag_sizes_raw = _parse_csv_ints(args.bag_sizes, "bag-sizes")
    y = np.asarray(data.y, dtype=str)
    counts = np.asarray([np.sum(y == pert) for pert in data.perturbations], dtype=int)
    budget = int(args.trial_cell_budget)
    usable_per_pert = np.minimum(counts, budget)
    max_valid = int(np.min(usable_per_pert) // 2)
    bag_sizes = [b for b in bag_sizes_raw if b <= max_valid]
    removed = [b for b in bag_sizes_raw if b > max_valid]
    if removed:
        print(
            "WARNING: Reducing bag sizes due to limited cells in at least one perturbation.",
            f"Requested={bag_sizes_raw}, max_valid={max_valid}, using={bag_sizes}",
        )
    if not bag_sizes:
        raise ValueError(f"No bag size is feasible; smallest perturbation supports at most bag-size={max_valid}.")

    deg_weights, _pds_weights, active_weights = _load_weight_families(args, dataset_name, data)
    n_top_deg = int(max(1, round(float(args.target_effective_genes)))) if args.target_effective_genes is not None else 100
    deg_ranked = _resolve_ranked_deg_gene_indices(deg_weights, n_genes=n_genes)

    if gene_ranking_source == "deg":
        gene_ranked = _resolve_ranked_deg_gene_indices(deg_weights, n_genes=n_genes)
        ranking_label = "DEG"
    else:
        gene_ranked = _resolve_ranked_pds_gene_indices(
            dataset_name=dataset_name,
            genes=data.genes,
            perturbations=data.perturbations,
            pds_result_file=args.pds_result_file,
            pds_metric=args.pds_metric,
        )
        ranking_label = "PDS"

    X = np.asarray(data.X, dtype=float)
    labels = np.asarray(data.perturbations, dtype=str)
    ctrl_idx = np.where(y == data.control_label)[0]
    control_mean = np.mean(X[ctrl_idx, :], axis=0) if ctrl_idx.size > 0 else np.mean(X, axis=0)

    n_trials = int(args.n_trials)
    n_jobs = _resolve_n_jobs(args)
    max_b = int(max(bag_sizes))

    # Pre-build bag pairs for every (trial, bag_size) so workers can fork them.
    bag_pairs_by_trial_size: Dict[Tuple[int, int], Any] = {}
    for trial in range(n_trials):
        for bag_size in bag_sizes:
            bag_pairs_by_trial_size[(trial, bag_size)] = _build_bag_pairs_for_trial(
                y=y,
                perturbations=data.perturbations,
                bag_size=bag_size,
                trial_cell_budget=budget,
                seed=int(args.seed) + trial * 1009 + bag_size,
                normalize_repeats=bool(args.bag_normalize_repeats),
                max_bag_size=max_b,
            )

    _set_worker_state(
        X=X, labels=labels, control_mean=control_mean,
        active_weights=active_weights, gene_ranked=gene_ranked,
        deg_ranked=deg_ranked, n_top_deg=n_top_deg, metrics=metrics,
        bag_pairs_by_trial_size=bag_pairs_by_trial_size,
    )

    tasks = [(trial, k, bag_size) for trial in range(n_trials) for k in topk_values for bag_size in bag_sizes]
    n_tasks = len(tasks)
    n_k = len(topk_values)
    print(
        f"  bag-rank mode: {n_trials} trials x {n_k} k-values x {len(bag_sizes)} bag sizes x {len(metrics)} metrics "
        f"({n_tasks} tasks, {n_jobs} jobs)"
    )

    results_by_bag: Dict[int, List[TrialResult]] = {bs: [] for bs in bag_sizes}

    if n_jobs > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            for i, result in enumerate(pool.imap_unordered(_bag_rank_worker, tasks)):
                results_by_bag[result["bag_size"]].extend(result["rows"])
                if (i + 1) % max(1, n_tasks // 20) == 0 or (i + 1) == n_tasks:
                    print(f"  completed {i + 1}/{n_tasks} tasks", end="\r", flush=True)
    else:
        for i, task in enumerate(tasks):
            t, k, bs = task
            print(f"  trial {t + 1}/{n_trials}  k={k}  bag_size={bs} ({i + 1}/{n_tasks})", end="\r", flush=True)
            result = _bag_rank_worker(task)
            results_by_bag[result["bag_size"]].extend(result["rows"])

    print()
    for bag_size in bag_sizes:
        rows = results_by_bag[bag_size]
        stem = f"bag_rank_curve_bagsize{bag_size}"
        _save_curve(rows, out_csv=out_dir / f"{stem}.csv", x_col="n_genes")
        _plot_curve(
            rows,
            out_png=out_dir / f"{stem}.png",
            x_label=f"Top-k genes from per-gene {ranking_label} ranking",
            title=str(bag_size),
            log_x=True,
            y_label="Mean PDS",
            show_eff_gene_box=False,
            annotation_box=None,
            figure_size=(4.8, 2.8),
            legend_outside_right=True,
        )
        # Also save a fixed-range variant for easier comparisons across bag sizes.
        _plot_curve(
            rows,
            out_png=out_dir / f"{stem}_ylim01.png",
            x_label=f"Top-k genes from per-gene {ranking_label} ranking",
            title=str(bag_size),
            log_x=True,
            y_label="Mean PDS",
            show_eff_gene_box=False,
            annotation_box=None,
            y_limits=(0.0, 1.0),
            figure_size=(4.8, 2.8),
            legend_outside_right=True,
        )


def _save_config(args: argparse.Namespace, dataset_name: str, out_path: Path) -> None:
    payload = {
        "dataset": dataset_name,
        "mode": args.mode,
        "metrics": _parse_metrics(args.metrics),
        "weight_source": args.weight_source,
        "weight_scheme": args.weight_scheme,
        "target_effective_genes": float(args.target_effective_genes) if args.target_effective_genes is not None else None,
        "target_effective_genes_mode": args.target_effective_genes_mode,
        "pds_target_effective_genes_mode": args.pds_target_effective_genes_mode,
        "base_weight_exponent": float(args.base_weight_exponent),
        "pds_result_file": args.pds_result_file,
        "pds_metric": args.pds_metric,
        "cells_per_perturbation": args.cells_per_perturbation,
        "max_cells": args.max_cells,
        "max_perturbations": args.max_perturbations,
        "perturbation_sort": str(getattr(args, "perturbation_sort", "top")),
        "n_trials": int(args.n_trials),
        "split_mode": str(getattr(args, "split_mode", "budget")),
        "trial_cell_budget": int(args.trial_cell_budget),
        "seed": int(args.seed),
        "heatmaps_enabled": bool(args.run_distance_heatmaps),
        "heatmap_metrics": _parse_metrics(args.heatmap_metrics) if args.heatmap_metrics else _parse_metrics(args.metrics),
        "heatmap_quantile_scope": args.heatmap_quantile_scope,
        "heatmap_max_perturbations": int(args.heatmap_max_perturbations),
        "heatmap_max_bags_per_perturbation": int(args.heatmap_max_bags_per_perturbation),
        "n_jobs": int(getattr(args, "n_jobs", 1)),
    }
    if args.mode in ("single_cell_pds_rank_curve", "bag_rank_curve"):
        payload["gene_ranking"] = str(getattr(args, "gene_ranking", "pds"))
        payload["topk_values"] = (
            _default_log10_topk(n_genes=int(args._n_genes))
            if str(args.topk_values).strip().lower() in {"auto", "auto_log2", "auto_log10", "default"}
            else _parse_csv_ints(args.topk_values, "topk-values")
        )
        if args.mode == "bag_rank_curve":
            payload["bag_sizes"] = _parse_csv_ints(args.bag_sizes, "bag-sizes")
    elif args.mode in ("effective_genes_curve", "snr_curve"):
        _eg_raw = str(args.effective_genes_values)
        _no_log = bool(getattr(args, "no_log_x", False))
        if _no_log and _eg_raw.strip().lower() in {"auto", "auto_log10", "default"}:
            _eg_raw = "auto_linear"
        _metrics = _parse_metrics(args.metrics)
        _eg_vals = _resolve_effective_genes_values(_eg_raw, n_genes=int(args._n_genes))
        _eg_vals = _apply_max_effective_genes_cap(_eg_vals, args, _metrics, silent=True)
        payload["effective_genes_values"] = _eg_vals
        payload["max_effective_genes"] = _resolved_max_effective_genes(args, _metrics)
        payload["no_log_x"] = _no_log
    elif args.mode in ("snr_rank_curve", "snr_bag_rank_curve"):
        payload["gene_ranking"] = str(getattr(args, "gene_ranking", "pds"))
        payload["topk_values"] = (
            _default_log10_topk(n_genes=int(args._n_genes))
            if str(args.topk_values).strip().lower() in {"auto", "auto_log2", "auto_log10", "default"}
            else _parse_csv_ints(args.topk_values, "topk-values")
        )
        payload["no_log_x"] = bool(getattr(args, "no_log_x", False))
    else:
        payload["bag_sizes"] = _parse_csv_ints(args.bag_sizes, "bag-sizes")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.cells_per_perturbation is not None and args.max_cells is not None:
        raise ValueError("Use either --cells-per-perturbation or --max-cells, not both.")
    if args.target_effective_genes is not None and args.target_effective_genes <= 0:
        raise ValueError("--target-effective-genes must be positive.")
    datasets = base._resolve_datasets(args)
    multi_dataset = len(datasets) > 1
    for dataset_name in datasets:
        print(f"\n=== Running mode={args.mode} dataset={dataset_name} ===")
        prepared = base._prepare_data(args=args, dataset_name=dataset_name)
        data = _drop_control_perturbation(prepared)
        args._n_genes = int(data.X.shape[1])
        if args.output_dir is not None:
            base_out = args.output_dir.resolve()
            # For multi-dataset runs, keep per-dataset/mode subdirectories to avoid collisions.
            out_dir = base_out / dataset_name / args.mode if multi_dataset else base_out
        else:
            out_dir = (
                base.ROOT
                / "analyses"
                / "perturbation_discrimination"
                / "results"
                / dataset_name
                / "signal_dilution_curves"
                / ("pds_rank_curve_different_bag_sizes" if args.mode == "bag_rank_curve" else args.mode)
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.mode == "single_cell_pds_rank_curve":
            run_single_cell_mode(args=args, dataset_name=dataset_name, data=data, out_dir=out_dir)
        elif args.mode == "bag_size_curve":
            run_bag_size_mode(args=args, dataset_name=dataset_name, data=data, out_dir=out_dir)
        elif args.mode == "effective_genes_curve":
            run_effective_genes_mode(args=args, dataset_name=dataset_name, data=data, out_dir=out_dir)
        elif args.mode == "bag_rank_curve":
            run_bag_rank_mode(args=args, dataset_name=dataset_name, data=data, out_dir=out_dir)
        elif args.mode == "snr_curve":
            run_snr_curve(args=args, dataset_name=dataset_name, data=data, out_dir=out_dir)
        elif args.mode == "snr_rank_curve":
            run_snr_rank_curve(args=args, dataset_name=dataset_name, data=data, out_dir=out_dir)
        elif args.mode == "snr_bag_rank_curve":
            run_snr_bag_rank_curve(args=args, dataset_name=dataset_name, data=data, out_dir=out_dir)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
        _save_config(args=args, dataset_name=dataset_name, out_path=out_dir / "run_config.json")
        print(f"Saved outputs in: {out_dir}")


if __name__ == "__main__":
    main()
