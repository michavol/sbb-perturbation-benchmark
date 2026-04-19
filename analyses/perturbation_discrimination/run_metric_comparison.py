#!/usr/bin/env python3
"""Cross-dataset metric comparison via Cohen's d' and AUC.

For each dataset, loads top-50 perturbations (by cell count), splits all cells
into two halves (bag means), computes pairwise distance matrices for each
metric, then derives Cohen's d' and AUC via ``_dprime_per_pert_pooled``.
Multiple trials provide error bars.

Outputs:
    - Bar plots (d' and AUC) with datasets as legend, metrics on x-axis
    - Heatmaps (datasets × metrics)
    - CSV with all per-trial results
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import cm as mpl_cm
from matplotlib import colors as mpl_colors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyses.perturbation_discrimination import metric_utils as base
from analyses.perturbation_discrimination.run_signal_dilution_curves import (
    _build_bag_mean_pairs_all_cells,
    _dprime_per_pert_pooled,
    _query_truth_scores,
    _resolve_weight_transforms,
    _singlecell_metric_distance,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATASETS: List[str] = [
    "adamson16",
    "frangieh21",
    "nadig25hepg2",
    "norman19",
    "replogle20",
    "replogle22k562",
    "wessels23",
]

DEFAULT_METRICS: List[str] = [
    "mse",
    "wmse",
    "wmse_syn",
    "wmse_dataset",
    "r2_delta",
    "w_r2_delta",
    "w_r2_delta_syn",
    "pearson_delta",
    "pearson_deltapert",
    "pds_pearson_deltapert",
    "w_pearson_delta",
    "w_pearson_delta_syn",
    "energy_distance",
    "weighted_energy_distance",
    "weighted_energy_distance_syn",
    "pds",
    "wpds",
    "wpds_syn",
]

METRIC_DISPLAY: Dict[str, str] = {
    "mse": "MSE",
    "wmse": "wMSE",
    "wmse_syn": "wMSE_syn",
    "wmse_dataset": "wMSE_dataset",
    "r2_delta": "R² Δ",
    "w_r2_delta": "wR² Δ",
    "w_r2_delta_syn": "wR² Δ_syn",
    "pearson_delta": "Pearson Δ",
    "pearson_deltapert": "Pearson Δ Pert",
    "pds_pearson_deltapert": "PDS Pearson Δ Pert",
    "w_pearson_delta": "wPearson Δ",
    "w_pearson_delta_syn": "wPearson Δ_syn",
    "energy_distance": "Energy",
    "weighted_energy_distance": "wEnergy",
    "weighted_energy_distance_syn": "wEnergy_syn",
    "pds": "PDS",
    "wpds": "wPDS",
    "wpds_syn": "wPDS_syn",
}

SYNTHETIC_WEIGHT_METRIC_MAP: Dict[str, str] = {
    "wmse_syn": "wmse",
    "w_r2_delta_syn": "w_r2_delta",
    "w_pearson_delta_syn": "w_pearson_delta",
    "weighted_energy_distance_syn": "weighted_energy_distance",
    "wpds_syn": "wpds",
}

DATASET_DISPLAY: Dict[str, str] = {
    "adamson16": "Adamson '16",
    "frangieh21": "Frangieh '21",
    "nadig25hepg2": "Nadig '25",
    "norman19": "Norman '19",
    "replogle20": "Replogle '20",
    "replogle22k562": "Replogle '22",
    "wessels23": "Wessels '23",
}

# Nature-style plotting constants
NATURE_FONT = "Arial"
NATURE_AXIS_LABEL_SIZE = 7
NATURE_TICK_SIZE = 6
NATURE_LEGEND_SIZE = 6
NATURE_TITLE_SIZE = 7
NATURE_DPI = 600
FIG_SIZE_BAR = (7.2, 2.8)
FIG_SIZE_HEATMAP = (5.5, 3.0)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Single (dataset, metric, trial) measurement."""

    dataset: str
    metric: str
    trial: int
    dprime: float
    auc: float
    pds: float
    within: float
    between: float


@dataclass
class PertResult:
    """Per-perturbation Cohen's d' for a single (dataset, metric, trial)."""

    dataset: str
    metric: str
    trial: int
    perturbation: str
    dprime: float
    auc: float
    pds: float


@dataclass
class SelfControlResult:
    """Self-vs-control discrimination result for (dataset, metric, trial).

    For each perturbation, compares the self-distance (technical duplicate,
    i.e., distance to the other half of the same perturbation) against the
    control-distance (distance to control mean). The accuracy is the ratio
    of perturbations where the self-distance is smaller than the control-distance
    (i.e., the metric correctly identifies that a perturbation is more similar
    to itself than to control).

    For PDS metrics (pds, wpds), stores the mean PDS scores for self and control
    queries instead of accuracy-based discrimination.
    """

    dataset: str
    metric: str
    trial: int
    accuracy: float  # Ratio of correct discriminations (0-1)
    n_total: int  # Total number of perturbations tested
    n_correct: int  # Number of perturbations where self < control
    pds_self: Optional[float] = None  # Mean PDS when querying with self (technical duplicate)
    pds_control: Optional[float] = None  # Mean PDS when querying with control mean


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


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


def _dataset_palette_viridis(datasets: Sequence[str]) -> Dict[str, str]:
    """Build a viridis color palette for datasets."""
    try:
        from matplotlib import colormaps
        cmap = colormaps["viridis"]
    except (ImportError, KeyError):
        cmap = mpl_cm.get_cmap("viridis")  # fallback for older matplotlib
    if len(datasets) <= 1:
        colors = [cmap(0.5)]
    else:
        colors = [cmap(i / (len(datasets) - 1)) for i in range(len(datasets))]
    return {ds: mpl_colors.to_hex(c) for ds, c in zip(datasets, colors)}


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _dprime_per_pert_pooled_rows(
    dist_list: List[np.ndarray],
) -> Tuple[List[float], List[float]]:
    """Per-perturbation Cohen's d' values (not averaged).

    Same computation as ``_dprime_per_pert_pooled`` but returns the full
    list of per-perturbation d' and AUC values instead of the mean.

    Args:
        dist_list: List of B distance matrices, each shape (n_perts, n_perts).

    Returns:
        Tuple of (dprime_per_pert, auc_per_pert), each of length n_perts.
    """
    from scipy.special import ndtr

    B = len(dist_list)
    n = dist_list[0].shape[0]
    D = np.stack(dist_list, axis=0)  # (B, n, n)

    dprime_per: List[float] = []
    auc_per: List[float] = []
    for i in range(n):
        within_i = D[:, i, i]  # (B,)
        between_i = np.delete(D[:, i, :].reshape(B, n), i, axis=1).ravel()  # (B*(n-1),)
        mu_w = float(np.mean(within_i))
        mu_b = float(np.mean(between_i))
        var_w = float(np.var(within_i, ddof=1)) if within_i.size > 1 else 0.0
        var_b = float(np.var(between_i, ddof=1)) if between_i.size > 1 else 0.0
        sigma_i = float(np.sqrt((var_w + var_b) / 2.0))
        if sigma_i < 1e-15:
            d_i = float("inf") if mu_b > mu_w else 0.0
        else:
            d_i = (mu_b - mu_w) / sigma_i
        dprime_per.append(d_i)
        auc_per.append(float(ndtr(d_i / np.sqrt(2.0))))
    return dprime_per, auc_per


def _load_deg_weights_from_csv(
    csv_path: Path,
    genes: Sequence[str],
    perturbations: Sequence[str],
    deg_weight_transform: str = "minmax",
    base_weight_exponent: float = 2.0,
) -> Dict[str, np.ndarray]:
    """Load per-perturbation DEG weights from an explicit CSV path."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing DEG weight file: {csv_path}")

    gene_to_idx: Dict[str, int] = {gene: idx for idx, gene in enumerate(genes)}
    active_perts = set(str(item) for item in perturbations)
    raw_abs: Dict[str, Dict[str, float]] = {str(pert): {} for pert in perturbations}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pert = str(row.get("Perturbation", "")).strip()
            gene = str(row.get("Gene", "")).strip()
            raw_score = row.get("score", "")
            if pert not in active_perts or gene not in gene_to_idx or raw_score in ("", None):
                continue
            try:
                score_value = abs(float(raw_score))
            except (TypeError, ValueError):
                continue
            prev = raw_abs[pert].get(gene)
            if prev is None or score_value > prev:
                raw_abs[pert][gene] = score_value

    weights_per_pert: Dict[str, np.ndarray] = {}
    skipped_perts: List[str] = []
    n_genes = len(genes)
    for pert in perturbations:
        pert_key = str(pert)
        w = np.zeros(n_genes, dtype=float)
        for gene, score in raw_abs.get(pert_key, {}).items():
            w[gene_to_idx[gene]] = score
        if deg_weight_transform == "minmax":
            w_min = float(np.min(w))
            w_max = float(np.max(w))
            denom = w_max - w_min
            if denom > 0:
                w = (w - w_min) / denom
            else:
                w = np.zeros_like(w)
        elif deg_weight_transform == "rank01":
            order = np.argsort(w)
            if w.size == 1:
                w = np.asarray([1.0], dtype=float)
            elif w.size > 1:
                w_ranked = np.zeros_like(w, dtype=float)
                w_ranked[order] = np.linspace(0.0, 1.0, w.size, endpoint=True)
                w = w_ranked
            else:
                w = np.zeros_like(w)
        else:
            raise ValueError(f"Unsupported deg_weight_transform: {deg_weight_transform}")

        if np.max(w) <= 0.0:
            skipped_perts.append(pert_key)
            continue

        w_out = np.power(w, float(base_weight_exponent))
        w_out = base._normalize_weight_vector_for_effective_genes(
            w_out,
            context=f"DEG weights for perturbation '{pert_key}' from {csv_path.name}",
        )
        base._validate_weight_vector_for_effective_genes(
            w_out,
            context=f"DEG weights for perturbation '{pert_key}' from {csv_path.name}",
        )
        weights_per_pert[pert_key] = w_out

    if skipped_perts:
        print(
            f"WARNING: Skipped {len(skipped_perts)} perturbations with no DEG data in {csv_path.name}: "
            f"{skipped_perts[:5]}{'...' if len(skipped_perts) > 5 else ''}"
        )

    return weights_per_pert


def _load_deg_weights_from_dataset_h5ad(
    dataset_path: Path,
    genes: Sequence[str],
    perturbations: Sequence[str],
    dataset_name: str,
) -> Dict[str, np.ndarray]:
    """Load per-perturbation DEG weights from dataset's precomputed h5ad storage.
    
    This loads weights from adata.uns['names_df_dict_gt'] and adata.uns['scores_df_dict_gt']
    following the same logic as cellsimbench DataManager._precompute_deg_weights():
    1. Take absolute scores
    2. Min-max normalize
    3. Square the weights
    
    Args:
        dataset_path: Path to the dataset h5ad file
        genes: Ordered gene names to align weights to
        perturbations: List of perturbation names to load weights for
        dataset_name: Dataset name for constructing cov_pert_key lookups
        
    Returns:
        Dictionary mapping perturbation name to weight vector aligned with genes
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    adata = sc.read_h5ad(dataset_path)
    
    # Check for required DEG data in uns
    if 'names_df_dict_gt' not in adata.uns or 'scores_df_dict_gt' not in adata.uns:
        raise ValueError(
            f"Dataset {dataset_name} missing required DEG data in uns. "
            f"Need 'names_df_dict_gt' and 'scores_df_dict_gt'."
        )
    
    names_dict = adata.uns['names_df_dict_gt']
    scores_dict = adata.uns['scores_df_dict_gt']
    
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    n_genes = len(genes)
    all_gene_names = list(adata.var_names)
    
    weights_per_pert: Dict[str, np.ndarray] = {}
    missing_perts: List[str] = []
    
    for pert in perturbations:
        # Try different cov_pert_key formats
        cov_pert_keys_to_try = [
            f"{dataset_name}_{pert}",
            pert,
        ]
        
        weights = None
        for cov_pert_key in cov_pert_keys_to_try:
            if cov_pert_key in scores_dict and cov_pert_key in names_dict:
                scores = scores_dict[cov_pert_key]
                gene_names = names_dict[cov_pert_key]
                
                # Compute weights following cellsimbench logic
                abs_scores = np.abs(scores)
                min_val = np.min(abs_scores)
                max_val = np.max(abs_scores)
                
                if max_val > min_val:
                    normalized = (abs_scores - min_val) / (max_val - min_val)
                else:
                    normalized = np.zeros_like(abs_scores)
                
                normalized = np.nan_to_num(normalized, nan=0.0)
                squared_weights = np.square(normalized)
                
                # Create DataFrame and handle duplicates
                weights_df = pd.DataFrame({
                    'gene': gene_names,
                    'weight': squared_weights
                })
                weights_aggregated = weights_df.groupby('gene')['weight'].max()
                
                # Reindex to match requested gene order
                aligned_weights = weights_aggregated.reindex(
                    pd.Index(genes, name='gene'), 
                    fill_value=0.0
                )
                weights = aligned_weights.values.astype(float)
                break
        
        if weights is not None and np.max(weights) > 0:
            weights_per_pert[str(pert)] = weights
        else:
            missing_perts.append(str(pert))
    
    if missing_perts:
        print(
            f"WARNING: No dataset DEG weights found for {len(missing_perts)} perturbations: "
            f"{missing_perts[:5]}{'...' if len(missing_perts) > 5 else ''}"
        )
    
    return weights_per_pert


def _resolve_metric_spec(
    metric: str,
    control_weights: Dict[str, np.ndarray],
    synthetic_weights: Dict[str, np.ndarray],
    dataset_weights: Dict[str, np.ndarray],
) -> Tuple[str, Dict[str, np.ndarray]]:
    """Return the base metric name plus the appropriate weight table."""
    if metric == "wmse_dataset":
        return "wmse", dataset_weights
    if metric in SYNTHETIC_WEIGHT_METRIC_MAP:
        return SYNTHETIC_WEIGHT_METRIC_MAP[metric], synthetic_weights
    return metric, control_weights


def _pds_distance_metric(metric: str) -> str:
    """Map a PDS-style metric to the underlying distance metric."""
    if metric == "pds":
        return "mse"
    if metric == "wpds":
        return "wmse"
    if metric == "pds_pearson_deltapert":
        return "pearson_deltapert"
    return metric


def _stable_dataset_seed(dataset_name: str, base_seed: int) -> int:
    """Deterministic dataset-specific seed without relying on Python hash randomization."""
    return int(base_seed + sum(ord(ch) for ch in dataset_name) * 9973)


def _parse_dataset_max_perturbations_overrides(text: str) -> Dict[str, int]:
    """Parse overrides like 'nadig25hepg2:300,replogle22k562:300'."""
    if not text or not text.strip():
        return {}
    out: Dict[str, int] = {}
    for item in text.split(","):
        token = item.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"Invalid override '{token}'. Expected format dataset:max_perturbations"
            )
        ds, val = token.split(":", 1)
        ds = ds.strip()
        try:
            n = int(val.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid max perturbations in override '{token}'") from exc
        if n <= 0:
            raise ValueError(f"Override must be >0 in '{token}'")
        out[ds] = n
    return out


def _select_perturbations_for_dataset(
    dataset_name: str,
    perts: List[str],
    counts: Dict[str, int],
    max_perturbations: Optional[int],
    seed: int,
    dataset_max_perturbations_overrides: Optional[Dict[str, int]],
    random_sample_datasets: Optional[List[str]],
) -> List[str]:
    """Select perturbations with optional per-dataset random max override."""
    perts_sorted = sorted(perts, key=lambda p: counts[p], reverse=True)
    overrides = dataset_max_perturbations_overrides or {}
    random_set = set(random_sample_datasets or [])
    effective_max = overrides.get(dataset_name, max_perturbations)

    if effective_max is None or effective_max >= len(perts_sorted):
        return perts_sorted

    if dataset_name in random_set:
        rng = np.random.default_rng(_stable_dataset_seed(dataset_name, seed))
        chosen = rng.choice(
            np.asarray(perts_sorted, dtype=object),
            size=int(effective_max),
            replace=False,
        ).tolist()
        return [str(x) for x in chosen]

    return perts_sorted[: int(effective_max)]


def _compute_self_control_dataset(
    dataset_name: str,
    metrics: List[str],
    n_trials: int,
    n_resamples: int,
    max_perturbations: Optional[int],
    seed: int,
    base_weight_exponent: float,
    weight_scheme: str,
    metric_jobs: int = 1,
    dataset_max_perturbations_overrides: Optional[Dict[str, int]] = None,
    random_sample_datasets: Optional[List[str]] = None,
) -> List[SelfControlResult]:
    """Compute self-vs-control discrimination accuracy for all metrics on a single dataset.

    For each perturbation, compares:
    - Self-distance: distance to technical duplicate (other half of same perturbation)
    - Control-distance: distance to control mean

    Accuracy = proportion of perturbations where self-distance < control-distance.

    When metric_jobs > 1, metrics within each trial are processed in parallel.

    Args:
        dataset_name: Name of the dataset to load.
        metrics: Distance metrics to evaluate.
        n_trials: Number of independent random trials.
        n_resamples: Number of half-cell resamples per trial.
        max_perturbations: Keep top-N perturbations by cell count.
        seed: Base random seed.
        base_weight_exponent: Exponent for DEG weight transform.
        weight_scheme: Weight transform scheme ("normal" or "rank").

    Returns:
        List of :class:`CalibrationResult` objects (one per metric × trial).
    """
    # Build a minimal argparse namespace for _prepare_data
    ns = argparse.Namespace(
        max_perturbations=max_perturbations,
        perturbation_sort="top",
        cells_per_perturbation=None,
        max_cells=None,
        seed=seed,
        control_label=None,
    )
    data = base._prepare_data(ns, dataset_name)
    print(f"  [{dataset_name}] Loaded {data.X.shape[0]} cells, "
          f"{len(data.perturbations)} perturbations, {data.X.shape[1]} genes")

    # Filter to top-N perturbations by cell count
    y = np.asarray(data.y, dtype=str)
    perts = [p for p in data.perturbations if p != data.control_label]
    counts = {p: int(np.sum(y == p)) for p in perts}
    filtered_perts = _select_perturbations_for_dataset(
        dataset_name=dataset_name,
        perts=perts,
        counts=counts,
        max_perturbations=max_perturbations,
        seed=seed,
        dataset_max_perturbations_overrides=dataset_max_perturbations_overrides,
        random_sample_datasets=random_sample_datasets,
    )
    effective_max = (
        (dataset_max_perturbations_overrides or {}).get(dataset_name, max_perturbations)
    )
    if effective_max is None:
        print(f"  [{dataset_name}] Using all {len(filtered_perts)} perturbations")
    elif dataset_name in set(random_sample_datasets or []):
        print(
            f"  [{dataset_name}] Using {len(filtered_perts)} perturbations "
            f"(random sample, seed={_stable_dataset_seed(dataset_name, seed)})"
        )
    else:
        print(
            f"  [{dataset_name}] Using {len(filtered_perts)} perturbations "
            f"(top by cell count)"
        )

    # Build label array and filter data
    labels = np.asarray(filtered_perts, dtype=str)
    mask = np.isin(y, filtered_perts) | (y == data.control_label)
    X = np.asarray(data.X, dtype=float)[mask]
    y_filt = y[mask]

    # Check control cells exist
    ctrl_idx = np.where(y_filt == data.control_label)[0]
    if ctrl_idx.size == 0:
        raise ValueError(f"No control cells found for dataset {dataset_name}")

    # Compute control mean
    ctrl_idx_full = np.where(y == data.control_label)[0]
    control_mean = np.mean(X[ctrl_idx_full], axis=0) if ctrl_idx_full.size > 0 else np.mean(X, axis=0)
    dataset_mean = np.mean(X, axis=0)

    # Load DEG weights
    deg_transform, _ = _resolve_weight_transforms(weight_scheme)
    raw_deg_weights = base._load_deg_weights(
        dataset_name=dataset_name,
        genes=data.genes,
        perturbations=filtered_perts,
        deg_weight_transform=deg_transform,
        base_weight_exponent=base_weight_exponent,
    )
    synthetic_deg_path = (
        PROJECT_ROOT
        / "analyses"
        / "perturbation_discrimination"
        / "results"
        / dataset_name
        / "deg_scanpy"
        / "deg_synthetic.csv"
    )
    raw_synthetic_deg_weights = _load_deg_weights_from_csv(
        csv_path=synthetic_deg_path,
        genes=data.genes,
        perturbations=filtered_perts,
        deg_weight_transform=deg_transform,
        base_weight_exponent=base_weight_exponent,
    )
    control_weights = {k: np.asarray(v, dtype=float) for k, v in raw_deg_weights.items()}
    synthetic_weights = {k: np.asarray(v, dtype=float) for k, v in raw_synthetic_deg_weights.items()}
    
    # Load dataset precomputed DEG weights from h5ad
    dataset_path = base.load_dataset_path(PROJECT_ROOT, dataset_name)
    try:
        raw_dataset_weights = _load_deg_weights_from_dataset_h5ad(
            dataset_path=dataset_path,
            genes=data.genes,
            perturbations=filtered_perts,
            dataset_name=dataset_name,
        )
        dataset_weights = {k: np.asarray(v, dtype=float) for k, v in raw_dataset_weights.items()}
        print(f"  [{dataset_name}] Loaded {len(dataset_weights)} dataset DEG weights from h5ad")
    except (FileNotFoundError, ValueError) as e:
        print(f"  [{dataset_name}] WARNING: Could not load dataset DEG weights: {e}")
        dataset_weights = {}
    
    top_deg_local = np.array([], dtype=int)

    results: List[SelfControlResult] = []
    print(
        f"  [{dataset_name}] Self-control config: {n_trials} trial(s), "
        f"{n_resamples} resample(s)/trial, {len(metrics)} metric(s)"
    )

    for trial in range(n_trials):
        trial_start = time.perf_counter()
        print(
            f"  [{dataset_name}] Trial {trial + 1}/{n_trials} started "
            f"({len(metrics)} metric(s), {n_resamples} resample(s)/metric)"
        )
        trial_seed = seed + trial

        # Build bag-mean pairs using all cells per perturbation
        pair_repeats = _build_bag_mean_pairs_all_cells(
            X=X,
            y=y_filt,
            perturbations=filtered_perts,
            seed=trial_seed,
            n_resamples=n_resamples,
        )

        # For each metric, compute distances and calibration accuracy
        if metric_jobs > 1 and len(metrics) > 1:
            # Parallel metric processing
            print(
                f"  [{dataset_name}] Trial {trial + 1}/{n_trials} "
                f"processing {len(metrics)} metrics in parallel ({metric_jobs} workers)"
            )
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=min(metric_jobs, len(metrics))) as pool:
                metric_tasks = [
                    (
                        metric,
                        *_resolve_metric_spec(metric, control_weights, synthetic_weights, dataset_weights),
                        pair_repeats,
                        X,
                        labels,
                        control_mean,
                        dataset_mean,
                        top_deg_local,
                        dataset_name,
                        trial,
                        n_trials,
                        n_resamples,
                        filtered_perts,
                    )
                    for metric in metrics
                ]
                trial_results = pool.starmap(
                    _compute_self_control_metric_worker, metric_tasks
                )
                results.extend(trial_results)
        else:
            # Sequential metric processing
            for metric_idx, metric in enumerate(metrics, start=1):
                metric_start = time.perf_counter()
                print(
                    f"  [{dataset_name}] Trial {trial + 1}/{n_trials} "
                    f"metric {metric_idx}/{len(metrics)}: {metric}"
                )
                metric_base, metric_weights = _resolve_metric_spec(
                    metric,
                    control_weights,
                    synthetic_weights,
                    dataset_weights,
                )
                n_correct_per_resample: List[int] = []
                pds_self_per_resample: List[float] = []
                pds_control_per_resample: List[float] = []

                # Check if this is a PDS metric
                is_pds_metric = metric_base in ("pds", "wpds", "pds_pearson_deltapert")

                for resample_idx, (q_arr, t_arr) in enumerate(pair_repeats, start=1):
                    Xq = np.asarray(q_arr, dtype=float)  # (n_perts, n_genes)
                    Xt = np.asarray(t_arr, dtype=float)  # (n_perts, n_genes) - technical duplicates

                    # Compute within-perturbation distances (positive baseline)
                    dist_self = _singlecell_metric_distance(
                        metric=metric_base,
                        Xq=Xq,
                        Xt=Xt,
                        labels=labels,
                        control_mean=control_mean,
                        dataset_mean=dataset_mean,
                        active_weights=metric_weights,
                        top_deg_local_idx=top_deg_local,
                    )  # Shape: (n_perts, n_perts)
                    # Diagonal contains distance to technical duplicate
                    pos_baseline = np.diag(dist_self)  # (n_perts,)

                    # Compute distance to control mean (negative baseline)
                    # Create a dummy matrix: query perturbations vs control mean
                    X_ctrl_mean = control_mean.reshape(1, -1)  # (1, n_genes)
                    dist_to_ctrl = _singlecell_metric_distance(
                        metric=metric_base,
                        Xq=Xq,
                        Xt=X_ctrl_mean,
                        labels=labels,
                        control_mean=control_mean,
                        dataset_mean=dataset_mean,
                        active_weights=metric_weights,
                        top_deg_local_idx=top_deg_local,
                    )  # Shape: (n_perts, 1)
                    neg_baseline = dist_to_ctrl[:, 0]  # (n_perts,)

                    if is_pds_metric:
                        # For PDS metrics: compute proper PDS scores from distance matrices
                        # Self PDS: from dist_self (queries Xq vs targets Xt, both are bag means)
                        self_pds_scores = _compute_pds_scores_from_distances(dist_self)

                        # Control PDS: query is control, targets are bag means (Xt)
                        # Compute distances from control to all targets
                        dist_control_to_targets = _singlecell_metric_distance(
                            metric=_pds_distance_metric(metric_base),
                            Xq=X_ctrl_mean,  # (1, n_genes)
                            Xt=Xt,  # (n_perts, n_genes)
                            labels=labels,
                            control_mean=control_mean,
                            dataset_mean=dataset_mean,
                            active_weights=metric_weights,
                            top_deg_local_idx=top_deg_local,
                        )  # Shape: (1, n_perts)

                        # For each perturbation, compute what its PDS would be if control was the query
                        n_perts = Xt.shape[0]
                        control_pds_scores = np.zeros(n_perts, dtype=float)
                        for i in range(n_perts):
                            dists = dist_control_to_targets[0, :]  # (n_perts,)
                            order = np.argsort(dists)
                            rank = int(np.where(order == i)[0][0])
                            if n_perts <= 1:
                                control_pds_scores[i] = 1.0
                            else:
                                control_pds_scores[i] = 1.0 - 2.0 * (rank / float(n_perts - 1))

                        pds_self_per_resample.append(float(np.mean(self_pds_scores)))
                        pds_control_per_resample.append(float(np.mean(control_pds_scores)))
                        # For PDS, accuracy is proportion where self PDS > control PDS
                        n_correct = int(np.sum(self_pds_scores > control_pds_scores))
                    else:
                        # For standard metrics: lower distance = better
                        n_correct = int(np.sum(pos_baseline < neg_baseline))
                    n_correct_per_resample.append(n_correct)
                    if n_resamples > 1 and (
                        resample_idx == n_resamples
                        or resample_idx % max(1, n_resamples // 5) == 0
                    ):
                        print(
                            f"    [{dataset_name}] Trial {trial + 1}/{n_trials} "
                            f"{metric}: resample {resample_idx}/{n_resamples}"
                        )

                # Average across resamples
                avg_n_correct = int(round(np.mean(n_correct_per_resample)))
                accuracy = avg_n_correct / len(filtered_perts)

                # Compute mean PDS scores for PDS metrics
                pds_self = None
                pds_control = None
                if is_pds_metric and pds_self_per_resample:
                    pds_self = float(np.mean(pds_self_per_resample))
                    pds_control = float(np.mean(pds_control_per_resample))

                results.append(SelfControlResult(
                    dataset=dataset_name,
                    metric=metric,
                    trial=trial,
                    accuracy=accuracy,
                    n_total=len(filtered_perts),
                    n_correct=avg_n_correct,
                    pds_self=pds_self,
                    pds_control=pds_control,
                ))
                metric_elapsed = time.perf_counter() - metric_start
                if is_pds_metric:
                    print(
                        f"    [{dataset_name}] {metric} done: accuracy={accuracy:.4f} "
                        f"({avg_n_correct}/{len(filtered_perts)}), "
                        f"self_PDS={pds_self:.3f}, control_PDS={pds_control:.3f} in {metric_elapsed:.1f}s"
                    )
                else:
                    print(
                        f"    [{dataset_name}] {metric} done: accuracy={accuracy:.4f} "
                        f"({avg_n_correct}/{len(filtered_perts)}) in {metric_elapsed:.1f}s"
                    )

        trial_elapsed = time.perf_counter() - trial_start
        print(
            f"  [{dataset_name}] Trial {trial + 1}/{n_trials} finished in "
            f"{trial_elapsed:.1f}s"
        )

    print(f"  [{dataset_name}] Done — {len(results)} self-control results")
    return results


def _compute_pds_scores_from_distances(distances: np.ndarray) -> np.ndarray:
    """Convert distance matrix to PDS scores.

    For each query (row), computes the PDS score based on ranking distances
    to all targets. PDS = 1.0 when query is closest to diagonal (correct target),
    PDS = 0.0 at chance level, PDS = -1.0 when farthest.

    Args:
        distances: (n_queries, n_targets) distance matrix

    Returns:
        (n_queries,) array of PDS scores in [-1, 1]
    """
    n_queries, n_targets = distances.shape
    pds_scores = np.zeros(n_queries, dtype=float)

    for i in range(n_queries):
        # Rank of target i among all targets for query i
        order = np.argsort(distances[i])
        # Find where the diagonal (target i) ranks
        rank = int(np.where(order == i)[0][0])
        # Convert rank to PDS score: rank 0 -> 1.0, middle -> 0.0, last -> -1.0
        if n_targets <= 1:
            pds_scores[i] = 1.0
        else:
            pds_scores[i] = 1.0 - 2.0 * (rank / float(n_targets - 1))

    return pds_scores


def _compute_self_control_metric(
    metric: str,
    metric_base: str,
    metric_weights: Dict[str, np.ndarray],
    pair_repeats: List[Tuple[np.ndarray, np.ndarray]],
    X: np.ndarray,
    labels: np.ndarray,
    control_mean: np.ndarray,
    dataset_mean: np.ndarray,
    top_deg_local: np.ndarray,
    dataset_name: str,
    trial: int,
    n_trials: int,
    n_resamples: int,
    filtered_perts: List[str],
) -> SelfControlResult:
    """Compute self-control accuracy for a single metric.

    This is a worker for metric-level parallelization.
    """
    metric_start = time.perf_counter()
    n_correct_per_resample: List[int] = []
    pds_self_per_resample: List[float] = []
    pds_control_per_resample: List[float] = []

    # Check if this is a PDS metric
    is_pds_metric = metric_base in ("pds", "wpds", "pds_pearson_deltapert")

    for resample_idx, (q_arr, t_arr) in enumerate(pair_repeats, start=1):
        Xq = np.asarray(q_arr, dtype=float)
        Xt = np.asarray(t_arr, dtype=float)

        if is_pds_metric:
            # For PDS metrics, compute distance matrices and then convert to PDS scores
            # Self PDS: queries are bag means (Xq), targets are bag means (Xt)
            dist_self = _singlecell_metric_distance(
                metric=_pds_distance_metric(metric_base),
                Xq=Xq,
                Xt=Xt,
                labels=labels,
                control_mean=control_mean,
                dataset_mean=dataset_mean,
                active_weights=metric_weights,
                top_deg_local_idx=top_deg_local,
            )
            self_pds_scores = _compute_pds_scores_from_distances(dist_self)

            # Control PDS: query is control mean, targets are bag means (Xt)
            # Create a distance matrix where control is compared to all targets
            X_ctrl_mean = control_mean.reshape(1, -1)
            dist_control_to_targets = _singlecell_metric_distance(
                metric=_pds_distance_metric(metric_base),
                Xq=X_ctrl_mean,  # (1, n_genes)
                Xt=Xt,  # (n_perts, n_genes)
                labels=labels,
                control_mean=control_mean,
                dataset_mean=dataset_mean,
                active_weights=metric_weights,
                top_deg_local_idx=top_deg_local,
            )  # Shape: (1, n_perts)

            # For control PDS, we need to simulate: if control was the "prediction"
            # for each perturbation, how well would it retrieve the correct ground truth?
            # We replicate the control distances for each perturbation query
            # and compute PDS scores
            n_perts = Xt.shape[0]
            control_pds_scores = np.zeros(n_perts, dtype=float)
            for i in range(n_perts):
                # Distance from control to all targets
                dists = dist_control_to_targets[0, :]  # (n_perts,)
                # Rank of target i among all targets
                order = np.argsort(dists)
                rank = int(np.where(order == i)[0][0])
                if n_perts <= 1:
                    control_pds_scores[i] = 1.0
                else:
                    control_pds_scores[i] = 1.0 - 2.0 * (rank / float(n_perts - 1))

            pds_self_per_resample.append(float(np.mean(self_pds_scores)))
            pds_control_per_resample.append(float(np.mean(control_pds_scores)))

            # Accuracy: proportion where self PDS > control PDS
            n_correct = int(np.sum(self_pds_scores > control_pds_scores))
            n_correct_per_resample.append(n_correct)
        else:
            # For standard metrics, use original logic
            dist_self = _singlecell_metric_distance(
                metric=metric_base,
                Xq=Xq,
                Xt=Xt,
                labels=labels,
                control_mean=control_mean,
                dataset_mean=dataset_mean,
                active_weights=metric_weights,
                top_deg_local_idx=top_deg_local,
            )
            pos_baseline = np.diag(dist_self)

            X_ctrl_mean = control_mean.reshape(1, -1)
            dist_to_ctrl = _singlecell_metric_distance(
                metric=metric_base,
                Xq=Xq,
                Xt=X_ctrl_mean,
                labels=labels,
                control_mean=control_mean,
                dataset_mean=dataset_mean,
                active_weights=metric_weights,
                top_deg_local_idx=top_deg_local,
            )
            neg_baseline = dist_to_ctrl[:, 0]

            n_correct = int(np.sum(pos_baseline < neg_baseline))
            n_correct_per_resample.append(n_correct)

    avg_n_correct = int(round(np.mean(n_correct_per_resample)))
    accuracy = avg_n_correct / len(filtered_perts)

    # Compute mean PDS scores for PDS metrics
    pds_self = None
    pds_control = None
    if is_pds_metric and pds_self_per_resample:
        pds_self = float(np.mean(pds_self_per_resample))
        pds_control = float(np.mean(pds_control_per_resample))

    metric_elapsed = time.perf_counter() - metric_start
    if is_pds_metric:
        print(
            f"    [{dataset_name}] {metric} done: accuracy={accuracy:.4f} "
            f"({avg_n_correct}/{len(filtered_perts)}), "
            f"self_PDS={pds_self:.3f}, control_PDS={pds_control:.3f} in {metric_elapsed:.1f}s"
        )
    else:
        print(
            f"    [{dataset_name}] {metric} done: accuracy={accuracy:.4f} "
            f"({avg_n_correct}/{len(filtered_perts)}) in {metric_elapsed:.1f}s"
        )

    return SelfControlResult(
        dataset=dataset_name,
        metric=metric,
        trial=trial,
        accuracy=accuracy,
        n_total=len(filtered_perts),
        n_correct=avg_n_correct,
        pds_self=pds_self,
        pds_control=pds_control,
    )


def _compute_self_control_metric_worker(
    metric: str,
    metric_base: str,
    metric_weights: Dict[str, np.ndarray],
    pair_repeats: List[Tuple[np.ndarray, np.ndarray]],
    X: np.ndarray,
    labels: np.ndarray,
    control_mean: np.ndarray,
    dataset_mean: np.ndarray,
    top_deg_local: np.ndarray,
    dataset_name: str,
    trial: int,
    n_trials: int,
    n_resamples: int,
    filtered_perts: List[str],
) -> SelfControlResult:
    """Wrapper for Pool.starmap that unpacks arguments for _compute_self_control_metric."""
    return _compute_self_control_metric(
        metric=metric,
        metric_base=metric_base,
        metric_weights=metric_weights,
        pair_repeats=pair_repeats,
        X=X,
        labels=labels,
        control_mean=control_mean,
        dataset_mean=dataset_mean,
        top_deg_local=top_deg_local,
        dataset_name=dataset_name,
        trial=trial,
        n_trials=n_trials,
        n_resamples=n_resamples,
        filtered_perts=filtered_perts,
    )


def _compute_standard_metric(
    metric: str,
    pair_repeats: List[Tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    control_mean: np.ndarray,
    dataset_mean: np.ndarray,
    control_weights: Dict[str, np.ndarray],
    synthetic_weights: Dict[str, np.ndarray],
    dataset_weights: Dict[str, np.ndarray],
    top_deg_local: np.ndarray,
    dataset_name: str,
    trial: int,
    n_trials: int,
    filtered_perts: List[str],
) -> Tuple[ComparisonResult, List[PertResult]]:
    """Compute one standard-mode metric for a single dataset/trial."""
    metric_start = time.perf_counter()
    print(
        f"  [{dataset_name}] Trial {trial + 1}/{n_trials} "
        f"metric: {metric}"
    )
    metric_base, metric_weights = _resolve_metric_spec(
        metric,
        control_weights,
        synthetic_weights,
        dataset_weights,
    )
    dist_list: List[np.ndarray] = []
    for q_arr, t_arr in pair_repeats:
        Xq = np.asarray(q_arr, dtype=float)
        Xt = np.asarray(t_arr, dtype=float)
        dist_list.append(
            _singlecell_metric_distance(
                metric=metric_base,
                Xq=Xq,
                Xt=Xt,
                labels=labels,
                control_mean=control_mean,
                dataset_mean=dataset_mean,
                active_weights=metric_weights,
                top_deg_local_idx=top_deg_local,
            )
        )

    n_perts = dist_list[0].shape[0]
    within_all = np.concatenate([np.diag(d) for d in dist_list])
    off_mask = ~np.eye(n_perts, dtype=bool)
    between_all = np.concatenate([d[off_mask] for d in dist_list])
    dprime, auc = _dprime_per_pert_pooled(dist_list)

    is_pds_metric = metric_base in ("pds", "wpds", "pds_pearson_deltapert")
    if is_pds_metric:
        pds = float("nan")
        pds_rows = [float("nan")] * n_perts
    else:
        pds_per_resample = [float(np.mean(_query_truth_scores(d))) for d in dist_list]
        pds = float(np.mean(pds_per_resample))
        pds_rows = [
            float(np.mean([_query_truth_scores(d)[i] for d in dist_list]))
            for i in range(n_perts)
        ]

    result = ComparisonResult(
        dataset=dataset_name,
        metric=metric,
        trial=trial,
        dprime=dprime,
        auc=auc,
        pds=pds,
        within=float(np.mean(within_all)),
        between=float(np.mean(between_all)),
    )
    dprime_rows, auc_rows = _dprime_per_pert_pooled_rows(dist_list)
    pert_results: List[PertResult] = []
    for i, pert in enumerate(filtered_perts):
        pert_results.append(PertResult(
            dataset=dataset_name,
            metric=metric,
            trial=trial,
            perturbation=pert,
            dprime=dprime_rows[i],
            auc=auc_rows[i],
            pds=pds_rows[i],
        ))

    metric_elapsed = time.perf_counter() - metric_start
    print(
        f"    [{dataset_name}] {metric} done: d'={dprime:.3f}, "
        f"AUC={auc:.3f}, PDS={pds:.3f} in {metric_elapsed:.1f}s"
    )
    return result, pert_results


def _compute_standard_metric_worker(
    metric: str,
    pair_repeats: List[Tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    control_mean: np.ndarray,
    dataset_mean: np.ndarray,
    control_weights: Dict[str, np.ndarray],
    synthetic_weights: Dict[str, np.ndarray],
    dataset_weights: Dict[str, np.ndarray],
    top_deg_local: np.ndarray,
    dataset_name: str,
    trial: int,
    n_trials: int,
    filtered_perts: List[str],
) -> Tuple[ComparisonResult, List[PertResult]]:
    """Wrapper for multiprocessing standard-mode metric computation."""
    return _compute_standard_metric(
        metric=metric,
        pair_repeats=pair_repeats,
        labels=labels,
        control_mean=control_mean,
        dataset_mean=dataset_mean,
        control_weights=control_weights,
        synthetic_weights=synthetic_weights,
        dataset_weights=dataset_weights,
        top_deg_local=top_deg_local,
        dataset_name=dataset_name,
        trial=trial,
        n_trials=n_trials,
        filtered_perts=filtered_perts,
    )


def _compute_dataset(
    dataset_name: str,
    metrics: List[str],
    n_trials: int,
    n_resamples: int,
    max_perturbations: Optional[int],
    seed: int,
    base_weight_exponent: float,
    weight_scheme: str,
    metric_jobs: int = 1,
    dataset_max_perturbations_overrides: Optional[Dict[str, int]] = None,
    random_sample_datasets: Optional[List[str]] = None,
) -> Tuple[List[ComparisonResult], List[PertResult]]:
    """Compute d' and AUC for all metrics on a single dataset.

    Args:
        dataset_name: Name of the dataset to load.
        metrics: Distance metrics to evaluate.
        n_trials: Number of independent random trials.
        n_resamples: Number of half-cell resamples per trial for variance.
        max_perturbations: Keep top-N perturbations by cell count.
        seed: Base random seed.
        base_weight_exponent: Exponent for DEG weight transform.
        weight_scheme: Weight transform scheme (``"normal"`` or ``"rank"``).

    Returns:
        Tuple of:
            - List of :class:`ComparisonResult` (one per metric × trial, aggregated).
            - List of :class:`PertResult` (one per metric × trial × perturbation).
    """
    # Build a minimal argparse namespace for _prepare_data
    ns = argparse.Namespace(
        max_perturbations=max_perturbations,
        perturbation_sort="top",
        cells_per_perturbation=None,
        max_cells=None,
        seed=seed,
        control_label=None,
    )
    data = base._prepare_data(ns, dataset_name)
    print(f"  [{dataset_name}] Loaded {data.X.shape[0]} cells, "
          f"{len(data.perturbations)} perturbations, {data.X.shape[1]} genes")

    # Filter to top-N perturbations by cell count
    y = np.asarray(data.y, dtype=str)
    perts = [p for p in data.perturbations if p != data.control_label]
    counts = {p: int(np.sum(y == p)) for p in perts}
    filtered_perts = _select_perturbations_for_dataset(
        dataset_name=dataset_name,
        perts=perts,
        counts=counts,
        max_perturbations=max_perturbations,
        seed=seed,
        dataset_max_perturbations_overrides=dataset_max_perturbations_overrides,
        random_sample_datasets=random_sample_datasets,
    )
    effective_max = (
        (dataset_max_perturbations_overrides or {}).get(dataset_name, max_perturbations)
    )
    if effective_max is None:
        print(f"  [{dataset_name}] Using all {len(filtered_perts)} perturbations")
    elif dataset_name in set(random_sample_datasets or []):
        print(
            f"  [{dataset_name}] Using {len(filtered_perts)} perturbations "
            f"(random sample, seed={_stable_dataset_seed(dataset_name, seed)})"
        )
    else:
        print(
            f"  [{dataset_name}] Using {len(filtered_perts)} perturbations "
            f"(top by cell count)"
        )

    # Build label array and filter data
    labels = np.asarray(filtered_perts, dtype=str)
    mask = np.isin(y, filtered_perts) | (y == data.control_label)
    X = np.asarray(data.X, dtype=float)[mask]
    y_filt = y[mask]

    ctrl_idx = np.where(y_filt == data.control_label)[0]
    control_mean = np.mean(X[ctrl_idx], axis=0) if ctrl_idx.size > 0 else np.mean(X, axis=0)
    dataset_mean = np.mean(X, axis=0)

    # Load DEG weights (default transform, no effective-gene calibration)
    deg_transform, _ = _resolve_weight_transforms(weight_scheme)
    raw_deg_weights = base._load_deg_weights(
        dataset_name=dataset_name,
        genes=data.genes,
        perturbations=filtered_perts,
        deg_weight_transform=deg_transform,
        base_weight_exponent=base_weight_exponent,
    )
    synthetic_deg_path = (
        PROJECT_ROOT
        / "analyses"
        / "perturbation_discrimination"
        / "results"
        / dataset_name
        / "deg_scanpy"
        / "deg_synthetic.csv"
    )
    raw_synthetic_deg_weights = _load_deg_weights_from_csv(
        csv_path=synthetic_deg_path,
        genes=data.genes,
        perturbations=filtered_perts,
        deg_weight_transform=deg_transform,
        base_weight_exponent=base_weight_exponent,
    )
    # No calibration — use raw weights directly
    control_weights = {k: np.asarray(v, dtype=float) for k, v in raw_deg_weights.items()}
    synthetic_weights = {k: np.asarray(v, dtype=float) for k, v in raw_synthetic_deg_weights.items()}
    
    # Load dataset precomputed DEG weights from h5ad
    dataset_path = base.load_dataset_path(PROJECT_ROOT, dataset_name)
    try:
        raw_dataset_weights = _load_deg_weights_from_dataset_h5ad(
            dataset_path=dataset_path,
            genes=data.genes,
            perturbations=filtered_perts,
            dataset_name=dataset_name,
        )
        dataset_weights = {k: np.asarray(v, dtype=float) for k, v in raw_dataset_weights.items()}
        print(f"  [{dataset_name}] Loaded {len(dataset_weights)} dataset DEG weights from h5ad")
        # Report effective gene stats for dataset weights
        if dataset_weights:
            eg_vals = [base._effective_genes_single(w) for w in dataset_weights.values()]
            print(f"  [{dataset_name}] Dataset weights effective genes: mean={np.mean(eg_vals):.1f}, median={np.median(eg_vals):.1f}")
    except (FileNotFoundError, ValueError) as e:
        print(f"  [{dataset_name}] WARNING: Could not load dataset DEG weights: {e}")
        dataset_weights = {}
    
    # Report effective gene stats for control weights
    if control_weights:
        eg_vals_ctrl = [base._effective_genes_single(w) for w in control_weights.values()]
        print(f"  [{dataset_name}] Control weights effective genes: mean={np.mean(eg_vals_ctrl):.1f}, median={np.median(eg_vals_ctrl):.1f}")
    if synthetic_weights:
        eg_vals_syn = [base._effective_genes_single(w) for w in synthetic_weights.values()]
        print(f"  [{dataset_name}] Synthetic weights effective genes: mean={np.mean(eg_vals_syn):.1f}, median={np.median(eg_vals_syn):.1f}")

    # Dummy top-deg indices (not used for our metric set)
    top_deg_local = np.array([], dtype=int)

    results: List[ComparisonResult] = []
    pert_results: List[PertResult] = []
    print(
        f"  [{dataset_name}] Standard config: {n_trials} trial(s), "
        f"{n_resamples} resample(s)/trial, {len(metrics)} metric(s)"
    )

    for trial in range(n_trials):
        trial_start = time.perf_counter()
        print(
            f"  [{dataset_name}] Trial {trial + 1}/{n_trials} started "
            f"({len(metrics)} metric(s), {n_resamples} resample(s)/metric)"
        )
        trial_seed = seed + trial

        # Build bag-mean pairs using all cells per perturbation
        pair_repeats = _build_bag_mean_pairs_all_cells(
            X=X,
            y=y_filt,
            perturbations=filtered_perts,
            seed=trial_seed,
            n_resamples=n_resamples,
        )

        if metric_jobs > 1 and len(metrics) > 1:
            print(
                f"  [{dataset_name}] Trial {trial + 1}/{n_trials} "
                f"processing {len(metrics)} metrics in parallel ({metric_jobs} workers)"
            )
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=min(metric_jobs, len(metrics))) as pool:
                metric_tasks = [
                    (
                        metric,
                        pair_repeats,
                        labels,
                        control_mean,
                        dataset_mean,
                        control_weights,
                        synthetic_weights,
                        dataset_weights,
                        top_deg_local,
                        dataset_name,
                        trial,
                        n_trials,
                        filtered_perts,
                    )
                    for metric in metrics
                ]
                trial_metric_results = pool.starmap(
                    _compute_standard_metric_worker,
                    metric_tasks,
                )
            for result_row, pert_rows in trial_metric_results:
                results.append(result_row)
                pert_results.extend(pert_rows)
        else:
            for metric in metrics:
                result_row, pert_rows = _compute_standard_metric(
                    metric=metric,
                    pair_repeats=pair_repeats,
                    labels=labels,
                    control_mean=control_mean,
                    dataset_mean=dataset_mean,
                    control_weights=control_weights,
                    synthetic_weights=synthetic_weights,
                    dataset_weights=dataset_weights,
                    top_deg_local=top_deg_local,
                    dataset_name=dataset_name,
                    trial=trial,
                    n_trials=n_trials,
                    filtered_perts=filtered_perts,
                )
                results.append(result_row)
                pert_results.extend(pert_rows)

        trial_elapsed = time.perf_counter() - trial_start
        print(
            f"  [{dataset_name}] Trial {trial + 1}/{n_trials} finished in "
            f"{trial_elapsed:.1f}s"
        )

    print(f"  [{dataset_name}] Done — {len(results)} results, {len(pert_results)} per-pert rows")
    return results, pert_results


def _compute_dataset_worker(
    item: Tuple[str, Dict[str, Any]],
) -> Tuple[str, List[ComparisonResult], List[PertResult]]:
    """Top-level worker for multiprocessing (must be picklable for spawn).

    Args:
        item: Tuple of (dataset_name, kwargs dict for _compute_dataset).

    Returns:
        Tuple of (dataset_name, aggregated results, per-perturbation results).
    """
    dataset_name, kwargs = item
    try:
        results, pert_results = _compute_dataset(**kwargs)
        return (dataset_name, results, pert_results)
    except Exception as exc:
        return (dataset_name, [], [])  # caller will see empty and can log


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _build_summary_df(results: List[ComparisonResult]) -> pd.DataFrame:
    """Aggregate per-trial results into mean ± std across trials.

    Returns:
        DataFrame with columns: Dataset, Metric, mean_dprime, std_dprime,
        mean_auc, std_auc, mean_pds, std_pds.
    """
    rows: List[Dict[str, Any]] = []
    for r in results:
        rows.append({
            "Dataset": DATASET_DISPLAY.get(r.dataset, r.dataset),
            "Metric": METRIC_DISPLAY.get(r.metric, r.metric),
            "dataset_raw": r.dataset,
            "metric_raw": r.metric,
            "dprime": r.dprime,
            "auc": r.auc,
            "pds": r.pds,
        })
    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["Dataset", "Metric", "dataset_raw", "metric_raw"])
        .agg(
            mean_dprime=("dprime", "mean"),
            std_dprime=("dprime", "std"),
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
            mean_pds=("pds", "mean"),
            std_pds=("pds", "std"),
        )
        .reset_index()
    )
    agg["std_dprime"] = agg["std_dprime"].fillna(0.0)
    agg["std_auc"] = agg["std_auc"].fillna(0.0)
    agg["std_pds"] = agg["std_pds"].fillna(0.0)
    return agg


def plot_bar(
    summary: pd.DataFrame,
    y_col: str,
    y_err_col: str,
    y_label: str,
    output_path: Path,
    metrics_order: List[str],
) -> None:
    """Grouped bar plot: x = metric, bars = dataset, sorted by mean y.

    Args:
        summary: Aggregated DataFrame from :func:`_build_summary_df`.
        y_col: Column name for the bar height (e.g. ``"mean_dprime"``).
        y_err_col: Column for error bar half-width (e.g. ``"std_dprime"``).
        y_label: Axis label for y.
        output_path: Where to save the figure.
        metrics_order: Ordered list of raw metric names for x-axis ordering.
    """
    _apply_nature_rc()

    # Sort metrics by their mean y value (ascending)
    metric_means = (
        summary.groupby("metric_raw")[y_col]
        .mean()
        .reindex(metrics_order)
        .sort_values()
    )
    sorted_metrics = metric_means.index.tolist()
    display_order = [METRIC_DISPLAY.get(m, m) for m in sorted_metrics]

    # Sort datasets by their overall mean y (ascending) for consistent legend
    ds_means = summary.groupby("dataset_raw")[y_col].mean().sort_values()
    sorted_datasets = ds_means.index.tolist()
    display_ds = [DATASET_DISPLAY.get(d, d) for d in sorted_datasets]

    palette = _dataset_palette_viridis(display_ds)

    n_metrics = len(sorted_metrics)
    n_datasets = len(sorted_datasets)
    bar_width = 0.72 / n_datasets
    x_base = np.arange(n_metrics, dtype=float)

    fig, ax = plt.subplots(figsize=FIG_SIZE_BAR)

    for di, ds_raw in enumerate(sorted_datasets):
        ds_label = DATASET_DISPLAY.get(ds_raw, ds_raw)
        sub = summary[summary["dataset_raw"] == ds_raw].set_index("metric_raw")
        heights = [float(sub.loc[m, y_col]) if m in sub.index else 0.0 for m in sorted_metrics]
        errs = [float(sub.loc[m, y_err_col]) if m in sub.index else 0.0 for m in sorted_metrics]
        offset = (di - (n_datasets - 1) / 2) * bar_width
        ax.bar(
            x_base + offset,
            heights,
            width=bar_width * 0.9,
            yerr=errs,
            capsize=1.5,
            error_kw={"linewidth": 0.5},
            color=palette[ds_label],
            label=ds_label,
            edgecolor="none",
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels(display_order, rotation=30, ha="right")
    ax.set_ylabel(y_label)
    ax.set_xlim(-0.6, n_metrics - 0.4)
    y_min = min(0, ax.get_ylim()[0])
    ax.set_ylim(bottom=y_min)
    ax.legend(
        fontsize=NATURE_LEGEND_SIZE,
        frameon=False,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=2, width=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=NATURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_heatmap(
    summary: pd.DataFrame,
    y_col: str,
    title: str,
    output_path: Path,
    metrics_order: List[str],
    cmap: str = "viridis",
    fmt: str = ".2f",
) -> None:
    """Heatmap with datasets as rows and metrics as columns.

    Args:
        summary: Aggregated DataFrame from :func:`_build_summary_df`.
        y_col: Column name for heatmap cell value.
        title: Plot title.
        output_path: Where to save the figure.
        metrics_order: Ordered list of raw metric names.
        cmap: Matplotlib colormap name.
        fmt: Format string for annotation values.
    """
    _apply_nature_rc()

    # Sort metrics by global mean (ascending left to right)
    metric_means = (
        summary.groupby("metric_raw")[y_col]
        .mean()
        .reindex(metrics_order)
        .sort_values()
    )
    sorted_metrics = metric_means.index.tolist()

    # Sort datasets by global mean (ascending top to bottom)
    ds_means = summary.groupby("dataset_raw")[y_col].mean().sort_values()
    sorted_datasets = ds_means.index.tolist()

    pivot = summary.pivot_table(
        index="dataset_raw", columns="metric_raw", values=y_col
    )
    pivot = pivot.reindex(index=sorted_datasets, columns=sorted_metrics)
    pivot.index = [DATASET_DISPLAY.get(d, d) for d in pivot.index]
    pivot.columns = [METRIC_DISPLAY.get(m, m) for m in pivot.columns]

    fig, ax = plt.subplots(figsize=FIG_SIZE_HEATMAP)
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)

    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val < (pivot.values.max() + pivot.values.min()) / 2 else "black"
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=NATURE_TICK_SIZE, color=text_color)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title, fontsize=NATURE_TITLE_SIZE, pad=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.tick_params(labelsize=NATURE_TICK_SIZE, length=2, width=0.4)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=NATURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_distribution(
    pert_results: List[PertResult],
    y_col: str,
    y_label: str,
    output_path: Path,
    metrics_order: List[str],
) -> None:
    """Violin + strip plot of per-perturbation values.

    One subplot per metric (sorted by median y, ascending left to right).
    Within each subplot, one violin per dataset (sorted by median y, ascending).
    Individual perturbation dots are overlaid as a strip plot.
    Trials are pooled (each perturbation × trial = one dot).

    Args:
        pert_results: Per-perturbation results from :class:`PertResult`.
        y_col: Which value to plot: ``"dprime"``, ``"auc"``, or ``"pds"``.
        y_label: Y-axis label.
        output_path: Where to save the figure.
        metrics_order: Ordered list of raw metric names used for filtering.
    """
    import seaborn as sns

    _apply_nature_rc()

    rows: List[Dict[str, Any]] = []
    for r in pert_results:
        if r.metric not in metrics_order:
            continue
        rows.append({
            "metric_raw": r.metric,
            "Metric": METRIC_DISPLAY.get(r.metric, r.metric),
            "dataset_raw": r.dataset,
            "Dataset": DATASET_DISPLAY.get(r.dataset, r.dataset),
            y_col: getattr(r, y_col),
        })
    if not rows:
        return
    df = pd.DataFrame(rows)

    # Sort metrics by median y (ascending)
    metric_medians = (
        df.groupby("metric_raw")[y_col].median().reindex(metrics_order).sort_values()
    )
    sorted_metrics = [m for m in metric_medians.index if m in df["metric_raw"].unique()]

    # Sort datasets by overall median y (ascending) — determines violin order and colour
    ds_medians = df.groupby("dataset_raw")[y_col].median().sort_values()
    sorted_ds_raw = ds_medians.index.tolist()
    sorted_ds_display = [DATASET_DISPLAY.get(d, d) for d in sorted_ds_raw]
    palette = _dataset_palette_viridis(sorted_ds_display)

    n_metrics = len(sorted_metrics)
    fig_w = max(3.5, n_metrics * 1.6)
    fig, axes = plt.subplots(1, n_metrics, figsize=(fig_w, 2.8), sharey=True)
    if n_metrics == 1:
        axes = [axes]

    rng = np.random.default_rng(0)
    for ax, metric_raw in zip(axes, sorted_metrics):
        sub = df[df["metric_raw"] == metric_raw]
        for xi, ds_raw in enumerate(sorted_ds_raw):
            ds_label = DATASET_DISPLAY.get(ds_raw, ds_raw)
            vals = sub[sub["dataset_raw"] == ds_raw][y_col].to_numpy()
            if vals.size == 0:
                continue
            color = palette[ds_label]
            # Violin
            parts = ax.violinplot(
                vals, positions=[xi], widths=0.7,
                showmedians=True, showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.55)
                pc.set_edgecolor("none")
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(0.7)
            # Strip (jittered dots)
            jitter = rng.uniform(-0.18, 0.18, size=vals.size)
            ax.scatter(
                xi + jitter, vals,
                color=color, s=2.5, alpha=0.6, linewidths=0, zorder=3,
            )
        ax.set_xticks(range(len(sorted_ds_raw)))
        ax.set_xticklabels(sorted_ds_display, rotation=40, ha="right",
                           fontsize=NATURE_TICK_SIZE)
        ax.set_title(METRIC_DISPLAY.get(metric_raw, metric_raw),
                     fontsize=NATURE_TITLE_SIZE, pad=4)
        ax.tick_params(axis="y", labelsize=NATURE_TICK_SIZE, length=2, width=0.4)
        ax.tick_params(axis="x", length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(y_label, fontsize=NATURE_AXIS_LABEL_SIZE)

    fig.tight_layout(w_pad=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=NATURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_self_control_heatmap(
    sc_results: List[SelfControlResult],
    output_path: Path,
    metrics_order: List[str],
) -> None:
    """Heatmap of self-vs-control discrimination accuracy (datasets × metrics).

    Shows the ratio of perturbations where the self-distance (technical
    duplicate) is smaller than the control-distance. Higher is better.

    Args:
        calib_results: List of :class:`CalibrationResult` objects.
        output_path: Where to save the figure.
        metrics_order: Ordered list of raw metric names.
    """
    _apply_nature_rc()

    # Build DataFrame
    rows: List[Dict[str, Any]] = []
    for r in sc_results:
        rows.append({
            "dataset_raw": r.dataset,
            "Dataset": DATASET_DISPLAY.get(r.dataset, r.dataset),
            "metric_raw": r.metric,
            "Metric": METRIC_DISPLAY.get(r.metric, r.metric),
            "accuracy": r.accuracy,
        })
    df = pd.DataFrame(rows)

    # Aggregate across trials
    summary = df.groupby(["dataset_raw", "Dataset", "metric_raw", "Metric"]).agg(
        mean_accuracy=("accuracy", "mean"),
    ).reset_index()

    # Sort metrics by global mean (ascending left to right)
    metric_means = (
        summary.groupby("metric_raw")["mean_accuracy"]
        .mean()
        .reindex(metrics_order)
        .sort_values()
    )
    sorted_metrics = [m for m in metric_means.index if m in summary["metric_raw"].unique()]

    # Sort datasets by global mean (ascending top to bottom)
    ds_means = summary.groupby("dataset_raw")["mean_accuracy"].mean().sort_values()
    sorted_datasets = ds_means.index.tolist()

    pivot = summary.pivot_table(
        index="dataset_raw", columns="metric_raw", values="mean_accuracy"
    )
    pivot = pivot.reindex(index=sorted_datasets, columns=sorted_metrics)
    pivot.index = [DATASET_DISPLAY.get(d, d) for d in pivot.index]
    pivot.columns = [METRIC_DISPLAY.get(m, m) for m in pivot.columns]

    fig, ax = plt.subplots(figsize=FIG_SIZE_HEATMAP)
    # Use RdYlGn colormap: red (bad) -> yellow -> green (good)
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                # Text color: white for extreme values, black for middle
                text_color = "white" if val < 0.3 or val > 0.8 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=NATURE_TICK_SIZE, color=text_color)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Self vs Control Discrimination\n(self-distance < control-distance, higher is better)",
                 fontsize=NATURE_TITLE_SIZE, pad=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.tick_params(labelsize=NATURE_TICK_SIZE, length=2, width=0.4)
    cbar.set_label("Discrimination Ratio", fontsize=NATURE_TICK_SIZE)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=NATURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_self_control_pds_heatmap(
    sc_results: List[SelfControlResult],
    output_path: Path,
    metrics_order: List[str],
) -> None:
    """Heatmap of self-vs-control discrimination accuracy for PDS metrics.

    Similar to plot_self_control_heatmap but specifically for PDS metrics (pds, wpds).
    Shows the ratio of perturbations where self PDS > control PDS.
    Also displays the mean self and control PDS values as annotations.

    Args:
        sc_results: List of :class:`SelfControlResult` objects.
        output_path: Where to save the figure.
        metrics_order: Ordered list of raw metric names.
    """
    _apply_nature_rc()

    # Filter to only results with PDS scores
    pds_results = [r for r in sc_results if r.pds_self is not None and r.pds_control is not None]
    if not pds_results:
        print("  No PDS results to plot, skipping PDS heatmap")
        return

    # Build DataFrame with accuracy and PDS values
    rows: List[Dict[str, Any]] = []
    for r in pds_results:
        rows.append({
            "dataset_raw": r.dataset,
            "Dataset": DATASET_DISPLAY.get(r.dataset, r.dataset),
            "metric_raw": r.metric,
            "Metric": METRIC_DISPLAY.get(r.metric, r.metric),
            "accuracy": r.accuracy,  # Ratio where self PDS > control PDS
            "pds_self": r.pds_self,
            "pds_control": r.pds_control,
        })
    df = pd.DataFrame(rows)

    # Aggregate across trials
    summary = df.groupby(["dataset_raw", "Dataset", "metric_raw", "Metric"]).agg(
        mean_accuracy=("accuracy", "mean"),
        mean_pds_self=("pds_self", "mean"),
        mean_pds_control=("pds_control", "mean"),
    ).reset_index()

    # Filter metrics_order to only PDS metrics that exist in results
    pds_metrics = {"pds", "wpds", "pds_pearson_deltapert"}
    filtered_metrics_order = [m for m in metrics_order if m in pds_metrics and m in summary["metric_raw"].unique()]

    if not filtered_metrics_order:
        print("  No PDS metrics in results, skipping PDS heatmap")
        return

    # Sort metrics by global mean accuracy (ascending left to right)
    metric_means = (
        summary.groupby("metric_raw")["mean_accuracy"]
        .mean()
        .reindex(filtered_metrics_order)
        .sort_values()
    )
    sorted_metrics = [m for m in metric_means.index if m in summary["metric_raw"].unique()]

    # Sort datasets by global mean accuracy (ascending top to bottom)
    ds_means = summary.groupby("dataset_raw")["mean_accuracy"].mean().sort_values()
    sorted_datasets = ds_means.index.tolist()

    pivot = summary.pivot_table(
        index="dataset_raw", columns="metric_raw", values="mean_accuracy"
    )
    pivot = pivot.reindex(index=sorted_datasets, columns=sorted_metrics)
    pivot.index = [DATASET_DISPLAY.get(d, d) for d in pivot.index]
    pivot.columns = [METRIC_DISPLAY.get(m, m) for m in pivot.columns]

    fig, ax = plt.subplots(figsize=FIG_SIZE_HEATMAP)
    # Use RdYlGn colormap: red (bad) -> yellow -> green (good), range [0, 1]
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    # Annotate cells with accuracy and PDS values
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                # Text color: white for extreme values, black for middle
                text_color = "white" if val < 0.3 or val > 0.8 else "black"
                # Get corresponding PDS values for annotation
                ds_raw = sorted_datasets[i]
                met_raw = sorted_metrics[j]
                row_data = summary[(summary["dataset_raw"] == ds_raw) & (summary["metric_raw"] == met_raw)]
                if not row_data.empty:
                    pds_self = row_data["mean_pds_self"].values[0]
                    pds_ctrl = row_data["mean_pds_control"].values[0]
                    # Show accuracy with self/control PDS below
                    ax.text(j, i, f"{val:.2f}\nS:{pds_self:.2f}\nC:{pds_ctrl:.2f}",
                            ha="center", va="center", fontsize=NATURE_TICK_SIZE - 1, color=text_color)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Self vs Control Discrimination (PDS metrics)\n(ratio self>control, S=self PDS, C=control PDS)",
                 fontsize=NATURE_TITLE_SIZE, pad=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.tick_params(labelsize=NATURE_TICK_SIZE, length=2, width=0.4)
    cbar.set_label("Discrimination Ratio", fontsize=NATURE_TICK_SIZE)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=NATURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def _save_results_csv(results: List[ComparisonResult], path: Path) -> None:
    """Write all per-trial aggregated results to CSV.

    Args:
        results: List of :class:`ComparisonResult` objects.
        path: Output CSV file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "metric", "trial", "dprime", "auc", "pds", "within", "between"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "dataset": r.dataset,
                "metric": r.metric,
                "trial": r.trial,
                "dprime": f"{r.dprime:.6f}",
                "auc": f"{r.auc:.6f}",
                "pds": f"{r.pds:.6f}",
                "within": f"{r.within:.6f}",
                "between": f"{r.between:.6f}",
            })
    print(f"  Saved {path}")


def _save_pert_results_csv(pert_results: List[PertResult], path: Path) -> None:
    """Write per-perturbation d', AUC, and PDS to CSV.

    Each row is one (dataset, metric, trial, perturbation) combination,
    enabling downstream distribution plots of Cohen's d' across perturbations.

    Args:
        pert_results: List of :class:`PertResult` objects.
        path: Output CSV file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "metric", "trial", "perturbation", "dprime", "auc", "pds"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in pert_results:
            writer.writerow({
                "dataset": r.dataset,
                "metric": r.metric,
                "trial": r.trial,
                "perturbation": r.perturbation,
                "dprime": f"{r.dprime:.6f}",
                "auc": f"{r.auc:.6f}",
                "pds": f"{r.pds:.6f}",
            })
    print(f"  Saved {path}")


def _save_self_control_csv(sc_results: List[SelfControlResult], path: Path) -> None:
    """Write self-vs-control summary to CSV.

    Args:
        sc_results: List of :class:`SelfControlResult` objects.
        path: Output CSV file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "metric",
        "mean_accuracy",
        "std_accuracy_across_trials",
        "n_trials",
        "n_total",
        "mean_n_correct",
        "mean_pds_self",
        "std_pds_self",
        "mean_pds_control",
        "std_pds_control",
    ]
    grouped: Dict[Tuple[str, str], List[SelfControlResult]] = {}
    for row in sc_results:
        key = (row.dataset, row.metric)
        grouped.setdefault(key, []).append(row)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for (dataset, metric), rows in sorted(grouped.items()):
            accuracies = np.asarray([r.accuracy for r in rows], dtype=float)
            n_correct_vals = np.asarray([r.n_correct for r in rows], dtype=float)
            n_trials = int(len(rows))
            mean_accuracy = float(np.mean(accuracies))
            std_accuracy = float(np.std(accuracies, ddof=1)) if n_trials > 1 else 0.0
            mean_n_correct = float(np.mean(n_correct_vals))
            n_total = int(rows[0].n_total)

            # Compute PDS statistics for PDS metrics
            pds_self_vals = np.asarray([r.pds_self for r in rows if r.pds_self is not None], dtype=float)
            pds_control_vals = np.asarray([r.pds_control for r in rows if r.pds_control is not None], dtype=float)

            row_data = {
                "dataset": dataset,
                "metric": metric,
                "mean_accuracy": f"{mean_accuracy:.4f}",
                "std_accuracy_across_trials": f"{std_accuracy:.4f}",
                "n_trials": n_trials,
                "n_total": n_total,
                "mean_n_correct": f"{mean_n_correct:.2f}",
                "mean_pds_self": f"{float(np.mean(pds_self_vals)):.4f}" if pds_self_vals.size > 0 else "",
                "std_pds_self": f"{float(np.std(pds_self_vals, ddof=1)):.4f}" if pds_self_vals.size > 1 else "",
                "mean_pds_control": f"{float(np.mean(pds_control_vals)):.4f}" if pds_control_vals.size > 0 else "",
                "std_pds_control": f"{float(np.std(pds_control_vals, ddof=1)):.4f}" if pds_control_vals.size > 1 else "",
            }
            writer.writerow(row_data)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Cross-dataset metric comparison via Cohen's d' and AUC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--metrics",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=DEFAULT_METRICS,
        help="Comma-separated list of distance metrics.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Number of independent random trials for error bars.",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=20,
        help="Number of half-cell resamples per trial.",
    )
    parser.add_argument(
        "--max-perturbations",
        type=int,
        default=None,
        help="Keep top-N perturbations by cell count. Default: all perturbations.",
    )
    parser.add_argument(
        "--dataset-max-perturbations-overrides",
        type=str,
        default="",
        help=(
            "Optional per-dataset max perturbation overrides, e.g. "
            "'nadig25hepg2:300,replogle22k562:300'."
        ),
    )
    parser.add_argument(
        "--random-sample-datasets",
        nargs="*",
        default=[],
        help=(
            "Datasets that should use random sampling when max perturbation limits "
            "apply (global or override). Others use top-by-cell-count."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    parser.add_argument(
        "--base-weight-exponent",
        type=float,
        default=2.0,
        help="Exponent for DEG weight transform.",
    )
    parser.add_argument(
        "--weight-scheme",
        type=str,
        default="normal",
        choices=["normal", "rank"],
        help="Weight transform scheme.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analyses/perturbation_discrimination/results/metric_comparison",
        help="Output directory for plots and CSV.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Output image formats.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of parallel processes for dataset-level parallelism (1 = sequential).",
    )
    parser.add_argument(
        "--metric-jobs",
        type=int,
        default=1,
        help="Number of parallel processes for metric-level parallelism within each dataset. 1 = process metrics sequentially.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["standard", "self_control"],
        help=(
            "Analysis mode: 'standard' computes d', AUC, PDS; "
            "'self_control' computes accuracy of self (technical duplicate) vs control discrimination."
        ),
    )
    return parser.parse_args()


def _compute_self_control_worker(
    item: Tuple[str, Dict[str, Any]],
) -> Tuple[str, List[SelfControlResult]]:
    """Worker for self-control mode (must be picklable for spawn)."""
    dataset_name, kwargs = item
    try:
        results = _compute_self_control_dataset(**kwargs)
        return (dataset_name, results)
    except Exception as exc:
        print(f"  ERROR on {dataset_name}: {exc}")
        return (dataset_name, [])


def main() -> None:
    """Run cross-dataset metric comparison."""
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_jobs = max(1, int(args.n_jobs))
    metric_jobs = max(1, int(args.metric_jobs))
    dataset_max_overrides = _parse_dataset_max_perturbations_overrides(
        str(args.dataset_max_perturbations_overrides)
    )
    base_kwargs: Dict[str, Any] = {
        "metrics": args.metrics,
        "n_trials": args.n_trials,
        "n_resamples": args.n_resamples,
        "max_perturbations": args.max_perturbations,
        "seed": args.seed,
        "base_weight_exponent": args.base_weight_exponent,
        "weight_scheme": args.weight_scheme,
        "metric_jobs": metric_jobs,
        "dataset_max_perturbations_overrides": dataset_max_overrides,
        "random_sample_datasets": list(args.random_sample_datasets),
    }

    if args.mode == "self_control":
        # Self-control mode: technical duplicate vs control accuracy
        all_sc_results: List[SelfControlResult] = []

        if n_jobs > 1:
            tasks = [
                (ds, {**base_kwargs, "dataset_name": ds})
                for ds in args.datasets
            ]
            try:
                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=min(n_jobs, len(args.datasets))) as pool:
                    print(f"Parallel: {n_jobs} workers, {len(tasks)} datasets (self-control mode)")
                    for dataset_name, results in pool.imap_unordered(
                        _compute_self_control_worker, tasks
                    ):
                        if not results:
                            print(f"  WARNING: {dataset_name} returned no results")
                        else:
                            all_sc_results.extend(results)
                            print(f"  Completed {dataset_name} ({len(results)} self-control results)")
            except (PermissionError, OSError) as exc:
                print(f"WARNING: Multiprocessing failed ({exc}), falling back to sequential")
                n_jobs = 1

        if n_jobs <= 1:
            for dataset in args.datasets:
                print(f"\n=== Processing {dataset} (self-control mode) ===")
                try:
                    results = _compute_self_control_dataset(
                        dataset_name=dataset,
                        **base_kwargs,
                    )
                    all_sc_results.extend(results)
                except Exception as exc:
                    print(f"  ERROR on {dataset}: {exc}")
                    import traceback
                    traceback.print_exc()

        if not all_sc_results:
            print("No self-control results collected. Exiting.")
            return

        # Save self-control CSV
        _save_self_control_csv(all_sc_results, out_dir / "self_control_results.csv")

        # Plot self-control heatmap
        metrics_order = [m for m in args.metrics if m in {r.metric for r in all_sc_results}]
        for fmt in args.formats:
            plot_self_control_heatmap(
                all_sc_results,
                output_path=out_dir / f"self_control_heatmap.{fmt}",
                metrics_order=metrics_order,
            )
            # Also plot PDS-specific heatmap for PDS metrics
            plot_self_control_pds_heatmap(
                all_sc_results,
                output_path=out_dir / f"self_control_pds_diff.{fmt}",
                metrics_order=metrics_order,
            )

        print(f"\nAll self-control outputs saved to {out_dir}")
        return

    # Standard mode: d', AUC, PDS
    all_results: List[ComparisonResult] = []
    all_pert_results: List[PertResult] = []

    standard_kwargs = dict(base_kwargs)

    if n_jobs > 1:
        tasks = [
            (ds, {**standard_kwargs, "dataset_name": ds})
            for ds in args.datasets
        ]
        try:
            # Use spawn to avoid fork PermissionError on restricted systems
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=min(n_jobs, len(args.datasets))) as pool:
                print(f"Parallel: {n_jobs} workers, {len(tasks)} datasets")
                for dataset_name, results, pert_results in pool.imap_unordered(
                    _compute_dataset_worker, tasks
                ):
                    if not results:
                        print(f"  WARNING: {dataset_name} returned no results")
                    else:
                        all_results.extend(results)
                        all_pert_results.extend(pert_results)
                        print(f"  Completed {dataset_name} ({len(results)} results, "
                              f"{len(pert_results)} per-pert rows)")
        except (PermissionError, OSError) as exc:
            print(f"WARNING: Multiprocessing failed ({exc}), falling back to sequential")
            n_jobs = 1

    if n_jobs <= 1:
        for dataset in args.datasets:
            print(f"\n=== Processing {dataset} ===")
            try:
                results, pert_results = _compute_dataset(
                    dataset_name=dataset,
                    **standard_kwargs,
                )
                all_results.extend(results)
                all_pert_results.extend(pert_results)
            except Exception as exc:
                print(f"  ERROR on {dataset}: {exc}")
                import traceback
                traceback.print_exc()

    if not all_results:
        print("No results collected. Exiting.")
        return

    # Save aggregated per-trial CSV
    _save_results_csv(all_results, out_dir / "metric_comparison.csv")
    # Save per-perturbation CSV (one row per dataset × metric × trial × perturbation)
    _save_pert_results_csv(all_pert_results, out_dir / "metric_comparison_per_pert.csv")

    # Build summary
    summary = _build_summary_df(all_results)

    metrics_order = [m for m in args.metrics if m in summary["metric_raw"].unique()]

    # Bar plots
    for fmt in args.formats:
        plot_bar(
            summary,
            y_col="mean_dprime",
            y_err_col="std_dprime",
            y_label="Cohen's d' (+/- trial s.d.)",
            output_path=out_dir / f"bar_dprime.{fmt}",
            metrics_order=metrics_order,
        )
        plot_bar(
            summary,
            y_col="mean_auc",
            y_err_col="std_auc",
            y_label="AUC (+/- trial s.d.)",
            output_path=out_dir / f"bar_auc.{fmt}",
            metrics_order=metrics_order,
        )
        plot_bar(
            summary,
            y_col="mean_pds",
            y_err_col="std_pds",
            y_label="Mean PDS (+/- trial s.d.)",
            output_path=out_dir / f"bar_pds.{fmt}",
            metrics_order=metrics_order,
        )

    # Heatmaps
    for fmt in args.formats:
        plot_heatmap(
            summary,
            y_col="mean_dprime",
            title="Cohen's d' (all genes, default weights)",
            output_path=out_dir / f"heatmap_dprime.{fmt}",
            metrics_order=metrics_order,
        )
        plot_heatmap(
            summary,
            y_col="mean_auc",
            title="AUC (all genes, default weights)",
            output_path=out_dir / f"heatmap_auc.{fmt}",
            metrics_order=metrics_order,
            fmt=".3f",
        )
        plot_heatmap(
            summary,
            y_col="mean_pds",
            title="Mean PDS (all genes, default weights)",
            output_path=out_dir / f"heatmap_pds.{fmt}",
            metrics_order=metrics_order,
            fmt=".3f",
        )

    # Distribution plots (violin + strip per metric, one panel per metric)
    for fmt in args.formats:
        plot_distribution(
            all_pert_results,
            y_col="dprime",
            y_label="Cohen's d' (per perturbation)",
            output_path=out_dir / f"dist_dprime.{fmt}",
            metrics_order=metrics_order,
        )
        plot_distribution(
            all_pert_results,
            y_col="auc",
            y_label="AUC (per perturbation)",
            output_path=out_dir / f"dist_auc.{fmt}",
            metrics_order=metrics_order,
        )
        plot_distribution(
            all_pert_results,
            y_col="pds",
            y_label="PDS (per perturbation)",
            output_path=out_dir / f"dist_pds.{fmt}",
            metrics_order=metrics_order,
        )

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
