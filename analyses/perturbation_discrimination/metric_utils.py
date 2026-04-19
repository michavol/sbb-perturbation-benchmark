"""Run all-genes cell-retrieval meta-metric comparison across distance metrics.

This script evaluates how different distance metrics support perturbation-level
cell retrieval using normalized-rank scoring (same retrieval objective used in
signal-dilution analysis). It uses all genes from the selected dataset subset.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analyses.perturbation_discrimination.discrimination_comparison import (  # noqa: E402
    load_dataset_path,
    load_pds_full,
)

DistanceMetric = str


@dataclass(frozen=True)
class PreparedData:
    """Prepared matrix/labels for metric evaluation.

    Attributes:
        X: Dense expression matrix with shape ``(n_cells, n_genes)``.
        y: Perturbation labels with shape ``(n_cells,)``.
        genes: Ordered gene names aligned to ``X``.
        perturbations: Active perturbation labels.
        control_label: Inferred or provided control label.
    """

    X: np.ndarray
    y: np.ndarray
    genes: List[str]
    perturbations: List[str]
    control_label: str


def _max_n_above_threshold(sorted_scores_desc: np.ndarray, threshold: float) -> int:
    """Return max N where cumulative mean(top-N) >= threshold."""
    if sorted_scores_desc.size == 0:
        return 0
    cumulative = np.cumsum(sorted_scores_desc)
    means = cumulative / (np.arange(sorted_scores_desc.size, dtype=float) + 1.0)
    valid = np.where(means >= threshold)[0]
    if valid.size == 0:
        return 0
    return int(valid[-1] + 1)


@dataclass(frozen=True)
class MetricResult:
    """Per-metric retrieval summary."""

    metric: str
    mean_normalized_rank: float
    std_bootstrap: float
    effective_genes_median: float
    effective_genes_mean: float
    n_cells: int
    n_perturbations: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="All-genes retrieval meta-metric comparison for perturbation data."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Single dataset name (backward compatible).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional list of datasets to process in one run.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=(
            "mse,wmse,pds_wmse,r2_delta_pert,r2_delta_ctrl,weighted_r2_delta_pert,"
            "weighted_r2_delta_ctrl,pds_weighted_r2_delta_pert,pds_weighted_r2_delta_ctrl,"
            "delta_pearson,deg_weighted_delta_pearson,"
            "pds_weighted_delta_pearson"
        ),
        help="Comma-separated metric list.",
    )
    parser.add_argument(
        "--pds-result-file",
        type=str,
        default="pds_full.h5ad",
        help="PDS results filename in results/<dataset>/, used for PDS-weighted metrics.",
    )
    parser.add_argument(
        "--pds-metric",
        type=str,
        default="Energy_Distance",
        help="PDS metric used for PDS-weighted metrics.",
    )
    parser.add_argument(
        "--pds-weight-transform",
        type=str,
        choices=("rank01", "clip01", "shift01", "abs_clip01"),
        default="rank01",
        help=(
            "Transform applied to raw PDS scores before base exponent and calibration. "
            "'rank01' uses per-perturbation percentile ranks (most robust default)."
        ),
    )
    parser.add_argument(
        "--pds-threshold-filter",
        type=float,
        default=0.8,
        help="PDS threshold used for threshold-filtered metric variants.",
    )
    parser.add_argument(
        "--base-weight-exponent",
        type=float,
        default=2.0,
        help=(
            "Base exponent for transforming DEG/PDS weights after the selected "
            "weight transform (default 2.0 reproduces previous behavior)."
        ),
    )
    parser.add_argument(
        "--deg-weight-transform",
        type=str,
        choices=("minmax", "rank01"),
        default="minmax",
        help=(
            "Transform applied to per-perturbation DEG scores before base exponent. "
            "'minmax' reproduces the historical behavior."
        ),
    )
    parser.add_argument(
        "--target-effective-genes",
        type=float,
        default=None,
        help=(
            "Optional target effective gene number for weighted metrics. "
            "If set, DEG and PDS weights are power-calibrated to match this target."
        ),
    )
    parser.add_argument(
        "--target-effective-genes-mode",
        type=str,
        choices=("mean", "per_pert"),
        default="mean",
        help=(
            "How to match target effective genes for DEG-weighted metrics: "
            "'mean' matches average across perturbations; "
            "'per_pert' matches each perturbation separately."
        ),
    )
    parser.add_argument(
        "--pds-target-effective-genes-mode",
        type=str,
        choices=("mean", "per_pert"),
        default="per_pert",
        help=(
            "How to match target effective genes for PDS-weighted metrics: "
            "'mean' matches average across perturbations; "
            "'per_pert' matches each perturbation separately (default)."
        ),
    )
    parser.add_argument(
        "--cells-per-perturbation",
        type=int,
        default=5,
        help="Optional fixed number of cells sampled per perturbation.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Optional max number of cells sampled globally.",
    )
    parser.add_argument(
        "--max-perturbations",
        type=int,
        default=None,
        help="Optional max number of perturbations sampled randomly.",
    )
    parser.add_argument(
        "--control-label",
        type=str,
        default=None,
        help="Optional explicit control label (used for delta_pearson baseline).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=3,
        help="Number of bootstrap resamples for score std estimation (0 disables).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--run-distance-heatmaps",
        action="store_true",
        help="If set, write pairwise distance heatmaps for the selected metrics.",
    )
    parser.add_argument(
        "--heatmap-max-perturbations",
        type=int,
        default=10,
        help="Max perturbations shown in heatmaps.",
    )
    parser.add_argument(
        "--heatmap-max-cells",
        type=int,
        default=200,
        help="Max cells shown in heatmaps.",
    )
    parser.add_argument(
        "--heatmap-quantile-bins",
        type=int,
        default=100,
        help="Quantile bins for heatmap visualization (e.g., 100 for percentiles).",
    )
    parser.add_argument(
        "--heatmap-quantile-scope",
        type=str,
        choices=("global", "per_perturbation"),
        default="per_perturbation",
        help=(
            "How to compute quantile bins for heatmaps. "
            "'per_perturbation' computes bins independently per perturbation row block."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory; defaults to dataset-specific results folder.",
    )
    return parser.parse_args()


def _resolve_perturbation_key(adata: sc.AnnData) -> str:
    """Resolve perturbation key from AnnData observations."""
    for key in ("perturbation", "condition"):
        if key in adata.obs.columns:
            return key
    raise KeyError("Could not resolve perturbation key from adata.obs.")


def _infer_control_label(labels: np.ndarray) -> str:
    """Infer a control label from perturbation names."""
    y = np.asarray(labels, dtype=str)
    y_lower = np.char.lower(y)
    is_control = np.logical_or(np.char.find(y_lower, "control") >= 0, np.char.find(y_lower, "ctrl") >= 0)
    if np.any(is_control):
        values, counts = np.unique(y[is_control], return_counts=True)
        return str(values[int(np.argmax(counts))])
    values, counts = np.unique(y, return_counts=True)
    return str(values[int(np.argmax(counts))])


def _sample_cells_per_perturbation(labels: np.ndarray, n_cells: int, seed: int) -> np.ndarray:
    """Sample exactly ``n_cells`` cells per perturbation."""
    if n_cells <= 0:
        raise ValueError("cells-per-perturbation must be positive.")
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for pert in sorted(np.unique(labels).tolist()):
        idx = np.where(labels == pert)[0]
        if idx.size < n_cells:
            raise ValueError(
                f"Perturbation '{pert}' has only {idx.size} cells; requires at least {n_cells}."
            )
        chosen = rng.choice(idx, size=n_cells, replace=False)
        selected.extend(chosen.tolist())
    return np.asarray(sorted(selected), dtype=int)


def _select_perturbations_by_pds(
    unique_perts: np.ndarray,
    dataset_name: str,
    args: argparse.Namespace,
    n: int,
    hardest: bool,
) -> np.ndarray:
    """Select perturbations ranked by their mean per-gene PDS discriminability.

    Loads the pds_full result file, computes for each perturbation its mean
    score across all genes (using ``args.pds_metric``), then returns the N
    perturbations with the lowest (hardest=True) or highest (hardest=False)
    mean score.  Perturbations not present in the PDS file are excluded.
    """
    pds_result_file = str(getattr(args, "pds_result_file", "pds_full.h5ad"))
    pds_metric = str(getattr(args, "pds_metric", "Energy_Distance"))
    result_path = (
        ROOT
        / "analyses"
        / "perturbation_discrimination"
        / "results"
        / dataset_name
        / pds_result_file
    )
    if not result_path.exists():
        raise FileNotFoundError(
            f"PDS result file not found for pds_bottom/pds_top sort: {result_path}"
        )
    pds_full = load_pds_full(result_path)
    pds_metrics = [str(m) for m in pds_full["metrics"]]
    if pds_metric not in pds_metrics:
        # Fall back to first available metric
        pds_metric = pds_metrics[0]
        print(f"  PDS metric not found; falling back to '{pds_metric}'.")
    metric_idx = pds_metrics.index(pds_metric)
    scores = np.asarray(pds_full["scores_mean"], dtype=float)[metric_idx]  # (n_perts, n_genes)
    pds_perts = [str(p) for p in pds_full["perturbations"]]
    pds_pert_to_idx = {p: i for i, p in enumerate(pds_perts)}

    # Compute mean per-gene PDS for each perturbation present in both sets
    candidates = []
    mean_pds = []
    for pert in unique_perts:
        idx = pds_pert_to_idx.get(str(pert))
        if idx is not None:
            candidates.append(pert)
            mean_pds.append(float(np.mean(scores[idx])))

    if not candidates:
        raise ValueError("No overlap between dataset perturbations and PDS result file.")

    candidates = np.asarray(candidates)
    mean_pds = np.asarray(mean_pds, dtype=float)

    if hardest:
        order = np.argsort(mean_pds)  # ascending: lowest PDS first
    else:
        order = np.argsort(mean_pds)[::-1]  # descending: highest PDS first

    chosen = candidates[order[:n]]
    print(
        f"  PDS-based sort ({'hardest' if hardest else 'easiest'}): "
        f"mean PDS range [{mean_pds[order[0]]:.3f}, {mean_pds[order[min(n-1, len(order)-1)]]:.3f}]"
    )
    return chosen


def _prepare_data(args: argparse.Namespace, dataset_name: str) -> PreparedData:
    """Load and subset dataset while retaining all genes."""
    dataset_path = load_dataset_path(ROOT, dataset_name)
    adata = sc.read_h5ad(dataset_path)
    pert_key = _resolve_perturbation_key(adata)
    y = adata.obs[pert_key].astype("string").fillna("NA").astype(str).to_numpy()

    if args.max_perturbations is not None:
        if args.max_perturbations <= 0:
            raise ValueError("--max-perturbations must be positive.")
        unique_perts, pert_counts = np.unique(y, return_counts=True)
        if unique_perts.size > args.max_perturbations:
            pert_sort = str(getattr(args, "perturbation_sort", "top"))
            if pert_sort in ("pds_bottom", "pds_top"):
                chosen = _select_perturbations_by_pds(
                    unique_perts=unique_perts,
                    dataset_name=dataset_name,
                    args=args,
                    n=args.max_perturbations,
                    hardest=(pert_sort == "pds_bottom"),
                )
                sort_label = "lowest PDS (hardest)" if pert_sort == "pds_bottom" else "highest PDS (easiest)"
            elif pert_sort == "bottom":
                sort_idx = np.argsort(pert_counts)[: args.max_perturbations]
                chosen = unique_perts[sort_idx]
                sort_label = "fewest cells"
            else:
                sort_idx = np.argsort(pert_counts)[::-1][: args.max_perturbations]
                chosen = unique_perts[sort_idx]
                sort_label = "most cells"
            mask = np.isin(y, chosen)
            adata = adata[mask, :].copy()
            y = adata.obs[pert_key].astype("string").fillna("NA").astype(str).to_numpy()
            print(f"Perturbation subsampling: kept {len(chosen)}/{len(unique_perts)} perturbations ({sort_label})")

    if args.cells_per_perturbation is not None:
        idx = _sample_cells_per_perturbation(
            labels=y,
            n_cells=args.cells_per_perturbation,
            seed=args.seed,
        )
        adata = adata[idx, :].copy()
        y = adata.obs[pert_key].astype("string").fillna("NA").astype(str).to_numpy()
    elif args.max_cells is not None and args.max_cells > 0 and adata.n_obs > args.max_cells:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(np.arange(adata.n_obs), size=args.max_cells, replace=False)
        adata = adata[np.sort(idx), :].copy()
        y = adata.obs[pert_key].astype("string").fillna("NA").astype(str).to_numpy()

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    genes = [str(gene) for gene in adata.var_names.tolist()]
    perturbations = sorted(np.unique(y).tolist())
    control_label = args.control_label if args.control_label is not None else _infer_control_label(y)
    if control_label not in set(y.tolist()):
        raise ValueError(f"Control label '{control_label}' not present in selected labels.")
    return PreparedData(
        X=X,
        y=y.astype(str),
        genes=genes,
        perturbations=perturbations,
        control_label=control_label,
    )


def _load_deg_weights(
    dataset_name: str,
    genes: Sequence[str],
    perturbations: Sequence[str],
    deg_weight_transform: str = "minmax",
    base_weight_exponent: float = 2.0,
) -> Dict[str, np.ndarray]:
    """Load per-pert DEG weights with CellSimBench transformation."""
    csv_path = (
        ROOT
        / "analyses"
        / "perturbation_discrimination"
        / "results"
        / dataset_name
        / "deg_scanpy"
        / "deg_control.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing DEG(control) file for WMSE weights: {csv_path}")

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

        # Skip perturbations with all-zero weights (no DEG data available)
        if np.max(w) <= 0.0:
            skipped_perts.append(pert_key)
            continue

        w_out = np.power(w, float(base_weight_exponent))
        w_out = _normalize_weight_vector_for_effective_genes(
            w_out,
            context=f"DEG weights for perturbation '{pert_key}'",
        )
        _validate_weight_vector_for_effective_genes(
            w_out,
            context=f"DEG weights for perturbation '{pert_key}'",
        )
        weights_per_pert[pert_key] = w_out

    if skipped_perts:
        print(
            f"WARNING: Skipped {len(skipped_perts)} perturbations with no DEG data: {skipped_perts[:5]}"
            f"{'...' if len(skipped_perts) > 5 else ''}"
        )

    return weights_per_pert


def _load_pds_weights(
    dataset_name: str,
    genes: Sequence[str],
    perturbations: Sequence[str],
    pds_result_file: str,
    pds_metric: str,
    pds_weight_transform: str = "rank01",
    base_weight_exponent: float = 2.0,
) -> Dict[str, np.ndarray]:
    """Load per-pert PDS weights aligned to the dataset genes.

    Args:
        dataset_name: Dataset identifier.
        genes: Ordered genes aligned to the expression matrix.
        perturbations: Active perturbations in the selected data.
        pds_result_file: PDS result filename under results/<dataset>/.
        pds_metric: Metric name from the PDS result file.

    Returns:
        Mapping perturbation -> transformed per-gene PDS weight vector.
    """
    result_path = (
        ROOT
        / "analyses"
        / "perturbation_discrimination"
        / "results"
        / dataset_name
        / pds_result_file
    )
    pds_full = load_pds_full(result_path)
    pds_genes = [str(item) for item in pds_full["genes"]]
    pds_perturbations = [str(item) for item in pds_full["perturbations"]]
    metric_names = [str(item) for item in list(pds_full["metrics"])]
    if pds_metric not in metric_names:
        raise ValueError(f"PDS metric '{pds_metric}' not in {metric_names}.")
    metric_idx = metric_names.index(pds_metric)
    score_tensor = np.asarray(pds_full["scores_mean"], dtype=float)
    metric_scores = score_tensor[metric_idx]  # (n_perts, n_genes_pds)

    gene_to_data_idx = {str(gene): idx for idx, gene in enumerate(genes)}
    pds_pert_to_idx = {str(pert): idx for idx, pert in enumerate(pds_perturbations)}
    n_genes = len(genes)
    weights_per_pert: Dict[str, np.ndarray] = {}

    for pert in perturbations:
        pert_key = str(pert)
        w = np.zeros(n_genes, dtype=float)
        p_idx = pds_pert_to_idx.get(pert_key)
        if p_idx is not None:
            raw_scores = np.asarray(metric_scores[p_idx], dtype=float)
            if pds_weight_transform == "rank01":
                order = np.argsort(raw_scores)
                transformed_scores = np.zeros_like(raw_scores, dtype=float)
                if raw_scores.size == 1:
                    transformed_scores[0] = 1.0
                elif raw_scores.size > 1:
                    transformed_scores[order] = np.linspace(0.0, 1.0, raw_scores.size, endpoint=True)
            elif pds_weight_transform == "clip01":
                transformed_scores = np.clip(raw_scores, 0.0, 1.0)
            elif pds_weight_transform == "shift01":
                transformed_scores = np.clip((raw_scores + 1.0) / 2.0, 0.0, 1.0)
            elif pds_weight_transform == "abs_clip01":
                transformed_scores = np.clip(np.abs(raw_scores), 0.0, 1.0)
            else:
                raise ValueError(f"Unsupported pds_weight_transform: {pds_weight_transform}")
            for p_gene_idx, gene in enumerate(pds_genes):
                data_idx = gene_to_data_idx.get(gene)
                if data_idx is not None:
                    w[data_idx] = transformed_scores[p_gene_idx]
        w_out = np.power(w, float(base_weight_exponent))
        w_out = _normalize_weight_vector_for_effective_genes(
            w_out,
            context=f"PDS weights for perturbation '{pert_key}'",
        )
        _validate_weight_vector_for_effective_genes(
            w_out,
            context=f"PDS weights for perturbation '{pert_key}'",
        )
        weights_per_pert[pert_key] = w_out
    return weights_per_pert


def _normalize_weight_vector_for_effective_genes(weights: np.ndarray, context: str) -> np.ndarray:
    """Normalize a weight vector so that its maximum value is 1."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError(f"{context}: weight vector must be 1D, got shape={w.shape}.")
    if w.size == 0:
        raise ValueError(f"{context}: weight vector is empty.")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{context}: weights contain non-finite values.")
    if np.any(w < 0):
        raise ValueError(f"{context}: negative weights are not allowed.")

    w_max = float(np.max(w))
    if w_max <= 0.0:
        # Some perturbations can legitimately have no DEG signal (all zeros) given the
        # current gene universe. In that case we fall back to a uniform, uninformative
        # weight vector rather than aborting the entire analysis.
        # This keeps those perturbations in the dataset while ensuring downstream
        # effective-gene logic still sees a well-defined [0, 1] weight profile.
        w = np.ones_like(w, dtype=float)
        return w
    return w / w_max


def _validate_weight_vector_for_effective_genes(weights: np.ndarray, context: str) -> None:
    """Validate assumptions required for meaningful effective-gene calculations."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError(f"{context}: weight vector must be 1D, got shape={w.shape}.")
    if w.size == 0:
        raise ValueError(f"{context}: weight vector is empty.")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{context}: weights contain non-finite values.")

    tol = 1e-8
    w_min = float(np.min(w))
    w_max = float(np.max(w))
    if w_min < -tol or w_max > 1.0 + tol:
        raise ValueError(
            f"{context}: weights must be in [0,1]. Observed min={w_min:.6g}, max={w_max:.6g}."
        )
    # Do not require that max is already 1.0 here; vectors can be re-scaled.


def _validate_weights_per_perturbation_for_effective_genes(
    weights_per_pert: Dict[str, np.ndarray],
    context: str,
) -> None:
    """Validate all perturbation weight vectors for effective-gene computations."""
    for pert, weights in weights_per_pert.items():
        _validate_weight_vector_for_effective_genes(
            np.asarray(weights, dtype=float),
            context=f"{context} perturbation '{pert}'",
        )


def _effective_genes_single(weights: np.ndarray) -> float:
    """Compute effective gene number for a single weight vector.

    Uses the same definition as `_effective_genes_for_metric`:
    N_eff = 1 / sum(p_i^2), where p is normalized nonnegative weights.
    """
    w = _normalize_weight_vector_for_effective_genes(
        np.asarray(weights, dtype=float),
        context="Effective-gene computation input",
    )
    _validate_weight_vector_for_effective_genes(w, context="Effective-gene computation input")
    w_sum = float(np.sum(w))
    if w_sum < 1e-12:
        raise ValueError("Effective-gene computation input: sum(weights) is zero after normalization.")
    p = w / w_sum
    denom = float(np.sum(np.square(p)))
    if denom < 1e-12:
        return float(w.shape[0])
    return float(1.0 / denom)


def _find_exponent_for_target_effective_genes(
    base_weights: np.ndarray,
    target_effective_genes: float,
    max_exponent: float = 128.0,
    tol: float = 0.5,
    max_iter: int = 120,
) -> float:
    """Find power exponent that matches target effective genes."""
    n_genes = int(base_weights.shape[0])
    target = float(target_effective_genes)
    if target >= float(n_genes):
        return 0.0
    if target <= 1.0:
        return float(max_exponent)
    lo, hi = 0.0, float(max_exponent)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        neff = _effective_genes_single(np.power(base_weights, mid))
        if abs(neff - target) <= tol:
            return float(mid)
        if neff > target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def _calibrate_weights_per_perturbation(
    weights_per_pert: Dict[str, np.ndarray],
    target_effective_genes: float,
    mode: str,
) -> Dict[str, np.ndarray]:
    """Calibrate weight sharpness to match a target effective-gene number."""
    _validate_weights_per_perturbation_for_effective_genes(
        weights_per_pert=weights_per_pert,
        context="Before calibration:",
    )
    if target_effective_genes is None:
        return {key: np.asarray(value, dtype=float) for key, value in weights_per_pert.items()}
    if mode not in {"mean", "per_pert"}:
        raise ValueError(f"Unsupported target-effective-genes mode: {mode}")
    if not weights_per_pert:
        return {}
    output: Dict[str, np.ndarray] = {}
    if mode == "per_pert":
        for pert, weights in weights_per_pert.items():
            w = np.asarray(weights, dtype=float)
            exponent = _find_exponent_for_target_effective_genes(
                base_weights=w,
                target_effective_genes=float(target_effective_genes),
            )
            output[pert] = _normalize_weight_vector_for_effective_genes(
                np.power(w, exponent),
                context=f"After calibration perturbation '{pert}'",
            )
        _validate_weights_per_perturbation_for_effective_genes(
            weights_per_pert=output,
            context="After calibration:",
        )
        return output
    # mode == "mean": one shared exponent across perturbations
    all_weights = np.stack([np.asarray(weights, dtype=float) for weights in weights_per_pert.values()], axis=0)
    n_genes = int(all_weights.shape[1])
    target = float(target_effective_genes)
    if target >= float(n_genes):
        exponent = 0.0
    elif target <= 1.0:
        exponent = 128.0
    else:
        lo, hi = 0.0, 128.0
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            neff_values = np.asarray(
                [_effective_genes_single(np.power(row, mid)) for row in all_weights],
                dtype=float,
            )
            mean_neff = float(np.mean(neff_values))
            if abs(mean_neff - target) <= 0.5:
                lo = hi = mid
                break
            if mean_neff > target:
                lo = mid
            else:
                hi = mid
        exponent = float(0.5 * (lo + hi))
    for pert, weights in weights_per_pert.items():
        output[pert] = _normalize_weight_vector_for_effective_genes(
            np.power(np.asarray(weights, dtype=float), exponent),
            context=f"After calibration perturbation '{pert}'",
        )
    _validate_weights_per_perturbation_for_effective_genes(
        weights_per_pert=output,
        context="After calibration:",
    )
    return output


def _resolve_pds_threshold_gene_indices(
    dataset_name: str,
    genes: Sequence[str],
    perturbations: Sequence[str],
    pds_result_file: str,
    pds_metric: str,
    threshold: float,
) -> np.ndarray:
    """Resolve global PDS threshold-filtered gene indices.

    Uses active perturbations only, then selects top-N genes where cumulative
    mean PDS over ranked genes stays above ``threshold``.
    """
    result_path = (
        ROOT
        / "analyses"
        / "perturbation_discrimination"
        / "results"
        / dataset_name
        / pds_result_file
    )
    pds_full = load_pds_full(result_path)
    pds_genes = [str(item) for item in pds_full["genes"]]
    pds_perturbations = [str(item) for item in pds_full["perturbations"]]
    metric_names = [str(item) for item in list(pds_full["metrics"])]
    if pds_metric not in metric_names:
        raise ValueError(f"PDS metric '{pds_metric}' not in {metric_names}.")
    metric_idx = metric_names.index(pds_metric)
    score_tensor = np.asarray(pds_full["scores_mean"], dtype=float)
    metric_scores = score_tensor[metric_idx]  # (n_perts, n_genes_pds)

    pert_to_idx = {pert: idx for idx, pert in enumerate(pds_perturbations)}
    active_idx = [pert_to_idx[pert] for pert in perturbations if pert in pert_to_idx]
    if not active_idx:
        raise ValueError("No overlap between selected perturbations and PDS perturbations.")
    active_scores = metric_scores[np.asarray(active_idx, dtype=int)]
    mean_scores = np.mean(active_scores, axis=0)

    gene_to_data_idx = {str(gene): idx for idx, gene in enumerate(genes)}
    pds_gene_data_idx: List[int] = []
    pds_gene_scores: List[float] = []
    for p_gene_idx, gene in enumerate(pds_genes):
        d_idx = gene_to_data_idx.get(gene)
        if d_idx is None:
            continue
        pds_gene_data_idx.append(d_idx)
        pds_gene_scores.append(float(mean_scores[p_gene_idx]))
    if not pds_gene_data_idx:
        raise ValueError("No overlapping genes between data matrix and PDS results.")

    aligned_idx = np.asarray(pds_gene_data_idx, dtype=int)
    aligned_scores = np.asarray(pds_gene_scores, dtype=float)
    order_local = np.argsort(aligned_scores)[::-1]
    sorted_scores = aligned_scores[order_local]
    n_keep = _max_n_above_threshold(sorted_scores_desc=sorted_scores, threshold=threshold)
    if n_keep <= 0:
        return np.asarray([], dtype=int)
    selected = aligned_idx[order_local[:n_keep]]
    return np.asarray(np.unique(selected), dtype=int)


def _rank_to_score(rank: int, n_labels: int) -> float:
    """Convert rank to normalized score in [-1, 1]."""
    if n_labels <= 1:
        return 1.0
    return float(1.0 - 2.0 * (rank / float(n_labels - 1)))


def _normalized_rank_per_sample(distances: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute normalized-rank score for each query sample."""
    y = np.asarray(labels, dtype=str)
    classes = np.unique(y)
    class_to_idx: Dict[str, np.ndarray] = {cls: np.where(y == cls)[0] for cls in classes}
    scores = np.full(y.shape[0], np.nan, dtype=float)
    for i in range(y.shape[0]):
        true_label = str(y[i])
        class_means: List[float] = []
        valid = True
        for cls in classes:
            idx = class_to_idx[str(cls)]
            if str(cls) == true_label:
                idx = idx[idx != i]
                if idx.size == 0:
                    valid = False
                    break
            class_means.append(float(np.mean(distances[i, idx])))
        if not valid:
            continue
        class_means_arr = np.asarray(class_means, dtype=float)
        order = np.argsort(class_means_arr)
        true_cls_idx = int(np.where(classes == true_label)[0][0])
        rank = int(np.where(order == true_cls_idx)[0][0])
        scores[i] = _rank_to_score(rank=rank, n_labels=int(classes.size))
    return scores


def _bootstrap_std(scores: np.ndarray, n_bootstrap: int, seed: int) -> float:
    """Estimate std of mean score via bootstrap over query samples."""
    valid = scores[np.isfinite(scores)]
    if valid.size == 0 or n_bootstrap <= 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, valid.size, size=valid.size)
        means[i] = float(np.mean(valid[idx]))
    return float(np.std(means, ddof=1)) if means.size > 1 else float("nan")


def _pairwise_mse_matrix(X: np.ndarray) -> np.ndarray:
    """Compute symmetric MSE distance matrix."""
    condensed = pdist(X, metric="sqeuclidean") / float(max(X.shape[1], 1))
    condensed = np.nan_to_num(condensed, nan=0.0, posinf=0.0, neginf=0.0)
    dist = squareform(condensed)
    np.fill_diagonal(dist, 0.0)
    return dist


def _pairwise_delta_pearson_matrix(X: np.ndarray, control_mean: np.ndarray) -> np.ndarray:
    """Compute correlation distance on control-centered expression."""
    centered = X - control_mean[None, :]
    condensed = pdist(centered, metric="correlation")
    condensed = np.nan_to_num(condensed, nan=1.0, posinf=1.0, neginf=1.0)
    dist = squareform(condensed)
    np.fill_diagonal(dist, 0.0)
    return dist


def _weighted_sqeuclidean_rows(X: np.ndarray, query_idx: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted squared Euclidean distances from query rows to all rows."""
    w = np.asarray(weights, dtype=float)
    w_sum = float(np.sum(w))
    if w_sum < 1e-12:
        w = np.full(X.shape[1], 1.0 / float(max(X.shape[1], 1)))
    else:
        w = w / w_sum
    sqrt_w = np.sqrt(w)
    X_scaled = X * sqrt_w[None, :]
    Q = X_scaled[query_idx]
    all_norm = np.sum(np.square(X_scaled), axis=1)
    q_norm = np.sum(np.square(Q), axis=1)
    dist = q_norm[:, None] + all_norm[None, :] - 2.0 * (Q @ X_scaled.T)
    return np.maximum(dist, 0.0)


def _subset_weights_per_pert(
    weights_per_pert: Dict[str, np.ndarray],
    gene_indices: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    """Subset per-perturbation weight vectors to selected genes if requested."""
    if gene_indices is None:
        return {pert: np.asarray(weights, dtype=float) for pert, weights in weights_per_pert.items()}
    return {
        pert: np.asarray(weights, dtype=float)[gene_indices]
        for pert, weights in weights_per_pert.items()
    }


def _effective_genes_for_metric(
    labels: np.ndarray,
    n_features: int,
    weights_per_pert: Optional[Dict[str, np.ndarray]],
) -> Tuple[float, float]:
    """Compute effective gene numbers (median/mean) for one metric.

    For unweighted metrics, the effective number is exactly ``n_features``.
    For weighted metrics, per-query effective genes are computed as
    ``N_eff = 1 / sum(p_i^2)`` where ``p`` are normalized nonnegative weights.
    """
    if weights_per_pert is None:
        return float(n_features), float(n_features)
    y = np.asarray(labels, dtype=str)
    per_pert_neff: Dict[str, float] = {}
    for pert, weights in weights_per_pert.items():
        w = np.asarray(weights, dtype=float)
        w_sum = float(np.sum(w))
        if w_sum < 1e-12:
            p = np.full(w.shape[0], 1.0 / float(max(w.shape[0], 1)))
        else:
            p = w / w_sum
        denom = float(np.sum(np.square(p)))
        if denom < 1e-12:
            per_pert_neff[str(pert)] = float(w.shape[0])
        else:
            per_pert_neff[str(pert)] = float(1.0 / denom)
    query_neff = np.asarray([per_pert_neff.get(str(label), float(n_features)) for label in y], dtype=float)
    return float(np.median(query_neff)), float(np.mean(query_neff))


def _weighted_correlation_rows(X: np.ndarray, query_idx: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted correlation distance from query rows to all rows."""
    w = np.asarray(weights, dtype=float)
    w_sum = float(np.sum(w))
    if w_sum < 1e-12:
        w = np.full(X.shape[1], 1.0 / float(max(X.shape[1], 1)))
    else:
        w = w / w_sum
    X_weighted = X * np.sqrt(w)[None, :]
    Q = X_weighted[query_idx]

    Q_center = Q - np.mean(Q, axis=1, keepdims=True)
    X_center = X_weighted - np.mean(X_weighted, axis=1, keepdims=True)
    numerator = Q_center @ X_center.T
    q_norm = np.linalg.norm(Q_center, axis=1)
    x_norm = np.linalg.norm(X_center, axis=1)
    denom = q_norm[:, None] * x_norm[None, :]
    corr = np.divide(
        numerator,
        denom,
        out=np.zeros_like(numerator),
        where=denom > 1e-12,
    )
    corr = np.clip(corr, -1.0, 1.0)
    return np.nan_to_num(1.0 - corr, nan=1.0, posinf=1.0, neginf=1.0)


def _pairwise_wmse_query_weighted(
    X: np.ndarray,
    labels: np.ndarray,
    weights_per_pert: Dict[str, np.ndarray],
) -> np.ndarray:
    """Compute query-perturbation-weighted WMSE distance matrix."""
    y = np.asarray(labels, dtype=str)
    dist = np.zeros((X.shape[0], X.shape[0]), dtype=float)
    for pert in np.unique(y):
        q_idx = np.where(y == pert)[0]
        weights = weights_per_pert.get(str(pert))
        if weights is None:
            weights = np.zeros(X.shape[1], dtype=float)
        dist[q_idx, :] = _weighted_sqeuclidean_rows(X=X, query_idx=q_idx, weights=weights)
    np.fill_diagonal(dist, 0.0)
    return dist


def _r2_distance_from_sse(sse: np.ndarray, sst_per_truth: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert SSE and per-truth SST to distance = 1 - R²."""
    dist = np.empty_like(sse, dtype=float)
    non_constant = sst_per_truth > eps
    if np.any(non_constant):
        dist[:, non_constant] = sse[:, non_constant] / sst_per_truth[None, non_constant]
    if np.any(~non_constant):
        zero_err = sse[:, ~non_constant] <= eps
        dist[:, ~non_constant] = np.where(zero_err, 0.0, 1.0)
    return np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=1.0)


def _pairwise_r2_delta_pert(X: np.ndarray) -> np.ndarray:
    """Compute query-to-truth distance = 1 - R² on dataset-mean deltas."""
    baseline = np.mean(X, axis=0)
    deltas = X - baseline[None, :]
    norms = np.sum(np.square(deltas), axis=1)
    sse = norms[:, None] + norms[None, :] - 2.0 * (deltas @ deltas.T)
    sse = np.maximum(sse, 0.0)
    truth_means = np.mean(deltas, axis=1)
    centered_truth = deltas - truth_means[:, None]
    sst = np.sum(np.square(centered_truth), axis=1)
    dist = _r2_distance_from_sse(sse=sse, sst_per_truth=sst)
    np.fill_diagonal(dist, 0.0)
    return dist


def _pairwise_r2_delta_ctrl(X: np.ndarray, control_mean: np.ndarray) -> np.ndarray:
    """Compute query-to-truth distance = 1 - R² on control-mean deltas."""
    deltas = X - control_mean[None, :]
    norms = np.sum(np.square(deltas), axis=1)
    sse = norms[:, None] + norms[None, :] - 2.0 * (deltas @ deltas.T)
    sse = np.maximum(sse, 0.0)
    truth_means = np.mean(deltas, axis=1)
    centered_truth = deltas - truth_means[:, None]
    sst = np.sum(np.square(centered_truth), axis=1)
    dist = _r2_distance_from_sse(sse=sse, sst_per_truth=sst)
    np.fill_diagonal(dist, 0.0)
    return dist


def _weighted_sst_per_truth(deltas: np.ndarray, weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute weighted SST per truth vector for weighted R²."""
    w = np.asarray(weights, dtype=float)
    w_sum = float(np.sum(w))
    if w_sum < eps:
        w = np.full(deltas.shape[1], 1.0 / float(max(deltas.shape[1], 1)))
    else:
        w = w / w_sum
    truth_means = deltas @ w
    centered = deltas - truth_means[:, None]
    return np.sum(np.square(centered) * w[None, :], axis=1)


def _pairwise_weighted_r2_delta_pert(
    X: np.ndarray,
    labels: np.ndarray,
    weights_per_pert: Dict[str, np.ndarray],
) -> np.ndarray:
    """Compute query-pert weighted distance = 1 - weighted R² on dataset deltas."""
    baseline = np.mean(X, axis=0)
    deltas = X - baseline[None, :]
    y = np.asarray(labels, dtype=str)
    dist = np.zeros((X.shape[0], X.shape[0]), dtype=float)
    for pert in np.unique(y):
        q_idx = np.where(y == pert)[0]
        weights = weights_per_pert.get(str(pert))
        if weights is None:
            weights = np.zeros(X.shape[1], dtype=float)
        sse_rows = _weighted_sqeuclidean_rows(X=deltas, query_idx=q_idx, weights=weights)
        sst = _weighted_sst_per_truth(deltas=deltas, weights=weights)
        dist[q_idx, :] = _r2_distance_from_sse(sse=sse_rows, sst_per_truth=sst)
    np.fill_diagonal(dist, 0.0)
    return dist


def _pairwise_weighted_r2_delta_ctrl(
    X: np.ndarray,
    labels: np.ndarray,
    control_mean: np.ndarray,
    weights_per_pert: Dict[str, np.ndarray],
) -> np.ndarray:
    """Compute query-pert weighted distance = 1 - weighted R² on control deltas."""
    deltas = X - control_mean[None, :]
    y = np.asarray(labels, dtype=str)
    dist = np.zeros((X.shape[0], X.shape[0]), dtype=float)
    for pert in np.unique(y):
        q_idx = np.where(y == pert)[0]
        weights = weights_per_pert.get(str(pert))
        if weights is None:
            weights = np.zeros(X.shape[1], dtype=float)
        sse_rows = _weighted_sqeuclidean_rows(X=deltas, query_idx=q_idx, weights=weights)
        sst = _weighted_sst_per_truth(deltas=deltas, weights=weights)
        dist[q_idx, :] = _r2_distance_from_sse(sse=sse_rows, sst_per_truth=sst)
    np.fill_diagonal(dist, 0.0)
    return dist


def _pairwise_weighted_delta_pearson(
    X: np.ndarray,
    labels: np.ndarray,
    control_mean: np.ndarray,
    weights_per_pert: Dict[str, np.ndarray],
) -> np.ndarray:
    """Compute query-pert weighted Delta Pearson distance matrix."""
    y = np.asarray(labels, dtype=str)
    centered = X - control_mean[None, :]
    dist = np.zeros((X.shape[0], X.shape[0]), dtype=float)
    for pert in np.unique(y):
        q_idx = np.where(y == pert)[0]
        weights = weights_per_pert.get(str(pert))
        if weights is None:
            weights = np.zeros(X.shape[1], dtype=float)
        dist[q_idx, :] = _weighted_correlation_rows(
            X=centered,
            query_idx=q_idx,
            weights=weights,
        )
    np.fill_diagonal(dist, 0.0)
    return dist


def _compute_distance_matrix(
    metric: DistanceMetric,
    X: np.ndarray,
    y: np.ndarray,
    control_label: str,
    deg_weights_per_pert: Optional[Dict[str, np.ndarray]],
    pds_weights_per_pert: Optional[Dict[str, np.ndarray]],
    gene_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Dispatch pairwise distance matrix computation by metric name."""
    X_eval = X if gene_indices is None else X[:, gene_indices]
    deg_eval: Optional[Dict[str, np.ndarray]] = None
    pds_eval: Optional[Dict[str, np.ndarray]] = None
    if deg_weights_per_pert is not None:
        deg_eval = _subset_weights_per_pert(deg_weights_per_pert, gene_indices=gene_indices)
    if pds_weights_per_pert is not None:
        pds_eval = _subset_weights_per_pert(pds_weights_per_pert, gene_indices=gene_indices)

    if metric == "mse":
        return _pairwise_mse_matrix(X_eval)
    if metric == "wmse":
        if deg_eval is None:
            raise ValueError("DEG weights are required for wmse.")
        return _pairwise_wmse_query_weighted(X=X_eval, labels=y, weights_per_pert=deg_eval)
    if metric == "pds_wmse":
        if pds_eval is None:
            raise ValueError("PDS weights are required for pds_wmse.")
        return _pairwise_wmse_query_weighted(X=X_eval, labels=y, weights_per_pert=pds_eval)
    if metric == "delta_pearson":
        control_idx = np.where(y == control_label)[0]
        if control_idx.size < 1:
            raise ValueError(f"No control cells for control_label='{control_label}'.")
        control_mean = np.mean(X_eval[control_idx], axis=0)
        return _pairwise_delta_pearson_matrix(X=X_eval, control_mean=control_mean)
    if metric == "deg_weighted_delta_pearson":
        if deg_eval is None:
            raise ValueError("DEG weights are required for deg_weighted_delta_pearson.")
        control_idx = np.where(y == control_label)[0]
        if control_idx.size < 1:
            raise ValueError(f"No control cells for control_label='{control_label}'.")
        control_mean = np.mean(X_eval[control_idx], axis=0)
        return _pairwise_weighted_delta_pearson(
            X=X_eval,
            labels=y,
            control_mean=control_mean,
            weights_per_pert=deg_eval,
        )
    if metric == "pds_weighted_delta_pearson":
        if pds_eval is None:
            raise ValueError("PDS weights are required for pds_weighted_delta_pearson.")
        control_idx = np.where(y == control_label)[0]
        if control_idx.size < 1:
            raise ValueError(f"No control cells for control_label='{control_label}'.")
        control_mean = np.mean(X_eval[control_idx], axis=0)
        return _pairwise_weighted_delta_pearson(
            X=X_eval,
            labels=y,
            control_mean=control_mean,
            weights_per_pert=pds_eval,
        )
    if metric == "r2_delta_pert":
        return _pairwise_r2_delta_pert(X_eval)
    if metric == "r2_delta_ctrl":
        control_idx = np.where(y == control_label)[0]
        if control_idx.size < 1:
            raise ValueError(f"No control cells for control_label='{control_label}'.")
        control_mean = np.mean(X_eval[control_idx], axis=0)
        return _pairwise_r2_delta_ctrl(X=X_eval, control_mean=control_mean)
    if metric == "weighted_r2_delta_pert":
        if deg_eval is None:
            raise ValueError("DEG weights are required for weighted_r2_delta_pert.")
        return _pairwise_weighted_r2_delta_pert(X=X_eval, labels=y, weights_per_pert=deg_eval)
    if metric == "weighted_r2_delta_ctrl":
        if deg_eval is None:
            raise ValueError("DEG weights are required for weighted_r2_delta_ctrl.")
        control_idx = np.where(y == control_label)[0]
        if control_idx.size < 1:
            raise ValueError(f"No control cells for control_label='{control_label}'.")
        control_mean = np.mean(X_eval[control_idx], axis=0)
        return _pairwise_weighted_r2_delta_ctrl(
            X=X_eval,
            labels=y,
            control_mean=control_mean,
            weights_per_pert=deg_eval,
        )
    if metric == "pds_weighted_r2_delta_pert":
        if pds_eval is None:
            raise ValueError("PDS weights are required for pds_weighted_r2_delta_pert.")
        return _pairwise_weighted_r2_delta_pert(X=X_eval, labels=y, weights_per_pert=pds_eval)
    if metric == "pds_weighted_r2_delta_ctrl":
        if pds_eval is None:
            raise ValueError("PDS weights are required for pds_weighted_r2_delta_ctrl.")
        control_idx = np.where(y == control_label)[0]
        if control_idx.size < 1:
            raise ValueError(f"No control cells for control_label='{control_label}'.")
        control_mean = np.mean(X_eval[control_idx], axis=0)
        return _pairwise_weighted_r2_delta_ctrl(
            X=X_eval,
            labels=y,
            control_mean=control_mean,
            weights_per_pert=pds_eval,
        )
    raise ValueError(f"Unsupported metric: {metric}")


def _parse_metrics(raw: str) -> List[str]:
    """Parse and validate metric list from CLI."""
    values = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {
        "mse",
        "wmse",
        "pds_wmse",
        "r2_delta_pert",
        "r2_delta_ctrl",
        "weighted_r2_delta_pert",
        "weighted_r2_delta_ctrl",
        "pds_weighted_r2_delta_pert",
        "pds_weighted_r2_delta_ctrl",
        "delta_pearson",
        "deg_weighted_delta_pearson",
        "pds_weighted_delta_pearson",
        "filtered_mse",
        "filtered_wmse",
        "filtered_pds_wmse",
        "filtered_r2_delta_pert",
        "filtered_r2_delta_ctrl",
        "filtered_weighted_r2_delta_pert",
        "filtered_weighted_r2_delta_ctrl",
        "filtered_pds_weighted_r2_delta_pert",
        "filtered_pds_weighted_r2_delta_ctrl",
        "filtered_delta_pearson",
        "filtered_deg_weighted_delta_pearson",
        "filtered_pds_weighted_delta_pearson",
    }
    if not values:
        raise ValueError("No metrics parsed from --metrics.")
    unknown = [item for item in values if item not in allowed]
    if unknown:
        raise ValueError(f"Unsupported metrics requested: {unknown}. Allowed: {sorted(allowed)}")
    deduped: List[str] = []
    for metric in values:
        if metric not in deduped:
            deduped.append(metric)
    return deduped


def _evaluate_metrics(
    data: PreparedData,
    metrics: Sequence[str],
    n_bootstrap: int,
    seed: int,
    deg_weights_per_pert: Optional[Dict[str, np.ndarray]],
    pds_weights_per_pert: Optional[Dict[str, np.ndarray]],
    filtered_gene_indices: Optional[np.ndarray],
) -> Tuple[List[MetricResult], Dict[str, np.ndarray]]:
    """Evaluate normalized-rank scores for all requested metrics."""
    results: List[MetricResult] = []
    matrices: Dict[str, np.ndarray] = {}
    n_cells = int(data.X.shape[0])
    n_perts = int(len(data.perturbations))
    for metric in metrics:
        print (f"Evaluating metric: {metric}")
        use_filtered = metric.startswith("filtered_")
        base_metric = metric[len("filtered_") :] if use_filtered else metric
        eval_gene_idx = filtered_gene_indices if use_filtered else None
        n_features_eval = int(data.X.shape[1] if eval_gene_idx is None else eval_gene_idx.size)
        deg_eval = (
            _subset_weights_per_pert(deg_weights_per_pert, gene_indices=eval_gene_idx)
            if deg_weights_per_pert is not None
            else None
        )
        pds_eval = (
            _subset_weights_per_pert(pds_weights_per_pert, gene_indices=eval_gene_idx)
            if pds_weights_per_pert is not None
            else None
        )
        metric_weights: Optional[Dict[str, np.ndarray]] = None
        if base_metric in {
            "wmse",
            "weighted_r2_delta_pert",
            "weighted_r2_delta_ctrl",
            "deg_weighted_delta_pearson",
        }:
            metric_weights = deg_eval
        elif base_metric in {
            "pds_wmse",
            "pds_weighted_r2_delta_pert",
            "pds_weighted_r2_delta_ctrl",
            "pds_weighted_delta_pearson",
        }:
            metric_weights = pds_eval
        if use_filtered and (filtered_gene_indices is None or filtered_gene_indices.size == 0):
            dist = np.full((n_cells, n_cells), np.nan, dtype=float)
            scores = _normalized_rank_per_sample(distances=dist, labels=data.y)
            results.append(
                MetricResult(
                    metric=metric,
                    mean_normalized_rank=float(np.nanmean(scores)),
                    std_bootstrap=float("nan"),
                    effective_genes_median=0.0,
                    effective_genes_mean=0.0,
                    n_cells=n_cells,
                    n_perturbations=n_perts,
                )
            )
            matrices[metric] = dist
            continue
        dist = _compute_distance_matrix(
            metric=base_metric,
            X=data.X,
            y=data.y,
            control_label=data.control_label,
            deg_weights_per_pert=deg_weights_per_pert,
            pds_weights_per_pert=pds_weights_per_pert,
            gene_indices=eval_gene_idx,
        )
        scores = _normalized_rank_per_sample(distances=dist, labels=data.y)
        mean_score = float(np.nanmean(scores))
        std_boot = _bootstrap_std(scores=scores, n_bootstrap=n_bootstrap, seed=seed + hash(metric) % 9973)
        eff_median, eff_mean = _effective_genes_for_metric(
            labels=data.y,
            n_features=n_features_eval,
            weights_per_pert=metric_weights,
        )
        results.append(
            MetricResult(
                metric=metric,
                mean_normalized_rank=mean_score,
                std_bootstrap=std_boot,
                effective_genes_median=eff_median,
                effective_genes_mean=eff_mean,
                n_cells=n_cells,
                n_perturbations=n_perts,
            )
        )
        matrices[metric] = dist
    return results, matrices


def _save_summary(results: Sequence[MetricResult], output_path: Path) -> None:
    """Write metric summary CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "metric",
                "mean_normalized_rank",
                "std_bootstrap",
                "effective_genes_median",
                "effective_genes_mean",
                "n_cells",
                "n_perturbations",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item.metric,
                    f"{item.mean_normalized_rank:.8f}",
                    f"{item.std_bootstrap:.8f}" if np.isfinite(item.std_bootstrap) else "",
                    f"{item.effective_genes_median:.8f}",
                    f"{item.effective_genes_mean:.8f}",
                    item.n_cells,
                    item.n_perturbations,
                ]
            )


def _save_config(
    args: argparse.Namespace,
    data: PreparedData,
    metrics: Sequence[str],
    output_path: Path,
    dataset_name: str,
) -> None:
    """Write run config JSON."""
    payload = {
        "dataset": dataset_name,
        "metrics": list(metrics),
        "pds_result_file": args.pds_result_file,
        "pds_metric": args.pds_metric,
        "deg_weight_transform": args.deg_weight_transform,
        "pds_weight_transform": args.pds_weight_transform,
        "pds_threshold_filter": float(args.pds_threshold_filter),
        "base_weight_exponent": float(args.base_weight_exponent),
        "target_effective_genes": (
            None if args.target_effective_genes is None else float(args.target_effective_genes)
        ),
        "target_effective_genes_mode": str(args.target_effective_genes_mode),
        "pds_target_effective_genes_mode": str(args.pds_target_effective_genes_mode),
        "seed": int(args.seed),
        "cells_per_perturbation": args.cells_per_perturbation,
        "max_cells": args.max_cells,
        "max_perturbations": args.max_perturbations,
        "control_label": data.control_label,
        "n_cells": int(data.X.shape[0]),
        "n_genes": int(data.X.shape[1]),
        "n_perturbations": int(len(data.perturbations)),
        "perturbations": data.perturbations,
        "n_bootstrap": int(args.n_bootstrap),
        "heatmap_quantile_scope": str(args.heatmap_quantile_scope),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_datasets(args: argparse.Namespace) -> List[str]:
    """Resolve dataset list from ``--datasets`` and/or ``--dataset``."""
    values: List[str] = []
    if args.datasets is not None:
        values.extend([str(item).strip() for item in args.datasets if str(item).strip()])
    if args.dataset is not None and str(args.dataset).strip():
        values.append(str(args.dataset).strip())
    deduped: List[str] = []
    for item in values:
        if item not in deduped:
            deduped.append(item)
    if not deduped:
        raise ValueError("Provide at least one dataset via --dataset or --datasets.")
    return deduped


def _plot_metric_heatmaps(
    matrices: Dict[str, np.ndarray],
    labels: np.ndarray,
    metrics: Sequence[str],
    output_path: Path,
    max_perturbations: int,
    max_cells: int,
    quantile_bins: int,
    quantile_scope: str,
    seed: int,
) -> None:
    """Plot pairwise distance heatmaps for all metrics on a common cell subset."""
    if quantile_bins <= 1:
        raise ValueError("heatmap_quantile_bins must be > 1.")

    def _quantile_bin_block(block: np.ndarray, n_bins: int) -> np.ndarray:
        """Map one matrix block to [0, 1] via quantile bins."""
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

    def _quantile_bin_matrix(
        matrix: np.ndarray,
        n_bins: int,
        scope: str,
        row_groups: Optional[List[np.ndarray]],
    ) -> np.ndarray:
        """Map matrix values to [0, 1] using global or per-perturbation bins."""
        if scope == "global":
            return _quantile_bin_block(matrix, n_bins=n_bins)
        if scope != "per_perturbation":
            raise ValueError(f"Unsupported heatmap quantile scope: {scope}")
        if not row_groups:
            return _quantile_bin_block(matrix, n_bins=n_bins)
        out = np.zeros_like(matrix, dtype=float)
        for group_rows in row_groups:
            out[group_rows, :] = _quantile_bin_block(matrix[group_rows, :], n_bins=n_bins)
        return out

    y = np.asarray(labels, dtype=str)
    rng = np.random.default_rng(seed)
    selected_perts = np.unique(y)
    if max_perturbations > 0 and selected_perts.size > max_perturbations:
        selected_perts = rng.choice(selected_perts, size=max_perturbations, replace=False)
    keep_idx = np.where(np.isin(y, selected_perts))[0]
    if max_cells > 0 and keep_idx.size > max_cells:
        keep_idx = np.sort(rng.choice(keep_idx, size=max_cells, replace=False))

    y_keep = y[keep_idx]
    order = np.argsort(y_keep, kind="stable")
    final_idx = keep_idx[order]
    y_final = y[final_idx]
    row_groups: List[np.ndarray] = []
    if y_final.size > 0:
        start = 0
        while start < y_final.size:
            end = start + 1
            while end < y_final.size and y_final[end] == y_final[start]:
                end += 1
            row_groups.append(np.arange(start, end, dtype=int))
            start = end

    n_metrics = len(metrics)
    n_cols = int(np.ceil(np.sqrt(n_metrics)))
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows))
    axes_arr = np.atleast_1d(axes).ravel()
    for ax in axes_arr[n_metrics:]:
        ax.axis("off")

    for i, metric in enumerate(metrics):
        ax = axes_arr[i]
        dist = matrices[metric][np.ix_(final_idx, final_idx)]
        # Optional stabilization for r2-like distances before visualization.
        if "r2" in str(metric).lower():
            dist = np.maximum(dist, 0.0)
        dist_vis = _quantile_bin_matrix(
            dist,
            n_bins=int(quantile_bins),
            scope=str(quantile_scope),
            row_groups=row_groups,
        )
        img = ax.imshow(
            dist_vis,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(metric)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Quantile rank (global)" if quantile_scope == "global" else "Quantile rank (per perturbation)")

    fig.suptitle("Pairwise Distance Heatmaps by Metric", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
