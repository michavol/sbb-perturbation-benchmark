"""
Core utilities for perturbation discrimination score (PDS) analysis.

This module provides:
- Distance metrics for 1D distributions
- Gene filtering strategies
- PDS computation using rank-based matching across perturbations
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import anndata
import numpy as np
import scanpy as sc
from scipy.stats import energy_distance, wasserstein_distance
from tqdm import tqdm


Array1D = np.ndarray
DistanceMetric = Callable[[Array1D, Array1D], float]


def resolve_perturbation_key(adata: anndata.AnnData, key: Optional[str]) -> str:
    """Resolve the perturbation column in adata.obs.

    Args:
        adata: Annotated data matrix.
        key: Preferred column name. If None, tries common defaults.

    Returns:
        Column name to use for perturbations.

    Raises:
        KeyError: If no valid perturbation column is found.
    """
    if key:
        if key not in adata.obs.columns:
            raise KeyError(f"Perturbation key '{key}' not found in adata.obs.")
        return key
    for candidate in ("perturbation", "condition"):
        if candidate in adata.obs.columns:
            return candidate
    raise KeyError("No perturbation column found in adata.obs.")


def mae_on_mean(cloud_a: Array1D, cloud_b: Array1D) -> float:
    """Compute MAE between the means of two point clouds.

    Args:
        cloud_a: Values for distribution A.
        cloud_b: Values for distribution B.

    Returns:
        Mean absolute error between the means.
    """
    return float(abs(np.mean(cloud_a) - np.mean(cloud_b)))


def mae_on_median(cloud_a: Array1D, cloud_b: Array1D) -> float:
    """Compute MAE between the medians of two point clouds.

    Args:
        cloud_a: Values for distribution A.
        cloud_b: Values for distribution B.

    Returns:
        Mean absolute error between the medians.
    """
    return float(abs(np.median(cloud_a) - np.median(cloud_b)))


def mae_on_quantile(cloud_a: Array1D, cloud_b: Array1D, quantile: float) -> float:
    """Compute MAE between the specified quantiles of two point clouds.

    Args:
        cloud_a: Values for distribution A.
        cloud_b: Values for distribution B.
        quantile: Quantile to compare (0.0 to 1.0).

    Returns:
        Mean absolute error between the quantiles.
    """
    return float(abs(np.quantile(cloud_a, quantile) - np.quantile(cloud_b, quantile)))


def energy_distance_1d(cloud_a: Array1D, cloud_b: Array1D) -> float:
    """Compute 1D energy distance using scipy.

    Args:
        cloud_a: Values for distribution A.
        cloud_b: Values for distribution B.

    Returns:
        Energy distance.
    """
    return float(energy_distance(np.asarray(cloud_a).ravel(), np.asarray(cloud_b).ravel()))


def wasserstein_distance_1d(cloud_a: Array1D, cloud_b: Array1D) -> float:
    """Compute 1D Wasserstein distance using scipy.

    Args:
        cloud_a: Values for distribution A.
        cloud_b: Values for distribution B.

    Returns:
        Wasserstein distance.
    """
    return float(wasserstein_distance(np.asarray(cloud_a).ravel(), np.asarray(cloud_b).ravel()))


def mmd_gaussian_1d(
    cloud_a: Array1D,
    cloud_b: Array1D,
    bandwidth: Optional[float] = None,
) -> float:
    """Compute MMD with a Gaussian kernel for 1D samples.

    Uses the unbiased estimator with optional median heuristic bandwidth.

    Args:
        cloud_a: Values for distribution A.
        cloud_b: Values for distribution B.
        bandwidth: Kernel bandwidth. If None, uses median heuristic.

    Returns:
        MMD value (non-negative).
    """
    x = np.asarray(cloud_a).ravel()
    y = np.asarray(cloud_b).ravel()

    if x.size < 2 or y.size < 2:
        raise ValueError("MMD requires at least 2 samples in each distribution.")

    if bandwidth is None:
        combined = np.concatenate([x, y])
        if combined.size < 2:
            bandwidth = 1.0
        else:
            diffs = np.abs(combined[:, None] - combined[None, :])
            median = np.median(diffs)
            bandwidth = float(median) if median > 0 else 1.0

    gamma = 1.0 / (2.0 * bandwidth * bandwidth)

    def kernel(a: Array1D, b: Array1D) -> np.ndarray:
        return np.exp(-gamma * (a[:, None] - b[None, :]) ** 2)

    k_xx = kernel(x, x)
    k_yy = kernel(y, y)
    k_xy = kernel(x, y)

    n = x.size
    m = y.size

    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)

    mmd = k_xx.sum() / (n * (n - 1)) + k_yy.sum() / (m * (m - 1)) - 2.0 * k_xy.mean()
    return float(max(mmd, 0.0))


def _is_combo_perturbation(name: str) -> bool:
    """Check if a perturbation label represents a combo perturbation.

    Args:
        name: Perturbation label.

    Returns:
        True if the label indicates a combo perturbation.
    """
    if "+" in name:
        return True
    if "_" in name:
        parts = [part for part in name.split("_") if part]
        return len(parts) >= 2
    return False


def select_top_perturbations(
    adata: anndata.AnnData,
    n_perturbations: int,
    perturbation_key: Optional[str] = None,
    exclude_controls: bool = True,
    prefer_combos: bool = False,
) -> Tuple[List[str], int]:
    """Select top perturbations by cell count and their minimum count.

    Args:
        adata: Annotated data matrix.
        n_perturbations: Number of perturbations to select.
        perturbation_key: Column in adata.obs with perturbation labels.
        exclude_controls: If True, exclude control perturbations.
        prefer_combos: If True, prefer combo perturbations when available.

    Returns:
        Tuple of (selected perturbations list, min cell count across selection).
    """
    if n_perturbations <= 0:
        raise ValueError("n_perturbations must be positive.")
    key = resolve_perturbation_key(adata, perturbation_key)
    counts = adata.obs[key].value_counts()
    if counts.empty:
        raise ValueError("No perturbations found in adata.obs.")

    if exclude_controls:
        valid_index = [
            p
            for p in counts.index
            if "control" not in str(p).lower() and "ctrl" not in str(p).lower()
        ]
        counts = counts.loc[valid_index]

    if prefer_combos:
        combo_labels = [p for p in counts.index if _is_combo_perturbation(str(p))]
        if combo_labels:
            counts = counts.loc[combo_labels]

    counts = counts.sort_values(ascending=False)
    selected = counts.index[:n_perturbations].tolist()
    if not selected:
        raise ValueError("No perturbations remain after filtering controls.")
    if len(selected) < n_perturbations:
        print(f"Requested {n_perturbations} perturbations, but only {len(selected)} available.")
        print(f"Selected perturbations: {selected}")
        print(f"Available perturbations: {counts.index.tolist()}")
        print(f"Available perturbations counts: {counts.values.tolist()}")
        print(f"Available perturbations counts: {counts.values.tolist()}")
        
    min_count = int(counts.loc[selected].min()) if selected else 0
    return selected, min_count


def sample_cells_per_perturbation(
    adata: anndata.AnnData,
    perturbations: Sequence[str],
    n_cells: int,
    seed: int,
    perturbation_key: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Sample a fixed number of cells per perturbation.

    Args:
        adata: Annotated data matrix.
        perturbations: Perturbations to sample from.
        n_cells: Number of cells per perturbation.
        seed: Random seed.
        perturbation_key: Column in adata.obs with perturbation labels.

    Returns:
        Mapping from perturbation to sampled cell indices.
    """
    if n_cells < 2:
        raise ValueError("n_cells must be at least 2 to allow half-splitting.")
    if not perturbations:
        raise ValueError("Perturbations list is empty.")
    key = resolve_perturbation_key(adata, perturbation_key)
    rng = np.random.default_rng(seed)
    sampled: Dict[str, np.ndarray] = {}
    for pert in perturbations:
        indices = np.where(adata.obs[key] == pert)[0]
        if indices.size < n_cells:
            raise ValueError(f"Perturbation {pert} has {indices.size} cells, needs {n_cells}.")
        sampled[pert] = rng.choice(indices, size=n_cells, replace=False)
    return sampled


def split_indices_half(indices: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices into two halves after shuffling.

    Args:
        indices: Array of indices to split.
        seed: Random seed.

    Returns:
        Tuple of (half A indices, half B indices).
    """
    if indices.size < 2:
        raise ValueError("Need at least 2 indices to split into halves.")
    rng = np.random.default_rng(seed)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    mid = shuffled.size // 2
    return shuffled[:mid], shuffled[mid:]


def get_expression_1d(
    adata: anndata.AnnData,
    cell_indices: np.ndarray,
    gene_idx: int,
) -> Array1D:
    """Extract 1D expression values for a gene from selected cells.

    Args:
        adata: Annotated data matrix.
        cell_indices: Indices of cells to include.
        gene_idx: Gene index.

    Returns:
        1D numpy array of expression values.
    """
    if gene_idx < 0 or gene_idx >= adata.shape[1]:
        raise IndexError(f"gene_idx {gene_idx} out of bounds for adata.var.")
    expr = adata.X[cell_indices, gene_idx]
    if hasattr(expr, "toarray"):
        return np.asarray(expr.toarray()).ravel()
    return np.asarray(expr).ravel()


def rank_to_score(rank: int, n_perturbations: int) -> float:
    """Convert a rank to a PDS score in [-1, 1].

    Args:
        rank: Zero-based rank (0 = best).
        n_perturbations: Total number of perturbations.

    Returns:
        Rank-based score where 1 is best and -1 is worst.
    """
    if n_perturbations <= 1:
        return 1.0
    if rank < 0 or rank >= n_perturbations:
        raise ValueError("rank must be within [0, n_perturbations).")
    return float(1.0 - 2.0 * (rank / float(n_perturbations - 1)))


def _make_tie_break_rng(
    base_seed: Optional[int],
    gene_idx: int,
) -> np.random.Generator:
    """Create a deterministic RNG for tie-breaking.

    Args:
        base_seed: Optional base seed for reproducibility.
        gene_idx: Gene index used to decorrelate RNG streams.

    Returns:
        NumPy random generator for tie-breaking.
    """
    if base_seed is None:
        return np.random.default_rng()
    seed = int((base_seed + gene_idx * 10007) % (2**32))
    return np.random.default_rng(seed)


def _apply_random_tie_break(
    distances: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add tiny random jitter to break ties without reordering non-ties.

    Args:
        distances: Distance vector to rank.
        rng: Random generator for jitter.

    Returns:
        Distance vector with tiny random jitter added.
    """
    jitter_scale = np.finfo(float).eps
    jitter = rng.uniform(-1.0, 1.0, size=distances.shape) * jitter_scale * (
        1.0 + np.abs(distances)
    )
    return distances + jitter


def compute_pds_for_gene(
    gene_idx: int,
    adata: anndata.AnnData,
    perturbations: Sequence[str],
    half_a_indices: Dict[str, np.ndarray],
    half_b_indices: Dict[str, np.ndarray],
    distance_metric: DistanceMetric,
    tie_break_seed: Optional[int] = None,
) -> Tuple[float, Dict[str, float], Dict[str, List[Tuple[str, float]]]]:
    """Compute PDS for a single gene across all perturbations.

    Args:
        gene_idx: Gene index to score.
        adata: Annotated data matrix.
        perturbations: Perturbation labels.
        half_a_indices: Mapping from perturbation to half-A cell indices.
        half_b_indices: Mapping from perturbation to half-B cell indices.
        distance_metric: Distance function between two 1D clouds.
        tie_break_seed: Optional seed for randomized tie-breaking.

    Returns:
        Tuple of (mean score across perturbations, per-pert scores, ranking details).
    """
    if not perturbations:
        raise ValueError("Perturbations list is empty.")
    if len(perturbations) < 2:
        raise ValueError("Need at least 2 perturbations for PDS.")
    for pert in perturbations:
        if pert not in half_a_indices or pert not in half_b_indices:
            raise KeyError(f"Missing half indices for perturbation '{pert}'.")
    scores: List[float] = []
    per_pert_scores: Dict[str, float] = {}
    ranking_details: Dict[str, List[Tuple[str, float]]] = {}
    rng = _make_tie_break_rng(tie_break_seed, gene_idx)

    half_b_clouds: Dict[str, Array1D] = {}
    for pert in perturbations:
        half_b_clouds[pert] = get_expression_1d(adata, half_b_indices[pert], gene_idx)

    for pert_a in perturbations:
        cloud_a = get_expression_1d(adata, half_a_indices[pert_a], gene_idx)
        distances: List[Tuple[str, float]] = []
        for pert_b in perturbations:
            dist = distance_metric(cloud_a, half_b_clouds[pert_b])
            distances.append((pert_b, dist))
        distance_values = np.asarray([item[1] for item in distances], dtype=float)
        distance_values = _apply_random_tie_break(distance_values, rng)
        order = np.argsort(distance_values)
        ordered_distances = [distances[idx] for idx in order]
        ranking_details[pert_a] = ordered_distances

        rank = next(i for i, (p, _) in enumerate(ordered_distances) if p == pert_a)
        score = rank_to_score(rank, len(perturbations))
        scores.append(score)
        per_pert_scores[pert_a] = score

    return float(np.mean(scores)), per_pert_scores, ranking_details


def compute_pds_for_gene_multi_metric(
    gene_idx: int,
    adata: anndata.AnnData,
    perturbations: Sequence[str],
    half_a_indices: Dict[str, np.ndarray],
    half_b_indices: Dict[str, np.ndarray],
    metric_funcs: Sequence[DistanceMetric],
    comparison_sets: Optional[Mapping[str, Sequence[str]]] = None,
    tie_break_seed: Optional[int] = None,
) -> np.ndarray:
    """Compute per-perturbation PDS for multiple metrics for one gene.

    Args:
        gene_idx: Gene index to score.
        adata: Annotated data matrix.
        perturbations: Perturbation labels.
        half_a_indices: Mapping from perturbation to half-A cell indices.
        half_b_indices: Mapping from perturbation to half-B cell indices.
        metric_funcs: Sequence of distance functions.
        comparison_sets: Optional mapping from perturbation to comparison set.
        tie_break_seed: Optional seed for randomized tie-breaking.

    Returns:
        Array of shape (n_metrics, n_perts) with PDS scores.
    """
    if not perturbations:
        raise ValueError("Perturbations list is empty.")
    if len(perturbations) < 2:
        raise ValueError("Need at least 2 perturbations for PDS.")
    for pert in perturbations:
        if pert not in half_a_indices or pert not in half_b_indices:
            raise KeyError(f"Missing half indices for perturbation '{pert}'.")

    n_metrics = len(metric_funcs)
    n_perts = len(perturbations)
    scores = np.zeros((n_metrics, n_perts), dtype=float)
    rng = _make_tie_break_rng(tie_break_seed, gene_idx)

    half_b_clouds: Dict[str, Array1D] = {}
    for pert in perturbations:
        half_b_clouds[pert] = get_expression_1d(adata, half_b_indices[pert], gene_idx)

    for pert_a_idx, pert_a in enumerate(perturbations):
        cloud_a = get_expression_1d(adata, half_a_indices[pert_a], gene_idx)
        comparison_perts = (
            list(comparison_sets.get(pert_a, perturbations))
            if comparison_sets is not None
            else list(perturbations)
        )
        if pert_a not in comparison_perts:
            comparison_perts.append(pert_a)
        distances = np.zeros((n_metrics, len(comparison_perts)), dtype=float)
        for pert_b_idx, pert_b in enumerate(comparison_perts):
            cloud_b = half_b_clouds[pert_b]
            for metric_idx, metric_func in enumerate(metric_funcs):
                distances[metric_idx, pert_b_idx] = metric_func(cloud_a, cloud_b)

        for metric_idx in range(n_metrics):
            distance_values = _apply_random_tie_break(distances[metric_idx], rng)
            order = np.argsort(distance_values)
            pert_a_local_idx = comparison_perts.index(pert_a)
            rank = int(np.where(order == pert_a_local_idx)[0][0])
            scores[metric_idx, pert_a_idx] = rank_to_score(rank, len(comparison_perts))

    return scores


def compute_pds_scores(
    adata: anndata.AnnData,
    perturbations: Sequence[str],
    gene_indices: Sequence[int],
    distance_metric: DistanceMetric,
    half_a_indices: Dict[str, np.ndarray],
    half_b_indices: Dict[str, np.ndarray],
    tie_break_seed: Optional[int] = None,
) -> Dict[str, object]:
    """Compute PDS scores for a set of genes.

    Args:
        adata: Annotated data matrix.
        perturbations: Perturbation labels.
        gene_indices: Gene indices to score.
        distance_metric: Distance function between two 1D clouds.
        half_a_indices: Mapping from perturbation to half-A cell indices.
        half_b_indices: Mapping from perturbation to half-B cell indices.
        tie_break_seed: Optional seed for randomized tie-breaking.

    Returns:
        Dictionary with gene scores, per-pert scores, and rankings.
    """
    if len(gene_indices) == 0:
        raise ValueError("gene_indices is empty.")
    gene_scores: List[float] = []
    gene_names: List[str] = []
    gene_rankings: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    per_pert_per_gene_scores: Dict[str, Dict[str, float]] = {p: {} for p in perturbations}

    for gene_idx in tqdm(
        gene_indices, desc="Scoring genes", leave=False, mininterval=2.0
    ):
        score, per_pert_scores, rankings = compute_pds_for_gene(
            gene_idx=gene_idx,
            adata=adata,
            perturbations=perturbations,
            half_a_indices=half_a_indices,
            half_b_indices=half_b_indices,
            distance_metric=distance_metric,
            tie_break_seed=tie_break_seed,
        )
        gene_name = str(adata.var_names[gene_idx])
        gene_names.append(gene_name)
        gene_scores.append(score)
        gene_rankings[gene_name] = rankings

        for pert, pert_score in per_pert_scores.items():
            per_pert_per_gene_scores[pert][gene_name] = pert_score

    return {
        "gene_scores": np.asarray(gene_scores),
        "gene_names": gene_names,
        "gene_rankings": gene_rankings,
        "per_pert_per_gene_scores": per_pert_per_gene_scores,
    }


def compute_mean_score_with_pert_specific_genes(
    perturbations: Sequence[str],
    per_pert_per_gene_scores: Dict[str, Dict[str, float]],
    pert_specific_genes_map: Optional[Dict[str, List[int]]],
    gene_names_by_index: Sequence[str],
) -> float:
    """Compute mean score using per-perturbation gene selections.

    Args:
        perturbations: Perturbation labels.
        per_pert_per_gene_scores: Per-perturbation gene scores.
        pert_specific_genes_map: Mapping of perturbation to selected gene indices.
        gene_names_by_index: Names aligned to adata.var_names.

    Returns:
        Mean score across perturbations using their own gene sets.
    """
    if not pert_specific_genes_map:
        raise ValueError("pert_specific_genes_map is empty.")

    pert_means: List[float] = []
    for pert in perturbations:
        gene_indices = pert_specific_genes_map.get(pert, [])
        if not gene_indices:
            continue
        for idx in gene_indices:
            if idx < 0 or idx >= len(gene_names_by_index):
                raise IndexError(f"Gene index {idx} out of bounds for gene_names_by_index.")
        gene_names = [gene_names_by_index[idx] for idx in gene_indices]
        scores = [
            per_pert_per_gene_scores[pert][gene]
            for gene in gene_names
            if gene in per_pert_per_gene_scores[pert]
        ]
        if scores:
            pert_means.append(float(np.mean(scores)))

    return float(np.mean(pert_means)) if pert_means else float("nan")


def select_highly_variable_genes(
    adata: anndata.AnnData,
    n_genes: int,
) -> np.ndarray:
    """Select top highly variable genes using scanpy.

    Args:
        adata: Annotated data matrix.
        n_genes: Number of genes to select.

    Returns:
        Array of selected gene indices.
    """
    if n_genes <= 0:
        raise ValueError("n_genes must be positive.")
    adata_copy = adata.copy()
    sc.pp.highly_variable_genes(
        adata_copy, n_top_genes=min(n_genes, adata_copy.shape[1]), subset=False
    )
    return np.where(adata_copy.var["highly_variable"])[0][:n_genes]


def _compute_mean_variance(X: object) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and variance for dense or sparse matrices.

    Args:
        X: Matrix-like object (dense or sparse).

    Returns:
        Tuple of (mean, variance) arrays.
    """
    if hasattr(X, "toarray"):
        X_dense = np.asarray(X.toarray())
        mean = np.mean(X_dense, axis=0)
        var = np.var(X_dense, axis=0)
        return mean, var

    mean = np.asarray(X.mean(axis=0)).ravel()
    if hasattr(X, "power"):
        mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
    else:
        X_dense = np.asarray(X)
        mean_sq = np.mean(X_dense**2, axis=0)
    var = np.maximum(mean_sq - mean**2, 0.0)
    return mean, var


def select_lowest_cv_from_hvg(
    adata: anndata.AnnData,
    n_genes: int,
    hvg_pool_size: int = 4000,
    perturbation_key: Optional[str] = None,
    min_cells_per_pert: int = 20,
) -> np.ndarray:
    """Select lowest average intra-perturbation CV genes from HVG pool.

    Args:
        adata: Annotated data matrix.
        n_genes: Number of genes to select.
        hvg_pool_size: HVG pool size before CV filtering.
        perturbation_key: Column in adata.obs with perturbation labels.
        min_cells_per_pert: Minimum cells per perturbation to include in CV averaging.

    Returns:
        Array of selected gene indices.
    """
    if n_genes <= 0:
        raise ValueError("n_genes must be positive.")
    if hvg_pool_size <= 0:
        raise ValueError("hvg_pool_size must be positive.")
    if min_cells_per_pert < 2:
        raise ValueError("min_cells_per_pert must be at least 2.")

    pool_size = min(hvg_pool_size, adata.shape[1])
    pool_indices = select_highly_variable_genes(adata, pool_size)

    key = resolve_perturbation_key(adata, perturbation_key)
    perturbations = adata.obs[key].unique()

    cv_sum = np.zeros(pool_indices.size, dtype=float)
    cv_count = np.zeros(pool_indices.size, dtype=float)

    for pert in perturbations:
        mask = adata.obs[key] == pert
        if np.sum(mask) < min_cells_per_pert:
            continue
        X_pert = adata[mask, :][:, pool_indices].X
        if hasattr(X_pert, "toarray"):
            X_pert = X_pert.toarray()
        X_pert = np.asarray(X_pert)
        mean = np.mean(X_pert, axis=0)
        std = np.std(X_pert, axis=0)

        valid = np.abs(mean) > 1e-8
        cv = np.full(pool_indices.size, np.inf, dtype=float)
        cv[valid] = std[valid] / mean[valid]

        cv_sum[valid] += cv[valid]
        cv_count[valid] += 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_cv = cv_sum / cv_count
    avg_cv[cv_count == 0] = np.inf
    avg_cv = np.nan_to_num(avg_cv, nan=np.inf)

    sorted_pool = np.argsort(avg_cv)
    selected = pool_indices[sorted_pool][:n_genes]
    return selected


def select_top_discriminating_genes(
    adata: anndata.AnnData,
    perturbations: Sequence[str],
    n_genes: int,
    distance_metric: DistanceMetric,
    half_a_indices: Dict[str, np.ndarray],
    half_b_indices: Dict[str, np.ndarray],
    candidate_gene_indices: Optional[np.ndarray] = None,
    tie_break_seed: Optional[int] = None,
) -> np.ndarray:
    """Select top discriminating genes by single-trial PDS.

    Args:
        adata: Annotated data matrix.
        perturbations: Perturbation labels.
        n_genes: Number of genes to select.
        distance_metric: Distance function between 1D clouds.
        half_a_indices: Mapping from perturbation to half-A indices.
        half_b_indices: Mapping from perturbation to half-B indices.
        candidate_gene_indices: Optional candidate subset to score.
        tie_break_seed: Optional seed for randomized tie-breaking.

    Returns:
        Array of selected gene indices.
    """
    if n_genes <= 0:
        raise ValueError("n_genes must be positive.")
    if len(perturbations) < 2:
        raise ValueError("Need at least 2 perturbations for discriminating selection.")
    if candidate_gene_indices is None:
        candidate_gene_indices = np.arange(adata.shape[1])
    if candidate_gene_indices.size == 0:
        raise ValueError("candidate_gene_indices is empty.")

    scores: Dict[int, float] = {}
    for gene_idx in tqdm(
        candidate_gene_indices,
        desc="Scoring candidates",
        leave=False,
        mininterval=2.0,
    ):
        mean_score, _, _ = compute_pds_for_gene(
            gene_idx=gene_idx,
            adata=adata,
            perturbations=perturbations,
            half_a_indices=half_a_indices,
            half_b_indices=half_b_indices,
            distance_metric=distance_metric,
            tie_break_seed=tie_break_seed,
        )
        scores[int(gene_idx)] = mean_score

    sorted_indices = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)
    selected = np.asarray(sorted_indices[:n_genes], dtype=int)
    if selected.size < n_genes:
        raise ValueError("Not enough candidate genes to satisfy n_genes.")
    return selected


def select_top_discriminating_genes_per_perturbation(
    adata: anndata.AnnData,
    perturbations: Sequence[str],
    n_genes: int,
    distance_metric: DistanceMetric,
    half_a_indices: Dict[str, np.ndarray],
    half_b_indices: Dict[str, np.ndarray],
    candidate_gene_indices: Optional[np.ndarray] = None,
    tie_break_seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, List[int]]]:
    """Select top discriminating genes per perturbation using single-trial PDS.

    Args:
        adata: Annotated data matrix.
        perturbations: Perturbation labels.
        n_genes: Number of genes per perturbation.
        distance_metric: Distance function between 1D clouds.
        half_a_indices: Mapping from perturbation to half-A indices.
        half_b_indices: Mapping from perturbation to half-B indices.
        candidate_gene_indices: Optional candidate subset to score.
        tie_break_seed: Optional seed for randomized tie-breaking.

    Returns:
        Tuple of (union indices, per-perturbation gene map).
    """
    if n_genes <= 0:
        raise ValueError("n_genes must be positive.")
    if len(perturbations) < 2:
        raise ValueError("Need at least 2 perturbations for discriminating selection.")
    if candidate_gene_indices is None:
        candidate_gene_indices = np.arange(adata.shape[1])
    if candidate_gene_indices.size == 0:
        raise ValueError("candidate_gene_indices is empty.")

    per_pert_scores: Dict[str, List[Tuple[int, float]]] = {p: [] for p in perturbations}

    for gene_idx in tqdm(
        candidate_gene_indices,
        desc="Scoring candidates per pert",
        leave=False,
        mininterval=2.0,
    ):
        _, per_pert, _ = compute_pds_for_gene(
            gene_idx=gene_idx,
            adata=adata,
            perturbations=perturbations,
            half_a_indices=half_a_indices,
            half_b_indices=half_b_indices,
            distance_metric=distance_metric,
            tie_break_seed=tie_break_seed,
        )
        for pert, score in per_pert.items():
            per_pert_scores[pert].append((int(gene_idx), float(score)))

    pert_gene_map: Dict[str, List[int]] = {}
    all_indices: List[int] = []
    for pert, scores in per_pert_scores.items():
        scores.sort(key=lambda item: item[1], reverse=True)
        top = [idx for idx, _ in scores[:n_genes]]
        if len(top) < n_genes:
            raise ValueError(f"Not enough genes to select for perturbation '{pert}'.")
        pert_gene_map[pert] = top
        all_indices.extend(top)

    union_indices = np.unique(np.asarray(all_indices, dtype=int))
    return union_indices, pert_gene_map

