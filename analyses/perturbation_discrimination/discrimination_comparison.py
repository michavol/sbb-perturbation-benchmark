"""
Aggregate filter-based PDS scores from full PDS tensors stored in results files.

This script computes filtering summaries without recomputing PDS, using
precomputed metrics × perturbations × genes tensors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analyses.perturbation_discrimination import pds_core  # noqa: E402
from analyses.perturbation_discrimination import deg_scanpy  # noqa: E402

FILTERS = [
    "highly_variable",
    "lowest_cv",
    "lowest_cv_per_perturbation",
    "deg_control",
    "deg_per_perturbation_control",
    "deg_synthetic",
    "deg_per_perturbation_synthetic",
    "top_discriminating",
    "top_discriminating_per_perturbation",
]


def _assign_pseudo_replicates(
    adata: sc.AnnData,
    n_groups: int,
    perturbation_key: str,
    key_name: str = "pseudo_rep",
    seed: int = 42,
) -> None:
    """Assign pseudo-replicate groups within each perturbation.

    Args:
        adata: Annotated data matrix to update in-place.
        n_groups: Number of pseudo-replicate groups per perturbation.
        perturbation_key: Column in adata.obs with perturbation labels.
        key_name: New column name for pseudo-replicate IDs.
        seed: Random seed for assignment.
    """
    rng = np.random.default_rng(seed)
    adata.obs[key_name] = "0"
    for pert in adata.obs[perturbation_key].unique():
        mask = adata.obs[perturbation_key] == pert
        indices = np.where(mask)[0]
        if indices.size == 0:
            continue
        rng.shuffle(indices)
        group_ids = np.arange(indices.size) % n_groups
        adata.obs.iloc[indices, adata.obs.columns.get_loc(key_name)] = group_ids.astype(str)


def _resolve_deg_group_keys(
    adata: sc.AnnData,
    dataset_name: str,
) -> Tuple[str, ...]:
    """Resolve replicate keys for DEG (used for warnings only).

    Args:
        adata: Annotated data matrix.
        dataset_name: Dataset identifier for special handling.

    Returns:
        Tuple of replicate key names (not used for scanpy DE).
    """
    if dataset_name == "norman19":
        if "gemgroup" not in adata.obs.columns:
            raise KeyError("Missing required obs column for DEG: 'gemgroup'")
        return ("gemgroup",)
    if dataset_name == "wessels23":
        if "HTO" not in adata.obs.columns:
            raise KeyError("Missing required obs column for DEG: 'HTO'")
        return ("HTO",)
    if dataset_name == "adamson16":
        print(
            "WARNING: Using pseudo-replicates for adamson16 "
            "(splitting each perturbation into 5 groups)."
        )
        _assign_pseudo_replicates(
            adata=adata,
            n_groups=5,
            perturbation_key="condition",
            key_name="pseudo_rep",
            seed=42,
        )
        return ("pseudo_rep",)
    for key in ("donor_id", "cell_type"):
        if key not in adata.obs.columns:
            adata.obs[key] = "unknown"
    return ("donor_id", "cell_type")


def _compute_lowest_cv_per_perturbation(
    adata: sc.AnnData,
    perturbations: Sequence[str],
    hvg_pool_size: int,
    perturbation_key: Optional[str],
    min_cells_per_pert: int = 2,
) -> Dict[str, List[str]]:
    """Compute lowest-CV gene rankings per perturbation from an HVG pool.

    Args:
        adata: Annotated data matrix.
        perturbations: Perturbation labels.
        hvg_pool_size: HVG pool size to consider.
        perturbation_key: Column in adata.obs with perturbation labels.
        min_cells_per_pert: Minimum cells per perturbation to include.

    Returns:
        Mapping perturbation -> ranked gene names by increasing CV.
    """
    if hvg_pool_size <= 0:
        raise ValueError("hvg_pool_size must be positive.")
    if min_cells_per_pert < 2:
        raise ValueError("min_cells_per_pert must be at least 2.")
    key = pds_core.resolve_perturbation_key(adata, perturbation_key)
    pool_indices = pds_core.select_highly_variable_genes(
        adata, min(hvg_pool_size, adata.shape[1])
    )
    rankings: Dict[str, List[str]] = {}
    for pert in perturbations:
        mask = adata.obs[key] == pert
        if int(np.sum(mask)) < min_cells_per_pert:
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
        order = np.argsort(cv)
        ranked_indices = pool_indices[order]
        rankings[pert] = [str(adata.var_names[i]) for i in ranked_indices]
    if not rankings:
        raise ValueError("No perturbations had enough cells for lowest CV.")
    return rankings


def load_dataset_paths(root: Path) -> List[Path]:
    """Load dataset paths from dataset configs."""
    config_dir = root / "cellsimbench" / "configs" / "dataset"
    paths: List[Path] = []
    for config_path in sorted(config_dir.glob("*.yaml")):
        with config_path.open("r") as handle:
            config = yaml.safe_load(handle)
        data_path = config.get("data_path")
        if not data_path:
            continue
        path = (root / data_path).resolve()
        if path.exists():
            paths.append(path)
    return paths


def load_dataset_path(root: Path, dataset_name: str) -> Path:
    """Load a single dataset path from dataset config.

    Args:
        root: Project root path.
        dataset_name: Dataset config stem.

    Returns:
        Absolute dataset path.
    """
    config_path = root / "cellsimbench" / "configs" / "dataset" / f"{dataset_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r") as handle:
        config = yaml.safe_load(handle)
    data_path = config.get("data_path")
    if not data_path:
        raise FileNotFoundError(f"No data_path in config: {config_path}")
    return (root / data_path).resolve()


def load_pds_full(result_path: Path) -> Dict[str, object]:
    """Load pds_full from a results h5ad file."""
    if not result_path.exists():
        raise FileNotFoundError(f"Missing results file: {result_path}")
    adata = ad.read_h5ad(result_path)
    if "pds_full" not in adata.uns:
        raise ValueError("Missing pds_full in results h5ad.")
    return adata.uns["pds_full"]


def compute_filter_rankings(
    adata: sc.AnnData,
    dataset_name: str,
    perturbations: Sequence[str],
    results_root: Path,
    filters: Sequence[str],
    max_perts: Optional[int],
) -> Dict[str, object]:
    """Compute ranked gene lists for filters and store as gene names.

    Args:
        adata: Annotated data matrix.
        dataset_name: Dataset name used for caching DEG results.
        perturbations: Perturbations included in PDS.
        results_root: Results root directory for DEG caching.
        filters: Filters requested for this run.
        max_perts: Optional limit on perturbations for debugging.

    Returns:
        Dictionary with ranked gene lists keyed by filter name.
    """
    rankings: Dict[str, object] = {}
    if "condition" not in adata.obs.columns:
        raise KeyError("Missing required obs column for DEG: 'condition'")
    adata_deg = adata.copy()
    group_keys = _resolve_deg_group_keys(adata_deg, dataset_name)
    adata_copy = adata.copy()
    sc.pp.highly_variable_genes(adata_copy, n_top_genes=adata.shape[1], subset=False)
    if "highly_variable_rank" in adata_copy.var:
        hvg_rank = adata_copy.var["highly_variable_rank"].to_numpy()
        order = np.argsort(np.nan_to_num(hvg_rank, nan=np.inf))
    elif "dispersions_norm" in adata_copy.var:
        order = np.argsort(-np.asarray(adata_copy.var["dispersions_norm"]))
    else:
        order = np.arange(adata.shape[1])
    rankings["highly_variable"] = [str(adata.var_names[i]) for i in order]

    try:
        low_cv_indices = pds_core.select_lowest_cv_from_hvg(
            adata, n_genes=min(adata.shape[1], 4000), perturbation_key=None
        )
        rankings["lowest_cv"] = [str(adata.var_names[i]) for i in low_cv_indices]
    except ValueError as exc:
        print(f"WARNING: lowest_cv ranking not available: {exc}")
        rankings["lowest_cv"] = []

    if "lowest_cv_per_perturbation" in filters:
        try:
            per_pert_low_cv = _compute_lowest_cv_per_perturbation(
                adata=adata,
                perturbations=perturbations,
                hvg_pool_size=min(adata.shape[1], 4000),
                perturbation_key=None,
            )
            rankings["lowest_cv_per_perturbation"] = per_pert_low_cv
        except ValueError as exc:
            print(f"WARNING: lowest_cv_per_perturbation ranking not available: {exc}")
            rankings["lowest_cv_per_perturbation"] = {}

    deg_modes = []
    if any(filt in filters for filt in ("deg_control", "deg_per_perturbation_control")):
        deg_modes.append("control")
    if any(
        filt in filters for filt in ("deg_synthetic", "deg_per_perturbation_synthetic")
    ):
        deg_modes.append("synthetic")
    cache_tag = f"_subset{max_perts}" if max_perts is not None else None
    for mode in deg_modes:
        try:
            cache_path = deg_scanpy.compute_deg_cache(
                adata=adata_deg,
                dataset_name=dataset_name,
                results_root=results_root,
                perturbation_key="condition",
                mode=mode,
                min_cells=4,
                perturbations=perturbations,
                cache_tag=cache_tag,
            )
            try:
                deg_df = deg_scanpy.load_deg_cache(cache_path)
            except Exception:
                cache_path = deg_scanpy.compute_deg_cache(
                    adata=adata_deg,
                    dataset_name=dataset_name,
                    results_root=results_root,
                    perturbation_key="condition",
                    mode=mode,
                    min_cells=4,
                    force=True,
                    perturbations=perturbations,
                    cache_tag=cache_tag,
                )
                deg_df = deg_scanpy.load_deg_cache(cache_path)
            metadata_path = cache_path.with_name(
                cache_path.name.replace(".csv", "_metadata.json")
            )
            try:
                metadata = deg_scanpy.load_deg_metadata(metadata_path)
            except Exception:
                metadata = {}
            if metadata.get("mapping_version") != deg_scanpy.MAPPING_VERSION:
                cache_path = deg_scanpy.compute_deg_cache(
                    adata=adata_deg,
                    dataset_name=dataset_name,
                    results_root=results_root,
                    perturbation_key="condition",
                    mode=mode,
                    min_cells=4,
                    force=True,
                    perturbations=perturbations,
                    cache_tag=cache_tag,
                )
                deg_df = deg_scanpy.load_deg_cache(cache_path)
            if mode == "synthetic" and not metadata.get("includes_synthetic_controls", False):
                cache_path = deg_scanpy.compute_deg_cache(
                    adata=adata_deg,
                    dataset_name=dataset_name,
                    results_root=results_root,
                    perturbation_key="condition",
                    mode=mode,
                    min_cells=4,
                    force=True,
                    perturbations=perturbations,
                    cache_tag=cache_tag,
                )
                deg_df = deg_scanpy.load_deg_cache(cache_path)
            if deg_df.empty:
                cache_path = deg_scanpy.compute_deg_cache(
                    adata=adata_deg,
                    dataset_name=dataset_name,
                    results_root=results_root,
                    perturbation_key="condition",
                    mode=mode,
                    min_cells=4,
                    force=True,
                    perturbations=perturbations,
                    cache_tag=cache_tag,
                )
                deg_df = deg_scanpy.load_deg_cache(cache_path)
            global_ranked, per_pert_ranked = deg_scanpy.build_deg_rankings(
                deg_df, perturbations
            )
            rankings[f"deg_{mode}"] = global_ranked
            rankings[f"deg_per_perturbation_{mode}"] = per_pert_ranked
        except ValueError as exc:
            print(f"WARNING: deg_{mode} ranking not available: {exc}")
            rankings[f"deg_{mode}"] = []
            rankings[f"deg_per_perturbation_{mode}"] = {}

    return rankings


def get_filter_gene_indices(
    pds_genes: Sequence[str],
    perturbations: Sequence[str],
    filter_name: str,
    n_genes: int,
    filter_rankings: Dict[str, object],
) -> Tuple[np.ndarray, Optional[Dict[str, List[int]]]]:
    """Resolve gene indices for a given filter using stored rankings."""
    gene_to_idx = {g: i for i, g in enumerate(pds_genes)}
    if filter_name == "highly_variable":
        ranked = filter_rankings.get("highly_variable", [])
        if len(ranked) < n_genes:
            print(f"WARNING: highly_variable ranking has only {len(ranked)} genes.")
        selected = [gene_to_idx[g] for g in ranked[:n_genes] if g in gene_to_idx]
        return np.asarray(selected, dtype=int), None
    if filter_name == "lowest_cv":
        ranked = filter_rankings.get("lowest_cv", [])
        if len(ranked) < n_genes:
            print(f"WARNING: lowest_cv ranking has only {len(ranked)} genes.")
        selected = [gene_to_idx[g] for g in ranked[:n_genes] if g in gene_to_idx]
        return np.asarray(selected, dtype=int), None
    if filter_name == "lowest_cv_per_perturbation":
        per_pert = filter_rankings.get("lowest_cv_per_perturbation", {})
        if not isinstance(per_pert, dict):
            raise ValueError("lowest_cv_per_perturbation rankings missing or invalid.")
        pert_gene_map: Dict[str, List[int]] = {}
        all_indices: List[int] = []
        missing_perts: List[str] = []
        for pert in perturbations:
            ranked = per_pert.get(pert, [])
            indices = [gene_to_idx[g] for g in ranked[:n_genes] if g in gene_to_idx]
            if not indices:
                missing_perts.append(pert)
                continue
            pert_gene_map[pert] = indices
            all_indices.extend(indices)
        if missing_perts:
            print(
                "WARNING: Missing lowest_cv_per_perturbation rankings for "
                f"{len(missing_perts)} perturbations; they will be skipped."
            )
        if not pert_gene_map:
            raise ValueError("No lowest_cv_per_perturbation rankings available.")
        return np.unique(np.asarray(all_indices, dtype=int)), pert_gene_map
    if filter_name in {"deg_control", "deg_synthetic"}:
        ranked = filter_rankings.get(filter_name, [])
        if len(ranked) < n_genes:
            print(f"WARNING: deg ranking has only {len(ranked)} genes.")
        selected = [gene_to_idx[g] for g in ranked[:n_genes] if g in gene_to_idx]
        return np.asarray(selected, dtype=int), None
    if filter_name in {
        "deg_per_perturbation_control",
        "deg_per_perturbation_synthetic",
    }:
        per_pert = filter_rankings.get(filter_name, {})
        if not isinstance(per_pert, dict):
            raise ValueError("deg_per_perturbation rankings missing or invalid.")
        pert_gene_map: Dict[str, List[int]] = {}
        all_indices: List[int] = []
        missing_perts: List[str] = []
        for pert in perturbations:
            ranked = per_pert.get(pert, [])
            indices = [gene_to_idx[g] for g in ranked[:n_genes] if g in gene_to_idx]
            if not indices:
                missing_perts.append(pert)
                continue
            pert_gene_map[pert] = indices
            all_indices.extend(indices)
        if missing_perts:
            print(
                f"WARNING: Missing DEG-per-perturbation rankings for "
                f"{len(missing_perts)} perturbations; they will be skipped."
            )
        if not pert_gene_map:
            raise ValueError("No DEG-per-perturbation rankings available.")
        return np.unique(np.asarray(all_indices, dtype=int)), pert_gene_map
    if filter_name == "top_discriminating":
        return np.array([], dtype=int), None
    if filter_name == "top_discriminating_per_perturbation":
        return np.array([], dtype=int), None
    raise ValueError(f"Unknown filter: {filter_name}")


def resolve_results_path(dataset_path: Path, results_root: Path) -> Path:
    """Resolve the results h5ad path for a dataset."""
    dataset_name = dataset_path.parent.name
    return results_root / dataset_name / "pds_full.h5ad"


def compute_trial_std_of_mean(gene_stds: np.ndarray) -> float:
    """Estimate trial-to-trial std of a mean across genes.

    Assumes per-gene trial noise is independent and uses RMS aggregation.

    Args:
        gene_stds: Per-gene std across trials.

    Returns:
        Estimated std of the mean across genes.
    """
    if gene_stds.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(gene_stds))) / np.sqrt(gene_stds.size))


def compute_summary_global(
    gene_means: np.ndarray,
    gene_stds: np.ndarray,
) -> Dict[str, float]:
    """Compute summary stats for global gene score vector.

    Args:
        gene_means: Mean scores per gene.
        gene_stds: Std across trials per gene.

    Returns:
        Summary statistics including per-pert and trial stds.
    """
    return {
        "mean_score": float(np.mean(gene_means)),
        "std_score_perts": float(np.std(gene_means)),
        "std_score_trials": compute_trial_std_of_mean(gene_stds),
        "min_score": float(np.min(gene_means)),
        "max_score": float(np.max(gene_means)),
    }


def compute_summary_per_pert(
    per_pert_scores: np.ndarray,
    per_pert_stds: np.ndarray,
    pert_gene_map: Dict[str, List[int]],
    perturbations: Sequence[str],
) -> Dict[str, float]:
    """Compute summary stats for per-perturbation gene maps.

    Args:
        per_pert_scores: Mean scores per perturbation and gene.
        per_pert_stds: Std across trials per perturbation and gene.
        pert_gene_map: Mapping perturbation -> gene indices.
        perturbations: Perturbation list to evaluate.

    Returns:
        Summary statistics including per-pert and trial stds.
    """
    perts_mean: List[float] = []
    perts_trial_std: List[float] = []
    for pert_idx, pert in enumerate(perturbations):
        gene_indices = pert_gene_map.get(pert, [])
        if not gene_indices:
            raise ValueError(f"No genes for perturbation '{pert}'.")
        scores = per_pert_scores[pert_idx, gene_indices]
        perts_mean.append(float(np.mean(scores)))
        per_gene_stds = per_pert_stds[pert_idx, gene_indices]
        perts_trial_std.append(compute_trial_std_of_mean(per_gene_stds))
    perts_mean_arr = np.asarray(perts_mean)
    perts_trial_std_arr = np.asarray(perts_trial_std)
    return {
        "mean_score": float(np.mean(perts_mean_arr)),
        "std_score_perts": float(np.std(perts_mean_arr)),
        "std_score_trials": float(np.nanmean(perts_trial_std_arr)),
        "min_score": float(np.min(perts_mean_arr)),
        "max_score": float(np.max(perts_mean_arr)),
    }


def compute_max_n_per_pert(
    per_pert_scores: np.ndarray,
    perturbations: Sequence[str],
    threshold: float,
) -> Tuple[Dict[str, int], float, float]:
    """Compute max n per perturbation where mean top-n score >= threshold.

    Args:
        per_pert_scores: Array of shape (n_perts, n_genes).
        perturbations: Perturbation labels aligned with per_pert_scores rows.
        threshold: Threshold for average PDS.

    Returns:
        Tuple of (per-pert max n mapping, mean max n, std max n).
    """
    if per_pert_scores.size == 0:
        return {}, float("nan"), float("nan")
    max_n_map: Dict[str, int] = {}
    for pert_idx, pert in enumerate(perturbations):
        sorted_scores = np.sort(per_pert_scores[pert_idx])[::-1]
        cumulative = np.cumsum(sorted_scores)
        avg_scores = cumulative / (np.arange(len(sorted_scores)) + 1)
        valid = np.where(avg_scores >= threshold)[0]
        max_n_map[pert] = int(valid[-1] + 1) if valid.size > 0 else 0
    max_values = np.asarray(list(max_n_map.values()), dtype=float)
    return max_n_map, float(np.mean(max_values)), float(np.std(max_values))


def write_filter_outputs(
    output_dir: Path,
    metric_name: str,
    perturbations: Sequence[str],
    per_pert_scores: np.ndarray,
    summary: Dict[str, float],
    gene_names: Sequence[str],
    top_n_per_pert: Optional[int],
) -> None:
    """Write per-metric outputs for a filter."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "trial_summary.json"
    with summary_path.open("w") as handle:
        json.dump({"metric": metric_name, "summary": summary}, handle, indent=2)

    per_pert_rows: List[Dict[str, object]] = []
    for pert_idx, pert in enumerate(perturbations):
        scores = per_pert_scores[pert_idx]
        order = np.argsort(scores)[::-1]
        if top_n_per_pert is not None:
            order = order[:top_n_per_pert]
        for rank, gene_idx in enumerate(order, start=1):
            per_pert_rows.append(
                {
                    "Perturbation": pert,
                    "Rank": rank,
                    "Gene": str(gene_names[gene_idx]),
                    "Mean_PDS": float(scores[gene_idx]),
                }
            )
    pd.DataFrame(per_pert_rows).to_csv(output_dir / "per_pert_top_genes.csv", index=False)


def aggregate_dataset(
    dataset_path: Path,
    output_root: Path,
    n_genes: int,
    filters: Sequence[str],
    top_n_per_pert: Optional[int],
    max_perts: Optional[int],
) -> None:
    """Aggregate filter-based scores for a dataset."""
    results_root = ROOT / "analyses" / "perturbation_discrimination" / "results"
    dataset_name = dataset_path.parent.name
    print(f"Starting dataset: {dataset_name}")
    result_path = resolve_results_path(dataset_path, results_root)
    pds_full = load_pds_full(result_path)
    adata = sc.read_h5ad(dataset_path)
    metric_names = list(pds_full["metrics"])
    perturbations = list(pds_full["perturbations"])
    scores_mean = np.asarray(pds_full["scores_mean"])
    scores_std = np.asarray(pds_full["scores_std"])
    genes = list(pds_full["genes"])
    if max_perts is not None and max_perts < len(perturbations):
        print(
            f"WARNING: Limiting perturbations to first {max_perts} "
            "for debugging."
        )
        perturbations = perturbations[:max_perts]
        scores_mean = scores_mean[:, :max_perts, :]
        scores_std = scores_std[:, :max_perts, :]
    print(f"[{dataset_name}] Computing filter rankings...")
    filter_rankings = compute_filter_rankings(
        adata=adata,
        dataset_name=dataset_name,
        perturbations=perturbations,
        results_root=results_root,
        filters=filters,
        max_perts=max_perts,
    )

    n_metrics, n_perts, _ = scores_mean.shape
    gene_means = scores_mean.mean(axis=1)

    print(
        f"Aggregating {dataset_name}: metrics={n_metrics}, perts={n_perts}, "
        f"genes={len(genes)}, filters={len(filters)}"
    )
    suffix = f"_subset{max_perts}" if max_perts is not None else ""
    dataset_dir = output_root / dataset_name / f"aggregates{suffix}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    deg_per_pert_summaries: Dict[Tuple[str, str], float] = {}
    top_disc_per_pert_summaries: Dict[str, float] = {}

    for filter_name in filters:
        print(f"[{dataset_name}] Filter: {filter_name}")
        filter_dir = dataset_dir / filter_name
        filter_dir.mkdir(parents=True, exist_ok=True)
        summary_rows: List[Dict[str, object]] = []

        try:
            gene_indices, pert_gene_map = get_filter_gene_indices(
                genes, perturbations, filter_name, n_genes, filter_rankings
            )
        except ValueError as exc:
            print(f"WARNING: Skipping filter '{filter_name}' due to: {exc}")
            continue
        if gene_indices.size == 0 and filter_name not in {
            "top_discriminating",
            "top_discriminating_per_perturbation",
        }:
            print(f"WARNING: No genes for filter '{filter_name}'. Skipping.")
            continue

        for metric_idx, metric_name in enumerate(metric_names):
            print(f"[{dataset_name}]   Metric: {metric_name}")
            per_pert_scores = scores_mean[metric_idx]
            per_pert_stds = scores_std[metric_idx]

            if filter_name == "top_discriminating":
                order = np.argsort(gene_means[metric_idx])[::-1]
                gene_indices = order[:n_genes]
                pert_gene_map = None

            if filter_name == "top_discriminating_per_perturbation":
                pert_gene_map = {}
                for pert_idx, pert in enumerate(perturbations):
                    order = np.argsort(per_pert_scores[pert_idx])[::-1]
                    pert_gene_map[pert] = order[:n_genes].tolist()

            if pert_gene_map is None:
                selected_scores = gene_means[metric_idx, gene_indices]
                selected_stds = scores_std[metric_idx].mean(axis=0)[gene_indices]
                summary = compute_summary_global(selected_scores, selected_stds)
            else:
                perts_with_genes = [p for p in perturbations if p in pert_gene_map]
                summary = compute_summary_per_pert(
                    per_pert_scores, per_pert_stds, pert_gene_map, perts_with_genes
                )

            summary_rows.append(
                {
                    "Metric": metric_name,
                    "Mean_Score": summary["mean_score"],
                    "Std_Score_Perts": summary["std_score_perts"],
                    "Std_Score_Trials": summary["std_score_trials"],
                    "Min_Score": summary["min_score"],
                    "Max_Score": summary["max_score"],
                }
            )

            metric_dir = filter_dir / metric_name
            write_filter_outputs(
                metric_dir,
                metric_name,
                perturbations,
                per_pert_scores,
                summary,
                genes,
                top_n_per_pert,
            )

            if filter_name in {
                "deg_per_perturbation_control",
                "deg_per_perturbation_synthetic",
            }:
                deg_per_pert_summaries[(filter_name, metric_name)] = summary["mean_score"]
            if filter_name == "top_discriminating_per_perturbation":
                top_disc_per_pert_summaries[metric_name] = summary["mean_score"]

        pd.DataFrame(summary_rows).to_csv(filter_dir / "summary.csv", index=False)

    # Sanity check: top discriminating per pert should dominate deg per pert
    for metric_name in metric_names:
        for deg_filter in (
            "deg_per_perturbation_control",
            "deg_per_perturbation_synthetic",
        ):
            deg_key = (deg_filter, metric_name)
            if deg_key in deg_per_pert_summaries and metric_name in top_disc_per_pert_summaries:
                if metric_name != "Energy_Distance":
                    continue
                if top_disc_per_pert_summaries[metric_name] < deg_per_pert_summaries[deg_key]:
                    print(
                        f"WARNING: top_discriminating_per_perturbation < {deg_filter} "
                        f"for {dataset_name}, metric {metric_name}."
                    )

    if scores_mean.shape[2] != len(genes):
        raise ValueError("pds_full gene dimension does not match gene list.")

    thresholds_rows: List[Dict[str, object]] = []
    thresholds_per_pert_rows: List[Dict[str, object]] = []
    thresholds_summary: Dict[str, Dict[str, object]] = {}
    for metric_idx, metric_name in enumerate(metric_names):
        per_pert_scores = scores_mean[metric_idx]
        max_n_per_pert, mean_n, std_n = compute_max_n_per_pert(
            per_pert_scores, perturbations, threshold=0.8
        )
        for pert, max_n in max_n_per_pert.items():
            thresholds_per_pert_rows.append(
                {"Metric": metric_name, "Perturbation": pert, "Max_N_Above_0p8": max_n}
            )
        thresholds_summary[metric_name] = {
            "Per_Pert_Max_N_Above_0p8": max_n_per_pert,
            "Mean_Max_N_Above_0p8": float(mean_n),
            "Std_Max_N_Above_0p8": float(std_n),
        }
        thresholds_rows.append(
            {
                "Metric": metric_name,
                "Mean_Max_N_Above_0p8": float(mean_n),
                "Std_Max_N_Above_0p8": float(std_n),
            }
        )

    pd.DataFrame(thresholds_rows).to_csv(dataset_dir / "thresholds.csv", index=False)
    if thresholds_per_pert_rows:
        pd.DataFrame(thresholds_per_pert_rows).to_csv(
            dataset_dir / "thresholds_per_perturbation.csv", index=False
        )
    with (dataset_dir / "thresholds_summary.json").open("w") as handle:
        json.dump({"Thresholds": thresholds_summary}, handle, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Aggregate filter scores from PDS.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["wessels23", "norman19", "adamson16", "frangieh21", "kaden25fibroblast", "nadig25hepg2", "replogle20"],
        help="Datasets to process (default: all datasets).",
    )
    parser.add_argument("adata_path", nargs="?", help="Path to a single h5ad file.")
    parser.add_argument("--n-genes", type=int, default=1000, help="Genes per filter.")
    parser.add_argument(
        "--filters", nargs="+", default=FILTERS, help="Filters to compute."
    )
    parser.add_argument(
        "--top-n-per-pert",
        type=int,
        default=None,
        help="Limit per-pert gene output (default: all genes).",
    )
    parser.add_argument(
        "--max-perts",
        type=int,
        default=None,
        help="Limit perturbations for debugging (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for aggregation."""
    args = parse_args()
    if args.datasets and args.adata_path:
        raise ValueError("Specify either --datasets or a single adata_path.")

    if args.adata_path:
        paths = [Path(args.adata_path)]
    elif args.datasets:
        paths = [load_dataset_path(ROOT, name) for name in args.datasets]
    else:
        paths = load_dataset_paths(ROOT)
    for path in paths:
        aggregate_dataset(
            dataset_path=path,
            output_root=ROOT / "analyses" / "perturbation_discrimination" / "results",
            n_genes=args.n_genes,
            filters=args.filters,
            top_n_per_pert=args.top_n_per_pert,
            max_perts=args.max_perts,
        )


if __name__ == "__main__":
    main()
