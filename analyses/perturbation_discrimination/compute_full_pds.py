"""
Compute full PDS tensors (metrics × perturbations × genes) and store in results.

This script computes PDS for all genes and all non-control perturbations,
for all configured metrics, using shared splits per trial. It also computes
an equalized-cells baseline for comparison. The output only contains PDS
scores and minimal metadata (no filter rankings).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analyses.perturbation_discrimination import pds_core  # noqa: E402



METRICS: Dict[str, pds_core.DistanceMetric] = {
    "MAE_Mean": pds_core.mae_on_mean,
    "MAE_Median": pds_core.mae_on_median,
    # "MAE_P95": lambda a, b: pds_core.mae_on_quantile(a, b, 0.95),
    "Energy_Distance": pds_core.energy_distance_1d,
    "Wasserstein_1D": pds_core.wasserstein_distance_1d,
}

_WORKER_ADATA: Optional[sc.AnnData] = None
_WORKER_PERTURBATIONS: Optional[Sequence[str]] = None
_WORKER_HALF_A: Optional[Dict[str, np.ndarray]] = None
_WORKER_HALF_B: Optional[Dict[str, np.ndarray]] = None
_WORKER_METRICS: Optional[Sequence[pds_core.DistanceMetric]] = None
_WORKER_COMPARISON_SETS: Optional[Dict[str, Sequence[str]]] = None
_WORKER_TIE_BREAK_SEED: Optional[int] = None


def _set_worker_state(
    adata: sc.AnnData,
    perturbations: Sequence[str],
    half_a: Dict[str, np.ndarray],
    half_b: Dict[str, np.ndarray],
    metric_funcs: Sequence[pds_core.DistanceMetric],
    comparison_sets: Optional[Dict[str, Sequence[str]]] = None,
    tie_break_seed: Optional[int] = None,
) -> None:
    """Set global worker state for forked processes."""
    global _WORKER_ADATA, _WORKER_PERTURBATIONS, _WORKER_HALF_A, _WORKER_HALF_B, _WORKER_METRICS
    global _WORKER_COMPARISON_SETS, _WORKER_TIE_BREAK_SEED
    _WORKER_ADATA = adata
    _WORKER_PERTURBATIONS = perturbations
    _WORKER_HALF_A = half_a
    _WORKER_HALF_B = half_b
    _WORKER_METRICS = metric_funcs
    _WORKER_COMPARISON_SETS = comparison_sets
    _WORKER_TIE_BREAK_SEED = tie_break_seed


def _compute_gene_scores(gene_idx: int) -> np.ndarray:
    """Compute metric×pert scores for one gene in a worker."""
    if _WORKER_ADATA is None or _WORKER_PERTURBATIONS is None:
        raise RuntimeError("Worker state not initialized.")
    if gene_idx == 0:
        print(f"Computing PDS for {len(_WORKER_METRICS)} metrics and {len(_WORKER_PERTURBATIONS)} perturbations.")
    return pds_core.compute_pds_for_gene_multi_metric(
        gene_idx=gene_idx,
        adata=_WORKER_ADATA,
        perturbations=_WORKER_PERTURBATIONS,
        half_a_indices=_WORKER_HALF_A or {},
        half_b_indices=_WORKER_HALF_B or {},
        metric_funcs=_WORKER_METRICS or [],
        comparison_sets=_WORKER_COMPARISON_SETS,
        tie_break_seed=_WORKER_TIE_BREAK_SEED,
    )


def load_dataset_paths(root: Path) -> List[Path]:
    """Load dataset paths from dataset configs.

    Args:
        root: Project root path.

    Returns:
        List of dataset h5ad paths.
    """
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


def compute_dataset_stats(
    adata: sc.AnnData, perturbations: Sequence[str], perturbation_key: str
) -> Dict[str, float]:
    """Compute dataset-level stats for metadata.

    Args:
        adata: Annotated data matrix.
        perturbations: Perturbations included in PDS.
        perturbation_key: Column name for perturbations.

    Returns:
        Dictionary with dataset statistics.
    """
    counts = adata.obs[perturbation_key].value_counts()
    counts = counts.loc[list(perturbations)]
    return {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_perturbations": int(len(perturbations)),
        "avg_cells_per_pert": float(counts.mean()) if not counts.empty else 0.0,
        "min_cells_per_pert": int(counts.min()) if not counts.empty else 0,
        "max_cells_per_pert": int(counts.max()) if not counts.empty else 0,
    }


def get_non_control_perturbations(
    adata: sc.AnnData, perturbation_key: str
) -> List[str]:
    """Return non-control perturbations.

    Args:
        adata: Annotated data matrix.
        perturbation_key: Column name for perturbations.

    Returns:
        Sorted list of non-control perturbation labels.
    """
    perts_raw = adata.obs[perturbation_key]
    valid_mask = ~perts_raw.isna()
    perts = perts_raw[valid_mask].astype(str)
    non_control_mask = ~perts.str.lower().str.contains("control|ctrl", regex=True)
    non_control_mask &= perts.str.strip() != "*"
    return sorted(perts[non_control_mask].unique())


def build_splits_all_cells(
    adata: sc.AnnData,
    perturbations: Sequence[str],
    seed: int,
    perturbation_key: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Build half splits using all available cells per perturbation."""
    rng = np.random.default_rng(seed)
    half_a: Dict[str, np.ndarray] = {}
    half_b: Dict[str, np.ndarray] = {}
    for pert in perturbations:
        indices = np.where(adata.obs[perturbation_key] == pert)[0]
        if indices.size < 2:
            raise ValueError(
                f"Perturbation '{pert}' has {indices.size} cells; need at least 2."
            )
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        mid = shuffled.size // 2
        half_a[pert] = shuffled[:mid]
        half_b[pert] = shuffled[mid:]
    return half_a, half_b


def build_splits_equalized(
    adata: sc.AnnData,
    perturbations: Sequence[str],
    n_cells_per_pert: int,
    seed: int,
    perturbation_key: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Build half splits using equalized cell counts per perturbation."""
    sampled = pds_core.sample_cells_per_perturbation(
        adata=adata,
        perturbations=perturbations,
        n_cells=n_cells_per_pert,
        seed=seed,
        perturbation_key=perturbation_key,
    )
    half_a: Dict[str, np.ndarray] = {}
    half_b: Dict[str, np.ndarray] = {}
    for pert, indices in sampled.items():
        split_a, split_b = pds_core.split_indices_half(indices, seed=seed + 13)
        half_a[pert] = split_a
        half_b[pert] = split_b
    return half_a, half_b


def compute_full_pds_for_dataset(
    adata: sc.AnnData,
    perturbations: Sequence[str],
    metric_names: Sequence[str],
    metric_funcs: Sequence[pds_core.DistanceMetric],
    gene_indices: Sequence[int],
    n_trials: int,
    seed: int,
    equalized: bool,
    n_jobs: int,
    comparison_sets: Optional[Dict[str, Sequence[str]]] = None,
    n_cells_per_pert_override: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute mean/std PDS tensors for one dataset.

    Args:
        adata: Annotated data matrix.
        perturbations: Perturbation labels.
        metric_names: Metric names.
        metric_funcs: Metric functions.
        n_trials: Number of trials.
        seed: Base random seed.
        equalized: Whether to use equalized cell counts per perturbation.
        comparison_sets: Optional mapping from perturbation to comparison set.

    Returns:
        Tuple of (mean_scores, std_scores, equalized_cells).
    """
    n_metrics = len(metric_names)
    n_perts = len(perturbations)
    n_genes = len(gene_indices)
    dtype = np.float32

    scores_sum = np.zeros((n_metrics, n_perts, n_genes), dtype=dtype)
    scores_sq_sum = np.zeros((n_metrics, n_perts, n_genes), dtype=dtype)

    counts = adata.obs[pds_core.resolve_perturbation_key(adata, None)].value_counts()
    min_cells = int(counts.loc[list(perturbations)].min())
    equalized_cells = int(n_cells_per_pert_override) if n_cells_per_pert_override is not None else (
        min_cells if equalized else 0
    )

    for trial in tqdm(range(n_trials), desc="Trials", mininterval=2.0):
        if n_cells_per_pert_override is not None:
            half_a, half_b = build_splits_equalized(
                adata=adata,
                perturbations=perturbations,
                n_cells_per_pert=int(n_cells_per_pert_override),
                seed=seed + trial,
                perturbation_key=pds_core.resolve_perturbation_key(adata, None),
            )
        elif equalized:
            half_a, half_b = build_splits_equalized(
                adata=adata,
                perturbations=perturbations,
                n_cells_per_pert=min_cells,
                seed=seed + trial,
                perturbation_key=pds_core.resolve_perturbation_key(adata, None),
            )
        else:
            half_a, half_b = build_splits_all_cells(
                adata=adata,
                perturbations=perturbations,
                seed=seed + trial,
                perturbation_key=pds_core.resolve_perturbation_key(adata, None),
            )

        trial_start = time.perf_counter()
        print(f"Trial {trial + 1}/{n_trials}: scoring {len(gene_indices)} genes.")
        tie_break_seed = seed + trial
        if n_jobs > 1:
            if mp.get_start_method(allow_none=True) != "fork":
                raise RuntimeError("Multiprocessing requires fork start method on this platform.")
            _set_worker_state(
                adata,
                perturbations,
                half_a,
                half_b,
                metric_funcs,
                comparison_sets,
                tie_break_seed=tie_break_seed,
            )
            # Larger chunksize reduces overhead; progress is logged by gene count.
            chunksize = max(1, len(gene_indices) // n_jobs)
            print(f"Multiprocessing chunksize: {chunksize}")
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=n_jobs) as pool:
                next_log = max(1, len(gene_indices) // 10)
                for idx, scores in enumerate(
                    pool.imap(_compute_gene_scores, map(int, gene_indices), chunksize=chunksize)
                ):
                    scores_sum[:, :, idx] += scores.astype(dtype)
                    scores_sq_sum[:, :, idx] += (scores.astype(dtype) ** 2)
                    if (idx + 1) % next_log == 0 or (idx + 1) == len(gene_indices):
                        print(f"Processed {idx + 1}/{len(gene_indices)} genes (mp).")
        else:
            for idx, gene_idx in enumerate(
                tqdm(gene_indices, desc="Scoring genes", leave=False, mininterval=2.0)
            ):
                scores = pds_core.compute_pds_for_gene_multi_metric(
                    gene_idx=int(gene_idx),
                    adata=adata,
                    perturbations=perturbations,
                    half_a_indices=half_a,
                    half_b_indices=half_b,
                    metric_funcs=metric_funcs,
                    comparison_sets=comparison_sets,
                    tie_break_seed=tie_break_seed,
                )
                scores_sum[:, :, idx] += scores.astype(dtype)
                scores_sq_sum[:, :, idx] += (scores.astype(dtype) ** 2)
        trial_elapsed = time.perf_counter() - trial_start
        print(f"Trial {trial + 1}/{n_trials} done in {trial_elapsed:.1f}s.")

    mean_scores = scores_sum / float(n_trials)
    var_scores = scores_sq_sum / float(n_trials) - mean_scores**2
    std_scores = np.sqrt(np.maximum(var_scores, 0.0)).astype(dtype)
    return mean_scores, std_scores, equalized_cells


def update_adata_with_pds(
    adata: ad.AnnData,
    perturbations: Sequence[str],
    metric_names: Sequence[str],
    mean_scores: np.ndarray,
    std_scores: np.ndarray,
    genes: Sequence[str],
    equalized: bool,
    equalized_cells: int,
    n_trials: int,
    dataset_stats: Dict[str, float],
) -> None:
    """Store PDS tensors and metadata in adata.uns.

    Args:
        adata: Results AnnData object to populate.
        perturbations: Perturbation labels.
        metric_names: Metric names.
        mean_scores: Mean PDS tensor.
        std_scores: Std PDS tensor.
        genes: Gene names in score order.
        equalized: Whether this stores equalized-cell results.
        equalized_cells: Equalized cell count (0 if not used).
        n_trials: Number of trials.
        dataset_stats: Dataset-level metadata.
    """
    key = "pds_full_equalized" if equalized else "pds_full"
    adata.uns[key] = {
        "metrics": list(metric_names),
        "perturbations": list(perturbations),
        "genes": list(genes),
        "scores_mean": mean_scores,
        "scores_std": std_scores,
        "n_trials": int(n_trials),
        "scale": "[-1,1]",
        "equalized_cells": int(equalized_cells),
        "dataset_stats": dataset_stats,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compute full PDS tensors.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["wessels23", "norman19", "adamson16", "frangieh21", "kaden25fibroblast", "nadig25hepg2", "replogle22k562"],
        help="Datasets to process (default: all datasets).",
    )
    parser.add_argument("adata_path", nargs="?", help="Path to a single h5ad file.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing PDS.")
    parser.add_argument("--n-trials", type=int, default=3, help="Number of trials.")
    parser.add_argument(
        "--cells-per-comparison",
        type=int,
        default=None,
        help="If set, sample exactly this many cells per split-half via equalized sampling.",
    )
    parser.add_argument(
        "--equalized-baseline",
        action="store_true",
        help="Also compute equalized-cell baseline.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(METRICS.keys()),
        help="Metrics to compute (default: all).",
    )
    parser.add_argument(
        "--max-genes",
        type=int,
        default=None,
        help="Limit number of genes for debugging.",
    )
    parser.add_argument(
        "--output-name",
        default="pds_full.h5ad",
        help="Output filename to write inside each dataset results directory.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Number of worker processes for gene scoring (default: 8).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for full PDS computation."""
    args = parse_args()
    if args.datasets and args.adata_path:
        raise ValueError("Specify either --datasets or a single adata_path.")

    metric_names = [m for m in args.metrics if m in METRICS]
    if not metric_names:
        raise ValueError("No valid metrics selected.")
    metric_funcs = [METRICS[m] for m in metric_names]

    if args.adata_path:
        paths = [Path(args.adata_path)]
    elif args.datasets:
        paths = [load_dataset_path(ROOT, name) for name in args.datasets]
    else:
        paths = load_dataset_paths(ROOT)
    for path in paths:
        print(f"\nProcessing {path}...")
        adata = sc.read_h5ad(path)
        pert_key = pds_core.resolve_perturbation_key(adata, None)
        perturbations = get_non_control_perturbations(adata, pert_key)
        if len(perturbations) < 2:
            raise ValueError("Need at least 2 perturbations for PDS.")
        comparison_sets: Optional[Dict[str, Sequence[str]]] = None
        if len(perturbations) > 200:
            rng = np.random.default_rng(42)
            perturbations_list = list(perturbations)
            comparison_sets = {}
            for pert in perturbations_list:
                others = [p for p in perturbations_list if p != pert]
                sample_size = min(200, len(others))
                sampled = rng.choice(others, size=sample_size, replace=False).tolist()
                comparison_sets[pert] = [pert] + sampled

        dataset_name = path.parent.name
        results_dir = ROOT / "analyses" / "perturbation_discrimination" / "results" / dataset_name
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / args.output_name
        if results_path.exists() and not args.force:
            print(f"{args.output_name} already present; skipping (use --force to overwrite).")
            continue

        if args.max_genes is not None and args.max_genes > 0:
            gene_indices = list(range(min(args.max_genes, adata.shape[1])))
        else:
            gene_indices = list(range(adata.shape[1]))
        genes = [str(adata.var_names[i]) for i in gene_indices]

        dataset_stats = compute_dataset_stats(adata, perturbations, pert_key)
        print(
            "Dataset overview: "
            f"n_obs={dataset_stats['n_obs']}, "
            f"n_vars={dataset_stats['n_vars']}, "
            f"n_perts={dataset_stats['n_perturbations']}, "
            f"avg_cells_per_pert={dataset_stats['avg_cells_per_pert']:.2f}, "
            f"min_cells_per_pert={dataset_stats['min_cells_per_pert']}, "
            f"max_cells_per_pert={dataset_stats['max_cells_per_pert']}"
        )
        mean_scores, std_scores, equalized_cells = compute_full_pds_for_dataset(
            adata=adata,
            perturbations=perturbations,
            metric_names=metric_names,
            metric_funcs=metric_funcs,
            gene_indices=gene_indices,
            n_trials=args.n_trials,
            seed=42,
            equalized=False,
            n_jobs=args.n_jobs,
            comparison_sets=comparison_sets,
            n_cells_per_pert_override=(
                2 * int(args.cells_per_comparison)
                if args.cells_per_comparison is not None
                else None
            ),
        )
        result_adata = ad.AnnData(X=sp.csr_matrix((0, len(genes))))
        result_adata.var_names = genes
        update_adata_with_pds(
            adata=result_adata,
            perturbations=perturbations,
            metric_names=metric_names,
            mean_scores=mean_scores,
            std_scores=std_scores,
            genes=genes,
            equalized=False,
            equalized_cells=equalized_cells,
            n_trials=args.n_trials,
            dataset_stats=dataset_stats,
        )

        if args.equalized_baseline:
            mean_eq, std_eq, equalized_cells = compute_full_pds_for_dataset(
                adata=adata,
                perturbations=perturbations,
                metric_names=metric_names,
                metric_funcs=metric_funcs,
                gene_indices=gene_indices,
                n_trials=args.n_trials,
                seed=42,
                equalized=True,
                n_jobs=args.n_jobs,
                comparison_sets=comparison_sets,
            )
            update_adata_with_pds(
                adata=result_adata,
                perturbations=perturbations,
                metric_names=metric_names,
                mean_scores=mean_eq,
                std_scores=std_eq,
                genes=genes,
                equalized=True,
                equalized_cells=equalized_cells,
                n_trials=args.n_trials,
                dataset_stats=dataset_stats,
            )

        result_adata.write_h5ad(results_path)
        print(f"Saved full PDS to {results_path}")


if __name__ == "__main__":
    main()
