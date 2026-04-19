#!/usr/bin/env python
"""
Add gene-wise perturbation discrimination scores (PDS) to AnnData objects.

This script extracts energy-distance PDS from the results
`analyses/perturbation_discrimination/results/<dataset>/pds_full.h5ad`, stores
mean/std scores in adata.uns, and records the per-perturbation maximal number
of top genes whose average PDS exceeds 0.8 (plus mean/std across perturbations).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import sys

import anndata
import numpy as np
import scanpy as sc
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))




def load_pds_full_from_results(
    adata_path: Path,
) -> Dict[str, object]:
    """Load pds_full from results h5ad.

    Args:
        adata_path: Path to the original dataset h5ad.

    Returns:
        pds_full dictionary from results.
    """
    dataset_name = adata_path.parent.name
    results_path = (
        ROOT
        / "analyses"
        / "perturbation_discrimination"
        / "results"
        / dataset_name
        / "pds_full.h5ad"
    )
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")
    results_adata = anndata.read_h5ad(results_path)
    if "pds_full" not in results_adata.uns:
        raise ValueError("Missing pds_full in results h5ad.")
    return results_adata.uns["pds_full"]


def extract_energy_pds(
    pds_full: Dict[str, object],
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """Extract energy PDS mean/std from pds_full dict.

    Args:
        pds_full: PDS metadata dictionary.

    Returns:
        Tuple of (perturbations, genes, mean_scores, std_scores).
    """
    metrics = list(pds_full.get("metrics", []))
    if "Energy_Distance" not in metrics:
        raise ValueError("Energy_Distance metric missing from pds_full.")
    metric_idx = metrics.index("Energy_Distance")
    mean_scores = np.asarray(pds_full["scores_mean"][metric_idx])
    std_scores = np.asarray(pds_full["scores_std"][metric_idx])
    return (
        list(pds_full["perturbations"]),
        list(pds_full["genes"]),
        mean_scores,
        std_scores,
    )


def max_n_above_threshold(mean_scores: np.ndarray, threshold: float) -> List[int]:
    """Compute maximal n for each perturbation with mean PDS above threshold.

    Args:
        mean_scores: Matrix of mean PDS scores (n_perts, n_genes).
        threshold: Threshold for mean PDS.

    Returns:
        List of maximal n values for each perturbation.
    """
    max_ns: List[int] = []
    for row in mean_scores:
        sorted_scores = np.sort(row)[::-1]
        cumulative = np.cumsum(sorted_scores)
        avg_scores = cumulative / (np.arange(len(sorted_scores)) + 1)
        valid = np.where(avg_scores >= threshold)[0]
        max_ns.append(int(valid[-1] + 1) if valid.size > 0 else 0)
    return max_ns


def add_pds_to_adata(
    adata_path: Path,
    force: bool = False,
) -> None:
    """Add gene-wise energy PDS to an AnnData file in-place.

    Args:
        adata_path: Path to the AnnData file.
        force: If True, recompute even if PDS is already present.
    """
    print(f"Loading AnnData from {adata_path}...")
    adata_full = sc.read_h5ad(adata_path)
    if "pds_energy_1d_mean" in adata_full.uns and not force:
        print(f"Skipping {adata_path} (PDS already present)")
        return

    pds_full = load_pds_full_from_results(adata_path)
    perturbations, genes, mean_scores, std_scores = extract_energy_pds(pds_full)

    max_ns = max_n_above_threshold(mean_scores, threshold=0.8)
    max_ns_arr = np.asarray(max_ns, dtype=float)
    mean_max_n = float(np.mean(max_ns_arr)) if max_ns_arr.size > 0 else 0.0
    std_max_n = float(np.std(max_ns_arr)) if max_ns_arr.size > 0 else 0.0

    adata_full.uns["pds_energy_1d_mean"] = {
        "perturbations": perturbations,
        "genes": genes,
        "scores": mean_scores.tolist(),
    }
    adata_full.uns["pds_energy_1d_std"] = {
        "perturbations": perturbations,
        "genes": genes,
        "scores": std_scores.tolist(),
    }
    adata_full.uns["pds_energy_1d_threshold"] = {
        "threshold": 0.8,
        "max_n_per_perturbation": dict(zip(perturbations, max_ns)),
        "mean_max_n_across_perturbations": mean_max_n,
        "std_max_n_across_perturbations": std_max_n,
    }

    temp_path = adata_path.with_suffix(".with_pds.h5ad")
    adata_full.write_h5ad(temp_path)
    temp_path.replace(adata_path)
    print(f"Updated PDS in {adata_path}")


def load_dataset_paths(root: Path) -> List[Path]:
    """Load dataset paths from dataset configs.

    Args:
        root: Project root path.

    Returns:
        List of dataset h5ad paths.
    """
    config_dir = root / "cellsimbench" / "configs" / "dataset"
    paths: List[Path] = []
    for config_path in config_dir.glob("*.yaml"):
        with config_path.open("r") as handle:
            config = yaml.safe_load(handle)
        data_path = config.get("data_path")
        if data_path:
            path = (root / data_path).resolve()
            if path.exists():
                paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Add gene-wise PDS to AnnData files.")
    parser.add_argument("adata_path", nargs="?", help="Path to a single h5ad file.")
    parser.add_argument("--all", action="store_true", help="Process all datasets.")
    parser.add_argument("--force", action="store_true", help="Force recomputation.")
    return parser.parse_args()


def main() -> None:
    """Entry point for adding PDS to AnnData files."""
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    if args.all and args.adata_path:
        raise ValueError("Specify either --all or a single adata_path.")
    if not args.all and not args.adata_path:
        raise ValueError("Specify --all or a single adata_path.")

    if args.all:
        for path in load_dataset_paths(root):
            add_pds_to_adata(adata_path=path, force=args.force)
    else:
        add_pds_to_adata(
            adata_path=Path(args.adata_path),
            force=args.force,
        )


if __name__ == "__main__":
    main()
