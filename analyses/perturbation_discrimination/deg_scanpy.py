"""
Scanpy-based differential expression (DEG) computation for PDS filtering.

This module computes perturbation-specific DEG scores using scanpy's
rank_genes_groups, caches results per dataset, and exposes rankings for
downstream filtering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from pandas.errors import PerformanceWarning
import warnings


CONTROL_LABEL_TOKENS = ("control", "ctrl")
SYNTHETIC_CONTROL_LABEL = "ctrl_synthetic_mean"
MAPPING_VERSION = 2


@dataclass(frozen=True)
class DegCachePaths:
    """Paths for DEG cache outputs."""

    csv_path: Path
    metadata_path: Path


def resolve_control_conditions(conditions: Iterable[str]) -> List[str]:
    """Identify control condition labels.

    Args:
        conditions: Condition labels.

    Returns:
        List of labels identified as control conditions.
    """
    controls: List[str] = []
    for cond in conditions:
        cond_lower = str(cond).lower()
        if any(token in cond_lower for token in CONTROL_LABEL_TOKENS):
            controls.append(str(cond))
    return sorted(set(controls))


def add_synthetic_mean_controls(
    adata: anndata.AnnData,
    perturbation_key: str,
    n_controls: int = 100,
    sample_size: int = 100,
    seed: int = 42,
) -> anndata.AnnData:
    """Add synthetic mean controls to an AnnData object.

    Synthetic controls are created by averaging random non-control cells.

    Args:
        adata: Annotated data matrix.
        perturbation_key: Column in adata.obs with perturbation labels.
        n_controls: Number of synthetic control cells to add.
        sample_size: Cells per synthetic control.
        seed: Random seed for sampling.

    Returns:
        AnnData with synthetic mean controls appended.
    """
    rng = np.random.default_rng(seed)
    control_conditions = resolve_control_conditions(adata.obs[perturbation_key].unique())
    non_ctrl_mask = ~adata.obs[perturbation_key].isin(control_conditions)
    non_ctrl_indices = np.where(non_ctrl_mask.to_numpy())[0]
    if non_ctrl_indices.size < sample_size:
        return adata
    n_controls = max(1, min(n_controls, 100))
    synthetic_controls: List[anndata.AnnData] = []
    for i in range(n_controls):
        sampled = rng.choice(non_ctrl_indices, size=sample_size, replace=False)
        mean_expr = adata[sampled].X.mean(axis=0)
        if hasattr(mean_expr, "A1"):
            mean_expr = mean_expr.A1
        obs_df = pd.DataFrame(
            {perturbation_key: SYNTHETIC_CONTROL_LABEL},
            index=[f"synthetic_mean_{i}"],
        )
        synthetic_controls.append(anndata.AnnData(X=mean_expr.reshape(1, -1), obs=obs_df, var=adata.var))
    synthetic_adata = anndata.concat(synthetic_controls, join="outer", index_unique="_")
    combined = anndata.concat([adata, synthetic_adata], join="outer", index_unique="_")
    combined.var = adata.var.copy()
    return combined


def _cache_paths(
    results_root: Path,
    dataset_name: str,
    mode: str,
    cache_tag: Optional[str],
) -> DegCachePaths:
    """Build cache paths for a dataset/mode."""
    output_dir = results_root / dataset_name / "deg_scanpy"
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = cache_tag or ""
    return DegCachePaths(
        csv_path=output_dir / f"deg_{mode}{tag}.csv",
        metadata_path=output_dir / f"deg_{mode}{tag}_metadata.json",
    )


def compute_deg_cache(
    adata: anndata.AnnData,
    dataset_name: str,
    results_root: Path,
    perturbation_key: str,
    mode: str,
    min_cells: int,
    force: bool = False,
    perturbations: Optional[Sequence[str]] = None,
    cache_tag: Optional[str] = None,
) -> Path:
    """Compute or reuse cached scanpy DEG scores.

    Args:
        adata: Annotated data matrix.
        dataset_name: Dataset identifier.
        results_root: Root results directory.
        perturbation_key: Column name for perturbation labels.
        mode: DEG reference mode ('control' or 'synthetic').
        min_cells: Minimum cells per perturbation.
        force: Recompute even if cache exists.
        perturbations: Optional perturbation subset to compute.
        cache_tag: Optional cache tag for filenames.

    Returns:
        Path to cached CSV file.
    """
    if mode not in {"control", "synthetic"}:
        raise ValueError(f"Unknown DEG mode: {mode}")
    paths = _cache_paths(results_root, dataset_name, mode, cache_tag)
    if paths.csv_path.exists() and not force:
        return paths.csv_path

    adata_work = adata.copy()
    print(f"[deg_scanpy] Computing {mode} DEG cache for {dataset_name}...")
    if mode == "synthetic":
        adata_work = add_synthetic_mean_controls(
            adata_work,
            perturbation_key=perturbation_key,
            n_controls=100,
            sample_size=100,
            seed=42,
        )

    all_conditions = list(map(str, adata_work.obs[perturbation_key].unique()))
    control_conditions = resolve_control_conditions(all_conditions)
    if perturbations is None:
        perturbations = [
            cond for cond in all_conditions if cond not in control_conditions
        ]
    perturbations = list(map(str, perturbations))
    obs_set = set(all_conditions)
    mapped: Dict[str, str] = {}
    for pert in perturbations:
        if pert in obs_set:
            mapped[pert] = pert
            continue
        candidate = pert.replace("_", "+")
        if candidate in obs_set:
            mapped[pert] = candidate
            continue
        candidate = pert.replace("+", "_")
        if candidate in obs_set:
            mapped[pert] = candidate
            continue

    pert_counts = adata_work.obs[perturbation_key].value_counts()
    valid_perts = [
        pert
        for pert in perturbations
        if pert in mapped and pert_counts.get(mapped[pert], 0) >= min_cells
    ]
    valid_obs_perts = [mapped[pert] for pert in valid_perts]
    
    # Determine reference for DEG computation
    if mode == "synthetic":
        synthetic_mask = adata_work.obs[perturbation_key] == SYNTHETIC_CONTROL_LABEL
        adata_deg = adata_work[
            adata_work.obs[perturbation_key].isin(valid_obs_perts) | synthetic_mask
        ].copy()
        reference = SYNTHETIC_CONTROL_LABEL  # Compare vs synthetic controls
    elif mode == "control":
        # Include control cells and compare against them
        adata_deg = adata_work[
            adata_work.obs[perturbation_key].isin(valid_obs_perts) | 
            adata_work.obs[perturbation_key].isin(control_conditions)
        ].copy()
        # Use the first control condition as reference
        reference = control_conditions[0] if control_conditions else "rest"
    else:
        adata_deg = adata_work[adata_work.obs[perturbation_key].isin(valid_obs_perts)].copy()
        reference = "rest"
    if not valid_perts:
        pd.DataFrame(columns=["Perturbation", "Gene", "score", "pvalue", "padj"]).to_csv(
            paths.csv_path, index=False
        )
        metadata = {
            "mode": mode,
            "perturbation_key": perturbation_key,
            "min_cells": min_cells,
            "n_perturbations": 0,
        }
        paths.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return paths.csv_path

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PerformanceWarning)
        print(
            f"[deg_scanpy] rank_genes_groups: perts={len(valid_perts)}, "
            f"cells={adata_deg.n_obs}, genes={adata_deg.n_vars}"
        )
        print(f"[deg_scanpy] Using reference='{reference}' for DEG computation")
        sc.tl.rank_genes_groups(
            adata_deg,
            perturbation_key,
            method="t-test_overestim_var",
            reference=reference,
        )
    names_df = pd.DataFrame(adata_deg.uns["rank_genes_groups"]["names"])
    scores_df = pd.DataFrame(adata_deg.uns["rank_genes_groups"]["scores"])
    pvals_df = pd.DataFrame(adata_deg.uns["rank_genes_groups"]["pvals"])
    pvals_adj_df = pd.DataFrame(adata_deg.uns["rank_genes_groups"]["pvals_adj"])

    rows: List[Dict[str, object]] = []
    for pert in valid_perts:
        obs_key = mapped.get(pert)
        if not obs_key or obs_key not in names_df.columns:
            continue
        if obs_key in control_conditions:
            continue
        genes = names_df[obs_key].tolist()
        scores = scores_df[obs_key].tolist()
        pvals = pvals_df[obs_key].tolist()
        pvals_adj = pvals_adj_df[obs_key].tolist()
        for gene, score, pval, padj in zip(genes, scores, pvals, pvals_adj):
            rows.append(
                {
                    "Perturbation": pert,
                    "Gene": str(gene),
                    "score": float(score),
                    "pvalue": float(pval),
                    "padj": float(padj),
                }
            )

    results_df = pd.DataFrame(
        rows, columns=["Perturbation", "Gene", "score", "pvalue", "padj"]
    )
    results_df.to_csv(paths.csv_path, index=False)
    metadata = {
        "mode": mode,
        "perturbation_key": perturbation_key,
        "min_cells": min_cells,
        "n_perturbations": len(valid_perts),
        "n_total_rows": len(results_df),
        "includes_synthetic_controls": bool(mode == "synthetic"),
        "mapping_version": MAPPING_VERSION,
        "mapped_perturbations": len(mapped),
        "unmapped_perturbations": len(perturbations) - len(mapped),
        "reference": reference,
        "control_conditions": control_conditions if mode == "control" else [],
    }
    paths.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths.csv_path


def load_deg_metadata(metadata_path: Path) -> Dict[str, object]:
    """Load cached DEG metadata.

    Args:
        metadata_path: Path to metadata JSON.

    Returns:
        Parsed metadata dictionary.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing DEG metadata: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_deg_cache(csv_path: Path) -> pd.DataFrame:
    """Load cached DEG results.

    Args:
        csv_path: Path to cached DEG CSV.

    Returns:
        DataFrame with DEG statistics.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing DEG cache: {csv_path}")
    return pd.read_csv(csv_path)


def build_deg_rankings(
    deg_df: pd.DataFrame,
    perturbations: Sequence[str],
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Build aggregated and per-perturbation DEG rankings.

    Rankings are based on absolute scanpy scores.

    Args:
        deg_df: Cached DEG DataFrame.
        perturbations: Perturbations to include.

    Returns:
        Tuple of (global ranked genes, per-perturbation ranked genes).
    """
    if deg_df.empty:
        return [], {}
    deg_df = deg_df.copy()
    deg_df["abs_score"] = deg_df["score"].abs()
    per_pert: Dict[str, List[str]] = {}
    for pert in perturbations:
        subset = deg_df[deg_df["Perturbation"] == pert]
        if subset.empty:
            continue
        ranked = (
            subset.sort_values("abs_score", ascending=False)["Gene"]
            .astype(str)
            .tolist()
        )
        per_pert[pert] = ranked
    global_ranked = (
        deg_df.groupby("Gene")["abs_score"].mean().sort_values(ascending=False).index
    )
    return global_ranked.astype(str).tolist(), per_pert
