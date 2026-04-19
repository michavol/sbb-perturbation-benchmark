"""
Distribution example plots for perturbation discrimination score (PDS).

Loads original datasets to display expression distributions for selected
perturbation/gene examples.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import anndata
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analyses.perturbation_discrimination import pds_core  # noqa: E402
from analyses.perturbation_discrimination.visualize_results import (  # noqa: E402
    _clean_metric_name,
    list_available_datasets,
    load_dataset_path,
    save_figure,
    set_publication_style,
)

DEFAULT_METRICS = ["MAE_Mean", "MAE_Median", "Energy_Distance", "Wasserstein_1D"]


def load_per_pert_scores(
    results_root: Path,
    dataset: str,
    filtering: str,
    metric: str,
) -> Dict[str, Dict[str, float]]:
    """Load per-perturbation mean PDS scores for a metric.

    Args:
        results_root: Root directory of analysis results.
        dataset: Dataset name.
        filtering: Filtering method.
        metric: Metric name.

    Returns:
        Mapping perturbation -> gene -> mean PDS.
    """
    scores_path = (
        results_root / dataset / "aggregates" / filtering / metric / "per_pert_top_genes.csv"
    )
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing per-pert scores file: {scores_path}")
    df = pd.read_csv(scores_path)
    per_pert: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        pert = row["Perturbation"]
        gene = row["Gene"]
        per_pert.setdefault(pert, {})[gene] = float(row["Mean_PDS"])
    return per_pert


def load_metric_scores(
    results_root: Path,
    dataset: str,
    filtering: str,
    metrics: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load per-pert scores for multiple metrics.

    Args:
        results_root: Root directory of analysis results.
        dataset: Dataset name.
        filtering: Filtering method.
        metrics: Metric names.

    Returns:
        Mapping metric -> perturbation -> gene -> mean PDS.
    """
    scores: Dict[str, Dict[str, Dict[str, float]]] = {}
    for metric in metrics:
        scores[metric] = load_per_pert_scores(results_root, dataset, filtering, metric)
    return scores


def select_discrepancy_examples(
    adata: anndata.AnnData,
    energy_scores: Dict[str, Dict[str, float]],
    mean_scores: Dict[str, Dict[str, float]],
    median_scores: Dict[str, Dict[str, float]],
    examples_per_category: int,
    max_zero_frac: float,
    max_candidates: int,
    max_perts: Optional[int],
    seed: int,
    max_checks: int,
) -> List[Tuple[str, str, str]]:
    """Select gene/pert examples with large/small score discrepancies.

    Args:
        adata: Annotated data matrix for zero-fraction filtering.
        energy_scores: Per-pert energy scores.
        mean_scores: Per-pert mean scores.
        median_scores: Per-pert median scores.
        examples_per_category: Number of examples per category.
        max_zero_frac: Exclude genes with >= this zero fraction.
        max_candidates: Max candidate pairs to consider (sampling for speed).
        max_perts: Optional cap on perturbations to sample.
        seed: Random seed for sampling.
        max_checks: Max adata checks per category.

    Returns:
        List of (category, perturbation, gene) tuples.
    """
    rng = np.random.default_rng(seed)
    perturbation_key = pds_core.resolve_perturbation_key(adata, None)
    all_perts = list(energy_scores.keys())
    if max_perts is not None and max_perts < len(all_perts):
        all_perts = rng.choice(all_perts, size=max_perts, replace=False).tolist()
    candidates: List[Tuple[str, str, float, float, float, float]] = []
    for pert in all_perts:
        gene_scores = energy_scores.get(pert, {})
        for gene, e_score in gene_scores.items():
            if gene not in mean_scores.get(pert, {}):
                continue
            if gene not in median_scores.get(pert, {}):
                continue
            m_score = mean_scores[pert][gene]
            med_score = median_scores[pert][gene]
            diff = e_score - m_score
            candidates.append((pert, gene, diff, e_score, m_score, med_score))

    if len(candidates) > max_candidates:
        indices = rng.choice(len(candidates), size=max_candidates, replace=False)
        candidates = [candidates[idx] for idx in indices]

    if not candidates:
        return []

    energy_vals = np.array([c[3] for c in candidates])
    mean_vals = np.array([c[4] for c in candidates])
    median_vals = np.array([c[5] for c in candidates])
    high_q = 0.75
    low_q = 0.25
    energy_hi = np.quantile(energy_vals, high_q)
    mean_hi = np.quantile(mean_vals, high_q)
    median_hi = np.quantile(median_vals, high_q)
    energy_lo = np.quantile(energy_vals, low_q)
    mean_lo = np.quantile(mean_vals, low_q)
    median_lo = np.quantile(median_vals, low_q)

    def passes_zero_filter(pert: str, gene: str) -> bool:
        mask = adata.obs[perturbation_key] == pert
        expr = adata[mask, gene].X
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        expr = np.asarray(expr).ravel()
        zero_frac = float(np.mean(expr <= 1e-6))
        return zero_frac < max_zero_frac

    def select_top(
        pool: Iterable[Tuple[str, str, float, float, float, float]],
        key_fn: callable,
    ) -> List[Tuple[str, str]]:
        sorted_pool = sorted(pool, key=key_fn, reverse=True)
        selected: List[Tuple[str, str]] = []
        checks = 0
        for pert, gene, _, _, _, _ in sorted_pool:
            if (pert, gene) in selected:
                continue
            checks += 1
            if checks > max_checks:
                break
            if passes_zero_filter(pert, gene):
                selected.append((pert, gene))
            if len(selected) >= examples_per_category:
                break
        return selected

    energy_better = [
        c
        for c in candidates
        if c[3] >= energy_hi
        and c[4] <= mean_lo
        and c[5] <= median_lo
        and c[4] < 0.0
        and c[5] < 0.0
        and c[2] > 0
    ]

    if not energy_better:
        return []

    examples: List[Tuple[str, str, str]] = []
    for pert, gene in select_top(energy_better, key_fn=lambda item: item[2]):
        examples.append(("Energy_Better", pert, gene))
    return examples


def select_comparison_perturbations(
    adata: anndata.AnnData,
    gene: str,
    target_pert: str,
    perturbation_key: str,
    top_k: int,
) -> List[str]:
    """Select comparison perturbations by largest mean expression difference."""
    expr = adata[:, gene].X
    if hasattr(expr, "toarray"):
        expr = expr.toarray()
    expr = np.asarray(expr).ravel()
    df = pd.DataFrame(
        {
            "Perturbation": adata.obs[perturbation_key].to_numpy(),
            "Expression": expr,
        }
    )
    means = df.groupby("Perturbation")["Expression"].mean()
    if target_pert not in means.index:
        return []
    diffs = (means - means.loc[target_pert]).abs().sort_values(ascending=False)
    comparison = [pert for pert in diffs.index if pert != target_pert][:top_k]
    return comparison


def plot_discrepancy_panel(
    adata: anndata.AnnData,
    gene: str,
    target_pert: str,
    metric_scores: Dict[str, float],
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
    top_k: int,
    log_expression: bool,
) -> None:
    """Plot discrepancy panel with metric scores and distributions."""
    perturbation_key = pds_core.resolve_perturbation_key(adata, None)
    comparison_perts = select_comparison_perturbations(
        adata, gene, target_pert, perturbation_key, top_k=top_k
    )
    perts_to_show = [target_pert] + comparison_perts
    nrows = max(1, len(perts_to_show))
    fig = plt.figure(figsize=(12.5, max(3.5, 1.6 * nrows)))
    gs = gridspec.GridSpec(nrows=nrows, ncols=2, width_ratios=[0.8, 3.0])

    ax_table = fig.add_subplot(gs[:, 0])
    ax_table.axis("off")
    table_df = pd.DataFrame(
        {
            "Metric": [_clean_metric_name(m) for m in metric_scores.keys()],
            "PDS": [f"{metric_scores[m]:.3f}" for m in metric_scores],
        }
    )
    table = ax_table.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    ax_table.set_title(f"{gene} | {target_pert}", fontsize=11, pad=10)

    expr_cache: Dict[str, np.ndarray] = {}
    nonzero_values: List[np.ndarray] = []
    for pert in perts_to_show:
        mask = adata.obs[perturbation_key] == pert
        expr = adata[mask, gene].X
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        expr = np.asarray(expr).ravel()
        expr_cache[pert] = expr
        non_zero = expr[expr > 1e-6]
        if non_zero.size:
            nonzero_values.append(non_zero)

    ax = fig.add_subplot(gs[:, 1])
    zero_threshold = 1e-6
    data: List[np.ndarray] = []
    labels: List[str] = []
    for pert in perts_to_show:
        expr = expr_cache[pert]
        zero_frac = float(np.mean(expr <= zero_threshold))
        if log_expression:
            expr = np.log1p(expr)
        data.append(expr)
        labels.append(f"{pert}\n<1e-6: {zero_frac:.1%}")

    parts = ax.violinplot(
        data,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        vert=True,
        bw_method=0.2,
    )
    for body in parts["bodies"]:
        body.set_facecolor("#4c72b0")
        body.set_alpha(0.5)
        body.set_edgecolor("#2c3e50")

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Expression (log1p)" if log_expression else "Expression")
    ax.set_title(f"{gene} | {target_pert}", fontsize=10)
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.7)
    fig.tight_layout()
    save_figure(fig, output_dir, stem, formats, dpi=300)
    plt.close(fig)


def plot_discrepancy_distributions(
    results_root: Path,
    dataset: str,
    filtering: str,
    output_dir: Path,
    metrics: Sequence[str],
    formats: Sequence[str],
    examples_per_category: int,
    top_k: int,
    max_candidates: int,
    max_perts: Optional[int],
    seed: int,
    max_checks: int,
    log_expression: bool,
) -> None:
    """Plot distribution examples for Energy vs Mean discrepancies."""
    print(f"[distributions] Dataset={dataset} starting.")
    dist_output_dir = output_dir / "distributions"
    filtering = "top_discriminating"
    energy_scores = load_per_pert_scores(results_root, dataset, filtering, "Energy_Distance")
    mean_scores = load_per_pert_scores(results_root, dataset, filtering, "MAE_Mean")
    median_scores = load_per_pert_scores(results_root, dataset, filtering, "MAE_Median")
    adata_path = load_dataset_path(dataset, ROOT)
    if not adata_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {adata_path}")
    print(f"[distributions] Loading adata: {adata_path}")
    adata = sc.read_h5ad(adata_path, backed="r")
    examples = select_discrepancy_examples(
        adata,
        energy_scores,
        mean_scores,
        median_scores,
        examples_per_category=examples_per_category,
        max_zero_frac=0.95,
        max_candidates=max_candidates,
        max_perts=max_perts,
        seed=seed,
        max_checks=max_checks,
    )
    if not examples:
        print(f"[distributions] Dataset={dataset} no examples selected; skipping.")
        return

    all_scores = load_metric_scores(results_root, dataset, filtering, metrics)
    print(f"[distributions] Dataset={dataset} examples={len(examples)}.")
    for category, pert, gene in examples:
        print(f"[distributions] Dataset={dataset} category={category} pert={pert} gene={gene}")
        metric_scores = {}
        for metric in metrics:
            metric_scores[metric] = all_scores.get(metric, {}).get(pert, {}).get(gene, np.nan)
        stem = f"{dataset}_{filtering}_{category}_{pert}_{gene}"
        plot_discrepancy_panel(
            adata,
            gene,
            pert,
            metric_scores,
            dist_output_dir,
            stem,
            formats,
            top_k=top_k,
            log_expression=log_expression,
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize PDS distribution examples for discrepancy panels."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to visualize.",
    )
    parser.add_argument(
        "--output-formats",
        nargs="+",
        default=["pdf", "png"],
        help="Output formats (e.g., pdf png).",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="nature",
        help="Plot style preset (nature or existing).",
    )
    parser.add_argument(
        "--filtering",
        type=str,
        default="highly_variable",
        help="Filtering method for discrepancy plots.",
    )
    parser.add_argument(
        "--top-k-comparisons",
        type=int,
        default=5,
        help="Number of comparison perturbations for discrepancy plots.",
    )
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=3,
        help="Number of discrepancy examples per category.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=5000,
        help="Max candidate gene/pert pairs to sample for scoring.",
    )
    parser.add_argument(
        "--max-perts",
        type=int,
        default=None,
        help="Optional cap on perturbations to sample.",
    )
    parser.add_argument(
        "--max-checks",
        type=int,
        default=200,
        help="Max adata checks per category.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--log-expression",
        action="store_true",
        help="Apply log1p to expression values before plotting.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for distribution visualization script."""
    args = parse_args()
    results_root = ROOT / "analyses" / "perturbation_discrimination" / "results"
    output_dir = ROOT / "analyses" / "perturbation_discrimination" / "figures" / "publication"
    set_publication_style(args.style)
    if not args.datasets:
        args.datasets = list_available_datasets(results_root)
    print(f"[distributions] Datasets: {', '.join(args.datasets)}")

    for dataset in args.datasets:
        plot_discrepancy_distributions(
            results_root,
            dataset,
            args.filtering,
            output_dir,
            metrics=DEFAULT_METRICS,
            formats=args.output_formats,
            examples_per_category=args.examples_per_category,
            top_k=args.top_k_comparisons,
            max_candidates=args.max_candidates,
            max_perts=args.max_perts,
            seed=args.seed,
            max_checks=args.max_checks,
            log_expression=args.log_expression,
        )


if __name__ == "__main__":
    main()
