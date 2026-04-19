#!/usr/bin/env python3
"""Synthetic-weight overlap diagnostics across perturbations.

Main outputs:
1) Per-dataset histogram of pairwise Jaccard overlap across perturbations.
2) Per-dataset *plain* (non-hierarchical) heatmap of pairwise Jaccard overlap.
3) Wessels-specific 50/50 combo split diagnostics (train/test style):
   - global union-vs-union Jaccard
   - cross-split pairwise Jaccard distribution and heatmap

Jaccard(A, B) = |A ∩ B| / |A ∪ B|
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_NATURE_FONT = "Arial"
_NATURE_AXIS_LABEL_SIZE = 7
_NATURE_TICK_SIZE = 6
_NATURE_LEGEND_SIZE = 6
_NATURE_TITLE_SIZE = 7
_NATURE_DPI = 450

ALL_DATASETS = [
    "wessels23",
    "norman19",
    "replogle20",
    "adamson16",
    "frangieh21",
    "replogle22k562",
    "nadig25hepg2",
]
COMBO_DATASETS = ["wessels23", "norman19", "replogle20"]

DATASET_COLORS = {
    "wessels23": "#1b5e20",
    "norman19": "#ff7f0e",
    "replogle20": "#9467bd",
    "adamson16": "#17becf",
    "frangieh21": "#d62728",
    "replogle22k562": "#e377c2",
    "nadig25hepg2": "#8c564b",
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


def _savefig(fig: plt.Figure, path: Path, formats: Tuple[str, ...]) -> None:
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=_NATURE_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"  Saved {out}")


def _is_combo_perturbation(name: str, *, allow_underscore: bool = False) -> bool:
    s = str(name)
    tokens = ["+", "__", "|", ";"]
    if allow_underscore:
        tokens.append("_")
    return any(tok in s for tok in tokens)


# ---------------------------------------------------------------------------
# Weight loading (mirrors DataManager._load_deg_weights_from_csv logic)
# ---------------------------------------------------------------------------

def load_synthetic_weights(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Load raw |score| per perturbation from deg_synthetic.csv.

    Returns {perturbation: {gene: |score|}}.
    """
    raw: Dict[str, Dict[str, float]] = defaultdict(dict)
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            pert = str(row.get("Perturbation", "")).strip()
            gene = str(row.get("Gene", "")).strip()
            score_str = row.get("score", "")
            if not pert or not gene or score_str in ("", None):
                continue
            try:
                score_val = abs(float(score_str))
            except (TypeError, ValueError):
                continue
            prev = raw[pert].get(gene)
            if prev is None or score_val > prev:
                raw[pert][gene] = score_val
    return dict(raw)


def normalise_weights(raw: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Apply per-perturbation min-max + square normalisation."""
    out: Dict[str, Dict[str, float]] = {}
    for pert, gene_scores in raw.items():
        vals = np.array(list(gene_scores.values()), dtype=float)
        mn, mx = float(vals.min()), float(vals.max())
        denom = mx - mn
        if denom <= 0:
            out[pert] = {g: 0.0 for g in gene_scores}
            continue
        out[pert] = {g: ((v - mn) / denom) ** 2 for g, v in gene_scores.items()}
    return out


def top_k_genes(weights: Dict[str, Dict[str, float]], k: int) -> Dict[str, Set[str]]:
    """Return top-k genes per perturbation by normalized weight."""
    out: Dict[str, Set[str]] = {}
    for pert, gene_w in weights.items():
        sorted_genes = sorted(gene_w, key=gene_w.get, reverse=True)[:k]
        out[pert] = set(sorted_genes)
    return out


def jaccard(a: Set[str], b: Set[str]) -> float:
    """Set overlap score in [0, 1]."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def pairwise_jaccard(pert_order: Sequence[str], gene_sets: Dict[str, Set[str]]) -> np.ndarray:
    """Pairwise Jaccard matrix for perturbations in the supplied order."""
    n = len(pert_order)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        ai = gene_sets[pert_order[i]]
        for j in range(i, n):
            val = jaccard(ai, gene_sets[pert_order[j]])
            mat[i, j] = val
            mat[j, i] = val
    return mat


def upper_triangle_values(mat: np.ndarray) -> np.ndarray:
    """Extract upper-triangle values (excluding diagonal)."""
    if mat.shape[0] < 2:
        return np.array([], dtype=float)
    return mat[np.triu_indices(mat.shape[0], k=1)]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_jaccard_histogram_from_values(
    values: np.ndarray,
    *,
    dataset: str,
    title_suffix: str,
    x_label: str,
    n_entities: int,
    output_path: Path,
    formats: Tuple[str, ...],
) -> None:
    """Histogram for a precomputed list of Jaccard values."""
    color = DATASET_COLORS.get(dataset, "#333333")
    fig, ax = plt.subplots(figsize=(3.6, 2.45))

    if values.size == 0:
        ax.text(0.5, 0.5, "No pairs to plot", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        ax.hist(values, bins=50, range=(0, 1), color=color, edgecolor="white", linewidth=0.3, alpha=0.85)
        mean_v = float(np.mean(values))
        med_v = float(np.median(values))
        ax.axvline(med_v, color="black", linewidth=0.8, linestyle="--", zorder=5)
        ax.axvline(mean_v, color="#c62828", linewidth=0.8, linestyle="-", zorder=5)
        ax.text(0.02, 0.98, f"mean={mean_v:.3f}\nmedian={med_v:.3f}", fontsize=_NATURE_LEGEND_SIZE,
                ha="left", va="top", transform=ax.transAxes)

    ax.text(
        0.98,
        0.98,
        f"n_entities={n_entities}\nn_pairs={values.size}",
        fontsize=_NATURE_LEGEND_SIZE,
        ha="right",
        va="top",
        transform=ax.transAxes,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of pairs")
    title = dataset if not title_suffix else f"{dataset}: {title_suffix}"
    ax.set_title(title)
    ax.set_xlim(0, 1)

    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="both", which="major", length=2.5, width=0.6, pad=2)

    fig.tight_layout()
    _savefig(fig, output_path, formats)
    plt.close(fig)


def plot_plain_heatmap(
    mat: np.ndarray,
    *,
    labels: Sequence[str],
    dataset: str,
    title_suffix: str,
    top_k: int,
    output_path: Path,
    formats: Tuple[str, ...],
) -> None:
    """Plain non-hierarchical heatmap."""
    n = len(labels)
    show_ticks = n <= 80
    if show_ticks:
        # Give labeled heatmaps more room so tick labels are readable.
        size = max(6.0, min(14.0, n * 0.14))
    else:
        size = max(4.0, min(12.0, n * 0.06))
    fig, ax = plt.subplots(figsize=(size, size))

    sns.heatmap(
        mat,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        cbar_kws={"label": f"Jaccard (top {top_k})", "shrink": 0.65},
        xticklabels=labels if show_ticks else False,
        yticklabels=labels if show_ticks else False,
        square=True,
        linewidths=0,
    )
    title = dataset if not title_suffix else f"{dataset}: {title_suffix}"
    ax.set_title(title, fontsize=_NATURE_TITLE_SIZE)
    if show_ticks:
        ax.tick_params(axis="x", labelrotation=90, labelsize=max(2, _NATURE_TICK_SIZE - 1), length=0)
        ax.tick_params(axis="y", labelrotation=0, labelsize=max(2, _NATURE_TICK_SIZE - 1), length=0)
    else:
        ax.set_xlabel(f"Perturbations (n={n})")
        ax.set_ylabel(f"Perturbations (n={n})")

    fig.tight_layout()
    _savefig(fig, output_path, formats)
    plt.close(fig)


def clustered_order_from_similarity(mat: np.ndarray) -> np.ndarray:
    """Return hierarchical-clustering leaf order from a similarity matrix."""
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
    except Exception:
        # Fallback: keep original order if scipy is unavailable.
        return np.arange(mat.shape[0], dtype=int)

    # Convert similarity [0,1] to distance [0,1].
    dist = 1.0 - np.clip(mat, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    return leaves_list(Z)


# ---------------------------------------------------------------------------
# 50/50 split diagnostics
# ---------------------------------------------------------------------------

def run_split50_cross_pairwise_diagnostics(
    *,
    dataset: str,
    gene_sets: Dict[str, Set[str]],
    top_k: int,
    seed: int,
    output_dir: Path,
    formats: Tuple[str, ...],
    include_singles_fallback: bool = False,
) -> None:
    """50/50 split diagnostics across combo perturbations for a dataset."""
    allow_us = dataset in COMBO_DATASETS
    combo_perts = sorted(
        [p for p in gene_sets if _is_combo_perturbation(p, allow_underscore=allow_us)]
    )
    if len(combo_perts) < 4:
        if include_singles_fallback and len(gene_sets) >= 4:
            combo_perts = sorted(gene_sets)
            print(f"[{dataset}] Using all perturbations for split diagnostics (combo count < 4)")
        else:
            print(f"[{dataset}] Not enough combo perturbations for split diagnostics; skipping")
            return

    rng = np.random.default_rng(seed)
    shuffled = combo_perts.copy()
    rng.shuffle(shuffled)
    split_idx = len(shuffled) // 2
    split_a = shuffled[:split_idx]
    split_b = shuffled[split_idx:]

    sets_a = [gene_sets[p] for p in split_a]
    sets_b = [gene_sets[p] for p in split_b]

    union_a = set().union(*sets_a) if sets_a else set()
    union_b = set().union(*sets_b) if sets_b else set()
    union_j = jaccard(union_a, union_b)

    # Cross-split pairwise matrix: A rows x B cols
    mat_ab = np.zeros((len(split_a), len(split_b)), dtype=float)
    vals_ab: List[float] = []
    for i, pa in enumerate(split_a):
        for j, pb in enumerate(split_b):
            v = jaccard(gene_sets[pa], gene_sets[pb])
            mat_ab[i, j] = v
            vals_ab.append(v)
    vals_ab_arr = np.array(vals_ab, dtype=float)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Histogram for cross-split pairwise overlaps
    plot_jaccard_histogram_from_values(
        vals_ab_arr,
        dataset=dataset,
        title_suffix="50/50 combo split cross-overlap",
        x_label=f"Cross-split Jaccard (top {top_k} weighted genes)",
        n_entities=len(combo_perts),
        output_path=output_dir / "split50_cross_pairwise_histogram",
        formats=formats,
    )

    # Cross-split heatmap
    # Make a square figure for consistent visual comparison.
    # If labels are shown, increase canvas so tick labels fit.
    label_count = max(len(split_a), len(split_b))
    if label_count <= 80:
        size = max(6.5, label_count * 0.10)
    else:
        size = max(5.0, label_count * 0.06)
    fig, ax = plt.subplots(figsize=(size, size))
    show_ticks = max(len(split_a), len(split_b)) <= 80
    sns.heatmap(
        mat_ab,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        cbar_kws={"label": f"Cross-split Jaccard (top {top_k})", "shrink": 0.6},
        xticklabels=split_b if show_ticks else False,
        yticklabels=split_a if show_ticks else False,
        linewidths=0,
        square=True,
    )
    ax.set_title(f"{dataset}: split A (rows) vs split B (cols)")
    if show_ticks:
        ax.tick_params(axis="x", labelrotation=90, labelsize=max(2, _NATURE_TICK_SIZE - 1), length=0)
        ax.tick_params(axis="y", labelrotation=0, labelsize=max(2, _NATURE_TICK_SIZE - 1), length=0)
    else:
        ax.set_xlabel(f"Split B perturbations (n={len(split_b)})")
        ax.set_ylabel(f"Split A perturbations (n={len(split_a)})")
    fig.tight_layout()
    _savefig(fig, output_dir / "split50_cross_pairwise_heatmap", formats)
    plt.close(fig)

    summary = {
        "dataset": dataset,
        "seed": int(seed),
        "top_k": int(top_k),
        "n_combo_perturbations_total": len(combo_perts),
        "n_split_a": len(split_a),
        "n_split_b": len(split_b),
        "global_union_jaccard": float(union_j),
        "split_a_union_size": len(union_a),
        "split_b_union_size": len(union_b),
        "cross_split_pairwise_mean": float(np.mean(vals_ab_arr)) if vals_ab_arr.size else None,
        "cross_split_pairwise_median": float(np.median(vals_ab_arr)) if vals_ab_arr.size else None,
        "cross_split_pairwise_min": float(np.min(vals_ab_arr)) if vals_ab_arr.size else None,
        "cross_split_pairwise_max": float(np.max(vals_ab_arr)) if vals_ab_arr.size else None,
    }
    (output_dir / "split50_overlap_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with (output_dir / "split50_partition.csv").open("w", encoding="utf-8", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["perturbation", "split"])
        for p in split_a:
            w.writerow([p, "A"])
        for p in split_b:
            w.writerow([p, "B"])

    print(f"[{dataset}] 50/50 split diagnostics: union_jaccard={union_j:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot overlap diagnostics for synthetic weighted genes across perturbations."
    )
    parser.add_argument("--datasets", nargs="+", default=COMBO_DATASETS)
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of top weighted genes per perturbation (default: 100).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for deterministic 50/50 split diagnostics (default: 42).")
    parser.add_argument("--combo-only", action="store_true", default=True,
                        help="Restrict analysis to combo perturbations (default: enabled).")
    parser.add_argument("--include-singles", action="store_true", default=False,
                        help="Include single perturbations as well (overrides --combo-only).")
    parser.add_argument("--max-combos-heatmap", type=int, default=400,
                        help="Maximum perturbations for heatmap; random subsample if exceeded.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("analyses/perturbation_discrimination/figures/weight_overlap"))
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()

    _apply_nature_rc()
    formats = tuple(args.formats)
    results_root = Path("analyses/perturbation_discrimination/results")

    print("Jaccard explanation: J(A,B)=|A∩B|/|A∪B|, 1=identical sets, 0=no overlap")

    summaries: Dict[str, Tuple[int, float, float]] = {}

    for ds in args.datasets:
        csv_path = results_root / ds / "deg_scanpy" / "deg_synthetic.csv"
        if not csv_path.exists():
            print(f"[{ds}] deg_synthetic.csv not found — skipping")
            continue

        print(f"\n[{ds}] Loading synthetic weights from {csv_path}")
        raw = load_synthetic_weights(csv_path)
        weights = normalise_weights(raw)
        gene_sets_all = top_k_genes(weights, args.top_k)

        if not args.include_singles:
            if args.combo_only:
                allow_us = ds in COMBO_DATASETS
                gene_sets = {
                    p: gs
                    for p, gs in gene_sets_all.items()
                    if _is_combo_perturbation(p, allow_underscore=allow_us)
                }
            else:
                gene_sets = gene_sets_all
        else:
            gene_sets = gene_sets_all

        perts = sorted(gene_sets)
        if len(perts) < 2:
            print(f"[{ds}] Not enough perturbations after filtering; skipping")
            continue

        mat = pairwise_jaccard(perts, gene_sets)
        vals = upper_triangle_values(mat)
        mean_j = float(np.mean(vals)) if vals.size else float("nan")
        med_j = float(np.median(vals)) if vals.size else float("nan")
        print(f"[{ds}] n_perts={len(perts)}, n_pairs={vals.size}, mean={mean_j:.4f}, median={med_j:.4f}")
        summaries[ds] = (len(perts), mean_j, med_j)

        out_base = args.output_dir / ds
        plot_jaccard_histogram_from_values(
            vals,
            dataset=ds,
            title_suffix="pairwise overlap",
            x_label=f"Pairwise Jaccard (top {args.top_k} weighted genes)",
            n_entities=len(perts),
            output_path=out_base / "jaccard_histogram",
            formats=formats,
        )

        # Non-hierarchical heatmap. Subsample if too large.
        heat_perts = perts
        if len(heat_perts) > args.max_combos_heatmap:
            rng = np.random.default_rng(args.seed)
            idx = np.sort(rng.choice(len(heat_perts), size=args.max_combos_heatmap, replace=False))
            heat_perts = [heat_perts[i] for i in idx]
            print(f"[{ds}] Heatmap subsampled to {len(heat_perts)} perturbations")
        heat_mat = pairwise_jaccard(heat_perts, gene_sets)
        plot_plain_heatmap(
            heat_mat,
            labels=heat_perts,
            dataset=ds,
            title_suffix="pairwise overlap heatmap (no clustering)",
            top_k=args.top_k,
            output_path=out_base / "jaccard_heatmap",
            formats=formats,
        )
        # Also save a heatmap with perturbations reordered by hierarchical clustering
        # (same heatmap style, no dendrogram shown).
        order = clustered_order_from_similarity(heat_mat)
        heat_mat_clustered = heat_mat[np.ix_(order, order)]
        heat_perts_clustered = [heat_perts[i] for i in order]
        plot_plain_heatmap(
            heat_mat_clustered,
            labels=heat_perts_clustered,
            dataset=ds,
            title_suffix="pairwise overlap heatmap (clustered order)",
            top_k=args.top_k,
            output_path=out_base / "jaccard_heatmap_clustered",
            formats=formats,
        )

        # 50/50 cross-split diagnostics for all selected datasets
        run_split50_cross_pairwise_diagnostics(
            dataset=ds,
            gene_sets=gene_sets_all,
            top_k=args.top_k,
            seed=args.seed,
            output_dir=out_base,
            formats=formats,
            include_singles_fallback=args.include_singles,
        )

    # Across-dataset summary
    if summaries:
        ds_order = sorted(summaries, key=lambda d: summaries[d][1], reverse=True)
        means = [summaries[d][1] for d in ds_order]
        medians = [summaries[d][2] for d in ds_order]
        colors = [DATASET_COLORS.get(d, "#666666") for d in ds_order]

        fig, ax = plt.subplots(figsize=(3.6, 2.45))
        x = np.arange(len(ds_order))
        w = 0.35
        ax.bar(x - w / 2, means, w, color=colors, edgecolor="white", linewidth=0.3, alpha=0.85, label="Mean")
        ax.bar(x + w / 2, medians, w, color=colors, edgecolor="white", linewidth=0.3, alpha=0.50, label="Median")
        ax.set_xticks(x)
        ax.set_xticklabels(ds_order, rotation=45, ha="right")
        ax.set_ylabel(f"Pairwise Jaccard (top {args.top_k})")
        ax.legend(frameon=False, fontsize=_NATURE_LEGEND_SIZE)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(axis="both", which="major", length=2.5, width=0.6, pad=2)
        fig.tight_layout()
        _savefig(fig, args.output_dir / "summary_across_datasets", formats)
        plt.close(fig)

    print("\nDone!")


if __name__ == "__main__":
    main()
