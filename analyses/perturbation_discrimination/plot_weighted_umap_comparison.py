"""Compare standard vs DEG-weight-weighted UMAP embeddings.

Generates a two-panel figure:
  - Standard UMAP of control cells and technical-duplicate-half cells from
    selected perturbations (UMAP on gene expression directly, no PCA).
  - Weighted UMAP where gene features are scaled like benchmark wMSE
    (nonnegative weights normalized to sum to 1, then ``sqrt(w)`` per gene),
    using DEG CSV weights with the same |score|→min-max→square transform as
    ``DataManager._load_deg_weights_from_csv``.

Designed for Nature-style publication figures.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import scanpy as sc
from scipy import sparse
from umap import UMAP

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analyses.perturbation_discrimination.discrimination_comparison import (
    load_dataset_path,
)
from cellsimbench.utils.effective_genes import _effective_genes_single

# ---------------------------------------------------------------------------
# Nature style defaults
# ---------------------------------------------------------------------------

NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.title_fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "lines.linewidth": 0.8,
    "lines.markersize": 2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

CONTROL_COLOUR = "#BFBFBF"
PERTURBATION_PALETTE = [
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#DC9E82",
    "#7E6148",
    "#B09C85",
    "#E377C2",
    "#BCBD22",
    "#17BECF",
    "#9467BD",
    "#D62728",
]

MM_TO_INCH = 1.0 / 25.4
NATURE_SINGLE_COL = 89.0 * MM_TO_INCH
NATURE_DOUBLE_COL = 183.0 * MM_TO_INCH

# Extra exponent on averaged weights (1 = none). CellSimBench DEG weights already
# apply a square after min-max; default 1.0 matches evaluation-style weights.
DEFAULT_SHARPENING_EXPONENT = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sharpening_stem_suffix(exponent: float) -> str:
    """Non-empty suffix when *exponent* differs from the default (keeps both runs on disk)."""
    if np.isclose(float(exponent), DEFAULT_SHARPENING_EXPONENT):
        return ""
    tag = f"{float(exponent):g}".replace(".", "p").replace("-", "m")
    return f"_sharpen{tag}"


def _is_control_label(label: str) -> bool:
    lowered = label.strip().lower()
    return bool(re.search(r"control|ctrl", lowered)) or lowered == "*"


def _select_top_perturbations(
    adata: sc.AnnData,
    perturbation_key: str,
    n_perturbations: int,
) -> List[str]:
    """Select top *n* non-control perturbations by cell count."""
    labels = adata.obs[perturbation_key].astype(str)
    counts = labels.value_counts(dropna=False)
    candidates = [lab for lab in counts.index if not _is_control_label(lab)]
    candidates.sort(key=lambda lab: (-int(counts[lab]), lab))
    return candidates[:n_perturbations]


def _resolve_perturbation_key(adata: sc.AnnData) -> str:
    for key in ("condition", "perturbation"):
        if key in adata.obs.columns:
            return key
    raise KeyError("Could not resolve perturbation column from adata.obs.")


def _densify(X) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X.toarray(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _map_csv_pert_to_selected(csv_pert: str, perturbations: Sequence[str]) -> Optional[str]:
    """Map a CSV ``Perturbation`` label to an entry in *perturbations*.

    DEG caches may use ``+`` or ``_`` as combo separators (e.g. ``DOT1L_EP300`` vs
    ``DOT1L+EP300``).
    """
    key = str(csv_pert).replace("+", "_")
    for p in perturbations:
        if str(p).replace("+", "_") == key:
            return str(p)
    return None


def load_benchmark_style_deg_weights(
    dataset_name: str,
    genes: Sequence[str],
    perturbations: Sequence[str],
    *,
    deg_source: str = "synthetic",
) -> Dict[str, np.ndarray]:
    """Load DEG weights using the same transform as ``DataManager._load_deg_weights_from_csv``.

    Pipeline: |score| (max over duplicate genes) → min-max per perturbation → square.
    Vectors are **not** max-renormalized after squaring (matches CellSimBench wmse inputs).

    Args:
        dataset_name: Dataset id under ``results/<name>/deg_scanpy/``.
        genes: Gene names aligned to ``adata.var_names``.
        perturbations: Perturbation labels to keep (e.g. selected conditions).
        deg_source: ``synthetic`` → ``deg_synthetic.csv``; ``vscontrol`` → ``deg_control.csv``.

    Returns:
        Mapping perturbation label → nonnegative weight vector aligned to *genes*.
    """
    fname = "deg_synthetic.csv" if deg_source == "synthetic" else "deg_control.csv"
    if deg_source not in {"synthetic", "vscontrol"}:
        raise ValueError(f"deg_source must be 'synthetic' or 'vscontrol', got {deg_source!r}")
    csv_path = (
        ROOT
        / "analyses"
        / "perturbation_discrimination"
        / "results"
        / dataset_name
        / "deg_scanpy"
        / fname
    )
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing DEG CSV ({deg_source}): {csv_path}")

    gene_to_idx: Dict[str, int] = {g: i for i, g in enumerate(genes)}
    raw_abs: Dict[str, Dict[str, float]] = {str(p): {} for p in perturbations}

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            csv_pert = str(row.get("Perturbation", "")).strip()
            target_pert = _map_csv_pert_to_selected(csv_pert, perturbations)
            if target_pert is None:
                continue
            gene = str(row.get("Gene", "")).strip()
            score_raw = row.get("score", "")
            if gene not in gene_to_idx or not score_raw:
                continue
            try:
                val = abs(float(score_raw))
            except (TypeError, ValueError):
                continue
            prev = raw_abs[target_pert].get(gene)
            if prev is None or val > prev:
                raw_abs[target_pert][gene] = val

    n_genes = len(genes)
    weights: Dict[str, np.ndarray] = {}
    for pert in perturbations:
        w = np.zeros(n_genes, dtype=np.float64)
        for gene, score in raw_abs.get(str(pert), {}).items():
            w[gene_to_idx[gene]] = score
        w_min, w_max = float(np.min(w)), float(np.max(w))
        denom = w_max - w_min
        if denom > 0:
            w = (w - w_min) / denom
        else:
            w = np.zeros_like(w)
        w = np.nan_to_num(np.square(w), nan=0.0)
        weights[str(pert)] = w
    return weights


def average_weights_for_umap(
    weights_per_pert: Dict[str, np.ndarray],
    perturbations: Sequence[str],
    sharpening_exponent: float = 1.0,
) -> np.ndarray:
    """Element-wise mean of per-perturbation weights, optional extra sharpening.

    Averaging mirrors a single global gene scaling for a joint UMAP; per-cell
    wmse in benchmarks still uses each perturbation's own vector.
    """
    vecs = [
        weights_per_pert[p]
        for p in perturbations
        if p in weights_per_pert and float(np.sum(weights_per_pert[p])) > 1e-12
    ]
    if not vecs:
        raise ValueError("No non-zero weight vectors to average.")
    avg = np.mean(np.stack(vecs, axis=0), axis=0)
    if sharpening_exponent != 1.0:
        w_max = float(np.max(avg))
        if w_max > 0:
            avg = avg / w_max
        avg = np.power(avg, sharpening_exponent)
        w_max = float(np.max(avg))
        if w_max > 0:
            avg = avg / w_max
    return avg


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def compute_umap(
    X: np.ndarray,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Run UMAP in the full gene space (no PCA)."""
    n_features = int(X.shape[1])
    if n_features < 2:
        raise ValueError("UMAP requires at least 2 features.")
    n_neigh = max(5, min(n_neighbors, X.shape[0] - 1))

    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neigh,
        min_dist=min_dist,
        random_state=seed,
        n_jobs=-1,
        low_memory=True,
    )
    return reducer.fit_transform(X)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _build_colour_map(
    perturbations: Sequence[str],
) -> Dict[str, str]:
    """Map perturbation labels to hex colours (controls get grey)."""
    cmap: Dict[str, str] = {}
    pi = 0
    for lab in perturbations:
        if _is_control_label(lab):
            cmap[lab] = CONTROL_COLOUR
        else:
            cmap[lab] = PERTURBATION_PALETTE[pi % len(PERTURBATION_PALETTE)]
            pi += 1
    return cmap


def plot_umap_panel(
    ax: matplotlib.axes.Axes,
    coords: np.ndarray,
    labels: np.ndarray,
    colour_map: Dict[str, str],
    perturbation_order: Sequence[str],
    title: str,
    point_size: float = 4.0,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Draw one UMAP panel on *ax*."""
    if rng is None:
        rng = np.random.default_rng(0)

    ctrl_mask = np.array([_is_control_label(str(l)) for l in labels])
    ctrl_idx = np.where(ctrl_mask)[0]
    if ctrl_idx.size:
        order = rng.permutation(ctrl_idx)
        ax.scatter(
            coords[order, 0],
            coords[order, 1],
            c=CONTROL_COLOUR,
            s=point_size,
            alpha=0.3,
            edgecolors="none",
            rasterized=True,
            label="Control",
            zorder=1,
        )

    for pert in perturbation_order:
        if _is_control_label(pert):
            continue
        mask = labels == pert
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        order = rng.permutation(idx)
        ax.scatter(
            coords[order, 0],
            coords[order, 1],
            c=colour_map.get(pert, "#333333"),
            s=point_size * 1.4,
            alpha=0.85,
            edgecolors="none",
            rasterized=True,
            label=pert,
            zorder=2,
        )

    ax.set_title(title, fontweight="bold", pad=4)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare standard vs weighted UMAP for perturbation data."
    )
    p.add_argument("--dataset", default="frangieh21", help="Dataset config name.")
    p.add_argument(
        "--deg-source",
        choices=("synthetic", "vscontrol"),
        default="synthetic",
        help=(
            "DEG CSV to use: synthetic → deg_synthetic.csv, vscontrol → deg_control.csv "
            "(matches CellSimBench deg_weight_source)."
        ),
    )
    p.add_argument(
        "--n-perturbations",
        type=int,
        default=10,
        help="Number of perturbations to display.",
    )
    p.add_argument(
        "--tech-dup-half",
        choices=("second_half", "first_half", "all"),
        default="second_half",
        help="Which tech_dup_split half to use for perturbation cells.",
    )
    p.add_argument(
        "--max-control-cells",
        type=int,
        default=500,
        help="Cap on control cells to avoid swamping the UMAP.",
    )
    p.add_argument(
        "--min-dist", type=float, default=0.5, help="UMAP min_dist."
    )
    p.add_argument(
        "--n-neighbors", type=int, default=15, help="UMAP n_neighbors."
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to figures/weighted_umap/<dataset>/).",
    )
    p.add_argument(
        "--output-formats",
        nargs="+",
        default=["png", "pdf"],
        help="Figure formats.",
    )
    p.add_argument(
        "--point-size",
        type=float,
        default=4.0,
        help="Scatter point size for UMAP.",
    )
    p.add_argument(
        "--sharpening-exponent",
        type=float,
        default=DEFAULT_SHARPENING_EXPONENT,
        help=(
            "Optional extra exponent on averaged weights after max-normalization "
            "(1 = evaluation-style; >1 sharpens for visualization only). "
            f"If not {DEFAULT_SHARPENING_EXPONENT:g}, output stem gets suffix _sharpen<value>."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    matplotlib.rcParams.update(NATURE_RC)
    rng = np.random.default_rng(args.seed)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            ROOT
            / "analyses"
            / "perturbation_discrimination"
            / "figures"
            / "weighted_umap"
            / args.dataset
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading dataset: {args.dataset}", flush=True)
    dataset_path = load_dataset_path(ROOT, args.dataset)
    adata = sc.read_h5ad(dataset_path)
    pert_key = _resolve_perturbation_key(adata)
    print(f"  Perturbation key: {pert_key}", flush=True)
    print(f"  Cells: {adata.n_obs}  Genes: {adata.n_vars}", flush=True)

    # ------------------------------------------------------------------
    # Select perturbations and subset cells
    # ------------------------------------------------------------------
    selected = _select_top_perturbations(
        adata, pert_key, args.n_perturbations
    )
    print(f"  Selected perturbations ({len(selected)}): {selected}", flush=True)

    labels_all = adata.obs[pert_key].astype(str).to_numpy()
    is_ctrl = np.array([_is_control_label(l) for l in labels_all])
    is_selected = np.isin(labels_all, selected)

    # Apply tech_dup_split filter for perturbation cells
    if args.tech_dup_half != "all" and "tech_dup_split" in adata.obs.columns:
        tech_split = adata.obs["tech_dup_split"].astype(str).to_numpy()
        pert_mask = is_selected & (tech_split == args.tech_dup_half)
        ctrl_mask = is_ctrl & (tech_split == args.tech_dup_half)
    else:
        pert_mask = is_selected
        ctrl_mask = is_ctrl

    # Cap control cells
    ctrl_idx = np.where(ctrl_mask)[0]
    if ctrl_idx.size > args.max_control_cells:
        ctrl_idx = rng.choice(ctrl_idx, size=args.max_control_cells, replace=False)
    pert_idx = np.where(pert_mask)[0]

    keep_idx = np.sort(np.concatenate([ctrl_idx, pert_idx]))
    adata_sub = adata[keep_idx, :].copy()
    labels = adata_sub.obs[pert_key].astype(str).to_numpy()
    X = _densify(adata_sub.X)

    n_ctrl = int(np.sum([_is_control_label(l) for l in labels]))
    n_pert = int(labels.size - n_ctrl)
    print(
        f"  Subset: {labels.size} cells ({n_ctrl} control, {n_pert} perturbation)",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Load DEG weights (same transform as CellSimBench DataManager CSV loader)
    # ------------------------------------------------------------------
    genes = [str(g) for g in adata_sub.var_names.tolist()]
    weights_per_pert = load_benchmark_style_deg_weights(
        args.dataset,
        genes,
        selected,
        deg_source=args.deg_source,
    )
    neff_per_pert: List[float] = []
    for pert in selected:
        w = weights_per_pert.get(str(pert))
        if w is None or float(np.sum(w)) < 1e-12:
            continue
        neff_per_pert.append(float(_effective_genes_single(w)))
    if not neff_per_pert:
        print(
            "  WARNING: no DEG weights for selected perturbations; using uniform weights.",
            flush=True,
        )
        avg_w = np.ones(len(genes), dtype=np.float64)
        eff_genes_mean = float(len(genes))
    else:
        eff_genes_mean = float(np.mean(neff_per_pert))
        avg_w = average_weights_for_umap(
            weights_per_pert,
            selected,
            sharpening_exponent=args.sharpening_exponent,
        )
    w_sum_avg = float(np.sum(avg_w))
    if w_sum_avg < 1e-12:
        print(
            "  WARNING: averaged weights are all zero; using uniform scaling.",
            flush=True,
        )
        avg_w = np.ones(len(genes), dtype=np.float64)
        if not neff_per_pert:
            eff_genes_mean = float(len(genes))
    print(
        f"  DEG weights ({args.deg_source}): mean n_eff over selected perts "
        f"with data = {eff_genes_mean:.1f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Compute embeddings
    # ------------------------------------------------------------------
    print("  Computing standard UMAP...", flush=True)
    coords_standard = compute_umap(
        X,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        seed=args.seed,
    )

    print("  Computing weighted UMAP...", flush=True)
    w_sum = float(np.sum(avg_w))
    if w_sum < 1e-12:
        w_norm = np.full(len(genes), 1.0 / max(len(genes), 1), dtype=np.float64)
    else:
        w_norm = avg_w / w_sum
    sqrt_w = np.sqrt(w_norm)
    X_weighted = X * sqrt_w[np.newaxis, :]
    coords_weighted = compute_umap(
        X_weighted,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    all_labels_sorted = sorted(
        set(labels.tolist()),
        key=lambda l: (0 if _is_control_label(l) else 1, l),
    )
    colour_map = _build_colour_map(all_labels_sorted)

    fig_width = NATURE_DOUBLE_COL
    fig_height = fig_width * 0.48
    fig, (ax_std, ax_wt) = plt.subplots(
        1, 2, figsize=(fig_width, fig_height),
        gridspec_kw={"wspace": 0.35},
    )

    eff_genes_label = f"$n_{{\\mathrm{{eff}}}}$\u2009=\u2009{eff_genes_mean:.0f}"
    plot_umap_panel(
        ax_std,
        coords_standard,
        labels,
        colour_map,
        all_labels_sorted,
        title="Standard UMAP",
        point_size=args.point_size,
        rng=rng,
    )
    plot_umap_panel(
        ax_wt,
        coords_weighted,
        labels,
        colour_map,
        all_labels_sorted,
        title=f"Weighted UMAP ({eff_genes_label})",
        point_size=args.point_size,
        rng=rng,
    )

    handles, lbls = ax_wt.get_legend_handles_labels()
    fig.legend(
        handles,
        lbls,
        loc="center right",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        markerscale=1.8,
        handletextpad=0.4,
        borderpad=0.3,
        labelspacing=0.35,
    )

    fig.subplots_adjust(right=0.80, wspace=0.35)

    # Save (suffix when sharpening != default so multiple settings coexist)
    stem = (
        f"{args.dataset}_weighted_umap_comparison"
        f"{_sharpening_stem_suffix(args.sharpening_exponent)}"
    )
    for fmt in args.output_formats:
        out_path = output_dir / f"{stem}.{fmt}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out_path}", flush=True)
    plt.close(fig)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
