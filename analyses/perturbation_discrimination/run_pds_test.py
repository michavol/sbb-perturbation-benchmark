#!/usr/bin/env python3
"""PDS-based testing: rank-based permutation test for inter-perturbation discrimination.

For each dataset, splits ground-truth cells into two half-bags per perturbation
(Q = query, R = reference), computes a bag-to-bag distance matrix **once**,
then runs a local rank-based permutation test following the Q/R formulation:

    For perturbation p, rank d(Q_p, R_p) among {d(Q_p, R_q) : q = 1..K}.
    Null: permute perturbation labels across all 2|P| half-bags.
    p-value = (1 + sum 1{r_null <= r_obs}) / (M + 1).

Lower p-values indicate better inter-perturbation discrimination.
No symmetrization of the distance matrix is performed; asymmetric metrics
are handled naturally by the Q/R formulation (Q bags always provide weights).

Metrics:
    mse              – MSE on bag means
    wmse             – weighted MSE (synthetic DEG weights, query-weighted)
    pearson          – 1 − Pearson on raw bag means
    wpearson         – 1 − weighted Pearson on raw bag means
    pearson_dp       – 1 − Pearson(Δpert)  (Δpert = mean − dataset_mean)
    wpearson_dp      – 1 − weighted Pearson(Δpert)
    pearson_dc       – 1 − Pearson(Δctrl)  (Δctrl = mean − control_mean)
    wpearson_dc      – 1 − weighted Pearson(Δctrl)
    r2_dp            – 1 − R²(Δpert)
    wr2_dp           – 1 − weighted R²(Δpert)
    top20_mse        – MSE on top 20 DEGs
    top20_pearson_dp – 1 − Pearson(Δpert) on top 20 DEGs
    top100_mse       – MSE on top 100 DEGs (by synthetic DEG weight, query row)
    top100_energy    – energy distance on cells restricted to query’s top 100 DEG genes
    energy           – energy distance on raw cell bags
    wenergy          – weighted energy distance on raw cell bags

Usage:
    python run_pds_test.py --datasets frangieh21
    python run_pds_test.py --bag-mode half --n-permutations 999 --n-trials 3
    python run_pds_test.py --resume   # continue after a failure (same args as the partial run)
    python run_pds_test.py --overwrite  # replace any existing output in --output-dir
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyses.perturbation_discrimination import metric_utils as base
from analyses.perturbation_discrimination.run_metric_comparison import (
    _load_deg_weights_from_csv,
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

METRICS: List[str] = [
    "mse",
    "wmse",
    "wmse_vc",
    "pearson",
    "wpearson",
    "pearson_dp",
    "wpearson_dp",
    "wpearson_dp_vc",
    "pearson_dc",
    "wpearson_dc",
    "r2_dp",
    "wr2_dp",
    "top20_mse",
    "top20_pearson_dp",
]

ENERGY_METRICS: List[str] = ["energy", "wenergy"]

ALL_KNOWN_METRICS: List[str] = METRICS + ENERGY_METRICS + ["top100_mse", "top100_energy"]

METRIC_DISPLAY: Dict[str, str] = {
    "mse": "MSE",
    "wmse": "wMSE (syn)",
    "wmse_vc": "wMSE (vc)",
    "pearson": "Pearson",
    "wpearson": "wPearson (syn)",
    "pearson_dp": "Pearson Δpert",
    "wpearson_dp": "wPearson Δpert (syn)",
    "wpearson_dp_vc": "wPearson Δpert (vc)",
    "pearson_dc": "Pearson Δctrl",
    "wpearson_dc": "wPearson Δctrl (syn)",
    "r2_dp": "R² Δpert",
    "wr2_dp": "wR² Δpert (syn)",
    "top20_mse": "top20 MSE (syn)",
    "top20_pearson_dp": "top20 Pearson Δpert (syn)",
    "top100_mse": "top100 MSE (syn)",
    "top100_energy": "top100 Energy (syn)",
    "energy": "Energy dist.",
    "wenergy": "wEnergy dist. (syn)",
}

VC_METRIC_MAP: Dict[str, str] = {
    "wmse_vc": "wmse",
    "wpearson_dp_vc": "wpearson_dp",
}

ASYMMETRIC_METRICS = {
    "wmse", "wmse_vc", "wpearson", "wpearson_dp", "wpearson_dp_vc",
    "wpearson_dc", "wr2_dp", "top20_mse", "top20_pearson_dp", "top100_mse",
    "wenergy", "top100_energy",
}

CHECKPOINT_DATASETS = "pds_test_completed.json"
CHECKPOINT_META = "pds_test_checkpoint_meta.json"
DETAIL_CSV = "pds_test_detail.csv"
SUMMARY_CSV = "pds_test_summary.csv"

MEAN_BASED_METRICS = {
    "mse", "wmse", "wmse_vc", "pearson", "wpearson",
    "pearson_dp", "wpearson_dp", "wpearson_dp_vc",
    "pearson_dc", "wpearson_dc",
    "r2_dp", "wr2_dp", "top20_mse", "top20_pearson_dp", "top100_mse",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_dataset(
    dataset_name: str,
    max_perturbations: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], str, List[str]]:
    """Load dataset and return (X, y, genes, control_label, perturbations)."""
    ns = argparse.Namespace(
        max_perturbations=max_perturbations,
        perturbation_sort="top",
        cells_per_perturbation=None,
        max_cells=None,
        seed=seed,
        control_label=None,
    )
    data = base._prepare_data(ns, dataset_name)
    X = np.asarray(data.X, dtype=float)
    y = np.asarray(data.y, dtype=str)
    perts = [p for p in data.perturbations if p != data.control_label]
    return X, y, data.genes, data.control_label, perts


def _load_synthetic_weights(
    dataset_name: str,
    genes: Sequence[str],
    perturbations: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Load synthetic DEG weights matching benchmark config (minmax, exponent=2)."""
    csv_path = (
        PROJECT_ROOT
        / "analyses"
        / "perturbation_discrimination"
        / "results"
        / dataset_name
        / "deg_scanpy"
        / "deg_synthetic.csv"
    )
    return _load_deg_weights_from_csv(
        csv_path=csv_path,
        genes=genes,
        perturbations=perturbations,
        deg_weight_transform="minmax",
        base_weight_exponent=2.0,
    )


def _load_vscontrol_weights(
    dataset_name: str,
    genes: Sequence[str],
    perturbations: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Load vs-control DEG weights (same transform as synthetic: minmax, exponent=2)."""
    csv_path = (
        PROJECT_ROOT
        / "analyses"
        / "perturbation_discrimination"
        / "results"
        / dataset_name
        / "deg_scanpy"
        / "deg_control.csv"
    )
    return _load_deg_weights_from_csv(
        csv_path=csv_path,
        genes=genes,
        perturbations=perturbations,
        deg_weight_transform="minmax",
        base_weight_exponent=2.0,
    )


def _build_topk_indices(
    weights_per_pert: Dict[str, np.ndarray],
    k: int,
) -> Dict[str, np.ndarray]:
    """For each perturbation, indices of the top ``k`` genes by synthetic DEG weight."""
    out: Dict[str, np.ndarray] = {}
    for pert, w in weights_per_pert.items():
        kk = min(k, len(w))
        out[pert] = np.argsort(w)[-kk:]
    return out


def _build_top20_indices(
    weights_per_pert: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """For each perturbation, get the indices of the top 20 genes by weight."""
    return _build_topk_indices(weights_per_pert, 20)


# ---------------------------------------------------------------------------
# Bag splitting
# ---------------------------------------------------------------------------


def _split_half_bags(
    X: np.ndarray,
    y: np.ndarray,
    perturbations: List[str],
    seed: int,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Split each perturbation's cells into two half-bags.

    Returns
    -------
    bag_indices : list of 2K arrays of cell indices (bag 2k, 2k+1 come from pert k)
    bag_labels  : (2K,) perturbation label per bag
    bag_groups  : (2K,) integer group id (0..K-1), two bags per group
    """
    rng = np.random.default_rng(seed)
    bag_indices: List[np.ndarray] = []
    bag_labels: List[str] = []
    bag_groups: List[int] = []

    for k, pert in enumerate(perturbations):
        idx = np.where(y == pert)[0]
        if idx.size < 4:
            continue
        rng.shuffle(idx)
        mid = len(idx) // 2
        bag_indices.append(idx[:mid])
        bag_indices.append(idx[mid:])
        bag_labels.extend([pert, pert])
        bag_groups.extend([k, k])

    return bag_indices, np.asarray(bag_labels, dtype=str), np.asarray(bag_groups, dtype=int)


def _split_two_bags_fixed_cells(
    X: np.ndarray,
    y: np.ndarray,
    perturbations: List[str],
    seed: int,
    cells_per_bag: int,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Two bags per perturbation, each bag exactly ``cells_per_bag`` cells."""
    if cells_per_bag < 1:
        raise ValueError("cells_per_bag must be >= 1")
    rng = np.random.default_rng(seed)
    need = 2 * cells_per_bag
    bag_indices: List[np.ndarray] = []
    bag_labels: List[str] = []
    bag_groups: List[int] = []

    for k, pert in enumerate(perturbations):
        idx = np.where(y == pert)[0]
        if idx.size < need:
            continue
        rng.shuffle(idx)
        take = idx[:need]
        bag_indices.append(take[:cells_per_bag].astype(int, copy=False))
        bag_indices.append(take[cells_per_bag:].astype(int, copy=False))
        bag_labels.extend([pert, pert])
        bag_groups.extend([k, k])

    return bag_indices, np.asarray(bag_labels, dtype=str), np.asarray(bag_groups, dtype=int)


def _split_single_cell_bags(
    X: np.ndarray,
    y: np.ndarray,
    perturbations: List[str],
    seed: int,
    max_cells_per_pert: Optional[int] = None,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """One cell per bag (no pooling)."""
    rng = np.random.default_rng(seed)
    bag_indices: List[np.ndarray] = []
    bag_labels: List[str] = []
    bag_groups: List[int] = []

    for k, pert in enumerate(perturbations):
        idx = np.where(y == pert)[0]
        if idx.size < 2:
            continue
        rng.shuffle(idx)
        if max_cells_per_pert is not None and idx.size > max_cells_per_pert:
            idx = idx[:max_cells_per_pert]
        for cell_idx in idx:
            bag_indices.append(np.array([int(cell_idx)], dtype=int))
            bag_labels.append(pert)
            bag_groups.append(k)

    return bag_indices, np.asarray(bag_labels, dtype=str), np.asarray(bag_groups, dtype=int)


def split_bags(
    mode: str,
    X: np.ndarray,
    y: np.ndarray,
    perturbations: List[str],
    seed: int,
    cells_per_bag: int = 2,
    max_cells_per_pert: Optional[int] = None,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Dispatch bag construction: ``subsample``, ``half``, or ``single``."""
    if mode == "half":
        return _split_half_bags(X, y, perturbations, seed)
    if mode == "subsample":
        return _split_two_bags_fixed_cells(
            X, y, perturbations, seed, cells_per_bag=cells_per_bag,
        )
    if mode == "single":
        return _split_single_cell_bags(
            X, y, perturbations, seed, max_cells_per_pert=max_cells_per_pert,
        )
    raise ValueError(
        f"Unknown bag mode: {mode!r} (use 'half', 'subsample', or 'single')",
    )


# ---------------------------------------------------------------------------
# Distance matrix construction
# ---------------------------------------------------------------------------


def _bag_means(X: np.ndarray, bag_indices: List[np.ndarray]) -> np.ndarray:
    """Compute mean expression per bag -> (n_bags, n_genes)."""
    return np.array([X[idx].mean(axis=0) for idx in bag_indices])


def _weighted_pearson_vec(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson correlation between two vectors (matches metrics_engine)."""
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        return 0.0
    mu_a = float(np.average(a, weights=w))
    mu_b = float(np.average(b, weights=w))
    ca, cb = a - mu_a, b - mu_b
    cov = float(np.average(ca * cb, weights=w))
    va = float(np.average(ca ** 2, weights=w))
    vb = float(np.average(cb ** 2, weights=w))
    if va <= 0.0 or vb <= 0.0:
        return 0.0
    return cov / np.sqrt(va * vb)


def _r2_score_safe(y_true: np.ndarray, y_pred: np.ndarray,
                   sample_weight: Optional[np.ndarray] = None) -> float:
    """R² with safety for constant vectors."""
    if y_true.size < 2:
        return 0.0
    if np.std(y_true) < 1e-12:
        return 0.0
    if sample_weight is not None:
        return float(r2_score(y_true, y_pred, sample_weight=sample_weight))
    return float(r2_score(y_true, y_pred))


def _pearson_vec(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation matching scipy.stats.pearsonr (center, dot/norms)."""
    ca = a - a.mean()
    cb = b - b.mean()
    denom = np.sqrt(np.sum(ca ** 2) * np.sum(cb ** 2))
    if denom < 1e-12:
        return 0.0
    return float(np.clip(np.sum(ca * cb) / denom, -1.0, 1.0))


def _energy_distance(Xa: np.ndarray, Xb: np.ndarray, w: np.ndarray) -> float:
    """Energy distance between two cell clouds with weighted Euclidean norm."""
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, None)
    w_sum = float(np.sum(w))
    if w_sum < 1e-12:
        wn = np.full(Xa.shape[1], 1.0 / max(Xa.shape[1], 1))
    else:
        wn = w / w_sum

    def _pdist(A: np.ndarray, B: np.ndarray) -> float:
        sq = np.sum(((A[:, None, :] - B[None, :, :]) ** 2) * wn[None, None, :], axis=2)
        return float(np.mean(np.sqrt(np.maximum(sq, 0.0))))

    return 2.0 * _pdist(Xa, Xb) - _pdist(Xa, Xa) - _pdist(Xb, Xb)


def _energy_pair_worker(args: Tuple) -> Tuple[int, int, float]:
    """Compute energy distance for one (i, j) bag pair (subprocess-friendly)."""
    Xi, Xj, w, i, j = args
    return i, j, _energy_distance(Xi, Xj, w)


def compute_distance_matrix(
    metric: str,
    X: np.ndarray,
    bag_indices: List[np.ndarray],
    bag_labels: np.ndarray,
    dataset_mean: np.ndarray,
    control_mean: np.ndarray,
    weights_per_pert: Dict[str, np.ndarray],
    top20_per_pert: Dict[str, np.ndarray],
    top100_per_pert: Optional[Dict[str, np.ndarray]] = None,
    n_workers: int = 1,
    show_progress: bool = True,
    progress_desc: str = "",
    vc_weights_per_pert: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Compute (possibly asymmetric) bag-to-bag distance matrix.

    For weighted/asymmetric metrics each row uses the row-bag's perturbation
    weights (query-weighted), matching the benchmark convention.
    The matrix is NOT symmetrized; the Q/R test handles asymmetry naturally.
    """
    if metric in VC_METRIC_MAP:
        if vc_weights_per_pert is None:
            raise ValueError(f"Metric {metric} requires vc_weights_per_pert")
        metric = VC_METRIC_MAP[metric]
        weights_per_pert = vc_weights_per_pert

    n = len(bag_indices)
    n_genes = X.shape[1]
    bar = progress_desc or "D_bags"
    t100 = top100_per_pert or {}

    if metric in MEAN_BASED_METRICS:
        means = _bag_means(X, bag_indices)

    # --- mse ---
    if metric == "mse":
        return np.mean(np.square(means[:, None, :] - means[None, :, :]), axis=2)

    # --- wmse ---
    if metric == "wmse":
        D = np.zeros((n, n), dtype=float)
        row_iter = range(n)
        if show_progress:
            row_iter = tqdm(row_iter, total=n, desc=f"{bar} rows", unit="row", leave=True)
        for i in row_iter:
            w = weights_per_pert.get(bag_labels[i], np.ones(n_genes, dtype=float))
            w_sum = float(np.sum(w))
            wn = w / w_sum if w_sum > 1e-12 else np.full(n_genes, 1.0 / max(n_genes, 1))
            diff_sq = np.square(means[i] - means)
            D[i, :] = diff_sq @ wn
        return D

    # --- pearson (raw expression, no delta) ---
    if metric == "pearson":
        norms = np.linalg.norm(means - means.mean(axis=1, keepdims=True), axis=1)
        centered = means - means.mean(axis=1, keepdims=True)
        gram = centered @ centered.T
        outer = np.maximum(norms[:, None] * norms[None, :], 1e-12)
        corr = np.clip(gram / outer, -1.0, 1.0)
        return np.nan_to_num(1.0 - corr, nan=1.0)

    # --- wpearson (raw expression, no delta, query-weighted) ---
    if metric == "wpearson":
        D = np.zeros((n, n), dtype=float)
        row_iter = range(n)
        if show_progress:
            row_iter = tqdm(row_iter, total=n, desc=f"{bar} rows", unit="row", leave=True)
        for i in row_iter:
            w = weights_per_pert.get(bag_labels[i], np.ones(n_genes, dtype=float))
            for j in range(n):
                D[i, j] = 1.0 - _weighted_pearson_vec(means[i], means[j], w)
        return D

    # --- pearson_dp (delta pert) ---
    if metric == "pearson_dp":
        deltas = means - dataset_mean[None, :]
        norms = np.linalg.norm(deltas - deltas.mean(axis=1, keepdims=True), axis=1)
        centered = deltas - deltas.mean(axis=1, keepdims=True)
        gram = centered @ centered.T
        outer = np.maximum(norms[:, None] * norms[None, :], 1e-12)
        corr = np.clip(gram / outer, -1.0, 1.0)
        return np.nan_to_num(1.0 - corr, nan=1.0)

    # --- wpearson_dp (weighted delta pert) ---
    if metric == "wpearson_dp":
        deltas = means - dataset_mean[None, :]
        D = np.zeros((n, n), dtype=float)
        row_iter = range(n)
        if show_progress:
            row_iter = tqdm(row_iter, total=n, desc=f"{bar} rows", unit="row", leave=True)
        for i in row_iter:
            w = weights_per_pert.get(bag_labels[i], np.ones(n_genes, dtype=float))
            for j in range(n):
                D[i, j] = 1.0 - _weighted_pearson_vec(deltas[i], deltas[j], w)
        return D

    # --- pearson_dc (delta ctrl) ---
    if metric == "pearson_dc":
        deltas = means - control_mean[None, :]
        norms = np.linalg.norm(deltas - deltas.mean(axis=1, keepdims=True), axis=1)
        centered = deltas - deltas.mean(axis=1, keepdims=True)
        gram = centered @ centered.T
        outer = np.maximum(norms[:, None] * norms[None, :], 1e-12)
        corr = np.clip(gram / outer, -1.0, 1.0)
        return np.nan_to_num(1.0 - corr, nan=1.0)

    # --- wpearson_dc (weighted delta ctrl) ---
    if metric == "wpearson_dc":
        deltas = means - control_mean[None, :]
        D = np.zeros((n, n), dtype=float)
        row_iter = range(n)
        if show_progress:
            row_iter = tqdm(row_iter, total=n, desc=f"{bar} rows", unit="row", leave=True)
        for i in row_iter:
            w = weights_per_pert.get(bag_labels[i], np.ones(n_genes, dtype=float))
            for j in range(n):
                D[i, j] = 1.0 - _weighted_pearson_vec(deltas[i], deltas[j], w)
        return D

    # --- r2_dp: 1 - R²(delta_pert). D[i,j] treats j as truth, i as pred. ---
    if metric == "r2_dp":
        deltas = means - dataset_mean[None, :]
        D = np.zeros((n, n), dtype=float)
        row_iter = range(n)
        if show_progress:
            row_iter = tqdm(row_iter, total=n, desc=f"{bar} rows", unit="row", leave=True)
        for i in row_iter:
            for j in range(n):
                D[i, j] = 1.0 - _r2_score_safe(deltas[j], deltas[i])
        return D

    # --- wr2_dp: 1 - wR²(delta_pert), query-weighted ---
    if metric == "wr2_dp":
        deltas = means - dataset_mean[None, :]
        D = np.zeros((n, n), dtype=float)
        row_iter = range(n)
        if show_progress:
            row_iter = tqdm(row_iter, total=n, desc=f"{bar} rows", unit="row", leave=True)
        for i in row_iter:
            w = weights_per_pert.get(bag_labels[i], np.ones(n_genes, dtype=float))
            for j in range(n):
                D[i, j] = 1.0 - _r2_score_safe(deltas[j], deltas[i], sample_weight=w)
        return D

    # --- top20_mse: MSE on top 20 DEGs (query-weighted selection) ---
    if metric == "top20_mse":
        D = np.zeros((n, n), dtype=float)
        row_iter = range(n)
        if show_progress:
            row_iter = tqdm(row_iter, total=n, desc=f"{bar} rows", unit="row", leave=True)
        for i in row_iter:
            idx20 = top20_per_pert.get(bag_labels[i])
            if idx20 is None:
                D[i, :] = np.nan
                continue
            mi = means[i, idx20]
            D[i, :] = np.mean(np.square(mi[None, :] - means[:, idx20]), axis=1)
        return D

    # --- top20_pearson_dp: 1 - Pearson(delta_pert) on top 20 DEGs ---
    if metric == "top20_pearson_dp":
        deltas = means - dataset_mean[None, :]
        D = np.zeros((n, n), dtype=float)
        row_iter = range(n)
        if show_progress:
            row_iter = tqdm(row_iter, total=n, desc=f"{bar} rows", unit="row", leave=True)
        for i in row_iter:
            idx20 = top20_per_pert.get(bag_labels[i])
            if idx20 is None:
                D[i, :] = np.nan
                continue
            di = deltas[i, idx20]
            for j in range(n):
                dj = deltas[j, idx20]
                D[i, j] = 1.0 - _pearson_vec(di, dj)
        return D

    # --- top100_mse: MSE on top 100 DEGs (query gene set, same convention as top20_mse) ---
    if metric == "top100_mse":
        D = np.zeros((n, n), dtype=float)
        row_iter = range(n)
        if show_progress:
            row_iter = tqdm(row_iter, total=n, desc=f"{bar} rows", unit="row", leave=True)
        for i in row_iter:
            idx100 = t100.get(bag_labels[i])
            if idx100 is None:
                D[i, :] = np.nan
                continue
            mi = means[i, idx100]
            D[i, :] = np.mean(np.square(mi[None, :] - means[:, idx100]), axis=1)
        return D

    # --- energy (symmetric, unweighted) ---
    if metric == "energy":
        uniform_w = np.ones(n_genes, dtype=float)
        D = np.zeros((n, n), dtype=float)
        pair_args = [
            (X[bag_indices[i]], X[bag_indices[j]], uniform_w, i, j)
            for i in range(n) for j in range(i + 1, n)
        ]
        n_pairs = len(pair_args)
        n_pool = max(1, min(n_workers, n_pairs))
        desc = f"{bar} pairs"

        if n_pool <= 1:
            pbar = tqdm(total=n_pairs, desc=desc, unit="pair", leave=True) if show_progress else None
            for pa in pair_args:
                i, j, d = _energy_pair_worker(pa)
                D[i, j] = d
                D[j, i] = d
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()
        else:
            with mp.Pool(processes=n_pool) as pool:
                if show_progress:
                    pbar = tqdm(total=n_pairs, desc=desc, unit="pair", leave=True)
                    for i, j, d in pool.imap_unordered(
                        _energy_pair_worker, pair_args, chunksize=max(1, n_pairs // (n_pool * 4)),
                    ):
                        D[i, j] = d
                        D[j, i] = d
                        pbar.update(1)
                    pbar.close()
                else:
                    for i, j, d in pool.imap_unordered(
                        _energy_pair_worker, pair_args, chunksize=max(1, n_pairs // (n_pool * 4)),
                    ):
                        D[i, j] = d
                        D[j, i] = d
        return D

    # --- wenergy (asymmetric, query-weighted) ---
    if metric == "wenergy":
        D = np.zeros((n, n), dtype=float)
        pair_args = []
        for i in range(n):
            Xi = X[bag_indices[i]]
            w = weights_per_pert.get(bag_labels[i], np.ones(n_genes, dtype=float))
            for j in range(n):
                if i == j:
                    continue
                pair_args.append((Xi, X[bag_indices[j]], w, i, j))
        n_pairs = len(pair_args)
        n_pool = max(1, min(n_workers, n_pairs))
        desc = f"{bar} pairs"

        if n_pool <= 1:
            pbar = tqdm(total=n_pairs, desc=desc, unit="pair", leave=True) if show_progress else None
            for pa in pair_args:
                i, j, d = _energy_pair_worker(pa)
                D[i, j] = d
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()
        else:
            with mp.Pool(processes=n_pool) as pool:
                if show_progress:
                    pbar = tqdm(total=n_pairs, desc=desc, unit="pair", leave=True)
                    for i, j, d in pool.imap_unordered(
                        _energy_pair_worker, pair_args, chunksize=max(1, n_pairs // (n_pool * 4)),
                    ):
                        D[i, j] = d
                        pbar.update(1)
                    pbar.close()
                else:
                    for i, j, d in pool.imap_unordered(
                        _energy_pair_worker, pair_args, chunksize=max(1, n_pairs // (n_pool * 4)),
                    ):
                        D[i, j] = d
        return D

    # --- top100_energy: energy on raw cells in query's top-100 DEG subspace (uniform within R^100) ---
    if metric == "top100_energy":
        D = np.zeros((n, n), dtype=float)
        pair_args: List[Tuple] = []
        for i in range(n):
            idx100 = t100.get(bag_labels[i])
            if idx100 is None:
                continue
            Xi = X[bag_indices[i]][:, idx100]
            uniform_w = np.ones(len(idx100), dtype=float)
            for j in range(n):
                if i == j:
                    continue
                Xj = X[bag_indices[j]][:, idx100]
                pair_args.append((Xi, Xj, uniform_w, i, j))
        n_pairs = len(pair_args)
        n_pool = max(1, min(n_workers, n_pairs)) if n_pairs else 1
        desc = f"{bar} pairs"

        if n_pairs == 0:
            return D

        if n_pool <= 1:
            pbar = tqdm(total=n_pairs, desc=desc, unit="pair", leave=True) if show_progress else None
            for pa in pair_args:
                i, j, d = _energy_pair_worker(pa)
                D[i, j] = d
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()
        else:
            with mp.Pool(processes=n_pool) as pool:
                if show_progress:
                    pbar = tqdm(total=n_pairs, desc=desc, unit="pair", leave=True)
                    for i, j, d in pool.imap_unordered(
                        _energy_pair_worker, pair_args, chunksize=max(1, n_pairs // (n_pool * 4)),
                    ):
                        D[i, j] = d
                        pbar.update(1)
                    pbar.close()
                else:
                    for i, j, d in pool.imap_unordered(
                        _energy_pair_worker, pair_args, chunksize=max(1, n_pairs // (n_pool * 4)),
                    ):
                        D[i, j] = d
        return D

    raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# Rank-based PDS test (Q/R formulation)
# ---------------------------------------------------------------------------


def _compute_pair_ranks(
    D_bags: np.ndarray,
    idx_Q: np.ndarray,
    idx_R: np.ndarray,
) -> np.ndarray:
    """Q/R rank computation: rank d(Q_p, R_p) within {d(Q_p, R_q) : q=1..K}.

    Parameters
    ----------
    D_bags : (2K, 2K) distance matrix (possibly asymmetric).
    idx_Q  : (K,) query bag indices.
    idx_R  : (K,) reference bag indices.

    Returns
    -------
    ranks : (K,) rank of within-pair distance in each Q row.
            Rank 1 = smallest distance = best discrimination.
    """
    C = D_bags[np.ix_(idx_Q, idx_R)]
    K = idx_Q.shape[0]
    ranks = np.empty(K, dtype=float)
    for k in range(K):
        ranks[k] = rankdata(C[k, :], method="average")[k]
    return ranks


def _rank_permutation_test(
    D_bags: np.ndarray,
    true_idx_Q: np.ndarray,
    true_idx_R: np.ndarray,
    perturbation_names: List[str],
    n_permutations: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run the rank-based permutation test on a precomputed D_bags matrix.

    Null: permute perturbation labels across all 2K half-bags (preserving
    bag structure), then re-assign Q/R roles.
    """
    t0 = time.perf_counter()
    n_bags = D_bags.shape[0]
    K = true_idx_Q.shape[0]
    rng = np.random.default_rng(seed)

    R_obs = _compute_pair_ranks(D_bags, true_idx_Q, true_idx_R)

    null_ranks = np.empty((n_permutations, K), dtype=float)
    all_bag_indices = np.arange(n_bags)

    for p in range(n_permutations):
        shuffled = rng.permutation(all_bag_indices)
        null_ranks[p] = _compute_pair_ranks(D_bags, shuffled[0::2], shuffled[1::2])

    pvals_raw = (np.sum(null_ranks <= R_obs[np.newaxis, :], axis=0) + 1) / (
        n_permutations + 1
    )

    _, pvals_adj, _, _ = multipletests(pvals_raw, alpha=alpha, method="fdr_bh")

    fraction_significant = float(np.mean(pvals_adj < alpha))
    mean_pvalue = float(np.mean(pvals_raw))

    return {
        "pvals_raw": pvals_raw,
        "pvals_adj": pvals_adj,
        "fraction_significant": fraction_significant,
        "mean_pvalue": mean_pvalue,
        "time_seconds": time.perf_counter() - t0,
        "perturbation_names": perturbation_names,
        "R_obs": R_obs,
        "null_ranks": null_ranks,
    }


# ---------------------------------------------------------------------------
# Per-metric driver
# ---------------------------------------------------------------------------


def run_metric_pds_test(
    metric: str,
    X: np.ndarray,
    bag_indices: List[np.ndarray],
    bag_labels: np.ndarray,
    bag_groups: np.ndarray,
    dataset_mean: np.ndarray,
    control_mean: np.ndarray,
    weights_per_pert: Dict[str, np.ndarray],
    top20_per_pert: Dict[str, np.ndarray],
    n_permutations: int,
    alpha: float,
    seed: int,
    n_workers: int = 1,
    show_progress: bool = True,
    vc_weights_per_pert: Optional[Dict[str, np.ndarray]] = None,
    top100_per_pert: Optional[Dict[str, np.ndarray]] = None,
) -> Dict:
    """Compute D_bags + PDS rank-based test for one metric.

    No symmetrization: the Q/R formulation handles asymmetry naturally.
    """
    t0 = time.perf_counter()
    progress_label = METRIC_DISPLAY.get(metric, metric)
    dbar = f"    {progress_label} D_bags"

    D_bags = compute_distance_matrix(
        metric, X, bag_indices, bag_labels, dataset_mean, control_mean,
        weights_per_pert, top20_per_pert,
        top100_per_pert=top100_per_pert,
        n_workers=n_workers, show_progress=show_progress,
        progress_desc=dbar,
        vc_weights_per_pert=vc_weights_per_pert,
    )
    np.fill_diagonal(D_bags, 0.0)

    dist_time = time.perf_counter() - t0

    unique_groups = np.unique(bag_groups)
    K = len(unique_groups)
    true_idx_Q = np.empty(K, dtype=int)
    true_idx_R = np.empty(K, dtype=int)
    pert_names: List[str] = []

    for k_out, g in enumerate(unique_groups):
        bag_idxs = np.where(bag_groups == g)[0]
        true_idx_Q[k_out] = bag_idxs[0]
        true_idx_R[k_out] = bag_idxs[1]
        pert_names.append(str(bag_labels[bag_idxs[0]]))

    result = _rank_permutation_test(
        D_bags, true_idx_Q, true_idx_R, pert_names,
        n_permutations=n_permutations, alpha=alpha, seed=seed,
    )
    result["metric"] = metric
    result["dist_time"] = dist_time
    result["time_seconds"] = time.perf_counter() - t0

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PDS-based testing: rank-based permutation test for metric "
        "inter-perturbation discrimination signal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DEFAULT_DATASETS,
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=METRICS,
        help="Metrics to compare.",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=1000,
        help="Number of permutations for the null distribution.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=1,
        help="Independent random half-splits (averaged results).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="FDR significance threshold.",
    )
    parser.add_argument(
        "--max-perturbations", type=int, default=None,
        help="Limit to top-N perturbations by cell count.",
    )
    parser.add_argument(
        "--bag-mode",
        choices=("subsample", "half", "single"),
        default="half",
        help="half (default): split all cells into two half-bags. "
        "subsample: two bags/pert, each with --cells-per-bag cells. "
        "single: one cell per bag.",
    )
    parser.add_argument(
        "--cells-per-bag",
        type=int,
        default=2,
        help="With --bag-mode subsample: cells per bag (requires >= 2x per pert).",
    )
    parser.add_argument(
        "--max-cells-per-pert",
        type=int,
        default=None,
        help="With --bag-mode single: cap cells per perturbation.",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel workers for distance matrix (default: cpu_count).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=str,
        default="analyses/perturbation_discrimination/results/pds_test",
    )
    parser.add_argument(
        "--include-energy",
        action="store_true",
        help="Include energy and wenergy metrics (slow for large datasets).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from existing output: skip datasets listed in the checkpoint, "
        "reload saved rows. Requires matching run settings (see checkpoint meta).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing PDS output files in --output-dir before running (ignored with --resume).",
    )
    return parser.parse_args()


def _checkpoint_meta_dict(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "metrics": sorted(args.metrics),
        "n_trials": args.n_trials,
        "n_permutations": args.n_permutations,
        "bag_mode": args.bag_mode,
        "cells_per_bag": args.cells_per_bag,
        "alpha": args.alpha,
        "seed": args.seed,
        "max_perturbations": args.max_perturbations,
        "max_cells_per_pert": args.max_cells_per_pert,
    }


def _meta_compatible(saved: Dict[str, Any], current: Dict[str, Any]) -> bool:
    return saved == current


def _rebuild_summary_rows_from_detail(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Reconstruct dataset-level summary rows from per-perturbation detail (for --resume)."""
    if df.empty:
        return []
    work = df.copy()
    if "significant" in work.columns:
        sig = work["significant"]
        work["significant"] = sig.apply(
            lambda x: x is True or x == "True" or x is np.bool_(True) or x == 1,
        )
    group_keys = [
        "dataset", "bag_mode", "cells_per_bag", "metric", "metric_display", "trial",
    ]
    rows: List[Dict[str, Any]] = []
    for _, g in work.groupby(group_keys, sort=False):
        first = g.iloc[0]
        n_perts = int(g["perturbation"].nunique())
        rows.append({
            "dataset": first["dataset"],
            "bag_mode": first["bag_mode"],
            "cells_per_bag": first["cells_per_bag"],
            "metric": first["metric"],
            "metric_display": first["metric_display"],
            "trial": int(first["trial"]),
            "fraction_significant": float(g["significant"].mean()),
            "mean_pvalue": float(g["raw_pval"].mean()),
            "n_perturbations": n_perts,
            "n_bags": n_perts * 2,
            "time_seconds": float("nan"),
        })
    return rows


def _aggregate_summary(summary_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    summary_df = pd.DataFrame(summary_rows)
    return (
        summary_df.groupby(["dataset", "bag_mode", "cells_per_bag", "metric", "metric_display"])
        .agg(
            mean_frac_sig=("fraction_significant", "mean"),
            mean_pvalue=("mean_pvalue", "mean"),
            n_perturbations=("n_perturbations", "first"),
        )
        .reset_index()
    )


def _atomic_write_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, float_format="%.6f")
    os.replace(tmp, path)


def _write_pds_checkpoint(
    out_dir: Path,
    detail_rows: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    completed_datasets: List[str],
    meta: Dict[str, Any],
) -> None:
    """Persist detail + aggregated summary and checkpoint sidecars (atomic CSV writes)."""
    detail_path = out_dir / DETAIL_CSV
    summary_path = out_dir / SUMMARY_CSV
    detail_df = pd.DataFrame(detail_rows)
    _atomic_write_dataframe(detail_path, detail_df)
    agg = _aggregate_summary(summary_rows)
    _atomic_write_dataframe(summary_path, agg)
    ck_datasets = out_dir / CHECKPOINT_DATASETS
    ck_meta = out_dir / CHECKPOINT_META
    ck_datasets.write_text(json.dumps(sorted(set(completed_datasets)), indent=2) + "\n")
    ck_meta.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  Checkpoint saved ({len(completed_datasets)} dataset(s)) → {detail_path}", flush=True)


def _clear_pds_outputs(out_dir: Path) -> None:
    for name in (DETAIL_CSV, SUMMARY_CSV, CHECKPOINT_DATASETS, CHECKPOINT_META):
        p = out_dir / name
        if p.exists():
            p.unlink()


def main() -> None:
    args = parse_args()
    if args.bag_mode == "subsample" and args.cells_per_bag < 1:
        raise SystemExit("--cells-per-bag must be >= 1 when using --bag-mode subsample")

    if args.include_energy:
        for em in ENERGY_METRICS:
            if em not in args.metrics:
                args.metrics.append(em)

    has_vc_metrics = any(m in VC_METRIC_MAP for m in args.metrics)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and args.overwrite:
        raise SystemExit("Use only one of --resume or --overwrite.")

    meta = _checkpoint_meta_dict(args)
    detail_path = out_dir / DETAIL_CSV
    meta_path = out_dir / CHECKPOINT_META
    ck_path = out_dir / CHECKPOINT_DATASETS

    if args.overwrite:
        _clear_pds_outputs(out_dir)

    completed_datasets: List[str] = []
    detail_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    if args.resume:
        if not detail_path.exists():
            print("Resume: no existing detail file; starting fresh.", flush=True)
        else:
            if meta_path.exists():
                saved_meta = json.loads(meta_path.read_text())
                if not _meta_compatible(saved_meta, meta):
                    raise SystemExit(
                        "Checkpoint meta does not match current arguments.\n"
                        f"  Saved:   {saved_meta}\n"
                        f"  Current: {meta}\n"
                        "Use --overwrite to discard the old run or match the original settings."
                    )
            else:
                print(
                    "WARNING: checkpoint meta file missing; cannot verify run settings.",
                    flush=True,
                )
            df_loaded = pd.read_csv(detail_path)
            detail_rows = df_loaded.to_dict("records")
            summary_rows = _rebuild_summary_rows_from_detail(df_loaded)
            if ck_path.exists():
                completed_datasets = list(json.loads(ck_path.read_text()))
            else:
                completed_datasets = sorted(df_loaded["dataset"].unique().tolist())
            print(
                f"Resume: loaded {len(completed_datasets)} completed dataset(s), "
                f"{len(detail_rows)} detail rows.",
                flush=True,
            )
    elif detail_path.exists() or ck_path.exists() or meta_path.exists():
        raise SystemExit(
            f"Output already exists under {out_dir}. "
            "Pass --resume to continue or --overwrite to replace."
        )

    pending = [d for d in args.datasets if d not in set(completed_datasets)]
    if not pending:
        print("All requested datasets are already in the checkpoint.", flush=True)
    else:
        print(f"Datasets to run ({len(pending)}): {', '.join(pending)}", flush=True)

    n_workers = args.workers or mp.cpu_count()
    print(f"Workers: {n_workers} (parallelises distance matrix; permutations are sequential)")
    print(f"Bag mode: {args.bag_mode}", flush=True)

    for ds_idx, dataset_name in enumerate(pending, 1):
        print(f"\n{'='*60}")
        print(f"  [{ds_idx}/{len(pending)}] Dataset: {dataset_name}")
        print(f"{'='*60}")
        t0 = time.perf_counter()

        try:
            X, y, genes, ctrl_label, perts = _load_dataset(
                dataset_name, max_perturbations=args.max_perturbations, seed=args.seed,
            )
        except Exception as e:
            print(f"  SKIP {dataset_name}: {e}")
            continue

        print(f"  Loaded {X.shape[0]} cells, {len(perts)} perturbations, {X.shape[1]} genes")

        try:
            syn_weights = _load_synthetic_weights(dataset_name, genes, perts)
        except FileNotFoundError as e:
            print(f"  SKIP {dataset_name}: no synthetic weights – {e}")
            continue
        print(f"  Synthetic weights for {len(syn_weights)}/{len(perts)} perturbations")

        vc_weights: Optional[Dict[str, np.ndarray]] = None
        if has_vc_metrics:
            try:
                vc_weights = _load_vscontrol_weights(dataset_name, genes, perts)
                print(f"  Vscontrol weights for {len(vc_weights)}/{len(perts)} perturbations")
            except FileNotFoundError as e:
                print(f"  WARNING: no vscontrol weights – {e}; vc metrics will be skipped")

        perts_with_weights = [p for p in perts if p in syn_weights]
        if len(perts_with_weights) < 3:
            print(f"  SKIP {dataset_name}: only {len(perts_with_weights)} perturbations with weights")
            continue

        mask = np.isin(y, perts_with_weights) | (y == ctrl_label)
        X_sub = X[mask]
        y_sub = y[mask]
        dataset_mean = X_sub.mean(axis=0)
        control_mean = X_sub[y_sub == ctrl_label].mean(axis=0)

        top20_per_pert = (
            _build_topk_indices(syn_weights, 20)
            if any(m in args.metrics for m in ("top20_mse", "top20_pearson_dp"))
            else {}
        )
        top100_per_pert = (
            _build_topk_indices(syn_weights, 100)
            if any(m in args.metrics for m in ("top100_mse", "top100_energy"))
            else {}
        )

        empty_split = False
        for trial in range(args.n_trials):
            trial_seed = args.seed + trial
            bag_indices, bag_labels, bag_groups = split_bags(
                args.bag_mode,
                X_sub,
                y_sub,
                perts_with_weights,
                seed=trial_seed,
                cells_per_bag=args.cells_per_bag,
                max_cells_per_pert=args.max_cells_per_pert,
            )
            n_bags = len(bag_indices)
            K = len(np.unique(bag_groups))
            if n_bags == 0:
                print(
                    f"  SKIP {dataset_name}: no bags",
                    flush=True,
                )
                empty_split = True
                break
            print(f"  Trial {trial+1}/{args.n_trials}: {n_bags} bags from {K} perturbations")

            show_progress = not args.no_progress
            print(
                f"    Running {len(args.metrics)} metrics sequentially "
                f"({n_workers} workers for distance matrices) ...",
                flush=True,
            )
            mt0 = time.perf_counter()
            results: List[Dict] = []
            for mi, metric in enumerate(args.metrics, 1):
                mlabel = METRIC_DISPLAY.get(metric, metric)
                print(
                    f"    [{mi}/{len(args.metrics)}] Starting {mlabel} ...",
                    flush=True,
                )
                if metric in VC_METRIC_MAP and vc_weights is None:
                    print(f"    SKIP {mlabel}: no vscontrol weights", flush=True)
                    continue
                res = run_metric_pds_test(
                    metric, X_sub, bag_indices, bag_labels, bag_groups,
                    dataset_mean, control_mean, syn_weights, top20_per_pert,
                    args.n_permutations, args.alpha,
                    seed=trial_seed + hash(metric) % 100000,
                    n_workers=n_workers,
                    show_progress=show_progress,
                    vc_weights_per_pert=vc_weights,
                    top100_per_pert=top100_per_pert or None,
                )
                results.append(res)

            for res in results:
                label = METRIC_DISPLAY.get(res["metric"], res["metric"])
                frac = res["fraction_significant"]
                mean_p = res["mean_pvalue"]
                print(
                    f"    {label:>25s}  frac_sig={frac:.1%}  mean_p={mean_p:.4f}  "
                    f"(dist {res['dist_time']:.1f}s, total {res['time_seconds']:.1f}s)",
                    flush=True,
                )

                pnames = res["perturbation_names"]
                pvals_raw = res["pvals_raw"]
                pvals_adj = res["pvals_adj"]
                R_obs = res["R_obs"]
                for i_p, pname in enumerate(pnames):
                    detail_rows.append({
                        "dataset": dataset_name,
                        "bag_mode": args.bag_mode,
                        "cells_per_bag": (
                            args.cells_per_bag if args.bag_mode == "subsample" else ""
                        ),
                        "metric": res["metric"],
                        "metric_display": label,
                        "trial": trial,
                        "perturbation": pname,
                        "R_obs": float(R_obs[i_p]),
                        "raw_pval": float(pvals_raw[i_p]),
                        "adj_pval": float(pvals_adj[i_p]),
                        "significant": bool(pvals_adj[i_p] < args.alpha),
                    })

                summary_rows.append({
                    "dataset": dataset_name,
                    "bag_mode": args.bag_mode,
                    "cells_per_bag": (
                        args.cells_per_bag if args.bag_mode == "subsample" else ""
                    ),
                    "metric": res["metric"],
                    "metric_display": label,
                    "trial": trial,
                    "fraction_significant": frac,
                    "mean_pvalue": mean_p,
                    "n_perturbations": K,
                    "n_bags": n_bags,
                    "time_seconds": res["time_seconds"],
                })

            print(f"    All metrics: {time.perf_counter()-mt0:.1f}s", flush=True)

        if empty_split:
            continue

        print(f"  {dataset_name} done in {time.perf_counter()-t0:.1f}s")
        completed_datasets.append(dataset_name)
        _write_pds_checkpoint(
            out_dir, detail_rows, summary_rows, completed_datasets, meta,
        )

    if not detail_rows:
        print("No results collected.")
        return

    detail_path_final = out_dir / DETAIL_CSV
    summary_path_final = out_dir / SUMMARY_CSV
    print(f"\nPer-perturbation detail: {detail_path_final}")
    print(f"Summary: {summary_path_final}")

    agg = _aggregate_summary(summary_rows)

    print(f"\n{'='*80}")
    print(f"Fraction significant (alpha={args.alpha}, FDR-corrected)")
    print(f"{'='*80}")
    pivot = agg.pivot_table(
        index="dataset", columns="metric_display", values="mean_frac_sig",
    )
    col_order = [METRIC_DISPLAY.get(m, m) for m in ALL_KNOWN_METRICS if METRIC_DISPLAY.get(m, m) in pivot.columns]
    pivot = pivot[col_order]
    print(pivot.to_string(float_format=lambda x: f"{x:.1%}"))

    print(f"\n{'='*80}")
    print("Mean raw p-value (lower = better signal detection)")
    print(f"{'='*80}")
    pivot_p = agg.pivot_table(
        index="dataset", columns="metric_display", values="mean_pvalue",
    )
    pivot_p = pivot_p[[c for c in col_order if c in pivot_p.columns]]
    print(pivot_p.to_string(float_format=lambda x: f"{x:.2e}" if x < 0.01 else f"{x:.4f}"))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
