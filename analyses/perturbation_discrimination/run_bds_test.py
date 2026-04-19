#!/usr/bin/env python3
"""BDS test: which metrics detect perturbation signal vs control?

For each perturbation k, we split its cells into two half-bags (B1, B2) and
compute a delta = d_neg - d_pos:

    d_pos = metric(B1, B2)                    (positive baseline / technical duplicate)
    d_neg = 0.5 * (metric(B1, Xc) + metric(B2, Xc))   (negative baseline / control)

A permutation test (pooling pert + control cells, reshuffling) yields a raw
p-value per perturbation.  After BH-FDR correction, sensitivity = fraction of
perturbations detected as significant.

Multiple trials with different random half-bag splits stabilize the result.
Permutations are parallelized across workers (especially helpful for energy
distance, where each permutation is expensive).

Metrics (14 default + 2 energy opt-in):
    mse, wmse, wmse_vc, pearson, wpearson,
    pearson_dp, wpearson_dp, wpearson_dp_vc, pearson_dc, wpearson_dc,
    r2_dp, wr2_dp, top20_mse, top20_pearson_dp,
    energy, wenergy (opt-in via --include-energy)

Usage:
    python run_bds_test.py --datasets frangieh21 --n-permutations 999
    python run_bds_test.py --n-trials 5 --alpha 0.05
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyses.perturbation_discrimination.run_pds_test import (
    DEFAULT_DATASETS,
    METRICS,
    ENERGY_METRICS,
    ALL_KNOWN_METRICS,
    METRIC_DISPLAY,
    MEAN_BASED_METRICS,
    VC_METRIC_MAP,
    _energy_distance,
    _load_dataset,
    _load_synthetic_weights,
    _load_vscontrol_weights,
    _build_top20_indices,
    _weighted_pearson_vec,
    _pearson_vec,
    _r2_score_safe,
)

UNWEIGHTED_METRICS = {
    "mse", "pearson", "pearson_dp", "pearson_dc", "r2_dp", "energy",
}

# ---------------------------------------------------------------------------
# Pairwise bag distance (single scalar)
# ---------------------------------------------------------------------------


def compute_bag_distance(
    metric: str,
    A: np.ndarray,
    B: np.ndarray,
    w: np.ndarray,
    dataset_mean: np.ndarray,
    control_mean: np.ndarray,
    top20_idx: Optional[np.ndarray] = None,
) -> float:
    """Scalar distance between two cell-bags for a given metric.

    Parameters
    ----------
    A, B : (n_a, genes) and (n_b, genes) cell arrays
    w    : per-gene weight vector (query-perturbation weights; ones for unweighted)
    dataset_mean : (genes,) used by delta-pert metrics
    control_mean : (genes,) used by delta-ctrl metrics
    top20_idx    : indices of top 20 DEGs (for top20 metrics)
    """
    metric = VC_METRIC_MAP.get(metric, metric)
    n_genes = A.shape[1]

    if metric == "mse":
        ma, mb = A.mean(axis=0), B.mean(axis=0)
        return float(np.mean(np.square(ma - mb)))

    if metric == "wmse":
        ma, mb = A.mean(axis=0), B.mean(axis=0)
        w_sum = float(np.sum(w))
        wn = w / w_sum if w_sum > 1e-12 else np.full(n_genes, 1.0 / max(n_genes, 1))
        return float(np.sum(np.square(ma - mb) * wn))

    if metric == "pearson":
        ma, mb = A.mean(axis=0), B.mean(axis=0)
        return float(1.0 - _pearson_vec(ma, mb))

    if metric == "wpearson":
        ma, mb = A.mean(axis=0), B.mean(axis=0)
        return float(1.0 - _weighted_pearson_vec(ma, mb, w))

    if metric == "pearson_dp":
        da = A.mean(axis=0) - dataset_mean
        db = B.mean(axis=0) - dataset_mean
        return float(1.0 - _pearson_vec(da, db))

    if metric == "wpearson_dp":
        da = A.mean(axis=0) - dataset_mean
        db = B.mean(axis=0) - dataset_mean
        return float(1.0 - _weighted_pearson_vec(da, db, w))

    if metric == "pearson_dc":
        da = A.mean(axis=0) - control_mean
        db = B.mean(axis=0) - control_mean
        return float(1.0 - _pearson_vec(da, db))

    if metric == "wpearson_dc":
        da = A.mean(axis=0) - control_mean
        db = B.mean(axis=0) - control_mean
        return float(1.0 - _weighted_pearson_vec(da, db, w))

    if metric == "r2_dp":
        da = A.mean(axis=0) - dataset_mean
        db = B.mean(axis=0) - dataset_mean
        return float(1.0 - _r2_score_safe(db, da))

    if metric == "wr2_dp":
        da = A.mean(axis=0) - dataset_mean
        db = B.mean(axis=0) - dataset_mean
        return float(1.0 - _r2_score_safe(db, da, sample_weight=w))

    if metric == "top20_mse":
        if top20_idx is None:
            return float("nan")
        ma, mb = A.mean(axis=0), B.mean(axis=0)
        return float(np.mean(np.square(ma[top20_idx] - mb[top20_idx])))

    if metric == "top20_pearson_dp":
        if top20_idx is None:
            return float("nan")
        da = A.mean(axis=0) - dataset_mean
        db = B.mean(axis=0) - dataset_mean
        return float(1.0 - _pearson_vec(da[top20_idx], db[top20_idx]))

    if metric == "energy":
        uniform_w = np.ones(n_genes, dtype=float)
        return float(_energy_distance(A, B, uniform_w))

    if metric == "wenergy":
        return float(_energy_distance(A, B, w))

    raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# Vectorized permutation deltas for mean-based metrics
# ---------------------------------------------------------------------------


def _vectorized_mean_deltas(
    metric: str,
    pool: np.ndarray,
    n_k: int,
    perm_indices: np.ndarray,
    w: np.ndarray,
    dataset_mean: np.ndarray,
    control_mean: np.ndarray,
    top20_idx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute deltas for all permutations at once (mean-based metrics only).

    Returns (n_perm,) array of delta = d_neg - d_pos values.
    """
    metric = VC_METRIC_MAP.get(metric, metric)
    n_perm = perm_indices.shape[0]
    n_genes = pool.shape[1]
    mid = n_k // 2

    idx_b1 = perm_indices[:, :mid]
    idx_b2 = perm_indices[:, mid:n_k]
    idx_c = perm_indices[:, n_k:]

    mean_b1 = np.mean(pool[idx_b1], axis=1)
    mean_b2 = np.mean(pool[idx_b2], axis=1)
    mean_c = np.mean(pool[idx_c], axis=1)

    if metric == "mse":
        d_pos = np.mean(np.square(mean_b1 - mean_b2), axis=1)
        d_neg_1 = np.mean(np.square(mean_b1 - mean_c), axis=1)
        d_neg_2 = np.mean(np.square(mean_b2 - mean_c), axis=1)
        return 0.5 * (d_neg_1 + d_neg_2) - d_pos

    if metric == "wmse":
        w_sum = float(np.sum(w))
        wn = w / w_sum if w_sum > 1e-12 else np.full(n_genes, 1.0 / max(n_genes, 1))
        d_pos = (np.square(mean_b1 - mean_b2) * wn[None, :]).sum(axis=1)
        d_neg_1 = (np.square(mean_b1 - mean_c) * wn[None, :]).sum(axis=1)
        d_neg_2 = (np.square(mean_b2 - mean_c) * wn[None, :]).sum(axis=1)
        return 0.5 * (d_neg_1 + d_neg_2) - d_pos

    if metric == "top20_mse":
        if top20_idx is None:
            return np.full(n_perm, np.nan)
        d_pos = np.mean(np.square(mean_b1[:, top20_idx] - mean_b2[:, top20_idx]), axis=1)
        d_neg_1 = np.mean(np.square(mean_b1[:, top20_idx] - mean_c[:, top20_idx]), axis=1)
        d_neg_2 = np.mean(np.square(mean_b2[:, top20_idx] - mean_c[:, top20_idx]), axis=1)
        return 0.5 * (d_neg_1 + d_neg_2) - d_pos

    def _pearson_batch(ma: np.ndarray, mb: np.ndarray, ref: Optional[np.ndarray] = None,
                       idx: Optional[np.ndarray] = None) -> np.ndarray:
        if ref is not None:
            ma = ma - ref[None, :]
            mb = mb - ref[None, :]
        if idx is not None:
            ma = ma[:, idx]
            mb = mb[:, idx]
        ca = ma - ma.mean(axis=1, keepdims=True)
        cb = mb - mb.mean(axis=1, keepdims=True)
        num = np.sum(ca * cb, axis=1)
        denom = np.sqrt(np.sum(ca ** 2, axis=1) * np.sum(cb ** 2, axis=1))
        corr = np.where(denom > 1e-12, np.clip(num / denom, -1.0, 1.0), 0.0)
        return 1.0 - corr

    if metric == "pearson":
        d_pos = _pearson_batch(mean_b1, mean_b2)
        d_neg_1 = _pearson_batch(mean_b1, mean_c)
        d_neg_2 = _pearson_batch(mean_b2, mean_c)
        return 0.5 * (d_neg_1 + d_neg_2) - d_pos

    if metric == "pearson_dp":
        d_pos = _pearson_batch(mean_b1, mean_b2, ref=dataset_mean)
        d_neg_1 = _pearson_batch(mean_b1, mean_c, ref=dataset_mean)
        d_neg_2 = _pearson_batch(mean_b2, mean_c, ref=dataset_mean)
        return 0.5 * (d_neg_1 + d_neg_2) - d_pos

    if metric == "pearson_dc":
        d_pos = _pearson_batch(mean_b1, mean_b2, ref=control_mean)
        d_neg_1 = _pearson_batch(mean_b1, mean_c, ref=control_mean)
        d_neg_2 = _pearson_batch(mean_b2, mean_c, ref=control_mean)
        return 0.5 * (d_neg_1 + d_neg_2) - d_pos

    if metric == "top20_pearson_dp":
        d_pos = _pearson_batch(mean_b1, mean_b2, ref=dataset_mean, idx=top20_idx)
        d_neg_1 = _pearson_batch(mean_b1, mean_c, ref=dataset_mean, idx=top20_idx)
        d_neg_2 = _pearson_batch(mean_b2, mean_c, ref=dataset_mean, idx=top20_idx)
        return 0.5 * (d_neg_1 + d_neg_2) - d_pos

    def _wpearson_batch(ma: np.ndarray, mb: np.ndarray,
                        ref: Optional[np.ndarray] = None) -> np.ndarray:
        if ref is not None:
            ma = ma - ref[None, :]
            mb = mb - ref[None, :]
        results = np.empty(ma.shape[0])
        for i in range(ma.shape[0]):
            results[i] = 1.0 - _weighted_pearson_vec(ma[i], mb[i], w)
        return results

    if metric == "wpearson":
        d_pos = _wpearson_batch(mean_b1, mean_b2)
        d_neg_1 = _wpearson_batch(mean_b1, mean_c)
        d_neg_2 = _wpearson_batch(mean_b2, mean_c)
        return 0.5 * (d_neg_1 + d_neg_2) - d_pos

    if metric == "wpearson_dp":
        d_pos = _wpearson_batch(mean_b1, mean_b2, ref=dataset_mean)
        d_neg_1 = _wpearson_batch(mean_b1, mean_c, ref=dataset_mean)
        d_neg_2 = _wpearson_batch(mean_b2, mean_c, ref=dataset_mean)
        return 0.5 * (d_neg_1 + d_neg_2) - d_pos

    if metric == "wpearson_dc":
        d_pos = _wpearson_batch(mean_b1, mean_b2, ref=control_mean)
        d_neg_1 = _wpearson_batch(mean_b1, mean_c, ref=control_mean)
        d_neg_2 = _wpearson_batch(mean_b2, mean_c, ref=control_mean)
        return 0.5 * (d_neg_1 + d_neg_2) - d_pos

    def _r2_batch(ma: np.ndarray, mb: np.ndarray,
                  sw: Optional[np.ndarray] = None) -> np.ndarray:
        da = ma - dataset_mean[None, :]
        db = mb - dataset_mean[None, :]
        results = np.empty(da.shape[0])
        for i in range(da.shape[0]):
            results[i] = 1.0 - _r2_score_safe(db[i], da[i], sample_weight=sw)
        return results

    if metric == "r2_dp":
        d_pos = _r2_batch(mean_b1, mean_b2)
        d_neg_1 = _r2_batch(mean_b1, mean_c)
        d_neg_2 = _r2_batch(mean_b2, mean_c)
        return 0.5 * (d_neg_1 + d_neg_2) - d_pos

    if metric == "wr2_dp":
        d_pos = _r2_batch(mean_b1, mean_b2, sw=w)
        d_neg_1 = _r2_batch(mean_b1, mean_c, sw=w)
        d_neg_2 = _r2_batch(mean_b2, mean_c, sw=w)
        return 0.5 * (d_neg_1 + d_neg_2) - d_pos

    raise ValueError(f"Cannot vectorize metric: {metric}")


# ---------------------------------------------------------------------------
# Permutation-chunk worker for energy metrics
# ---------------------------------------------------------------------------


def _compute_delta_deterministic(
    metric: str,
    X_k: np.ndarray,
    X_c: np.ndarray,
    w: np.ndarray,
    dataset_mean: np.ndarray,
    control_mean: np.ndarray,
    top20_idx: Optional[np.ndarray] = None,
) -> float:
    """Delta = d_neg - d_pos with deterministic first-half / second-half split."""
    n_k = X_k.shape[0]
    mid = n_k // 2
    B1, B2 = X_k[:mid], X_k[mid:]

    d_pos = compute_bag_distance(metric, B1, B2, w, dataset_mean, control_mean, top20_idx)
    d_neg = 0.5 * (
        compute_bag_distance(metric, B1, X_c, w, dataset_mean, control_mean, top20_idx)
        + compute_bag_distance(metric, B2, X_c, w, dataset_mean, control_mean, top20_idx)
    )
    return d_neg - d_pos


def _perm_chunk_worker(args: Tuple) -> Tuple[int, int]:
    """Run a chunk of permutations for one perturbation (energy metrics)."""
    (metric, pool_cells, n_k, w, dataset_mean, control_mean, top20_idx,
     delta_obs, n_perms, seed) = args
    rng = np.random.default_rng(seed)
    n_total = pool_cells.shape[0]

    if metric in MEAN_BASED_METRICS:
        perm_indices = np.array(
            [rng.permutation(n_total) for _ in range(n_perms)]
        )
        null_deltas = _vectorized_mean_deltas(
            metric, pool_cells, n_k, perm_indices, w, dataset_mean,
            control_mean, top20_idx,
        )
        return int(np.sum(null_deltas >= delta_obs)), n_perms

    n_ge = 0
    for _ in range(n_perms):
        perm = rng.permutation(n_total)
        pseudo_k = pool_cells[perm[:n_k]]
        pseudo_c = pool_cells[perm[n_k:]]
        delta_null = _compute_delta_deterministic(
            metric, pseudo_k, pseudo_c, w, dataset_mean, control_mean, top20_idx,
        )
        if delta_null >= delta_obs:
            n_ge += 1
    return n_ge, n_perms


# ---------------------------------------------------------------------------
# Single-perturbation test
# ---------------------------------------------------------------------------


def test_single_perturbation(
    metric: str,
    X_k: np.ndarray,
    X_c: np.ndarray,
    w: np.ndarray,
    dataset_mean: np.ndarray,
    control_mean: np.ndarray,
    top20_idx: Optional[np.ndarray],
    n_permutations: int,
    seed: int,
    process_pool: Optional[Any] = None,
    n_workers: int = 1,
    perm_pbar: Optional[Any] = None,
) -> Tuple[float, float, float, float]:
    """Permutation test for one perturbation vs control.

    Returns (raw_pval, d_pos, d_neg, delta_obs).
    """
    rng = np.random.default_rng(seed)

    n_k = X_k.shape[0]
    mid = n_k // 2
    B1, B2 = X_k[:mid], X_k[mid:]
    d_pos = compute_bag_distance(metric, B1, B2, w, dataset_mean, control_mean, top20_idx)
    d_neg = 0.5 * (
        compute_bag_distance(metric, B1, X_c, w, dataset_mean, control_mean, top20_idx)
        + compute_bag_distance(metric, B2, X_c, w, dataset_mean, control_mean, top20_idx)
    )
    delta_obs = d_neg - d_pos

    pool_cells = np.vstack([X_k, X_c])
    n_total = pool_cells.shape[0]

    if metric in MEAN_BASED_METRICS:
        if perm_pbar is not None:
            n_chunks = min(100, max(1, n_permutations))
            chunk_sz = max(1, (n_permutations + n_chunks - 1) // n_chunks)
            n_ge = 0
            for start in range(0, n_permutations, chunk_sz):
                end = min(start + chunk_sz, n_permutations)
                this_chunk = end - start
                perm_indices = np.array(
                    [rng.permutation(n_total) for _ in range(this_chunk)]
                )
                null_deltas = _vectorized_mean_deltas(
                    metric, pool_cells, n_k, perm_indices, w, dataset_mean,
                    control_mean, top20_idx,
                )
                n_ge += int(np.sum(null_deltas >= delta_obs))
                perm_pbar.update(this_chunk)
        else:
            perm_indices = np.array(
                [rng.permutation(n_total) for _ in range(n_permutations)]
            )
            null_deltas = _vectorized_mean_deltas(
                metric, pool_cells, n_k, perm_indices, w, dataset_mean,
                control_mean, top20_idx,
            )
            n_ge = int(np.sum(null_deltas >= delta_obs))

    elif process_pool is not None and n_workers > 1:
        effective = min(n_workers, n_permutations)
        base_n, rem = divmod(n_permutations, effective)
        chunk_args = [
            (metric, pool_cells, n_k, w, dataset_mean, control_mean, top20_idx,
             delta_obs, base_n + (1 if i < rem else 0), seed + i * 7919)
            for i in range(effective)
            if base_n + (1 if i < rem else 0) > 0
        ]
        n_ge = 0
        for chunk_nge, chunk_n in process_pool.imap_unordered(
            _perm_chunk_worker, chunk_args,
        ):
            n_ge += chunk_nge
            if perm_pbar is not None:
                perm_pbar.update(chunk_n)
    else:
        n_ge = 0
        for _ in range(n_permutations):
            perm = rng.permutation(n_total)
            pseudo_k = pool_cells[perm[:n_k]]
            pseudo_c = pool_cells[perm[n_k:]]
            delta_null = _compute_delta_deterministic(
                metric, pseudo_k, pseudo_c, w, dataset_mean, control_mean, top20_idx,
            )
            if delta_null >= delta_obs:
                n_ge += 1
            if perm_pbar is not None:
                perm_pbar.update(1)

    raw_pval = (n_ge + 1) / (n_permutations + 1)
    return raw_pval, d_pos, d_neg, delta_obs


# ---------------------------------------------------------------------------
# Per (metric, trial) driver
# ---------------------------------------------------------------------------


def _run_metric_trial(
    metric: str,
    trial: int,
    X: np.ndarray,
    y: np.ndarray,
    control_label: str,
    perturbations: List[str],
    dataset_mean: np.ndarray,
    control_mean: np.ndarray,
    weights_per_pert: Dict[str, np.ndarray],
    top20_per_pert: Dict[str, np.ndarray],
    n_permutations: int,
    seed: int,
    alpha: float,
    show_pert_progress: bool,
    show_perm_progress: bool,
    max_control_cells: Optional[int],
    process_pool: Optional[Any] = None,
    n_workers: int = 1,
    vc_weights_per_pert: Optional[Dict[str, np.ndarray]] = None,
) -> Dict:
    """Run BDS test for one metric + one trial."""
    effective_weights = weights_per_pert
    if metric in VC_METRIC_MAP:
        if vc_weights_per_pert is None:
            raise ValueError(f"Metric {metric} requires vc_weights_per_pert")
        effective_weights = vc_weights_per_pert

    n_genes = X.shape[1]
    X_c = X[y == control_label]
    if max_control_cells is not None and max_control_cells > 0:
        n_c = X_c.shape[0]
        if n_c > max_control_cells:
            rng_c = np.random.default_rng(seed)
            X_c = X_c[rng_c.choice(n_c, size=max_control_cells, replace=False)]
    uniform_w = np.ones(n_genes, dtype=float)

    t0 = time.perf_counter()
    raw_pvals: List[float] = []
    d_pos_list: List[float] = []
    d_neg_list: List[float] = []
    delta_obs_list: List[float] = []
    pert_names: List[str] = []

    perts_to_test = [p for p in perturbations if np.sum(y == p) >= 4]
    n_pert = len(perts_to_test)
    label = METRIC_DISPLAY.get(metric, metric)

    use_combined_perm_bar = show_perm_progress and n_pert > 0 and n_permutations > 0
    use_pert_bar_only = show_pert_progress and not use_combined_perm_bar

    pert_pbar: Optional[Any] = None
    perm_pbar: Optional[Any] = None
    if use_combined_perm_bar:
        pert_pbar = tqdm(
            total=n_pert,
            desc=f"    {label} trial {trial + 1} perts",
            unit="pert", position=0, leave=True,
            bar_format=("{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                        "[{elapsed}<{remaining}] {postfix}"),
        )
        perm_pbar = tqdm(
            total=n_pert * n_permutations,
            desc=f"    {label} trial {trial + 1} perms (all)",
            unit="perm", position=1, leave=False,
            bar_format=("{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                        "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"),
        )

    pert_iter: Any
    if use_pert_bar_only:
        pert_iter = tqdm(perts_to_test, desc=f"    {label} t{trial + 1}",
                         leave=False, unit="pert")
    else:
        pert_iter = perts_to_test

    try:
        for pert_i, pert in enumerate(pert_iter):
            if pert_pbar is not None:
                pert_pbar.set_postfix_str(f"{pert_i + 1}/{n_pert}")
            if perm_pbar is not None:
                perm_pbar.set_postfix_str(f"pert {pert_i + 1}/{n_pert}")

            X_k = X[y == pert]
            w = (
                uniform_w
                if metric in UNWEIGHTED_METRICS
                else effective_weights.get(pert, uniform_w)
            )
            t20 = top20_per_pert.get(pert)

            pv, d_pos, d_neg, delta_obs = test_single_perturbation(
                metric, X_k, X_c, w, dataset_mean, control_mean, t20,
                n_permutations=n_permutations,
                seed=seed + hash(pert) % 100000,
                process_pool=process_pool,
                n_workers=n_workers,
                perm_pbar=perm_pbar,
            )
            raw_pvals.append(pv)
            d_pos_list.append(d_pos)
            d_neg_list.append(d_neg)
            delta_obs_list.append(delta_obs)
            pert_names.append(pert)
            if pert_pbar is not None:
                pert_pbar.update(1)
    finally:
        if perm_pbar is not None:
            perm_pbar.close()
        if pert_pbar is not None:
            pert_pbar.close()

    if not raw_pvals:
        return {
            "metric": metric, "trial": trial,
            "sensitivity": 0.0, "n_significant": 0, "n_tested": 0,
            "elapsed": time.perf_counter() - t0, "detail": [],
        }

    reject, pvals_corr, _, _ = multipletests(raw_pvals, alpha=alpha, method="fdr_bh")

    detail = [
        {
            "perturbation": p, "raw_pval": rp, "corrected_pval": cp,
            "significant": bool(sig),
            "d_pos": dp, "d_neg": dn, "delta_obs": do,
        }
        for p, rp, cp, sig, dp, dn, do in zip(
            pert_names, raw_pvals, pvals_corr, reject,
            d_pos_list, d_neg_list, delta_obs_list,
        )
    ]

    return {
        "metric": metric,
        "trial": trial,
        "sensitivity": float(np.mean(reject)),
        "n_significant": int(np.sum(reject)),
        "n_tested": len(raw_pvals),
        "elapsed": time.perf_counter() - t0,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BDS test: metric ability to detect perturbation signal vs control.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--metrics", nargs="+", default=METRICS)
    parser.add_argument("--n-permutations", type=int, default=999)
    parser.add_argument(
        "--n-trials", type=int, default=1,
        help="Repeat with different half-bag splits to stabilize.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-perturbations", type=int, default=50)
    parser.add_argument(
        "--max-control-cells", type=int, default=500,
        help="Subsample control cells (0 = no subsampling).",
    )
    parser.add_argument(
        "--workers", "--n-jobs", type=int, default=None, dest="workers",
        metavar="N",
        help="Process pool size for permutation parallelism (default: cpu_count).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=str,
        default="analyses/perturbation_discrimination/results/bds_test",
    )
    parser.add_argument(
        "--include-energy", action="store_true",
        help="Include energy and wenergy metrics (slow).",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--pert-progress", action="store_true")
    parser.add_argument("--perm-progress", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args.include_energy:
        for em in ENERGY_METRICS:
            if em not in args.metrics:
                args.metrics.append(em)

    has_vc_metrics = any(m in VC_METRIC_MAP for m in args.metrics)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_metric_trials = len(args.metrics) * args.n_trials
    n_workers = args.workers or mp.cpu_count()
    ctrl_cap = (
        f"max {args.max_control_cells} control cells"
        if args.max_control_cells and args.max_control_cells > 0
        else "all control cells"
    )
    print(
        f"Config: {args.n_permutations} permutations/pert, {args.n_trials} trials, "
        f"alpha={args.alpha}, {ctrl_cap}, max_perts={args.max_perturbations}.",
        flush=True,
    )
    print(
        f"Parallelism: {n_workers} worker(s); "
        f"{n_metric_trials} (metric x trial) jobs run sequentially.",
        flush=True,
    )

    detail_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for ds_idx, dataset_name in enumerate(args.datasets, 1):
        print(f"\n{'='*60}")
        print(f"  [{ds_idx}/{len(args.datasets)}] Dataset: {dataset_name}")
        print(f"{'='*60}")
        t0 = time.perf_counter()

        try:
            X, y, genes, ctrl_label, perts = _load_dataset(
                dataset_name,
                max_perturbations=args.max_perturbations,
                seed=args.seed,
            )
        except Exception as e:
            print(f"  SKIP {dataset_name}: {e}")
            continue

        n_ctrl = int(np.sum(y == ctrl_label))
        print(
            f"  Loaded {X.shape[0]} cells, {len(perts)} perturbations, "
            f"{n_ctrl} control cells, {X.shape[1]} genes",
        )

        try:
            syn_weights = _load_synthetic_weights(dataset_name, genes, perts)
        except FileNotFoundError as e:
            print(f"  SKIP {dataset_name}: no synthetic weights – {e}")
            continue

        vc_weights: Optional[Dict[str, np.ndarray]] = None
        if has_vc_metrics:
            try:
                vc_weights = _load_vscontrol_weights(dataset_name, genes, perts)
                print(f"  Vscontrol weights for {len(vc_weights)}/{len(perts)} perturbations")
            except FileNotFoundError as e:
                print(f"  WARNING: no vscontrol weights – {e}; vc metrics will be skipped")

        perts_with_weights = [p for p in perts if p in syn_weights]
        if len(perts_with_weights) < 3:
            print(
                f"  SKIP {dataset_name}: only {len(perts_with_weights)} "
                f"perturbations with weights",
            )
            continue

        mask = np.isin(y, perts_with_weights) | (y == ctrl_label)
        X_sub = X[mask]
        y_sub = y[mask]
        dataset_mean = X_sub.mean(axis=0)
        control_mean = X_sub[y_sub == ctrl_label].mean(axis=0)

        top20_per_pert = _build_top20_indices(syn_weights)

        print(
            f"  {len(perts_with_weights)} perturbations with weights, "
            f"running {len(args.metrics)} metrics x {args.n_trials} trials",
            flush=True,
        )

        pool_ctx = mp.Pool(n_workers) if n_workers > 1 else None
        try:
            results: List[Dict] = []
            job_i = 0

            for trial in range(args.n_trials):
                for metric in args.metrics:
                    job_i += 1
                    label = METRIC_DISPLAY.get(metric, metric)
                    if not args.no_progress:
                        print(
                            f"  [{job_i}/{n_metric_trials}] {label} trial {trial + 1}",
                            flush=True,
                        )

                    if metric in VC_METRIC_MAP and vc_weights is None:
                        print(f"    SKIP {label}: no vscontrol weights", flush=True)
                        continue

                    res = _run_metric_trial(
                        metric=metric,
                        trial=trial,
                        X=X_sub,
                        y=y_sub,
                        control_label=ctrl_label,
                        perturbations=perts_with_weights,
                        dataset_mean=dataset_mean,
                        control_mean=control_mean,
                        weights_per_pert=syn_weights,
                        top20_per_pert=top20_per_pert,
                        n_permutations=args.n_permutations,
                        seed=args.seed + trial,
                        alpha=args.alpha,
                        show_pert_progress=args.pert_progress,
                        show_perm_progress=args.perm_progress,
                        max_control_cells=(
                            args.max_control_cells
                            if args.max_control_cells > 0 else None
                        ),
                        process_pool=pool_ctx,
                        n_workers=n_workers,
                        vc_weights_per_pert=vc_weights,
                    )
                    results.append(res)

                    if not args.no_progress:
                        print(
                            f"    -> sensitivity={res['sensitivity']:.1%}  "
                            f"({res['n_significant']}/{res['n_tested']} sig)  "
                            f"{res['elapsed']:.1f}s",
                            flush=True,
                        )
        finally:
            if pool_ctx is not None:
                pool_ctx.close()
                pool_ctx.join()

        for res in results:
            label = METRIC_DISPLAY.get(res["metric"], res["metric"])
            for d in res["detail"]:
                detail_rows.append({
                    "dataset": dataset_name,
                    "metric": res["metric"],
                    "metric_display": label,
                    "trial": res["trial"],
                    "perturbation": d["perturbation"],
                    "d_pos": d["d_pos"],
                    "d_neg": d["d_neg"],
                    "delta_obs": d["delta_obs"],
                    "raw_pval": d["raw_pval"],
                    "corrected_pval": d["corrected_pval"],
                    "significant": d["significant"],
                })

        metric_trial: Dict[str, List[Dict]] = {}
        for res in results:
            metric_trial.setdefault(res["metric"], []).append(res)

        for metric, trials in sorted(metric_trial.items()):
            label = METRIC_DISPLAY.get(metric, metric)
            sensitivities = [t["sensitivity"] for t in trials]
            n_sigs = [t["n_significant"] for t in trials]
            n_tested = trials[0]["n_tested"]
            mean_sens = float(np.mean(sensitivities))
            all_raw: List[float] = []
            for t in trials:
                for d in t["detail"]:
                    all_raw.append(float(d["raw_pval"]))
            mean_raw_p = float(np.mean(all_raw)) if all_raw else float("nan")
            elapsed = max(t["elapsed"] for t in trials)
            print(
                f"    {label:>25s}  sensitivity={mean_sens:.1%}  "
                f"mean_raw_p={mean_raw_p:.4f}  "
                f"(sig {np.mean(n_sigs):.1f}/{n_tested})  "
                f"{elapsed:.1f}s",
                flush=True,
            )
            summary_rows.append({
                "dataset": dataset_name,
                "metric": metric,
                "metric_display": label,
                "n_tested": n_tested,
                "mean_sensitivity": mean_sens,
                "mean_raw_pval": mean_raw_p,
                "min_sensitivity": float(np.min(sensitivities)),
                "max_sensitivity": float(np.max(sensitivities)),
                "mean_n_significant": float(np.mean(n_sigs)),
            })

        print(f"  {dataset_name} done in {time.perf_counter()-t0:.1f}s")

    if not summary_rows:
        print("No results collected.")
        return

    detail_df = pd.DataFrame(detail_rows)
    detail_path = out_dir / "bds_test_detail.csv"
    detail_df.to_csv(detail_path, index=False, float_format="%.6f")
    print(f"\nDetail saved to {detail_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "bds_test_summary.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"Summary saved to {summary_path}")

    print(f"\n{'='*80}")
    print(f"Sensitivity summary (fraction detected, alpha={args.alpha})")
    print(f"{'='*80}")
    pivot = summary_df.pivot_table(
        index="dataset", columns="metric_display", values="mean_sensitivity",
    )
    col_order = [
        METRIC_DISPLAY.get(m, m) for m in ALL_KNOWN_METRICS
        if METRIC_DISPLAY.get(m, m) in pivot.columns
    ]
    pivot = pivot[col_order]
    print(pivot.to_string(float_format=lambda x: f"{x:.1%}"))

    print(f"\n{'='*80}")
    print("Mean raw p-value (lower = better detection)")
    print(f"{'='*80}")
    pivot_p = summary_df.pivot_table(
        index="dataset", columns="metric_display", values="mean_raw_pval",
    )
    pivot_p = pivot_p[col_order]
    print(pivot_p.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
