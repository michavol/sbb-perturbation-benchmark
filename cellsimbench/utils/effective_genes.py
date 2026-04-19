"""Effective gene count for weight vectors (N_eff = 1 / sum(p_i^2)).

Duplicated from ``analyses/perturbation_discrimination/metric_utils.py`` so
CellSimBench (including Docker model images) does not depend on the optional
``analyses`` package at import time.
"""

from __future__ import annotations

import numpy as np


def _normalize_weight_vector_for_effective_genes(weights: np.ndarray, context: str) -> np.ndarray:
    """Normalize a weight vector so that its maximum value is 1."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError(f"{context}: weight vector must be 1D, got shape={w.shape}.")
    if w.size == 0:
        raise ValueError(f"{context}: weight vector is empty.")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{context}: weights contain non-finite values.")
    if np.any(w < 0):
        raise ValueError(f"{context}: negative weights are not allowed.")

    w_max = float(np.max(w))
    if w_max <= 0.0:
        w = np.ones_like(w, dtype=float)
        return w
    return w / w_max


def _validate_weight_vector_for_effective_genes(weights: np.ndarray, context: str) -> None:
    """Validate assumptions required for meaningful effective-gene calculations."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError(f"{context}: weight vector must be 1D, got shape={w.shape}.")
    if w.size == 0:
        raise ValueError(f"{context}: weight vector is empty.")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{context}: weights contain non-finite values.")

    tol = 1e-8
    w_min = float(np.min(w))
    w_max = float(np.max(w))
    if w_min < -tol or w_max > 1.0 + tol:
        raise ValueError(
            f"{context}: weights must be in [0,1]. Observed min={w_min:.6g}, max={w_max:.6g}."
        )


def _effective_genes_single(weights: np.ndarray) -> float:
    """Compute effective gene number for a single weight vector.

    N_eff = 1 / sum(p_i^2), where p is normalized nonnegative weights.
    """
    w = _normalize_weight_vector_for_effective_genes(
        np.asarray(weights, dtype=float),
        context="Effective-gene computation input",
    )
    _validate_weight_vector_for_effective_genes(w, context="Effective-gene computation input")
    w_sum = float(np.sum(w))
    if w_sum < 1e-12:
        raise ValueError("Effective-gene computation input: sum(weights) is zero after normalization.")
    p = w / w_sum
    denom = float(np.sum(np.square(p)))
    if denom < 1e-12:
        return float(w.shape[0])
    return float(1.0 / denom)
