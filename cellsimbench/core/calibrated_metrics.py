"""
Calibrated metrics for CellSimBench.

This module implements calibration for gene-wise metrics using positive and negative
baselines. The calibration formula is:
    m_pred_cal = (m_pos - m_pred) / (m_pos - m_neg)

where:
- m_pred: Metric computed on predicted vs. ground truth
- m_pos: Metric on positive baseline (technical duplicates, opposite split)
- m_neg: Metric on negative baseline (control samples)

Calibrated values are capped to [0, 1] and invalid inputs raise errors.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Per-metric diagnostics for calibrate_aggregated_metric (reset each benchmark run).
_aggregated_calibration_unexpected_denominator_sign_counts: Dict[str, int] = {}
_aggregated_calibration_denominator_too_small_counts: Dict[str, int] = {}
_aggregated_calibration_n_evaluations: Dict[str, int] = {}


def reset_aggregated_calibration_denominator_issue_counts() -> None:
    """Clear aggregated-calibration counters (call once per benchmark run)."""
    _aggregated_calibration_unexpected_denominator_sign_counts.clear()
    _aggregated_calibration_denominator_too_small_counts.clear()
    _aggregated_calibration_n_evaluations.clear()


def get_aggregated_calibration_denominator_report() -> Dict[str, Any]:
    """Snapshot of per-metric evaluation counts and denominator-issue rates."""
    per_metric: Dict[str, Any] = {}
    names = (
        set(_aggregated_calibration_n_evaluations.keys())
        | set(_aggregated_calibration_unexpected_denominator_sign_counts.keys())
        | set(_aggregated_calibration_denominator_too_small_counts.keys())
    )
    for name in sorted(names):
        n_eval = int(_aggregated_calibration_n_evaluations.get(name, 0))
        n_sign = int(_aggregated_calibration_unexpected_denominator_sign_counts.get(name, 0))
        n_small = int(_aggregated_calibration_denominator_too_small_counts.get(name, 0))
        per_metric[name] = {
            "n_evaluations": n_eval,
            "unexpected_denominator_sign": n_sign,
            "unexpected_denominator_sign_fraction": (n_sign / n_eval) if n_eval else 0.0,
            "denominator_too_small": n_small,
            "denominator_too_small_fraction": (n_small / n_eval) if n_eval else 0.0,
        }
    return {"per_metric": per_metric}

def calibrate_metric(
    m_pred: float,
    m_pos: float,
    m_neg: float,
    perturbation: Optional[str] = None,
    gene_name: Optional[str] = None,
    min_denominator: float = 1e-10,
) -> float:
    """Calibrate a single metric value using positive and negative baselines.
    
    Formula: m_pred_cal = (m_pos - m_pred) / (m_pos - m_neg)
    
    This yields:
      - 0.0 when m_pred == m_pos (positive baseline / best case).
      - 1.0 when m_pred == m_neg (negative baseline / worst case).
    
    Args:
        m_pred: Metric value for prediction vs. ground truth.
        m_pos: Metric value for positive baseline (technical duplicates).
        m_neg: Metric value for negative baseline (controls).
        perturbation: Optional perturbation name for error context.
        gene_name: Optional gene name for error context.
        min_denominator: Minimum denominator threshold to avoid division by zero.
        
    Returns:
        Calibrated metric value.
        
    Raises:
        ValueError: If inputs contain NaN or baselines are invalid.
    """
    if np.isnan(m_pred) or np.isnan(m_pos) or np.isnan(m_neg):
        context = f"perturbation={perturbation}, gene={gene_name}" if perturbation or gene_name else ""
        raise ValueError(
            f"NaN values in calibration inputs ({context}): m_pred={m_pred}, m_pos={m_pos}, m_neg={m_neg}"
        )

    if m_pos >= m_neg:
        context = f"perturbation={perturbation}, gene={gene_name}" if perturbation or gene_name else ""
        raise ValueError(
            f"Invalid baselines for calibration ({context}): m_pos={m_pos}, m_neg={m_neg}"
        )

    denominator = m_pos - m_neg
    if abs(denominator) < min_denominator:
        context = f"perturbation={perturbation}, gene={gene_name}" if perturbation or gene_name else ""
        raise ValueError(
            f"Denominator too small in calibration ({context}): "
            f"m_pos={m_pos:.6f}, m_neg={m_neg:.6f}, denominator={denominator:.6e}"
        )

    numerator = m_pos - m_pred
    return float(numerator / denominator)


def calibrate_genewise_metrics(
    genewise_pred: np.ndarray,
    genewise_pos: np.ndarray,
    genewise_neg: np.ndarray,
    gene_names: Optional[List[str]] = None,
    perturbation: Optional[str] = None,
    min_denominator: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """Calibrate gene-wise metric values and compute average.
    
    Applies calibration to each gene independently, then averages across genes
    (ignoring NaN values).
    
    Args:
        genewise_pred: Array of gene-wise metric values for predictions.
        genewise_pos: Array of gene-wise metric values for positive baseline.
        genewise_neg: Array of gene-wise metric values for negative baseline.
        gene_names: Optional list of gene names for logging.
        perturbation: Optional perturbation name for logging.
        
    Returns:
        Tuple of (calibrated_genewise_values, average_calibrated_value).
        The calibrated values include only genes with valid baselines.
        
    Raises:
        ValueError: If input arrays have different shapes or contain NaNs,
            or if no genes have valid baselines.
    """
    # Validate inputs
    if not (genewise_pred.shape == genewise_pos.shape == genewise_neg.shape):
        raise ValueError(
            f"Input arrays must have the same shape. Got: "
            f"pred={genewise_pred.shape}, pos={genewise_pos.shape}, neg={genewise_neg.shape}"
        )
    
    if (
        np.isnan(genewise_pred).any()
        or np.isnan(genewise_pos).any()
        or np.isnan(genewise_neg).any()
    ):
        context = f"perturbation={perturbation}" if perturbation else ""
        raise ValueError(f"NaN values in calibration inputs ({context}).")
    
    denominators = genewise_pos - genewise_neg
    valid_mask = (genewise_pos < genewise_neg) & (np.abs(denominators) >= min_denominator)
    dropped_count = int(np.size(valid_mask) - np.count_nonzero(valid_mask))
    if dropped_count > 0:
        context = f"perturbation={perturbation}" if perturbation else "perturbation=unknown"
        # logger.warning(
        #     "Dropped %d/%d genes with invalid baselines (%s).",
        #     dropped_count,
        #     len(valid_mask),
        #     context,
        # )
    
    if not np.any(valid_mask):
        context = f"perturbation={perturbation}" if perturbation else ""
        raise ValueError(f"No valid baselines for calibration ({context}).")
    
    pred_valid = genewise_pred[valid_mask]
    pos_valid = genewise_pos[valid_mask]
    neg_valid = genewise_neg[valid_mask]
    
    calibrated_values = (pred_valid - pos_valid) / (neg_valid - pos_valid)
    out_of_range_mask = (calibrated_values < 0) | (calibrated_values > 1)
    if np.any(out_of_range_mask):
        context = f"perturbation={perturbation}" if perturbation else "perturbation=unknown"
        out_values = calibrated_values[out_of_range_mask]
        # logger.warning(
        #     "Calibrated values out of range (%s): count=%d, min=%.6f, max=%.6f",
        #     context,
        #     out_values.size,
        #     float(np.min(out_values)),
        #     float(np.max(out_values)),
        # )
    calibrated_values = np.clip(calibrated_values, 0, 1)
    
    
    avg_calibrated = float(np.mean(calibrated_values))
    
    return calibrated_values, avg_calibrated


def calibrate_aggregated_metric(
    m_pred: float,
    m_pos: float,
    m_neg: float,
    perturbation: Optional[str] = None,
    min_denominator: float = 1e-8,
    higher_is_better: bool = False,
    metric_name: Optional[str] = None,
) -> float:
    """Calibrate an aggregated metric value without capping.

    Formula: m_cal = (m_pred - m_pos) / (m_neg - m_pos)

    This yields:
      - 0.0 when m_pred == m_pos (positive baseline / best case).
      - 1.0 when m_pred == m_neg (negative baseline / worst case).

    Args:
        m_pred: Aggregated metric for prediction vs. ground truth.
        m_pos: Aggregated metric for positive baseline vs. ground truth.
        m_neg: Aggregated metric for negative baseline vs. ground truth.
        perturbation: Optional perturbation name for error context.
        min_denominator: Minimum denominator threshold to avoid division by zero.
        higher_is_better: Whether the metric increases with better performance.
        metric_name: If set, record diagnostics for benchmark reporting.

    Returns:
        Calibrated metric value (uncapped).

    Raises:
        ValueError: If inputs contain NaN or denominator is too small.
    """
    if np.isnan(m_pred) or np.isnan(m_pos) or np.isnan(m_neg):
        context = f"perturbation={perturbation}" if perturbation else ""
        raise ValueError(f"NaN values in calibration inputs ({context}).")

    if metric_name is not None:
        _aggregated_calibration_n_evaluations[metric_name] = (
            _aggregated_calibration_n_evaluations.get(metric_name, 0) + 1
        )

    denominator = m_neg - m_pos
    if abs(denominator) < min_denominator:
        context = f"perturbation={perturbation}" if perturbation else ""
        logger.warning(
            "Denominator too small in aggregated calibration "
            f"({context}): m_pos={m_pos:.6f}, m_neg={m_neg:.6f}, "
            f"denominator={denominator:.6e}"
        )
        if metric_name is not None:
            _aggregated_calibration_denominator_too_small_counts[metric_name] = (
                _aggregated_calibration_denominator_too_small_counts.get(metric_name, 0) + 1
            )
        return float("nan")
    if higher_is_better and denominator > 0:
        context = f"perturbation={perturbation}" if perturbation else ""
        logger.warning(
            "Unexpected denominator sign for higher-is-better calibration "
            f"({context}): m_pos={m_pos:.6f}, m_neg={m_neg:.6f}, "
            f"denominator={denominator:.6e}"
        )
        if metric_name is not None:
            _aggregated_calibration_unexpected_denominator_sign_counts[metric_name] = (
                _aggregated_calibration_unexpected_denominator_sign_counts.get(metric_name, 0) + 1
            )
        return float("nan")
    if not higher_is_better and denominator < 0:
        context = f"perturbation={perturbation}" if perturbation else ""
        logger.warning(
            "Unexpected denominator sign for lower-is-better calibration "
            f"({context}): m_pos={m_pos:.6f}, m_neg={m_neg:.6f}, "
            f"denominator={denominator:.6e}"
        )
        if metric_name is not None:
            _aggregated_calibration_unexpected_denominator_sign_counts[metric_name] = (
                _aggregated_calibration_unexpected_denominator_sign_counts.get(metric_name, 0) + 1
            )
        return float("nan")

    # Use standard formula for both lower-is-better and higher-is-better metrics:
    # m_cal = (m_pred - m_pos) / (m_neg - m_pos)
    # This yields 0.0 when m_pred == m_pos (best/achieved positive baseline)
    # and 1.0 when m_pred == m_neg (worst/achieved negative baseline)
    calibrated_value = float((m_pred - m_pos) / denominator)

    calibrated_value = max(0, calibrated_value)
    # calibrated_value = np.clip(calibrated_value, 0, 1)
    return calibrated_value


def calibrate_aggregated_metric_higher_is_better(
    m_pred: float,
    m_pos: float,
    m_neg: float,
    perturbation: Optional[str] = None,
    min_denominator: float = 1e-8,
    metric_name: Optional[str] = None,
) -> float:
    """Calibrate an aggregated metric using the standard baseline formula.

    Formula: m_cal = (m_pred - m_pos) / (m_neg - m_pos)

    This yields:
      - 0.0 when m_pred == m_pos (positive baseline / best case).
      - 1.0 when m_pred == m_neg (negative baseline / worst case).

    Args:
        m_pred: Aggregated metric for prediction vs. ground truth.
        m_pos: Aggregated metric for positive baseline vs. ground truth.
        m_neg: Aggregated metric for negative baseline vs. ground truth.
        perturbation: Optional perturbation name for error context.
        min_denominator: Minimum denominator threshold to avoid division by zero.

    Returns:
        Calibrated metric value (capped to [0, 1]).
    """
    return calibrate_aggregated_metric(
        m_pred=m_pred,
        m_pos=m_pos,
        m_neg=m_neg,
        perturbation=perturbation,
        min_denominator=min_denominator,
        higher_is_better=True,
        metric_name=metric_name,
    )


def calibrate_mse(
    pred_expr: np.ndarray,
    truth_expr: np.ndarray,
    pos_baseline_expr: np.ndarray,
    neg_baseline_expr: np.ndarray,
    gene_names: Optional[List[str]] = None,
    perturbation: Optional[str] = None,
) -> float:
    """Compute calibrated MSE (cMSE) on gene-wise means.
    
    Computes gene-wise MSE on means and calibrates using baselines.
    
    Args:
        pred_expr: Predicted expression array (n_cells x n_genes).
        truth_expr: Ground truth expression array (n_cells x n_genes).
        pos_baseline_expr: Positive baseline expression (mean vector, n_genes).
        neg_baseline_expr: Negative baseline expression (mean vector, n_genes).
        gene_names: Optional list of gene names.
        perturbation: Optional perturbation name for logging.
        
    Returns:
        Calibrated MSE value (average across genes).
    """
    from cellsimbench.core.genewise_metrics import compute_genewise_metrics
    
    # Compute gene-wise MSE for prediction vs. truth
    genewise_pred = compute_genewise_metrics(pred_expr, truth_expr, metric_name="mse")
    
    truth_mean = np.mean(truth_expr, axis=0)
    genewise_pos = (pos_baseline_expr - truth_mean) ** 2
    genewise_neg = (neg_baseline_expr - truth_mean) ** 2
    
    # Calibrate
    _, avg_cmse = calibrate_genewise_metrics(
        genewise_pred=genewise_pred,
        genewise_pos=genewise_pos,
        genewise_neg=genewise_neg,
        gene_names=gene_names,
        perturbation=perturbation,
    )
    
    return avg_cmse


def calibrate_weighted_mse(
    pred_expr: np.ndarray,
    truth_expr: np.ndarray,
    pos_baseline_expr: np.ndarray,
    neg_baseline_expr: np.ndarray,
    weights: np.ndarray,
    perturbation: Optional[str] = None,
) -> float:
    """Compute calibrated weighted MSE (cwMSE) on gene-wise means.

    Computes weighted MSE between prediction and ground truth means, then
    calibrates it using weighted positive/negative baselines.

    Args:
        pred_expr: Predicted expression array (n_cells x n_genes).
        truth_expr: Ground truth expression array (n_cells x n_genes).
        pos_baseline_expr: Positive baseline expression (mean vector, n_genes).
        neg_baseline_expr: Negative baseline expression (mean vector, n_genes).
        weights: Per-gene weights used by weighted MSE.
        perturbation: Optional perturbation name for logging.

    Returns:
        Calibrated weighted MSE value.
    """
    from cellsimbench.core.data_manager import wmse

    pred_mean = np.mean(pred_expr, axis=0)
    truth_mean = np.mean(truth_expr, axis=0)

    m_pred = float(wmse(pred_mean, truth_mean, weights))
    m_pos = float(wmse(pos_baseline_expr, truth_mean, weights))
    m_neg = float(wmse(neg_baseline_expr, truth_mean, weights))

    return calibrate_aggregated_metric(
        m_pred=m_pred,
        m_pos=m_pos,
        m_neg=m_neg,
        perturbation=perturbation,
        metric_name="cwmse",
    )


def calibrate_energy_distance(
    pred_expr: np.ndarray,
    truth_expr: np.ndarray,
    pos_baseline_expr: np.ndarray,
    neg_baseline_expr: np.ndarray,
    gene_names: Optional[List[str]] = None,
    perturbation: Optional[str] = None,
) -> float:
    """Compute calibrated Energy Distance (cED) gene-wise.
    
    Computes gene-wise energy distance and calibrates using baselines.
    Uses point clouds (full cell distributions) for all comparisons.
    
    Args:
        pred_expr: Predicted expression array (n_cells_pred x n_genes).
        truth_expr: Ground truth expression array (n_cells_truth x n_genes).
        pos_baseline_expr: Positive baseline point cloud (n_cells_pos x n_genes).
            Should be from opposite technical duplicate split.
        neg_baseline_expr: Negative baseline point cloud (n_cells_neg x n_genes).
            Should be randomly sampled controls, n_cells_neg = n_cells_pred.
        gene_names: Optional list of gene names.
        perturbation: Optional perturbation name for logging.
        
    Returns:
        Calibrated energy distance value (average across genes).
    """
    from cellsimbench.core.genewise_metrics import compute_genewise_metrics
    
    # Compute gene-wise energy distance for all comparisons
    # All are point cloud vs point cloud comparisons
    genewise_pred = compute_genewise_metrics(pred_expr, truth_expr, metric_name="ed")
    genewise_pos = compute_genewise_metrics(pos_baseline_expr, truth_expr, metric_name="ed")
    genewise_neg = compute_genewise_metrics(neg_baseline_expr, truth_expr, metric_name="ed")
    
    # Calibrate
    _, avg_ced = calibrate_genewise_metrics(
        genewise_pred=genewise_pred,
        genewise_pos=genewise_pos,
        genewise_neg=genewise_neg,
        gene_names=gene_names,
        perturbation=perturbation,
    )
    
    return avg_ced
