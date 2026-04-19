"""
Gene-wise metrics for calibrated metric computation in CellSimBench.

This module provides gene-wise metric computations that can be used for
calibrated metrics. The reported score for a perturbation is the average
over the genes.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import energy_distance


def mse_on_means(cloud_a: np.ndarray, cloud_b: np.ndarray) -> float:
    """Compute MSE between the means of two point clouds.
    
    Args:
        cloud_a: Values for distribution A (1D array of cell-level expression).
        cloud_b: Values for distribution B (1D array of cell-level expression).
        
    Returns:
        Mean squared error between the means.
    """
    mean_a = np.mean(cloud_a)
    mean_b = np.mean(cloud_b)
    return float((mean_a - mean_b) ** 2)


def energy_distance_1d(cloud_a: np.ndarray, cloud_b: np.ndarray) -> float:
    """Compute 1D energy distance using scipy.
    
    Args:
        cloud_a: Values for distribution A (1D array of cell-level expression).
        cloud_b: Values for distribution B (1D array of cell-level expression).
        
    Returns:
        Energy distance between the two distributions.
    """
    return float(energy_distance(
        np.asarray(cloud_a).ravel(), 
        np.asarray(cloud_b).ravel()
    ))


def compute_genewise_metrics(
    predictions: np.ndarray,
    truth: np.ndarray,
    metric_name: Literal["mse", "ed"] = "mse",
) -> np.ndarray:
    """Compute gene-wise metrics for all genes.
    
    The reported score for a perturbation is the average over the genes.
    
    Args:
        predictions: Predicted expression values of shape (n_cells, n_genes).
        truth: Ground truth expression values of shape (n_cells, n_genes).
        metric_name: Name of the metric to compute. Options: "mse" (MSE on means),
            "ed" (energy distance).
            
    Returns:
        Array of shape (n_genes,) containing the metric value for each gene.
        
    Raises:
        ValueError: If metric_name is not supported or if shapes don't match.
    """
    if predictions.shape != truth.shape:
        raise ValueError(
            f"Predictions shape {predictions.shape} does not match "
            f"truth shape {truth.shape}"
        )
    
    if len(predictions.shape) != 2:
        raise ValueError(
            f"Expected 2D arrays (n_cells, n_genes), got shape {predictions.shape}"
        )
    
    if metric_name == "mse":
        pred_mean = np.mean(predictions, axis=0)
        truth_mean = np.mean(truth, axis=0)
        return (pred_mean - truth_mean) ** 2
    elif metric_name == "ed":
        metric_fn = energy_distance_1d
    else:
        raise ValueError(
            f"Unsupported metric_name: {metric_name}. "
            f"Supported metrics: 'mse', 'ed'"
        )
    
    n_cells, n_genes = predictions.shape
    
    # Compute metric for each gene
    genewise_values = np.zeros(n_genes, dtype=float)
    for gene_idx in range(n_genes):
        pred_gene = predictions[:, gene_idx]
        truth_gene = truth[:, gene_idx]
        genewise_values[gene_idx] = metric_fn(pred_gene, truth_gene)
    
    return genewise_values
