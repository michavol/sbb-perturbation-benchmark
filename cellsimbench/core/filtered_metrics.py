"""
Gene filtering for calibrated metrics based on Perturbation Discrimination Scores (PDS).

This module provides functionality to filter genes based on their PDS scores,
which measure how well a gene discriminates between perturbation and control conditions.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import scanpy as sc


class GeneFilterLoader:
    """Load and apply gene filters based on PDS scores from pds_full.h5ad."""
    
    def __init__(self, results_dir: Path) -> None:
        """Initialize the gene filter loader.
        
        Args:
            results_dir: Path to perturbation_discrimination results directory.
                Expected structure: results_dir/{dataset}/pds_full.h5ad
        """
        self.results_dir = Path(results_dir)
        self._pds_cache: Dict[str, Dict] = {}
    
    def load_pds_data(self, dataset_name: str) -> Dict:
        """Load PDS data from pds_full.h5ad.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'wessels23', 'norman19').
            
        Returns:
            Dictionary containing:
                - 'metrics': List of PDS metric names (e.g., ['MAE_Mean', 'Energy_Distance'])
                - 'perturbations': List of perturbation names
                - 'genes': List of gene names
                - 'scores_mean': Array of shape (n_metrics, n_perturbations, n_genes)
                
        Raises:
            FileNotFoundError: If pds_full.h5ad does not exist for the dataset.
            KeyError: If required keys are missing from pds_full.h5ad.
        """
        # Check cache first
        if dataset_name in self._pds_cache:
            return self._pds_cache[dataset_name]
        
        # Load from file
        pds_path = self.results_dir / dataset_name / "pds_full.h5ad"
        if not pds_path.exists():
            raise FileNotFoundError(
                f"PDS file not found: {pds_path}. "
                f"Available datasets in {self.results_dir}: "
                f"{[d.name for d in self.results_dir.iterdir() if d.is_dir()]}"
            )
        
        adata = sc.read_h5ad(pds_path)
        
        # Extract PDS data from uns
        if 'pds_full' not in adata.uns:
            raise KeyError(f"Key 'pds_full' not found in {pds_path}.uns")
        
        pds_full = adata.uns['pds_full']

        # Replace _ with + in perturbation names
        pds_full['perturbations'] = [p.replace('_', '+') for p in pds_full['perturbations']]
        
        # Validate required keys
        required_keys = ['metrics', 'perturbations', 'genes', 'scores_mean']
        missing_keys = [k for k in required_keys if k not in pds_full]
        if missing_keys:
            raise KeyError(
                f"Missing required keys in pds_full: {missing_keys}. "
                f"Available keys: {list(pds_full.keys())}"
            )
        
        # Store in cache
        pds_data = {
            'metrics': pds_full['metrics'],
            'perturbations': pds_full['perturbations'],
            'genes': pds_full['genes'],
            'scores_mean': pds_full['scores_mean'],
        }
        self._pds_cache[dataset_name] = pds_data
        
        return pds_data
    
    def get_filtered_genes(
        self,
        dataset_name: str,
        pds_metric: str,
        perturbation: str,
        pds_threshold: float = 0.8,
    ) -> List[str]:
        """Get genes that pass PDS threshold for a specific perturbation.
        
        Args:
            dataset_name: Name of the dataset.
            pds_metric: PDS metric to use for filtering (e.g., 'Energy_Distance').
            perturbation: Perturbation identifier.
            pds_threshold: Minimum PDS score threshold (default: 0.8).
            
        Returns:
            List of gene names where PDS >= threshold for this perturbation.
            
        Raises:
            ValueError: If metric or perturbation not found in PDS data, or
                if no genes pass the threshold.
        """
        pds_data = self.load_pds_data(dataset_name)
        
        # Find metric index
        metrics = list(np.asarray(pds_data['metrics']).tolist())
        if pds_metric not in metrics:
            raise ValueError(
                f"PDS metric '{pds_metric}' not found. Available: {metrics}"
            )
        metric_idx = metrics.index(pds_metric)
        
        # Find perturbation index
        perturbations = list(np.asarray(pds_data['perturbations']).tolist())
        normalized_perturbation = perturbation.replace("_", "+")
        if normalized_perturbation not in perturbations:
            raise ValueError(
                f"Perturbation '{perturbation}' not found. Available: {perturbations}"
            )
        pert_idx = perturbations.index(normalized_perturbation)
        
        # Get PDS scores for this metric and perturbation
        scores = np.asarray(pds_data['scores_mean'])[metric_idx, pert_idx, :]
        scores = np.asarray(scores).squeeze()
        if scores.ndim != 1:
            raise ValueError(
                "Expected 1D PDS scores for a single perturbation. "
                f"Got shape {scores.shape} for {dataset_name}/{pds_metric}/{perturbation}."
            )
        genes = np.asarray(pds_data['genes'])
        
        # Filter genes by threshold
        passing_mask = scores >= pds_threshold
        filtered_genes = genes[passing_mask].tolist()
        if len(filtered_genes) == 0:
            raise ValueError(
                f"No genes pass PDS threshold {pds_threshold} for "
                f"{dataset_name}/{pds_metric}/{perturbation}."
            )
        
        return filtered_genes
    
    def get_all_filtered_genes(
        self,
        dataset_name: str,
        pds_metric: str,
        pds_threshold: float = 0.8,
    ) -> Dict[str, List[str]]:
        """Get filtered genes for all perturbations.
        
        Args:
            dataset_name: Name of the dataset.
            pds_metric: PDS metric to use for filtering.
            pds_threshold: Minimum PDS score threshold.
            
        Returns:
            Dictionary mapping perturbation -> list of gene names.
            Each list has variable length based on threshold.
        """
        pds_data = self.load_pds_data(dataset_name)
        perturbations = pds_data['perturbations']
        
        filtered_genes_dict = {}
        for perturbation in perturbations:
            filtered_genes_dict[perturbation] = self.get_filtered_genes(
                dataset_name=dataset_name,
                pds_metric=pds_metric,
                perturbation=perturbation,
                pds_threshold=pds_threshold,
            )
        
        return filtered_genes_dict


def apply_gene_filter(
    genewise_values: np.ndarray,
    gene_names: List[str],
    filter_genes: List[str],
) -> np.ndarray:
    """Filter gene-wise values to only include specified genes.
    
    Args:
        genewise_values: Array of gene-wise metric values (n_genes,).
        gene_names: List of all gene names corresponding to genewise_values.
        filter_genes: List of genes to keep.
        
    Returns:
        Filtered array containing only values for genes in filter_genes.
        
    Raises:
        ValueError: If shapes don't match or filter_genes not found.
    """
    if len(genewise_values) != len(gene_names):
        raise ValueError(
            f"Length mismatch: genewise_values has {len(genewise_values)} elements "
            f"but gene_names has {len(gene_names)} elements"
        )
    
    # Find indices of filter_genes in gene_names
    gene_names_array = np.array(gene_names)
    filter_genes_array = np.array(filter_genes)
    
    # Get mask for genes that are in filter list
    mask = np.isin(gene_names_array, filter_genes_array)
    
    if not np.any(mask):
        raise ValueError(
            f"None of the filter genes found in gene_names. "
            f"Filter genes: {filter_genes[:5]}..., Gene names: {gene_names[:5]}..."
        )
    
    # Apply filter
    filtered_values = genewise_values[mask]
    
    return filtered_values


def apply_perturbation_specific_filter(
    genewise_values: np.ndarray,
    gene_names: List[str],
    perturbation: str,
    gene_filter_map: Dict[str, List[str]],
) -> np.ndarray:
    """Apply perturbation-specific gene filtering.
    
    Args:
        genewise_values: Array of gene-wise metric values.
        gene_names: List of all gene names.
        perturbation: Perturbation identifier.
        gene_filter_map: Dictionary mapping perturbation -> gene list.
        
    Returns:
        Filtered array for the specific perturbation.
        
    Raises:
        KeyError: If perturbation not found in gene_filter_map.
    """
    if perturbation not in gene_filter_map:
        raise KeyError(
            f"Perturbation '{perturbation}' not found in gene_filter_map. "
            f"Available: {list(gene_filter_map.keys())[:10]}..."
        )
    
    filter_genes = gene_filter_map[perturbation]
    
    return apply_gene_filter(genewise_values, gene_names, filter_genes)
