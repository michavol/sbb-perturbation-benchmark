"""
Data management for CellSimBench framework.

This module provides the central data management system for handling AnnData objects,
DEG weights, baselines, and perturbation conditions.
"""

from typing import Dict, List, Tuple, Optional, Union
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

# Import metrics functions from existing codebase
def mse(x1: np.ndarray, x2: np.ndarray) -> float:
    """Calculate Mean Squared Error.
    
    Args:
        x1: First array of values.
        x2: Second array of values.
        
    Returns:
        Mean squared error between x1 and x2.
    """
    return np.mean((x1 - x2) ** 2)

def wmse(x1: np.ndarray, x2: np.ndarray, weights: np.ndarray) -> float:
    """Calculate Weighted Mean Squared Error.
    
    Args:
        x1: First array of values.
        x2: Second array of values.
        weights: Weight array for each element.
        
    Returns:
        Weighted mean squared error between x1 and x2.
    """
    weights_arr = np.array(weights)
    x1_arr = np.array(x1)
    x2_arr = np.array(x2)
    normalized_weights = weights_arr / np.sum(weights_arr)
    return np.sum(normalized_weights * ((x1_arr - x2_arr) ** 2))

def pearson(x1: np.ndarray, x2: np.ndarray) -> float:
    """Calculate Pearson correlation coefficient.
    
    Args:
        x1: First array of values.
        x2: Second array of values.
        
    Returns:
        Pearson correlation coefficient between x1 and x2.
    """
    return np.corrcoef(x1, x2)[0, 1]

def r2_score_on_deltas(delta_true: np.ndarray, delta_pred: np.ndarray, 
                      weights: Optional[np.ndarray] = None) -> float:
    """Calculate R² score on deltas with optional weighting.
    
    Args:
        delta_true: True delta values.
        delta_pred: Predicted delta values.
        weights: Optional weight array for weighted R².
        
    Returns:
        R² score between true and predicted deltas.
    """
    from sklearn.metrics import r2_score
    if len(delta_true) < 2 or len(delta_pred) < 2 or delta_true.shape != delta_pred.shape:
        return np.nan
    if weights is not None and np.sum(weights) != 0:
        return r2_score(delta_true, delta_pred, sample_weight=weights)
    else:
        return r2_score(delta_true, delta_pred)


class DataManager:
    """Handles data loading and DEG weights extraction for CellSimBench.
    
    This class manages AnnData objects containing perturbation response data,
    computes and caches DEG weights, and provides utilities for accessing
    baselines, splits, and perturbation conditions.
    
    Attributes:
        config: Dataset configuration dictionary.
        adata: Loaded AnnData object with expression data.
        deg_names_dict: Dictionary mapping perturbations to DEG names.
        deg_scores_dict: Dictionary mapping perturbations to DEG scores.
        deg_pvals_dict: Dictionary mapping perturbations to DEG p-values.
        pert_normalized_abs_scores_vsrest: Precomputed normalized DEG weights.
        
    Example:
        >>> config = {'data_path': 'data.h5ad', 'covariate_key': 'donor_id'}
        >>> dm = DataManager(config)
        >>> adata = dm.load_dataset()
        >>> weights = dm.get_deg_weights('donor1', 'GENE1')
    """
    
    def __init__(self, dataset_config: Dict) -> None:
        """Initialize DataManager with dataset configuration.
        
        Args:
            dataset_config: Dictionary containing dataset configuration including
                           data_path, covariate_key, and baseline keys.
        """
        self.config = dataset_config
        self.adata: Optional[sc.AnnData] = None
        self.deg_names_dict: Optional[Dict[str, List[str]]] = None
        self.deg_scores_dict: Optional[Dict[str, np.ndarray]] = None
        self.deg_pvals_dict: Optional[Dict[str, np.ndarray]] = None
        self.pert_normalized_abs_scores_vsrest: Dict[str, np.ndarray] = {}
        self.pert_deg_weights_vscontrol: Dict[str, np.ndarray] = {}
        self.pert_deg_weights_synthetic: Dict[str, np.ndarray] = {}
        # Which weight source to use in get_deg_weights():
        #   'vsrest'    – adata.uns DEGs computed vs all other perturbations (default)
        #   'vscontrol' – deg_control.csv DEGs computed vs control cells
        #   'synthetic' – deg_synthetic.csv DEGs computed vs synthetic control cells
        self.deg_weight_source: str = 'vsrest'
        
    def load_dataset(self) -> sc.AnnData:
        """Load the h5ad file and precompute DEG weights.
        
        Returns:
            Loaded AnnData object with precomputed DEG weights.
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist.
            ValueError: If required DEG data is not found in dataset.
        """
        path = Path(self.config['data_path'])
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        print(f"Loading dataset from {path}...")
        self.adata = sc.read_h5ad(path)
        self.adata.var.index.name = None
        print(f"Loaded AnnData with shape: {self.adata.shape}")
        
        # Extract DEG dictionaries - MANDATORY for CellSimBench operation
        try:
            self.deg_names_dict = self.adata.uns['names_df_dict_gt']
            self.deg_scores_dict = self.adata.uns['scores_df_dict_gt']
            self.deg_pvals_dict = self.adata.uns.get('pvals_adj_df_dict_gt', self.adata.uns.get('pvals_adj_df_dict_gt'))
                        
            # Precompute normalized weights for all perturbations
            self._precompute_deg_weights()
            print(f"Precomputed DEG weights for {len(self.pert_normalized_abs_scores_vsrest)} perturbations")
            
        except KeyError as e:
            raise ValueError(f"Required DEG data not found in dataset: {e}. "
                           f"Dataset must contain precomputed DEG information in uns['names_df_dict_gt'] "
                           f"and uns['scores_df_dict_gt']. Please use a properly processed dataset.")

        # Load vs-control DEG weights (optional - used by PDS_wMSE default and deg_weight_source='vscontrol')
        self._precompute_deg_weights_vscontrol()
        if self.pert_deg_weights_vscontrol:
            print(f"Loaded vs-control DEG weights for {len(self.pert_deg_weights_vscontrol)} perturbations")
        else:
            print("No vs-control DEG weights found (deg_control.csv missing)")

        # Load synthetic-control DEG weights (optional - used when deg_weight_source='synthetic')
        self._precompute_deg_weights_synthetic()
        if self.pert_deg_weights_synthetic:
            print(f"Loaded synthetic DEG weights for {len(self.pert_deg_weights_synthetic)} perturbations")
        else:
            print("No synthetic DEG weights found (deg_synthetic.csv missing)")
        
        return self.adata
    
    def _precompute_deg_weights(self) -> None:
        """Precompute normalized DEG weights following plotting.py logic.
        
        Computes min-max normalized and squared weights for each perturbation
        based on DEG scores. Weights are cached in pert_normalized_abs_scores_vsrest.
        """
        self.pert_normalized_abs_scores_vsrest_df = {}
        for cov_pert_key in tqdm(self.deg_scores_dict.keys(), desc="Calculating Weights"):
            if 'control' in cov_pert_key.lower():
                continue
            
            # Get scores and names for this covariate-perturbation combination
            scores = self.deg_scores_dict[cov_pert_key]
            gene_names = self.deg_names_dict[cov_pert_key]
            
            # Convert to absolute scores
            abs_scores = np.abs(scores)
            
            # Min-max normalization
            min_val = np.min(abs_scores)
            max_val = np.max(abs_scores)

            normalized_weights = (abs_scores - min_val) / (max_val - min_val)
            
            # Handle NaNs
            normalized_weights = np.nan_to_num(normalized_weights, nan=0.0)
            
            # Square the weights for stronger emphasis
            normalized_weights = np.square(normalized_weights)
            
            # Create series and handle duplicates by taking the maximum weight for each gene
            weights_df = pd.DataFrame({
                'gene': gene_names,
                'weight': normalized_weights
            })
            
            # Group by gene and take the maximum weight in case of duplicates
            weights_aggregated = weights_df.groupby('gene')['weight'].max()
            
            # Reindex to match adata.var_names
            weights = weights_aggregated.reindex(self.adata.var_names, fill_value=0.0)
            
            self.pert_normalized_abs_scores_vsrest[cov_pert_key] = weights.values
            self.pert_normalized_abs_scores_vsrest_df[cov_pert_key] = weights
    
    def _load_deg_weights_from_csv(self, csv_path: Path) -> Dict[str, np.ndarray]:
        """Load and normalise DEG weights from a discrimination-analysis CSV file.

        The CSV must have columns: ``Perturbation``, ``Gene``, ``score``.
        Weights are computed as: |score| → min-max normalisation → square.
        Returns a dict mapping ``<dataset_name>_<perturbation>`` → weight array
        aligned to ``self.adata.var_names``.
        """
        import csv as _csv
        dataset_name = self.config.get('name', '')
        gene_to_idx = {g: i for i, g in enumerate(self.adata.var_names)}
        n_genes = len(self.adata.var_names)

        raw_abs: Dict[str, Dict[str, float]] = {}
        with csv_path.open('r', encoding='utf-8', newline='') as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                pert = str(row.get('Perturbation', '')).strip()
                gene = str(row.get('Gene', '')).strip()
                raw_score = row.get('score', '')
                if not pert or not gene or raw_score in ('', None):
                    continue
                try:
                    score_val = abs(float(raw_score))
                except (TypeError, ValueError):
                    continue
                if gene not in gene_to_idx:
                    continue
                if pert not in raw_abs:
                    raw_abs[pert] = {}
                prev = raw_abs[pert].get(gene)
                if prev is None or score_val > prev:
                    raw_abs[pert][gene] = score_val

        result: Dict[str, np.ndarray] = {}
        for pert_name, gene_scores in raw_abs.items():
            w = np.zeros(n_genes, dtype=float)
            for gene, score in gene_scores.items():
                w[gene_to_idx[gene]] = score

            w_min, w_max = float(np.min(w)), float(np.max(w))
            denom = w_max - w_min
            if denom > 0:
                w = (w - w_min) / denom
            else:
                w = np.zeros_like(w)

            w = np.nan_to_num(np.square(w), nan=0.0)
            result[f"{dataset_name}_{pert_name}"] = w

        return result

    def _precompute_deg_weights_vscontrol(self) -> None:
        """Load DEG weights from ``deg_control.csv`` (perturbation vs control cells).

        Reads the precomputed CSV produced by the perturbation discrimination
        analysis pipeline:

            analyses/perturbation_discrimination/results/<dataset>/deg_scanpy/deg_control.csv
        """
        dataset_name = self.config.get('name', '')
        if not dataset_name:
            return
        root = Path(__file__).parents[2]
        csv_path = (root / "analyses" / "perturbation_discrimination" / "results"
                    / dataset_name / "deg_scanpy" / "deg_control.csv")
        if not csv_path.exists():
            return
        self.pert_deg_weights_vscontrol = self._load_deg_weights_from_csv(csv_path)

    def _precompute_deg_weights_synthetic(self) -> None:
        """Load DEG weights from ``deg_synthetic.csv`` (perturbation vs synthetic controls).

        Reads the precomputed CSV produced by the perturbation discrimination
        analysis pipeline:

            analyses/perturbation_discrimination/results/<dataset>/deg_scanpy/deg_synthetic.csv
        """
        dataset_name = self.config.get('name', '')
        if not dataset_name:
            return
        root = Path(__file__).parents[2]
        csv_path = (root / "analyses" / "perturbation_discrimination" / "results"
                    / dataset_name / "deg_scanpy" / "deg_synthetic.csv")
        if not csv_path.exists():
            return
        self.pert_deg_weights_synthetic = self._load_deg_weights_from_csv(csv_path)

    def _lookup_csv_weights(
        self,
        store: Dict[str, np.ndarray],
        covariate_value: str,
        perturbation: str,
        gene_order: List[str],
    ) -> np.ndarray:
        """Look up weights from a pre-loaded CSV weight store.

        Handles the ``+`` / ``_`` combo-separator ambiguity automatically.
        Returns a zero vector when no entry is found (caller should treat this
        as "not found").
        """
        cov_pert_key = f"{covariate_value}_{perturbation}"
        if cov_pert_key not in store:
            alt_pert = perturbation.replace('+', '_')
            cov_pert_key = f"{covariate_value}_{alt_pert}"
        if cov_pert_key not in store:
            return np.zeros(len(gene_order))

        w_series = pd.Series(store[cov_pert_key], index=list(self.adata.var_names))
        return w_series.reindex(gene_order, fill_value=0.0).values

    def get_deg_weights_vscontrol(self, covariate_value: str, perturbation: str, gene_order: List[str]) -> np.ndarray:
        """Return vs-control DEG weights aligned to ``gene_order``.

        Returns a zero vector when no vs-control data is available for this
        perturbation.  Callers should treat an all-zero vector as "not found"
        and either raise an error or fall back to vs-rest weights.

        Handles combo-separator ambiguity: some datasets store combos with ``_``
        (e.g. ``DOT1L_EP300`` in deg_control.csv) while the AnnData uses ``+``
        (e.g. ``DOT1L+EP300``).  Both variants are tried automatically.
        """
        return self._lookup_csv_weights(
            self.pert_deg_weights_vscontrol, covariate_value, perturbation, gene_order
        )

    def get_deg_weights_synthetic(self, covariate_value: str, perturbation: str, gene_order: List[str]) -> np.ndarray:
        """Return synthetic-control DEG weights aligned to ``gene_order``.

        Returns a zero vector when no synthetic DEG data is available for this
        perturbation.  Callers should treat an all-zero vector as "not found".

        Handles combo-separator ambiguity identically to ``get_deg_weights_vscontrol``.
        """
        return self._lookup_csv_weights(
            self.pert_deg_weights_synthetic, covariate_value, perturbation, gene_order
        )

    def get_available_controls(self) -> List[str]:
        """Get all available control types from the data.
        
        Returns:
            List of control condition names.
        """
        control_conditions = self.adata.obs['condition'][
            self.adata.obs['condition'].str.contains('ctrl', case=False, na=False)
        ].unique()
        return control_conditions.tolist()
    
    def get_available_splits(self) -> List[str]:
        """Get all available split columns from the data.
        
        Returns:
            List of split column names.
        """
        split_columns = [col for col in self.adata.obs.columns if 'split' in col.lower()]
        return split_columns
    
    def get_deg_weights(self, covariate_value: str, perturbation: str, gene_order: List[str]) -> np.ndarray:
        """Return DEG-based weights for a covariate-perturbation pair, aligned to ``gene_order``.

        The weight source is determined by ``self.deg_weight_source``:

        * ``'vsrest'`` (default) – weights from ``adata.uns['scores_df_dict_gt']``
          (computed during dataset preprocessing, reference = rest).
        * ``'vscontrol'`` – weights from ``deg_control.csv`` (perturbation vs control
          cells), raising when unavailable for a perturbation.
        * ``'synthetic'`` – weights from ``deg_synthetic.csv`` (perturbation vs
          synthetic control cells), raising when unavailable for a perturbation.
        """
        source = self.deg_weight_source

        if source == 'vscontrol':
            w = self.get_deg_weights_vscontrol(covariate_value, perturbation, gene_order)
            if np.sum(w) > 1e-12:
                return w
            raise ValueError(
                f"Missing DEG weights for '{covariate_value}_{perturbation}' with "
                f"deg_weight_source='vscontrol'. Expected entry in deg_control.csv."
            )

        elif source == 'synthetic':
            w = self.get_deg_weights_synthetic(covariate_value, perturbation, gene_order)
            if np.sum(w) > 1e-12:
                return w
            raise ValueError(
                f"Missing DEG weights for '{covariate_value}_{perturbation}' with "
                f"deg_weight_source='synthetic'. Expected entry in deg_synthetic.csv."
            )

        elif source != 'vsrest':
            raise ValueError(
                f"Unsupported deg_weight_source='{source}'. "
                "Valid options: 'vsrest', 'vscontrol', 'synthetic'."
            )

        # vsrest (default)
        cov_pert_key = f"{covariate_value}_{perturbation}"
        if cov_pert_key in self.pert_normalized_abs_scores_vsrest:
            weights = self.pert_normalized_abs_scores_vsrest_df[cov_pert_key]
            return weights.reindex(gene_order, fill_value=0.0).values

        return np.zeros(len(gene_order))
    
    def get_deg_mask(self, covariate_value: str, perturbation: str, gene_order: List[str], pval_threshold: float = 0.05) -> np.ndarray:
        """
        Get DEG mask for a specific covariate-perturbation combination.
        
        Args:
            covariate_value: Value of the covariate (e.g., donor ID)
            perturbation: Perturbation identifier
            gene_order: Ordered list of gene names to align the mask to.
            pval_threshold: P-value threshold for significance
            
        Returns:
            Boolean array indicating DEG positions, aligned to gene_order
        """
        cov_pert_key = f"{covariate_value}_{perturbation}"
        
        if cov_pert_key not in self.deg_pvals_dict:
            return np.zeros(len(gene_order), dtype=bool)
        
        # Get p-values and gene names (these are in DEG rank order, not var_names order)
        pvals = self.deg_pvals_dict[cov_pert_key]
        gene_names = self.deg_names_dict[cov_pert_key]
        
        # Create boolean mask for significant genes
        sig_mask = pvals < pval_threshold
        
        # Handle duplicates by taking the minimum p-value (most significant) for each gene
        pvals_df = pd.DataFrame({
            'gene': gene_names,
            'pval': pvals,
            'significant': sig_mask
        })
        
        # Group by gene and take minimum p-value, then check significance
        pvals_aggregated = pvals_df.groupby('gene')['pval'].min()
        deg_mask_aggregated = pvals_aggregated < pval_threshold

        
        # Reindex to match the target gene order
        deg_mask = deg_mask_aggregated.reindex(gene_order, fill_value=False)
        
        return deg_mask.values
    
    def get_control_baseline(
        self, 
        donor_id: Optional[str] = None,
        gene_order: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Get control baseline expression from uns using dataset-specific key.
        
        Args:
            donor_id: Optional donor/covariate ID for donor-specific baseline.
            
        Returns:
            Control baseline expression array, optionally aligned to gene_order.
            
        Raises:
            ValueError: If donor not found or control baseline not available.
        """
        control_baseline_key = self.config['control_baseline_key']
        if control_baseline_key not in self.adata.uns:
            # Fallback to calculating control mean from data
            warnings.warn(f"Control baseline key '{control_baseline_key}' not found in uns. Calculating from data.")
            control_conditions = self.get_available_controls()
            if control_conditions:
                baseline = (
                    self.adata[self.adata.obs['condition'] == control_conditions[0]]
                    .X.mean(axis=0)
                    .A1
                )
                return self._reorder_baseline_if_needed(baseline, gene_order)
            else:
                raise ValueError("No control conditions found in data")
        
        baseline_data = self.adata.uns[control_baseline_key]
        
        # Handle DataFrame format (donor-specific baselines)
        if hasattr(baseline_data, 'index') and hasattr(baseline_data, 'columns'):
            baseline_data = self._align_baseline_dataframe_columns(
                baseline_data, gene_order=gene_order, baseline_key=control_baseline_key
            )
            if donor_id is not None:
                # Return specific donor baseline
                if donor_id in baseline_data.index:
                    baseline = baseline_data.loc[donor_id].values
                    return self._reorder_baseline_if_needed(baseline, gene_order)
                else:
                    raise ValueError(f"Donor '{donor_id}' not found in control baseline. Available: {list(baseline_data.index)}")
            else:
                # Return mean across all donors
                baseline = baseline_data.mean(axis=0).values
                return self._reorder_baseline_if_needed(baseline, gene_order)
        else:
            # Handle array format (single baseline)
            return self._reorder_baseline_if_needed(np.asarray(baseline_data), gene_order)

    def _reorder_baseline_if_needed(
        self, 
        baseline: np.ndarray, 
        gene_order: Optional[List[str]],
    ) -> np.ndarray:
        """Reorder baseline vector to match the provided gene order.

        Args:
            baseline: Baseline expression array aligned to ``self.adata.var_names`` or
                already aligned to ``gene_order``.
            gene_order: Optional target gene order.

        Returns:
            Baseline expression aligned to ``gene_order`` if provided, otherwise unchanged.

        Raises:
            ValueError: If ``gene_order`` contains unknown genes or if the baseline
                length is incompatible with ``gene_order`` and ``adata.var_names``.
        """
        if gene_order is None:
            return baseline

        baseline = np.asarray(baseline)
        var_names_list = list(self.adata.var_names)
        var_name_set = set(var_names_list)
        gene_order_set = set(gene_order)

        if len(gene_order_set) != len(gene_order):
            raise ValueError("gene_order contains duplicate gene names.")

        if not gene_order_set.issubset(var_name_set):
            missing = sorted(gene_order_set - var_name_set)
            raise ValueError(
                "gene_order contains genes not present in adata.var_names: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )

        if baseline.shape[0] == len(gene_order):
            return baseline

        if baseline.shape[0] != len(var_names_list):
            raise ValueError(
                "baseline length does not match adata.var_names or gene_order length: "
                f"{baseline.shape[0]} vs {len(var_names_list)} or {len(gene_order)}"
            )

        var_name_to_idx = {name: idx for idx, name in enumerate(var_names_list)}
        index_map = np.fromiter(
            (var_name_to_idx[gene] for gene in gene_order),
            dtype=int,
            count=len(gene_order),
        )
        return baseline[index_map]

    def _align_baseline_dataframe_columns(
        self,
        baseline_data: pd.DataFrame,
        gene_order: Optional[List[str]],
        baseline_key: str,
    ) -> pd.DataFrame:
        """Align baseline DataFrame columns to ``adata.var_names`` or ``gene_order``.

        Args:
            baseline_data: Baseline DataFrame with genes as columns.
            gene_order: Optional target gene order.
            baseline_key: Baseline key for error context.

        Returns:
            Baseline DataFrame with columns ordered to match ``gene_order`` if provided,
            otherwise ``adata.var_names``.

        Raises:
            ValueError: If baseline columns do not match the expected gene set.
        """
        target_order = list(gene_order) if gene_order is not None else list(self.adata.var_names)
        baseline_cols = list(baseline_data.columns)
        missing = sorted(set(target_order) - set(baseline_cols))
        if missing:
            raise ValueError(
                f"Baseline '{baseline_key}' columns do not match expected genes. "
                f"Missing: {missing[:10]}{'...' if len(missing) > 10 else ''}"
            )
        if baseline_cols == target_order:
            return baseline_data
        return baseline_data.reindex(columns=target_order)
    
    def get_control_baseline_dict(self) -> Dict[str, np.ndarray]:
        """Get all donor-specific control baselines as a dictionary.
        
        Returns:
            Dictionary mapping donor IDs to control baseline arrays.
            
        Raises:
            ValueError: If control baseline not found in dataset.
        """
        control_baseline_key = self.config['control_baseline_key']
        if control_baseline_key not in self.adata.uns:
            raise ValueError(f"Control baseline key '{control_baseline_key}' not found in uns")
        
        baseline_data = self.adata.uns[control_baseline_key]
        
        # Handle DataFrame format (donor-specific baselines)
        if hasattr(baseline_data, 'index') and hasattr(baseline_data, 'columns'):
            baseline_data = self._align_baseline_dataframe_columns(
                baseline_data, gene_order=None, baseline_key=control_baseline_key
            )
            return {donor_id: baseline_data.loc[donor_id].values for donor_id in baseline_data.index}
        else:
            # Handle array format (single baseline) - return with generic key
            return {'default': baseline_data}
    
    def get_ground_truth_baseline(self) -> Dict[str, np.ndarray]:
        """Get ground truth baseline expressions from uns using dataset-specific key.
        
        Returns:
            Dictionary mapping covariate-perturbation keys to ground truth arrays.
            
        Raises:
            ValueError: If ground truth baseline not found in dataset.
        """
        ground_truth_baseline_key = self.config['ground_truth_baseline_key']
        if ground_truth_baseline_key not in self.adata.uns:
            raise ValueError(f"Ground truth baseline key '{ground_truth_baseline_key}' not found in uns")
        
        baseline_data = self.adata.uns[ground_truth_baseline_key]
        
        # Handle DataFrame format (covariate_perturbation-specific baselines)
        if hasattr(baseline_data, 'index') and hasattr(baseline_data, 'columns'):
            return {cov_pert_key: baseline_data.loc[cov_pert_key].values 
                    for cov_pert_key in baseline_data.index}
        else:
            # Handle dictionary format (already properly structured)
            return baseline_data
    
    def get_dataset_mean_baseline(self) -> Dict[str, np.ndarray]:
        """Get technical duplicate second half baseline expressions from uns.
        
        This is ALWAYS required as it serves as the control for all delta metrics.
        
        Returns:
            Dictionary mapping covariate-perturbation keys to baseline arrays.
            
        Raises:
            ValueError: If dataset mean baseline not found (required for delta metrics).
        """
        # Try to get from config first, with fallback to standard key name
        if hasattr(self.config, "dataset"):
            dataset_mean_baseline_key = self.config.dataset.dataset_mean_baseline_key
        else:
            dataset_mean_baseline_key = self.config.get("dataset_mean_baseline_key")
            if dataset_mean_baseline_key is None and isinstance(self.config.get("dataset"), dict):
                dataset_mean_baseline_key = self.config["dataset"].get("dataset_mean_baseline_key")
        
        if dataset_mean_baseline_key not in self.adata.uns:
            raise ValueError(f"Dataset mean baseline key '{dataset_mean_baseline_key}' not found in uns. "
                           f"This baseline is REQUIRED for delta metrics calculation.")
        
        baseline_data = self.adata.uns[dataset_mean_baseline_key]
        
        # Handle DataFrame format (covariate_perturbation-specific baselines)
        if hasattr(baseline_data, 'index') and hasattr(baseline_data, 'columns'):
            return {cov_pert_key: baseline_data.loc[cov_pert_key].values 
                    for cov_pert_key in baseline_data.index}
        else:
            # Handle dictionary format (already properly structured)
            return baseline_data
    
    def load_obs_only(self) -> pd.DataFrame:
        """Load only the observation metadata without the full expression matrix.
        
        Efficient method for accessing metadata without loading expression data.
        
        Returns:
            DataFrame containing observation metadata.
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist.
        """
        import h5py
        from anndata.experimental import read_elem
        
        path = Path(self.config['data_path'])
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
                
        print(f"Loading obs metadata from {path}...")
        with h5py.File(path, 'r') as f:
            obs = read_elem(f['obs'])
        
        return obs
    
    def get_perturbation_conditions(self, split_name: str, obs: Optional[pd.DataFrame] = None) -> Dict[str, List[str]]:
        """
        Get train/test perturbation conditions for a given split.
        
        Args:
            split_name: Name of the split column
            obs: Optional obs DataFrame. If not provided, uses self.adata.obs
            
        Returns:
            Dict with 'train' and 'test' lists of conditions
        """
        if obs is None:
            if self.adata is None:
                raise ValueError("Either provide obs parameter or load dataset first")
            obs = self.adata.obs
            
        if split_name not in obs.columns:
            raise ValueError(f"Split '{split_name}' not found in obs")
        
        split_data = obs[split_name]
        
        train_conditions = obs[split_data == 'train']['condition'].unique().tolist()
        val_conditions = obs[split_data == 'val']['condition'].unique().tolist()
        test_conditions = obs[split_data == 'test']['condition'].unique().tolist()

        # Remove control and placeholder conditions
        train_conditions = [
            condition
            for condition in train_conditions
            if 'ctrl' not in condition and condition != '*'
        ]
        val_conditions = [
            condition
            for condition in val_conditions
            if 'ctrl' not in condition and condition != '*'
        ]
        test_conditions = [
            condition
            for condition in test_conditions
            if 'ctrl' not in condition and condition != '*'
        ]
        
        return {
            'train': train_conditions,
            'val': val_conditions,
            'test': test_conditions
        }
    
    def get_covariate_condition_pairs(self, split_name: str, split_type: str = 'test') -> List[Tuple[str, str]]:
        """Get all covariate-condition pairs for a given split.
        
        Args:
            split_name: Name of the split column.
            split_type: Either 'train', 'val', or 'test'.
            
        Returns:
            List of (covariate_value, condition) tuples.
            
        Raises:
            ValueError: If split_name not found in obs columns.
        """
        if split_name not in self.adata.obs.columns:
            raise ValueError(f"Split '{split_name}' not found in obs")
        
        split_mask = self.adata.obs[split_name] == split_type
        split_data = self.adata.obs[split_mask]
        
        covariate_key = self.config['covariate_key']
        if covariate_key not in self.adata.obs.columns:
            raise ValueError(f"Covariate key '{covariate_key}' not found in obs")
        
        pairs = split_data[[covariate_key, 'condition']].drop_duplicates()
        
        return [(str(row[covariate_key]), row['condition']) for _, row in pairs.iterrows()]
    
    # ===== Baseline Access Methods for Calibrated Metrics =====
    
    def get_positive_baseline(
        self, 
        covariate: str, 
        perturbation: str,
        split_column: str = "tech_dup_split",
        current_split: str = "first_half",
        gene_order: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Get positive baseline (technical duplicate from opposite split).
        
        For calibrated metrics, the positive baseline represents the best-case
        scenario: technical duplicate split. This should be from the OPPOSITE
        split to the prediction data.
        
        Args:
            covariate: Covariate value (e.g., donor ID).
            perturbation: Perturbation identifier.
            split_column: Column name for technical duplicate split.
            current_split: The split used for prediction/truth (to get opposite one).
            
        Returns:
            Mean expression array for the opposite technical duplicate split,
            optionally aligned to gene_order.
            
        Raises:
            ValueError: If split column not found or perturbation not available.
            
        Example:
            If pred/truth are from 'first_half', this returns mean of 'second_half'.
        """
        if split_column not in self.adata.obs.columns:
            raise ValueError(f"Split column '{split_column}' not found in obs")
        
        # Determine opposite split
        if current_split == "first_half":
            opposite_split = "second_half"
        elif current_split == "second_half":
            opposite_split = "first_half"
        else:
            raise ValueError(
                f"Unknown split value: {current_split}. "
                f"Expected 'first_half' or 'second_half'"
            )

        baseline_key = None
        if current_split == "first_half":
            baseline_key = self.config.get("technical_duplicate_baseline_key")
        elif current_split == "second_half":
            baseline_key = self.config.get("ground_truth_baseline_key")

        if baseline_key and baseline_key in self.adata.uns:
            baseline_data = self.adata.uns[baseline_key]
            if hasattr(baseline_data, "index") and hasattr(baseline_data, "columns"):
                baseline_data = self._align_baseline_dataframe_columns(
                    baseline_data, gene_order=gene_order, baseline_key=baseline_key
                )
                cov_pert_key = f"{covariate}_{perturbation}"
                if cov_pert_key in baseline_data.index:
                    baseline = baseline_data.loc[cov_pert_key].values
                    return self._reorder_baseline_if_needed(baseline, gene_order)
                if perturbation in baseline_data.index:
                    baseline = baseline_data.loc[perturbation].values
                    return self._reorder_baseline_if_needed(baseline, gene_order)
        
        # Get covariate key from config
        covariate_key = self.config['covariate_key']
        
        # Filter for the specific covariate, perturbation, and opposite split
        mask = (
            (self.adata.obs[covariate_key] == covariate) &
            (self.adata.obs['condition'] == perturbation) &
            (self.adata.obs[split_column] == opposite_split)
        )
        
        if mask.sum() == 0:
            raise ValueError(
                f"No cells found for {covariate}_{perturbation} "
                f"in {split_column}={opposite_split}"
            )
        
        # Compute mean expression across cells
        subset = self.adata[mask]
        mean_expr = subset.X.mean(axis=0)
        
        # Handle sparse matrices
        if hasattr(mean_expr, 'A1'):
            mean_expr = mean_expr.A1
        
        baseline = np.asarray(mean_expr)
        return self._reorder_baseline_if_needed(baseline, gene_order)
    
    def get_negative_baseline(
        self,
        covariate: str,
        perturbation: str,
        n_samples_pred: int,
    ) -> np.ndarray:
        """Get negative baseline (control samples, capped to prediction sample count).
        
        For calibrated metrics, the negative baseline represents the worst-case
        scenario: control (untreated) samples. The number of control samples is
        capped to match the prediction sample count for fair comparison.
        
        Args:
            covariate: Covariate value (e.g., donor ID).
            perturbation: Perturbation identifier (not used for controls, but kept
                for consistency).
            n_samples_pred: Number of samples in the prediction to cap controls to.
            
        Returns:
            Mean expression array for control samples.
            
        Raises:
            ValueError: If no control samples found.
        """
        # Get covariate key from config
        covariate_key = self.config['covariate_key']
        
        # Get control conditions
        control_conditions = self.get_available_controls()
        if not control_conditions:
            raise ValueError("No control conditions found in data")
        
        # Filter for control samples from the same covariate
        mask = (
            (self.adata.obs[covariate_key] == covariate) &
            (self.adata.obs['condition'].isin(control_conditions))
        )
        
        if mask.sum() == 0:
            raise ValueError(f"No control samples found for {covariate}")
        
        # Cap the number of control samples to n_samples_pred
        control_indices = np.where(mask)[0]
        # if len(control_indices) > n_samples_pred:
        #     # Randomly sample to match prediction sample count
        #     rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        #     control_indices = rng.choice(
        #         control_indices, 
        #         size=n_samples_pred,
        #         replace=False
        #     )
        
        # Compute mean expression across selected control cells
        subset = self.adata[control_indices]
        mean_expr = subset.X.mean(axis=0)
        
        # Handle sparse matrices
        if hasattr(mean_expr, 'A1'):
            mean_expr = mean_expr.A1
        
        return np.asarray(mean_expr)
    
    def get_baseline_pointclouds(
        self,
        covariate: str,
        perturbation: str,
        gene_idx: int,
        split_column: str = "tech_dup_split",
        current_split: str = "first_half",
        n_samples_pred: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get cell-level point clouds for positive and negative baselines.
        
        This is needed for computing energy distance and Wasserstein distance,
        which operate on distributions (point clouds) rather than means.
        
        Args:
            covariate: Covariate value (e.g., donor ID).
            perturbation: Perturbation identifier.
            gene_idx: Index of the gene to extract point clouds for.
            split_column: Column name for technical duplicate split.
            current_split: The split used for prediction/truth.
            n_samples_pred: Number of samples to cap negative baseline to.
            
        Returns:
            Tuple of (positive_baseline_pointcloud, negative_baseline_pointcloud),
            each as 1D numpy arrays with cell-level expression values.
            
        Raises:
            ValueError: If split column not found or no cells available.
        """
        # Get positive baseline point cloud (opposite technical duplicate split)
        if split_column not in self.adata.obs.columns:
            raise ValueError(f"Split column '{split_column}' not found in obs")
        
        # Determine opposite split
        if current_split == "first_half":
            opposite_split = "second_half"
        elif current_split == "second_half":
            opposite_split = "first_half"
        else:
            raise ValueError(
                f"Unknown split value: {current_split}. "
                f"Expected 'first_half' or 'second_half'"
            )
        
        covariate_key = self.config['covariate_key']
        
        # Get positive baseline point cloud
        mask_pos = (
            (self.adata.obs[covariate_key] == covariate) &
            (self.adata.obs['condition'] == perturbation) &
            (self.adata.obs[split_column] == opposite_split)
        )
        
        if mask_pos.sum() == 0:
            raise ValueError(
                f"No cells found for positive baseline: {covariate}_{perturbation} "
                f"in {split_column}={opposite_split}"
            )
        
        pos_cells = self.adata[mask_pos].X[:, gene_idx]
        if hasattr(pos_cells, 'A1'):
            pos_cells = pos_cells.A1
        else:
            pos_cells = np.asarray(pos_cells).ravel()
        
        # Get negative baseline point cloud (controls, capped)
        control_conditions = self.get_available_controls()
        if not control_conditions:
            raise ValueError("No control conditions found in data")
        
        mask_neg = (
            (self.adata.obs[covariate_key] == covariate) &
            (self.adata.obs['condition'].isin(control_conditions))
        )
        
        if mask_neg.sum() == 0:
            raise ValueError(f"No control samples found for {covariate}")
        
        control_indices = np.where(mask_neg)[0]
        # if len(control_indices) > n_samples_pred:
        #     rng = np.random.RandomState(42)
        #     control_indices = rng.choice(
        #         control_indices,
        #         size=n_samples_pred,
        #         replace=False
        #     )
        
        neg_cells = self.adata[control_indices].X[:, gene_idx]
        if hasattr(neg_cells, 'A1'):
            neg_cells = neg_cells.A1
        else:
            neg_cells = np.asarray(neg_cells).ravel()
        
        return pos_cells, neg_cells
 