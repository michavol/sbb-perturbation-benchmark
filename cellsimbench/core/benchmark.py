"""
Main benchmark orchestration for CellSimBench framework.

Coordinates the complete benchmarking pipeline including model execution,
metrics calculation, and result visualization.
"""

import logging
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Sequence, Set
import numpy as np
import pandas as pd
import json
import warnings
from tqdm import tqdm
import pickle
import scanpy as sc
import re
from omegaconf import DictConfig, OmegaConf

from .data_manager import DataManager
from .model_runner import ModelRunner
from .metrics_engine import MetricsEngine
from .baseline_runner import BaselineRunner
from .plotting_engine import PlottingEngine
from .gpu_utils import get_available_gpus, calculate_gpu_assignment

from cellsimbench.utils.effective_genes import _effective_genes_single
from cellsimbench.core.calibrated_metrics import (
    get_aggregated_calibration_denominator_report,
    reset_aggregated_calibration_denominator_issue_counts,
)

log = logging.getLogger(__name__)


class BenchmarkRunner:
    """Main orchestration class for running benchmarks.
    
    Coordinates the complete benchmarking pipeline including data loading,
    model execution, metrics calculation, and visualization generation.
    
    Attributes:
        config: Hydra configuration object.
        data_manager: DataManager instance for data handling.
        model_runner: ModelRunner instance for model execution.
        
    Example:
        >>> runner = BenchmarkRunner(config)
        >>> results = runner.run_benchmark()
    """
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize BenchmarkRunner with configuration.
        
        Args:
            config: Hydra configuration containing dataset, model, and output settings.
        """
        self.config = config
        self.data_manager: Optional[DataManager] = None
        self.model_runner = ModelRunner()
        
    def run_benchmark(self) -> Dict[str, Any]:
        """Execute the complete benchmark pipeline."""
        log.info(f"Starting benchmark: {self.config.experiment.name}")
        
        # ALWAYS use k-fold logic (even if just 1 fold)
        return self._run_kfold_benchmark()

    
    def _run_kfold_benchmark(self) -> Dict[str, Any]:
        """Run benchmark across specified folds."""
        # Determine which folds to run - default to ALL folds
        if hasattr(self.config, 'fold_indices') and self.config.fold_indices is not None:
            fold_indices = self.config.fold_indices
        elif hasattr(self.config.dataset, 'folds'):
            # Default to ALL folds (consistent with training behavior)
            fold_indices = list(range(len(self.config.dataset.folds)))
        else:
            # Backward compatibility - no folds defined
            raise ValueError("No folds defined in dataset config")
        
        log.info(f"Running benchmark on fold(s): {fold_indices}")
        
        # Load dataset manager
        dataset_config = OmegaConf.to_object(self.config.dataset)
        
        self.data_manager = DataManager(dataset_config)
        _ = self.data_manager.load_dataset()

        # Apply deg_weight_source from metrics config (defaults to 'vsrest' if not set)
        deg_weight_source = self.config.get('metrics', {}).get('deg_weight_source', 'vsrest')
        self.data_manager.deg_weight_source = deg_weight_source
        if deg_weight_source != 'vsrest':
            print(f"[benchmark] deg_weight_source='{deg_weight_source}' — all weighted metrics will use this source")
        
        # Gather predictions from specified folds
        all_predictions = self._gather_fold_predictions(fold_indices)
        
        # Always use aggregated_folds for consistency, even with single fold
        split_name = 'aggregated_folds'
        
        output_dir = self._get_output_dir()
        
        # Calculate metrics on concatenated predictions
        log.info("Calculating metrics on aggregated predictions...")
        results = self._calculate_all_metrics(all_predictions, split_name, output_dir)
        
        # Generate plots if configured
        if self.config.output['generate_plots']:
            plotting_engine = PlottingEngine(
                self.data_manager,
                results,
                all_predictions,
                split_name,
                output_dir,
                self.config
            )
            
            # Generate all plots for k-fold results (always aggregated_folds now)
            log.info("Generating plots for k-fold benchmark results")
            plotting_engine.generate_all_plots()
        
        log.info("K-fold benchmark completed successfully")
        return results

    def _get_effective_enabled_baselines(self) -> Optional[List[str]]:
        """Return enabled baselines after applying dataset-specific exclusions."""
        if not hasattr(self.config, 'baselines') or self.config.baselines is None:
            return None

        enabled_baselines = self.config.baselines.get('enabled')
        if enabled_baselines is None:
            return None

        enabled_list = list(enabled_baselines)

        # Additive baseline should only be used for combinatorial datasets.
        additive_allowed_datasets = {'replogle20', 'norman19', 'wessels23'}
        dataset_name = str(self.config.dataset.get('name', ''))
        if dataset_name not in additive_allowed_datasets and 'additive' in enabled_list:
            enabled_list = [baseline for baseline in enabled_list if baseline != 'additive']
            log.info(
                "Excluding additive baseline for single-perturbation dataset '%s'.",
                dataset_name,
            )

        return enabled_list
    
    def _gather_fold_predictions(self, fold_indices: List[int]) -> Dict[str, sc.AnnData]:
        """Gather predictions with optional parallelism."""
        
        should_use_parallel = self._should_use_parallel_inference(fold_indices)
        
        if should_use_parallel:
            log.info("🚀 Using parallel fold inference")
            return self._gather_fold_predictions_parallel(fold_indices)
        else:
            log.info("🔄 Using sequential fold inference")
            return self._gather_fold_predictions_sequential(fold_indices)
    
    def _should_use_parallel_inference(self, fold_indices: List[int]) -> bool:
        """Determine whether to use parallel inference."""
        
        # Use same execution config as training
        parallel_enabled = getattr(self.config.execution, 'parallel_folds', True)
        
        if not parallel_enabled:
            log.info("Parallel inference disabled in execution config")
            return False
        
        if len(fold_indices) <= 1:
            log.info("Single fold inference - using sequential")
            return False
        
        available_gpus = get_available_gpus()
        if not available_gpus:
            log.info("No GPUs available - falling back to sequential inference")
            return False
        
        return True
    
    def _gather_fold_predictions_parallel(self, fold_indices: List[int]) -> Dict[str, sc.AnnData]:
        """Gather predictions from folds in parallel."""
        from concurrent.futures import ThreadPoolExecutor
        
        # GPU assignment  
        available_gpus = get_available_gpus()
        gpu_assignment = calculate_gpu_assignment(fold_indices, available_gpus)
        
        # Use as many workers as folds (GPUs will be assigned round-robin)
        max_workers = len(fold_indices)
        
        # Limit parallelism if requested by user
        max_parallel = getattr(self.config.execution, 'max_parallel_folds', None)
        if max_parallel is not None:
            max_workers = min(max_workers, max_parallel)
        
        log.info(f"Processing {len(fold_indices)} folds in parallel using {max_workers} workers")
        
        # Parallel execution
        fold_predictions = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all fold inference jobs
            future_to_fold = {}
            for fold_idx in fold_indices:
                gpu_id = gpu_assignment[fold_idx]
                future = executor.submit(self._process_single_fold_with_gpu, fold_idx, gpu_id)
                future_to_fold[future] = fold_idx
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_fold):
                fold_idx = future_to_fold[future]
                try:
                    predictions = future.result()  # No timeout
                    fold_predictions[fold_idx] = predictions
                    log.info(f"✅ Fold {fold_idx} inference completed")
                except Exception as e:
                    log.error(f"❌ Fold {fold_idx} inference failed: {e}")
                    raise RuntimeError(f"Inference failed for fold {fold_idx}: {e}")
        
        # Concatenate predictions (existing logic)
        return self._concatenate_fold_predictions(fold_predictions, fold_indices)
    
    def _process_single_fold_with_gpu(self, fold_idx: int, gpu_id: int) -> Dict[str, sc.AnnData]:
        """Process single fold predictions with specific GPU."""
        fold_config = self.config.dataset.folds[fold_idx]
        fold_split = fold_config.split
        
        log.info(f"Processing fold {fold_idx} on GPU {gpu_id} (split: {fold_split})")
        
        return self._process_single_fold(fold_idx, gpu_id)
    
    def _process_single_fold(self, fold_idx: int, gpu_id: Optional[int] = None) -> Dict[str, sc.AnnData]:
        """Process single fold predictions (shared logic for both parallel and sequential)."""
        fold_config = self.config.dataset.folds[fold_idx]
        fold_split = fold_config.split
        
        # Get fold output directory
        output_dir = self._get_output_dir()
        fold_output_dir = output_dir / f'fold_{fold_split}'
        fold_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load fold-specific baselines
        fold_baselines = self._load_fold_baselines(fold_config)
        
        # Load universal baselines for this fold's test conditions
        baseline_runner = BaselineRunner(self.data_manager)
        universal_baselines = {}
        
        # Ground truth is REQUIRED
        if not self.config.dataset.get('ground_truth_baseline_key'):
            raise ValueError("ground_truth_baseline_key is required in dataset config")
        universal_baselines['ground_truth'] = baseline_runner.load_baseline(
            self.config.dataset.ground_truth_baseline_key,
            'ground_truth',
            fold_split
        )

        universal_baselines['control'] = baseline_runner.load_baseline(
            self.config.dataset.control_baseline_key,
            'control',
            fold_split
        )
        
        # Technical duplicate is REQUIRED
        if not self.config.dataset.get('technical_duplicate_baseline_key'):
            raise ValueError("technical_duplicate_baseline_key is required in dataset config")
        universal_baselines['technical_duplicate'] = baseline_runner.load_baseline(
            self.config.dataset.technical_duplicate_baseline_key,
            'technical_duplicate',
            fold_split
        )
        
        if self.config.dataset.get('interpolated_duplicate_baseline_key'):
            universal_baselines['interpolated_duplicate'] = baseline_runner.load_baseline(
                self.config.dataset.interpolated_duplicate_baseline_key,
                'interpolated_duplicate',
                fold_split
            )
        
        # Additive baseline is optional
        if self.config.dataset.get('additive_baseline_key'):
            universal_baselines['additive'] = baseline_runner.load_baseline(
                self.config.dataset.additive_baseline_key,
                'additive',
                fold_split
            )

        # Random perturbation baseline (uses random training conditions)
        universal_baselines['random_perturbation'] = baseline_runner.generate_random_perturbation_baseline(
            fold_split
        )
            
        fold_predictions = {}
        
        # Run model predictions for this fold
        # Check multi-model case FIRST (to avoid accessing deleted cfg.model)
        if hasattr(self.config, 'models'):
            # Multi-model case - NEW feature
            from omegaconf import open_dict
            from ..utils.hash_utils import get_model_path_for_config
            
            for model_config in self.config.models:
                if model_config.type == 'baselines_only':
                    continue
                
                # For multi-model: Calculate path directly without modifying self.config (thread-safe)
                from omegaconf import open_dict
                from ..utils.hash_utils import get_model_path_for_config
                
                # Clean the model config: remove model_path if CLI added it (it shouldn't be in hash)
                clean_model_config = OmegaConf.to_object(model_config)
                if 'model_path' in clean_model_config:
                    del clean_model_config['model_path']
                clean_model_config = OmegaConf.create(clean_model_config)
                
                # Create fold-specific dataset config
                fold_dataset_config = OmegaConf.create(OmegaConf.to_object(self.config.dataset))
                with open_dict(fold_dataset_config):
                    fold_dataset_config.split = fold_split
                
                # Create individual training config for this model
                individual_training_config = OmegaConf.create({
                    'output_dir': f'models/{model_config.name}_{self.config.dataset.name}/',
                    'save_intermediate': True
                })
                
                # Calculate model path (same logic as training)
                model_path = get_model_path_for_config(
                    fold_dataset_config,
                    clean_model_config,
                    individual_training_config
                )
                log.info(f"  Calculated model path for fold {fold_split}: {model_path}")
                
                log.info(f"  Calculated model path for fold {fold_split}: {model_path}")
                
                # Create model config dict with path
                model_config_dict = OmegaConf.to_object(clean_model_config)
                model_config_dict['model_path'] = str(model_path)
                
                # Run predictions
                predictions_path = self.model_runner.run_model(
                    model_config_dict,
                    self.data_manager,
                    fold_split,
                    fold_output_dir,
                    gpu_id=gpu_id
                )
                
                # Load predictions
                model_predictions = sc.read_h5ad(predictions_path)
                
                # Add covariate column if missing (model predictions may not have it)
                if 'covariate' not in model_predictions.obs.columns:
                    log.warning("Covariate column not found in predictions, using ground truth covariate!!")
                    model_predictions.obs['covariate'] = universal_baselines['ground_truth'].obs['covariate'][0]
                
                # Fix "none" covariate values with real covariate from ground truth
                # TODO: Eventually we should fix this hack -- it's only for scgpt
                if 'covariate' in model_predictions.obs.columns:
                    none_mask = model_predictions.obs['covariate'] == 'none'
                    if none_mask.any():
                        log.warning(f"Found {none_mask.sum()} 'none' covariate values, replacing with ground truth covariate")
                        real_covariate = universal_baselines['ground_truth'].obs['covariate'].iloc[0]
                        
                        # Handle categorical covariate column
                        if hasattr(model_predictions.obs['covariate'], 'cat'):
                            # Add the new category if it doesn't exist
                            if real_covariate not in model_predictions.obs['covariate'].cat.categories:
                                model_predictions.obs['covariate'] = model_predictions.obs['covariate'].cat.add_categories([real_covariate])
                        
                        model_predictions.obs.loc[none_mask, 'covariate'] = real_covariate
                        
                        # Also fix pair_key if it exists and contains "none_"
                        if 'pair_key' in model_predictions.obs.columns:
                            none_pair_mask = model_predictions.obs['pair_key'].str.startswith('none_')
                            if none_pair_mask.any():
                                # Handle categorical pair_key column
                                new_pair_keys = (
                                    model_predictions.obs.loc[none_pair_mask, 'pair_key']
                                    .str.replace('none_', f'{real_covariate}_', regex=False)
                                )
                                
                                if hasattr(model_predictions.obs['pair_key'], 'cat'):
                                    # Add new categories if they don't exist
                                    new_categories = set(new_pair_keys) - set(model_predictions.obs['pair_key'].cat.categories)
                                    if new_categories:
                                        model_predictions.obs['pair_key'] = model_predictions.obs['pair_key'].cat.add_categories(list(new_categories))
                                
                                model_predictions.obs.loc[none_pair_mask, 'pair_key'] = new_pair_keys
                
                # Add delta calculations using fold-specific baselines
                model_predictions = self._add_delta_calculations(
                    model_predictions, fold_baselines, universal_baselines
                )
      
                model_display_name = self.get_model_display_name(
                    OmegaConf.to_object(model_config)
                )
                fold_predictions[model_display_name] = model_predictions
        elif hasattr(self.config, 'model') and self.config.model is not None and self.config.model.type != 'baselines_only':
            # Single model case - UNCHANGED from original code
            fold_model_config = self._create_fold_model_config(fold_split)
            
            # Run predictions
            predictions_path = self.model_runner.run_model(
                OmegaConf.to_object(fold_model_config.model),
                self.data_manager,
                fold_split,
                fold_output_dir,
                gpu_id=gpu_id
            )
            
            # Load predictions
            model_predictions = sc.read_h5ad(predictions_path)
            
            # Add covariate column if missing (model predictions may not have it)
            if 'covariate' not in model_predictions.obs.columns:
                log.warning("Covariate column not found in predictions, using ground truth covariate!!")
                model_predictions.obs['covariate'] = universal_baselines['ground_truth'].obs['covariate'][0]
            
            # Fix "none" covariate values with real covariate from ground truth
            # TODO: Eventually we should fix this hack -- it's only for scgpt
            if 'covariate' in model_predictions.obs.columns:
                none_mask = model_predictions.obs['covariate'] == 'none'
                if none_mask.any():
                    log.warning(f"Found {none_mask.sum()} 'none' covariate values, replacing with ground truth covariate")
                    real_covariate = universal_baselines['ground_truth'].obs['covariate'].iloc[0]
                    
                    # Handle categorical covariate column
                    if hasattr(model_predictions.obs['covariate'], 'cat'):
                        # Add the new category if it doesn't exist
                        if real_covariate not in model_predictions.obs['covariate'].cat.categories:
                            model_predictions.obs['covariate'] = model_predictions.obs['covariate'].cat.add_categories([real_covariate])
                    
                    model_predictions.obs.loc[none_mask, 'covariate'] = real_covariate
                    
                    # Also fix pair_key if it exists and contains "none_"
                    if 'pair_key' in model_predictions.obs.columns:
                        none_pair_mask = model_predictions.obs['pair_key'].str.startswith('none_')
                        if none_pair_mask.any():
                            # Handle categorical pair_key column
                            new_pair_keys = (
                                model_predictions.obs.loc[none_pair_mask, 'pair_key']
                                .str.replace('none_', f'{real_covariate}_', regex=False)
                            )
                            
                            if hasattr(model_predictions.obs['pair_key'], 'cat'):
                                # Add new categories if they don't exist
                                new_categories = set(new_pair_keys) - set(model_predictions.obs['pair_key'].cat.categories)
                                if new_categories:
                                    model_predictions.obs['pair_key'] = model_predictions.obs['pair_key'].cat.add_categories(list(new_categories))
                            
                            model_predictions.obs.loc[none_pair_mask, 'pair_key'] = new_pair_keys
            
            # Add delta calculations using fold-specific baselines
            model_predictions = self._add_delta_calculations(
                model_predictions, fold_baselines, universal_baselines
            )
  
            model_display_name = self.get_model_display_name(
                OmegaConf.to_object(self.config.model)
            )
            fold_predictions[model_display_name] = model_predictions
        
        # Determine what predictions to use for baseline filtering
        if fold_predictions:
            # Use first model's predictions for filtering baselines
            model_predictions = list(fold_predictions.values())[0]
        else:
            # No model predictions - need to use ground truth for filtering
            model_predictions = universal_baselines['ground_truth']
        
        # Process universal baselines - filter to match model's actual predictions
        for baseline_name, baseline_pred in tqdm(universal_baselines.items(), desc="Processing universal baselines"):
            # Filter baseline to match exact covariate-condition pairs that the model predicted
            # (some pairs may be skipped due to missing embeddings)
            model_pairs = model_predictions.obs[['covariate', 'condition']].drop_duplicates()
            
            # Create mask for matching covariate-condition pairs
            baseline_mask = pd.Series(False, index=baseline_pred.obs.index)
            for _, row in model_pairs.iterrows():
                pair_mask = (baseline_pred.obs['covariate'] == row['covariate']) & \
                           (baseline_pred.obs['condition'] == row['condition'])
                baseline_mask |= pair_mask
            
            filtered_baseline = baseline_pred[baseline_mask].copy()
            
            # Apply fold-specific delta calculations
            baseline_with_deltas = self._add_delta_calculations(
                filtered_baseline, fold_baselines, universal_baselines
            )
            
            fold_predictions[baseline_name] = baseline_with_deltas
        
        # Process fold-specific baselines - filter to match model predictions
        for baseline_name, baseline_pred in tqdm(fold_baselines.items(), desc="Processing fold-specific baselines"):
            # Filter baseline to match exact covariate-condition pairs that the model predicted
            model_pairs = model_predictions.obs[['covariate', 'condition']].drop_duplicates()
            
            # Create mask for matching covariate-condition pairs
            baseline_mask = pd.Series(False, index=baseline_pred.obs.index)
            for _, row in model_pairs.iterrows():
                pair_mask = (baseline_pred.obs['covariate'] == row['covariate']) & \
                           (baseline_pred.obs['condition'] == row['condition'])
                baseline_mask |= pair_mask
            
            filtered_baseline = baseline_pred[baseline_mask].copy()
            
            # Apply fold-specific delta calculations
            baseline_with_deltas = self._add_delta_calculations(
                filtered_baseline, fold_baselines, universal_baselines
            )
            fold_predictions[baseline_name] = baseline_with_deltas

        enabled_baselines = self._get_effective_enabled_baselines()

        if enabled_baselines is not None:
            metrics_subset = None
            if hasattr(self.config, 'metrics') and self.config.metrics is not None:
                metrics_subset = self.config.metrics.get('enabled')

            required_baselines = self._get_required_baselines(metrics_subset)

            baseline_names = set(universal_baselines.keys()) | set(fold_baselines.keys())
            unknown_baselines = set(enabled_baselines) - baseline_names
            if unknown_baselines:
                raise ValueError(
                    f"Unknown baselines requested: {sorted(unknown_baselines)}. "
                    f"Available baselines: {sorted(baseline_names)}"
                )
            enabled_set = set(enabled_baselines)
            missing_required = required_baselines - enabled_set
            if missing_required:
                raise ValueError(
                    "baselines.enabled must include required baselines for selected metrics. "
                    f"Missing: {sorted(missing_required)}"
                )
            enabled_set.update(required_baselines)
            for baseline_name in list(fold_predictions.keys()):
                if baseline_name in baseline_names and baseline_name not in enabled_set:
                    del fold_predictions[baseline_name]
        
        return fold_predictions
    
    def _gather_fold_predictions_sequential(self, fold_indices: List[int]) -> Dict[str, sc.AnnData]:
        """Sequential fold predictions (existing implementation)."""
        all_fold_predictions = {}  # Will store lists of predictions per model/baseline
        
        # Process each fold
        for fold_idx in fold_indices:
            fold_config = self.config.dataset.folds[fold_idx]
            fold_split = fold_config.split
            
            log.info(f"Processing fold {fold_idx} sequentially: {fold_split}")
            
            # Get predictions for this fold
            fold_predictions = self._process_single_fold(fold_idx)
            
            # Accumulate predictions
            for model_name, predictions in fold_predictions.items():
                if model_name not in all_fold_predictions:
                    all_fold_predictions[model_name] = []
                all_fold_predictions[model_name].append(predictions)
        
        # Concatenate predictions if multiple folds
        if len(fold_indices) > 1:
            return self._concatenate_fold_predictions_from_lists(all_fold_predictions)
        else:
            # Single fold - just unwrap the lists
            return {name: preds[0] for name, preds in all_fold_predictions.items()}
    
    def _concatenate_fold_predictions(self, fold_predictions: Dict[int, Dict[str, sc.AnnData]], 
                                     fold_indices: List[int]) -> Dict[str, sc.AnnData]:
        """Concatenate predictions from multiple folds (parallel result format)."""
        # Convert to list format
        all_fold_predictions = {}
        for fold_idx in fold_indices:
            for model_name, predictions in fold_predictions[fold_idx].items():
                if model_name not in all_fold_predictions:
                    all_fold_predictions[model_name] = []
                all_fold_predictions[model_name].append(predictions)
        
        return self._concatenate_fold_predictions_from_lists(all_fold_predictions)
    
    def _concatenate_fold_predictions_from_lists(self, all_fold_predictions: Dict[str, List[sc.AnnData]]) -> Dict[str, sc.AnnData]:
        """Concatenate predictions from list format."""
        log.info("Concatenating predictions from all folds...")
        concatenated_predictions = {}
        for model_name, fold_predictions_list in all_fold_predictions.items():
            # Concatenate along axis 0 (observations), preserving obsm fields
            concatenated = sc.concat(fold_predictions_list, axis=0, merge='same')
            concatenated_predictions[model_name] = concatenated
            log.info(f"  {model_name}: {concatenated.shape[0]} total predictions")
        return concatenated_predictions
    

    def _load_fold_baselines(self, fold_config: Dict) -> Dict[str, sc.AnnData]:
        """Load baselines specific to a fold using the main DataManager."""
        baselines = {}
        baseline_runner = BaselineRunner(self.data_manager)
        
        baselines['dataset_mean'] = baseline_runner.load_baseline(
            fold_config.dataset_mean_baseline_key,
            'dataset_mean',
            fold_config.split
        )
        
        if fold_config.get('linear_baseline_key'):
            baselines['linear'] = baseline_runner.load_baseline(
                fold_config.linear_baseline_key,
                'linear',
                fold_config.split
            )
        
        return baselines
    
    def _add_delta_calculations(self, predictions: sc.AnnData,
                               fold_baselines: Dict[str, sc.AnnData],
                               universal_baselines: Dict[str, sc.AnnData]) -> sc.AnnData:
        """Add delta calculations to predictions using fold-specific baselines."""
        
        # Add delta from control if available
        if 'control' in fold_baselines or 'control' in universal_baselines:
            control_pred = fold_baselines['control'] if 'control' in fold_baselines else universal_baselines['control']
            delta_ctrl = np.zeros_like(predictions.X)
            # Find intersection of control and predictions var_names
            # Use sorted list to ensure deterministic, reproducible ordering
            common_var_names = sorted(set(control_pred.var_names) & set(predictions.var_names))
            if not common_var_names:
                raise ValueError("No common variable names found between control and predictions")
            
            # Filter control and predictions to only include common var_names
            control_pred = control_pred[:, common_var_names]
            predictions = predictions[:, common_var_names]
            
            for i, cov in enumerate(tqdm(predictions.obs['covariate'], desc="Adding delta from control")):
                # Find matching control - MUST exist
                mask = (control_pred.obs['covariate'] == cov)
                if not mask.any():
                    # TODO: Why would this happen? I mean it's fine because all the values are identical, but still...
                    warnings.warn(f"No control baseline found for covariate={cov}")
                delta_ctrl[i] = predictions.X[i] - control_pred[mask].X[0]
            
            predictions.obsm['delta_ctrl'] = delta_ctrl
        else:
            raise ValueError("No control baseline found")
        
        # Add delta from dataset mean if available
        if 'dataset_mean' in fold_baselines:
            dataset_mean_pred = fold_baselines['dataset_mean']
            delta_mean = np.zeros_like(predictions.X)
            # Find intersection of dataset mean and predictions var_names
            # Use sorted list to ensure deterministic, reproducible ordering
            common_var_names = sorted(set(dataset_mean_pred.var_names) & set(predictions.var_names))
            if not common_var_names:
                raise ValueError("No common variable names found between dataset mean and predictions")
            
            # Filter dataset mean and predictions to only include common var_names
            dataset_mean_pred = dataset_mean_pred[:, common_var_names]
            predictions = predictions[:, common_var_names]
            
            for i, cov in enumerate(tqdm(predictions.obs['covariate'], desc="Adding delta from dataset mean")):
                # Find matching dataset mean - MUST exist
                mask = (dataset_mean_pred.obs['covariate'] == cov)
                if not mask.any():
                    warnings.warn(f"No dataset mean baseline found for covariate={cov}")
                delta_mean[i] = predictions.X[i] - dataset_mean_pred[mask].X[0]
            
            predictions.obsm['delta_mean'] = delta_mean
        else:
            raise ValueError("No dataset mean baseline found")
        
        return predictions
    

    def _create_fold_model_config(self, fold_split: str) -> DictConfig:
        """Create model config for specific fold."""
        from omegaconf import open_dict
        from ..utils.hash_utils import get_model_path_for_config
        
        fold_config = OmegaConf.create(OmegaConf.to_object(self.config))
        with open_dict(fold_config):
            # Update the dataset split
            fold_config.dataset.split = fold_split
            
            # Update training output_dir to match the actual dataset name
            fold_config.training.output_dir = f'models/{fold_config.model.name}_{fold_config.dataset.name}/'
            
            # Calculate and add the model path for this fold's trained model
            model_path = get_model_path_for_config(
                fold_config.dataset,
                fold_config.model,
                fold_config.training
            )
            fold_config.model.model_path = str(model_path)
            log.info(f"  Calculated model path for fold {fold_split}: {model_path}")
        
        return fold_config
    
    def _get_generic_baseline_name(self, fold_specific_name: str) -> str:
        """Map fold-specific baseline name to generic name.
        
        Only applies to truly fold-specific baselines (ctrl, split_mean, linear).
        Universal baselines (additive, technical_duplicate) don't get mapped.
        
        E.g., 'split_fold_0_ctrl_baseline' -> 'ctrl_baseline'
             'split_fold_1_split_mean_baseline' -> 'split_mean_baseline'
        """
        # Handle technical duplicate renaming (universal baseline)
        if 'technical_duplicate_second_half' in fold_specific_name:
            return 'technical duplicate'
        
        # Remove split_fold_N_ prefix using regex (for fold-specific baselines)
        generic_name = re.sub(r'^split_fold_\d+_', '', fold_specific_name)
        
        # Also handle older _foldN suffix pattern if it exists
        generic_name = re.sub(r'_fold\d+$', '', generic_name)
        
        return generic_name

    def _get_required_baselines(self, metrics_subset: Optional[Sequence[str]]) -> Set[str]:
        """Determine required baselines based on selected metrics.

        Args:
            metrics_subset: Optional list of metrics to compute.

        Returns:
            Set of baseline names that must be enabled for the selected metrics.
        """
        required_baselines = {'ground_truth'}
        if metrics_subset is None:
            return required_baselines

        calibrated_metrics = {
            'cmse',
            'cwmse',
            'cpearson',
            'cr2',
            'fcmse',
            'fcpearson',
            'fcr2',
            'cpearson_deltactrl',
            'cr2_deltactrl',
            'cweighted_pearson_deltactrl',
            'fcpearson_deltactrl',
            'fcr2_deltactrl',
            'cpearson_deltapert',
            'cr2_deltapert',
            'cweighted_pearson_deltapert',
            'fcpearson_deltapert',
            'fcr2_deltapert',
            'cweighted_r2_deltactrl',
            'cweighted_r2_deltapert',
            'cpds',
            'cpds_wmse',
        }
        deltapert_metrics = {
            'pearson_deltapert',
            'pearson_deltapert_degs',
            'cpearson_deltapert',
            'weighted_pearson_deltapert',
            'cweighted_pearson_deltapert',
            'fpearson_deltapert',
            'fcpearson_deltapert',
            'r2_deltapert',
            'r2_deltapert_degs',
            'cr2_deltapert',
            'fr2_deltapert',
            'fcr2_deltapert',
            'weighted_r2_deltapert',
            'cweighted_r2_deltapert',
        }
        deltactrl_metrics = {
            'pearson_deltactrl',
            'pearson_deltactrl_degs',
            'weighted_pearson_deltactrl',
            'fpearson_deltactrl',
            'fcpearson_deltactrl',
            'r2_deltactrl',
            'r2_deltactrl_degs',
            'fr2_deltactrl',
            'fcr2_deltactrl',
            'weighted_r2_deltactrl',
        }

        if any(metric in metrics_subset for metric in calibrated_metrics):
            required_baselines.update({'control', 'technical_duplicate'})
        if any(metric in metrics_subset for metric in deltactrl_metrics):
            required_baselines.add('control')
        if any(metric in metrics_subset for metric in deltapert_metrics):
            required_baselines.add('dataset_mean')

        return required_baselines
    
    def _get_output_dir(self) -> Path:
        """Get output directory based on experiment name."""
        from datetime import datetime
        
        # Create output directory based on experiment name (set by CLI for multi-model)
        experiment_name = self.config.experiment.name
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_dir = Path(f"outputs/{experiment_name}/{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def _compute_effective_gene_stats(self) -> Dict[str, float]:
        """Compute effective gene number statistics for the current weight source.
        
        Returns:
            Dictionary with mean, median, std, min, max of effective gene numbers
            across all perturbations with valid weights.
        """
        deg_weight_source = getattr(self.data_manager, 'deg_weight_source', 'vsrest')
        
        # Get all perturbations from the data manager
        adata = self.data_manager.adata
        if adata is None:
            return {}
        
        # Get perturbation keys
        covariate_key = self.data_manager.config.get('covariate_key', 'donor_id')
        perturbation_key = 'condition'
        
        if covariate_key not in adata.obs.columns or perturbation_key not in adata.obs.columns:
            return {}
        
        # Get unique covariate-perturbation combinations
        obs_df = adata.obs[[covariate_key, perturbation_key]].drop_duplicates()
        gene_names = list(adata.var_names)
        
        effective_genes = []
        missing_weights = []
        
        for _, row in obs_df.iterrows():
            covariate = row[covariate_key]
            perturbation = row[perturbation_key]
            
            # Skip control conditions
            if 'control' in str(perturbation).lower() or 'ctrl' in str(perturbation).lower():
                continue
            
            try:
                # Get weights using the current weight source
                weights = self.data_manager.get_deg_weights(covariate, perturbation, gene_names)
                weights = np.asarray(weights, dtype=float)
                
                # Check if weights are valid (not all zeros)
                if np.sum(weights) > 1e-12:
                    eff_genes = _effective_genes_single(weights)
                    effective_genes.append(float(eff_genes))
                else:
                    missing_weights.append(f"{covariate}_{perturbation}")
            except Exception as e:
                missing_weights.append(f"{covariate}_{perturbation} ({str(e)[:50]})")
        
        if not effective_genes:
            return {
                'deg_weight_source': deg_weight_source,
                'n_perts_with_weights': 0,
                'n_perts_missing': len(missing_weights),
                'error': 'No valid weights found'
            }
        
        eff_array = np.array(effective_genes)
        
        return {
            'deg_weight_source': deg_weight_source,
            'n_perts_with_weights': len(effective_genes),
            'n_perts_missing': len(missing_weights),
            'mean': float(np.mean(eff_array)),
            'median': float(np.median(eff_array)),
            'std': float(np.std(eff_array)),
            'min': float(np.min(eff_array)),
            'max': float(np.max(eff_array)),
            'missing_perturbations': missing_weights[:10]  # First 10 only to avoid huge output
        }
    
    def _calculate_all_metrics(self, all_predictions: Dict[str, sc.AnnData], split_name: str, output_dir: Path) -> Dict[str, Any]:
        """Calculate metrics for all models."""
        reset_aggregated_calibration_denominator_issue_counts()
        metrics_engine = MetricsEngine(self.data_manager)
        metrics_subset = None
        if hasattr(self.config, 'metrics') and self.config.metrics is not None:
            metrics_subset = self.config.metrics.get('enabled')
        
        # Handle aggregated_folds special case
        if split_name == 'aggregated_folds':
            # For aggregated folds, count unique covariate-condition pairs from predictions
            sample_pred = list(all_predictions.values())[0]
            unique_pairs = sample_pred.obs[['covariate', 'condition']].drop_duplicates()
            n_test_pairs = len(unique_pairs)
        else:
            n_test_pairs = len(self.data_manager.get_covariate_condition_pairs(split_name, 'test'))
        
        results = {
            'config': OmegaConf.to_object(self.config),
            'split_used': split_name,
            'models': {},
            'metadata': {
                'n_models_run': len(all_predictions),
                'n_genes': self.data_manager.adata.n_vars,
                'n_test_cov_pert_pairs': n_test_pairs
            }
        }
        
        # Extract ground truth from predictions
        if 'ground_truth' not in all_predictions:
            raise ValueError("ground_truth baseline is required but not found in predictions")
        
        ground_truth_adata = all_predictions['ground_truth']
        # Convert AnnData predictions to DataFrame format for metrics engine
        ground_truth_df, ground_truth_deltas = self._extract_dataframes_and_deltas(ground_truth_adata)
        
        # Create PDS cache directory
        # Use a persistent location (not timestamped) so cache is reused across runs
        dataset_name = self.config.dataset.name
        pds_cache_dir = Path(f"outputs/.pds_cache/{dataset_name}")
        pds_cache_dir.mkdir(exist_ok=True, parents=True)
        
        gene_mismatch_reports: List[str] = []

        pds_metrics_all = {
            'pds', 'pds_wmse', 'cpds', 'cpds_wmse',
            'pds_pearson_deltapert', 'cpds_pearson_deltapert',
            'pds_weighted_pearson_deltapert', 'cpds_weighted_pearson_deltapert',
            'pds_r2_deltapert', 'cpds_r2_deltapert',
            'pds_weighted_r2_deltapert', 'cpds_weighted_r2_deltapert',
        }
        need_pds = metrics_subset is None or any(m in metrics_subset for m in pds_metrics_all)

        for model_name, predictions_adata in all_predictions.items():
            if model_name == 'ground_truth':
                continue  # Skip ground truth in model processing

            log.info(f"Calculating metrics for: {model_name}")

            # Check for gene-set mismatches and impute missing genes with dataset mean
            gt_genes = set(ground_truth_df.columns)
            pred_genes = set(predictions_adata.var_names)
            missing_genes = gt_genes - pred_genes

            if missing_genes:
                log.warning(f"[{model_name}] Missing {len(missing_genes)} genes. Imputing with dataset mean.")
                predictions_adata = self._impute_missing_genes(
                    predictions_adata, ground_truth_adata, list(missing_genes)
                )
                gene_mismatch_reports.append(
                    f"[{model_name}] missing {len(missing_genes)}/{len(gt_genes)} "
                    f"ground-truth genes ({100.0 * len(missing_genes) / len(gt_genes):.1f}%). "
                    f"Missing genes were IMPUTED with dataset mean expression."
                )

            # Convert AnnData to DataFrame format
            predictions_df, predictions_deltas = self._extract_dataframes_and_deltas(predictions_adata)
            
            # Check for cached PDS results if any PDS metrics are enabled
            cached_pds_all = None
            if need_pds:
                cached_pds_data = self._load_cached_pds_scores(
                    model_name, predictions_df, ground_truth_df, pds_cache_dir
                )
                if cached_pds_data is not None:
                    cached_pds_all = cached_pds_data

            # Calculate metrics
            model_metrics, pds_all_scores = (
                metrics_engine.calculate_all_metrics(
                    predictions_df,
                    predictions_deltas,
                    ground_truth_df,
                    ground_truth_deltas,
                    cached_pds_all=cached_pds_all,
                    metrics_subset=metrics_subset,
                )
            )

            # Cache PDS scores for future runs if we calculated them
            if need_pds and cached_pds_all is None:
                self._save_pds_scores_to_cache(
                    model_name, predictions_df, ground_truth_df,
                    pds_all_scores, pds_cache_dir
                )
            
            model_summary = self._calculate_summary_stats(model_metrics)
            results['models'][model_name] = {
                'metrics': model_metrics,
                'summary_stats': model_summary
            }

        # Write gene-mismatch warnings next to the results
        if gene_mismatch_reports:
            report_path = output_dir / "gene_mismatch_warnings.txt"
            report_path.write_text(
                "GENE-SET MISMATCH WARNINGS\n"
                "==========================\n"
                "The following models predicted fewer genes than the ground truth.\n"
                "Missing genes were IMPUTED with dataset mean expression before\n"
                "computing metrics. This ensures all models are evaluated on the\n"
                "same complete gene set.\n\n"
                + "\n\n".join(gene_mismatch_reports)
                + "\n"
            )
            log.warning("Gene-set mismatch report written to %s", report_path)
        
        # Save results
        self._save_results(results, output_dir)
        
        # Print multi-model summary
        self._print_multi_model_summary(results)
        
        return results

    def _impute_missing_genes(
        self,
        predictions_adata: sc.AnnData,
        ground_truth_adata: sc.AnnData,
        missing_genes: List[str]
    ) -> sc.AnnData:
        """Impute missing genes in predictions with dataset mean from ground truth.

        Args:
            predictions_adata: AnnData with model predictions (may have missing genes).
            ground_truth_adata: AnnData with ground truth (has all genes).
            missing_genes: List of gene names to impute.

        Returns:
            AnnData with missing genes imputed with covariate-specific dataset mean.
        """
        if not missing_genes:
            return predictions_adata

        # Get gene lists
        existing_genes = list(predictions_adata.var_names)
        all_genes = existing_genes + list(missing_genes)

        # Build gene to index mappings
        gene_to_idx_new = {gene: i for i, gene in enumerate(all_genes)}
        gt_var_list = list(ground_truth_adata.var_names)
        gt_gene_to_idx = {gene: i for i, gene in enumerate(gt_var_list)}

        # Create new expression matrix with all genes
        n_obs = predictions_adata.n_obs
        n_genes = len(all_genes)
        new_X = np.zeros((n_obs, n_genes))

        # Fill in existing predictions
        for i, gene in enumerate(existing_genes):
            new_X[:, gene_to_idx_new[gene]] = predictions_adata.X[:, i]

        # Compute covariate-specific means from ground truth and impute missing genes
        for cov in ground_truth_adata.obs['covariate'].unique():
            # Get ground truth rows for this covariate
            gt_mask = ground_truth_adata.obs['covariate'] == cov
            gt_cov_data = ground_truth_adata[gt_mask]

            # Compute dataset mean for this covariate: mean(X) - mean(delta_mean) = dataset_mean
            if 'delta_mean' in gt_cov_data.obsm:
                mean_X = gt_cov_data.X.mean(axis=0)
                mean_delta = gt_cov_data.obsm['delta_mean'].mean(axis=0)
                if hasattr(mean_X, 'A1'):
                    mean_X = mean_X.A1
                if hasattr(mean_delta, 'A1'):
                    mean_delta = mean_delta.A1
                dataset_mean = np.asarray(mean_X) - np.asarray(mean_delta)
            else:
                # Fallback: just use mean of X
                dataset_mean = gt_cov_data.X.mean(axis=0)
                if hasattr(dataset_mean, 'A1'):
                    dataset_mean = dataset_mean.A1
                dataset_mean = np.asarray(dataset_mean)

            # Impute missing genes for rows with this covariate
            pred_mask = predictions_adata.obs['covariate'] == cov
            pred_indices = np.where(pred_mask)[0]

            for gene in missing_genes:
                if gene in gt_gene_to_idx:
                    gt_idx = gt_gene_to_idx[gene]
                    new_idx = gene_to_idx_new[gene]
                    new_X[pred_indices, new_idx] = dataset_mean[gt_idx]

        # Create new AnnData with imputed values
        new_var = pd.DataFrame(index=all_genes)
        imputed_adata = sc.AnnData(
            X=new_X,
            obs=predictions_adata.obs.copy(),
            var=new_var
        )

        # Copy and expand obsm data
        for key in ['delta_ctrl', 'delta_mean']:
            if key in predictions_adata.obsm:
                new_obsm = np.zeros((n_obs, n_genes))
                for i, gene in enumerate(existing_genes):
                    new_obsm[:, gene_to_idx_new[gene]] = predictions_adata.obsm[key][:, i]
                imputed_adata.obsm[key] = new_obsm

        return imputed_adata

    def _extract_dataframes_and_deltas(self, adata: sc.AnnData) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Extract expression DataFrames and delta DataFrames from AnnData.
        
        Args:
            adata: AnnData object with already mean-aggregated expression data and delta calculations in obsm
            
        Returns:
            Tuple of (expressions_df, deltas_dict) where:
            - expressions_df: DataFrame with cov_pert_key as index
            - deltas_dict: {'deltactrl': DataFrame, 'deltamean': DataFrame} with cov_pert_key as index
        """
        # Create covariate-condition key column
        adata.obs['cov_pert_key'] = adata.obs['covariate'].astype(str) + '_' + adata.obs['condition'].astype(str)
        
        # Convert entire AnnData to DataFrame in one go
        expressions_df = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs['cov_pert_key'])
        
        deltas = {}
        
        deltas['deltactrl'] = pd.DataFrame(adata.obsm['delta_ctrl'], columns=adata.var_names, index=adata.obs['cov_pert_key'])        
        deltas['deltamean'] = pd.DataFrame(adata.obsm['delta_mean'], columns=adata.var_names, index=adata.obs['cov_pert_key'])

        # Remove any keys containing "ctrl" or "control" from the dataframes
        expressions_df = expressions_df[~expressions_df.index.str.contains('ctrl')]
        expressions_df = expressions_df[~expressions_df.index.str.contains('control')]
        deltas['deltactrl'] = deltas['deltactrl'][~deltas['deltactrl'].index.str.contains('ctrl')]
        deltas['deltamean'] = deltas['deltamean'][~deltas['deltamean'].index.str.contains('ctrl')]
        deltas['deltactrl'] = deltas['deltactrl'][~deltas['deltactrl'].index.str.contains('control')]
        deltas['deltamean'] = deltas['deltamean'][~deltas['deltamean'].index.str.contains('control')]

        
        return expressions_df, deltas
    
    def _print_multi_model_summary(self, results: Dict[str, Any]):
        """Print a nice summary table for all models."""
        if not results['models']:
            print("No models were run.")
            return
        
        # Filter out ground_truth and rename control to control_mean for display
        display_models = {}
        for model_name, model_data in results['models'].items():
            if model_name == 'ground_truth':
                continue  # Skip ground truth - it's the reference, not a model to compare
            display_name = 'control_mean' if model_name == 'control' else model_name
            display_models[display_name] = model_data
        
        if not display_models:
            print("No models to display (ground_truth is hidden).")
            return
        
        # Calculate dynamic column width for model names
        max_model_name_len = max(len(name) for name in display_models.keys())
        model_col_width = max(max_model_name_len + 2, len("Model") + 2)  # At least as wide as "Model" header
        
        # Calculate total table width for ALL metrics
        col_widths = [model_col_width, 8, 8, 10, 12, 10, 12, 10, 12, 10, 12, 10, 10, 9, 12, 12]  # All column widths
        total_width = sum(col_widths) + len(col_widths) - 1  # +spaces between columns
        
        print("\n" + "="*(total_width))
        print(f"BENCHMARK RESULTS SUMMARY (split: {results['split_used']})")
        print("="*(total_width+2))
        
        # Header with ALL metrics
        print(f"{'Model':<{model_col_width}} {'MSE':<8} {'WMSE':<8} {'rΔ Ctrl':<10} {'rΔ Ctrl DEG':<12} {'rΔ Pert':<10} {'rΔ Pert DEG':<12} {'R²Δ Ctrl':<10} {'R²Δ Ctrl DEG':<12} {'R²Δ Pert':<10} {'R²Δ Pert DEG':<12} {'WR²Δ Ctrl':<10} {'WR²Δ Pert':<10} {'PDS':<9}")
        print("-" * total_width)
        
        # Model rows
        model_scores = {}
        for model_name, model_results in display_models.items():
            stats = model_results['summary_stats']
            
            # Get mean values for display - ALL metrics
            mse = stats.get('mse_mean', float('nan'))
            wmse = stats.get('wmse_mean', float('nan'))
            pearson_deltactrl = stats.get('pearson_deltactrl_mean', float('nan'))
            pearson_deltactrl_degs = stats.get('pearson_deltactrl_degs_mean', float('nan'))
            pearson_deltapert = stats.get('pearson_deltapert_mean', float('nan'))
            pearson_deltapert_degs = stats.get('pearson_deltapert_degs_mean', float('nan'))
            r2_deltactrl = stats.get('r2_deltactrl_mean', float('nan'))
            r2_deltactrl_degs = stats.get('r2_deltactrl_degs_mean', float('nan'))
            r2_deltapert = stats.get('r2_deltapert_mean', float('nan'))
            r2_deltapert_degs = stats.get('r2_deltapert_degs_mean', float('nan'))
            weighted_r2_deltactrl = stats.get('weighted_r2_deltactrl_mean', float('nan'))
            weighted_r2_deltapert = stats.get('weighted_r2_deltapert_mean', float('nan'))
            pds = stats.get('pds_mean', float('nan'))

            # Store for ranking (using control-based DEGs metric)
            model_scores[model_name] = {
                'pearson_deltactrl_degs': pearson_deltactrl_degs,
                'pearson_deltapert_degs': pearson_deltapert_degs,
                'r2_deltactrl_degs': r2_deltactrl_degs,
                'r2_deltapert_degs': r2_deltapert_degs,
                'weighted_r2_deltactrl': weighted_r2_deltactrl,
                'weighted_r2_deltapert': weighted_r2_deltapert,
                'mse': mse,
                'wmse': wmse,
                'pearson_deltactrl': pearson_deltactrl,
                'pearson_deltapert': pearson_deltapert,
                'r2_deltactrl': r2_deltactrl,
                'r2_deltapert': r2_deltapert,
                'pds': pds,
            }

            print(f"{model_name:<{model_col_width}} {mse:<8.4f} {wmse:<8.4f} {pearson_deltactrl:<10.4f} {pearson_deltactrl_degs:<12.4f} {pearson_deltapert:<10.4f} {pearson_deltapert_degs:<12.4f} {r2_deltactrl:<10.4f} {r2_deltactrl_degs:<12.4f} {r2_deltapert:<10.4f} {r2_deltapert_degs:<12.4f} {weighted_r2_deltactrl:<10.4f} {weighted_r2_deltapert:<10.4f} {pds:<9.4f}")
        
        # Find best model
        if model_scores:
            # Get the best model by weighted R² Δ
            best_model_name = None
            best_weighted_r2_deltapert = float('-inf')
            
            for model_name, model_scores in model_scores.items():
                if 'technical duplicate' in model_name:
                    continue
                current_weighted_r2_deltapert = model_scores['weighted_r2_deltapert']
                if current_weighted_r2_deltapert > best_weighted_r2_deltapert:
                    best_weighted_r2_deltapert = current_weighted_r2_deltapert
                    best_model_name = model_name
            
            if best_model_name:
                print(f"\nBest performing model 🎖️: {best_model_name} (Weighted R² Δ Pert mean: {best_weighted_r2_deltapert:.4f})")
            else:
                print("\nNo valid model scores found")
        
        print("="*total_width)
    
    def _calculate_summary_stats(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate summary statistics across covariate-perturbation pairs."""
        summary = {}
        if not metrics:
            return summary
        total_count = len(next(iter(metrics.values())))

        for metric_name, cov_pert_scores in metrics.items():
            if not cov_pert_scores:
                continue

            # Deal with cases where scores are all nan or 0 (0 can happen probably due to tolerance issues for pearson delta DEGs)
            if len(cov_pert_scores) == 0 or np.sum(list(cov_pert_scores.values())) == 0:
                summary[f"{metric_name}_mean"] = np.nan
                summary[f"{metric_name}_median"] = np.nan
                summary[f"{metric_name}_std"] = np.nan
                summary[f"{metric_name}_nan_prop"] = 1
                continue
                
            scores = list(cov_pert_scores.values())
            scores_non_nan = [s for s in scores if not np.isnan(s)]
            
            if scores_non_nan:
                summary[f"{metric_name}_mean"] = np.mean(scores_non_nan)
                summary[f"{metric_name}_median"] = np.median(scores_non_nan) 
                summary[f"{metric_name}_std"] = np.std(scores_non_nan)
                summary[f"{metric_name}_nan_prop"] = 1 - len(scores_non_nan) / total_count
            else:
                # All scores are NaN
                summary[f"{metric_name}_mean"] = np.nan
                summary[f"{metric_name}_median"] = np.nan
                summary[f"{metric_name}_std"] = np.nan
                summary[f"{metric_name}_nan_prop"] = 1.0
        
        return summary
    
    def _create_detailed_metrics_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create a long-format table with perturbation-level metrics for all models.
        
        Args:
            results: Dictionary containing all model results with metrics.
            
        Returns:
            DataFrame with columns: model, perturbation, metric, value
        """
        detailed_rows = []
        
        for model_name, model_data in results.get('models', {}).items():
            # Skip ground_truth in detailed metrics - it's the reference, not a model to compare
            if model_name == 'ground_truth':
                continue
            # Rename control to control_mean for consistency
            display_name = 'control_mean' if model_name == 'control' else model_name
            model_metrics = model_data.get('metrics', {})
            
            for metric_name, perturbation_scores in model_metrics.items():
                for perturbation, value in perturbation_scores.items():
                    detailed_rows.append({
                        'model': display_name,
                        'perturbation': perturbation,
                        'metric': metric_name,
                        'value': value
                    })
        
        return pd.DataFrame(detailed_rows)
    
    def get_model_display_name(self, model_config: Dict) -> str:
        """Get display name for benchmarking outputs."""
        return model_config.get('display_name', model_config.get('name'))
    
    def _load_cached_pds_scores(
        self, model_name: str, predictions_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame, cache_dir: Path
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """Load cached PDS scores if they exist and are valid.

        Returns dict with 'pds' and 'pds_wmse' keys, or None if not cached.
        """
        import hashlib

        pred_keys = sorted(predictions_df.index.tolist())
        gt_keys = sorted(ground_truth_df.index.tolist())

        pred_hash = hashlib.md5(json.dumps(pred_keys, sort_keys=True).encode()).hexdigest()[:12]
        gt_hash = hashlib.md5(json.dumps(gt_keys, sort_keys=True).encode()).hexdigest()[:12]
        weight_source = getattr(self.data_manager, 'deg_weight_source', 'vsrest')
        cache_key = f"{model_name}_{pred_hash}_{gt_hash}_{weight_source}"
        cache_file = cache_dir / f"{cache_key}_pds.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                # Check if it's the new format (dict with 'pds' key) or old format (flat dict)
                if isinstance(cached_data, dict) and 'pds' in cached_data:
                    log.info(f"  ✓ Loaded cached PDS scores for {model_name} from {cache_file.name}")
                    return cached_data
                else:
                    # Old format - convert to new format
                    log.info(f"  ✓ Loaded cached PDS scores (legacy format) for {model_name}")
                    return {'pds': cached_data, 'pds_wmse': {}}
            except Exception as e:
                log.warning(f"  Failed to load cached PDS scores: {e}")
                return None

        log.info(f"  No cached PDS scores found for {model_name}, will calculate")
        return None

    def _save_pds_scores_to_cache(
        self, model_name: str, predictions_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame, pds_data: Dict[str, Dict[str, float]], cache_dir: Path
    ) -> None:
        """Save PDS scores to cache for future runs.

        pds_data should be dict with 'pds' and 'pds_wmse' keys.
        """
        import hashlib

        pred_keys = sorted(predictions_df.index.tolist())
        gt_keys = sorted(ground_truth_df.index.tolist())

        pred_hash = hashlib.md5(json.dumps(pred_keys, sort_keys=True).encode()).hexdigest()[:12]
        gt_hash = hashlib.md5(json.dumps(gt_keys, sort_keys=True).encode()).hexdigest()[:12]
        weight_source = getattr(self.data_manager, 'deg_weight_source', 'vsrest')
        cache_key = f"{model_name}_{pred_hash}_{gt_hash}_{weight_source}"
        cache_file = cache_dir / f"{cache_key}_pds.json"

        try:
            with open(cache_file, 'w') as f:
                json.dump(pds_data, f, indent=2)
            log.info(f"  ✓ Saved PDS scores to cache: {cache_file.name}")
        except Exception as e:
            log.warning(f"  Failed to save PDS scores to cache: {e}")

            
    
    def _save_results(self, results: Dict[str, Any], output_dir: Path):
        """Save benchmark results to files."""
        
        # Compute effective gene stats for the current weight source
        effective_gene_stats = self._compute_effective_gene_stats()
        
        # Save JSON results (for easy inspection)
        json_path = output_dir / 'results.json'
        json_results = results.copy()
        
        # Add effective gene stats to the results
        json_results['effective_gene_stats'] = effective_gene_stats

        calib_report = get_aggregated_calibration_denominator_report()
        calib_report['n_test_cov_pert_pairs'] = json_results.get('metadata', {}).get(
            'n_test_cov_pert_pairs'
        )
        json_results.setdefault('metadata', {})['aggregated_calibration_denominators'] = calib_report
        results.setdefault('metadata', {})['aggregated_calibration_denominators'] = calib_report
        
        # Convert numpy types to native Python types for JSON serialization
        for model_name, model_data in json_results.get('models', {}).items():
            if 'summary_stats' in model_data:
                model_data['summary_stats'] = {k: float(v) if isinstance(v, np.number) else v 
                                             for k, v in model_data['summary_stats'].items()}
            if 'metrics' in model_data:
                model_data['metrics'] = {
                    metric_name: {k: float(v) if isinstance(v, np.number) else v 
                                 for k, v in metric_dict.items()}
                    for metric_name, metric_dict in model_data['metrics'].items()
                }
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        log.info(f"Results saved to {json_path}")
        
        # Also save effective gene stats to a separate file for easy access
        eg_path = output_dir / 'effective_gene_stats.json'
        with open(eg_path, 'w') as f:
            json.dump(effective_gene_stats, f, indent=2)
        log.info(f"Effective gene stats saved to {eg_path}")
        
        # Save pickle results (preserves all data types)
        pickle_path = output_dir / 'results.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        log.info(f"Results saved to {pickle_path}")
        
        # Save summary stats as CSV for easy analysis  
        if results.get('models'):
            summary_rows = []
            for model_name, model_data in results['models'].items():
                # Skip ground_truth in CSV output - it's the reference, not a model to compare
                if model_name == 'ground_truth':
                    continue
                # Rename control to control_mean for consistency
                display_name = 'control_mean' if model_name == 'control' else model_name
                row = {'model': display_name}
                row.update(model_data.get('summary_stats', {}))
                
                # Add effective gene stats to each row
                if effective_gene_stats:
                    row['deg_weight_source'] = effective_gene_stats.get('deg_weight_source', 'unknown')
                    row['effective_genes_mean'] = effective_gene_stats.get('mean', float('nan'))
                    row['effective_genes_median'] = effective_gene_stats.get('median', float('nan'))
                    row['effective_genes_std'] = effective_gene_stats.get('std', float('nan'))
                    row['effective_genes_min'] = effective_gene_stats.get('min', float('nan'))
                    row['effective_genes_max'] = effective_gene_stats.get('max', float('nan'))
                    row['n_perts_with_weights'] = effective_gene_stats.get('n_perts_with_weights', 0)
                
                summary_rows.append(row)
            
            summary_df = pd.DataFrame(summary_rows)
            csv_path = output_dir / 'summary_stats.csv'
            summary_df.to_csv(csv_path, index=False)
            log.info(f"Summary stats saved to {csv_path}")
            
            # Save detailed perturbation-level metrics
            detailed_df = self._create_detailed_metrics_table(results)
            detailed_csv_path = output_dir / 'detailed_metrics.csv'
            detailed_df.to_csv(detailed_csv_path, index=False)
            log.info(f"Detailed perturbation-level metrics saved to {detailed_csv_path}")

            calib_json_path = output_dir / 'calibration_aggregated_denominators.json'
            with open(calib_json_path, 'w') as f:
                json.dump(calib_report, f, indent=2)
            log.info(f"Aggregated calibration denominator stats saved to {calib_json_path}")
        
        # Print handled by _print_multi_model_summary 