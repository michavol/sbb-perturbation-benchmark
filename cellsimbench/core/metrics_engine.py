"""
Metrics calculation engine for CellSimBench framework.

Provides comprehensive metrics computation for evaluating perturbation
response predictions against ground truth.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Sequence
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import scanpy as sc
from tqdm import tqdm

log = logging.getLogger(__name__)

# Import metrics functions from data_manager
from .data_manager import mse, wmse, pearson, r2_score_on_deltas, DataManager


class MetricsEngine:

    def __init__(self, data_manager: DataManager) -> None:
        """Initialize MetricsEngine with a DataManager instance.

        Args:
            data_manager: DataManager for accessing ground truth data and DEG weights.
        """
        self.data_manager = data_manager
        self._pos_baseline_cache: Dict[Tuple[str, str, str, str, Tuple[str, ...]], np.ndarray] = {}
        self._neg_baseline_cache: Dict[Tuple[str, Tuple[str, ...]], np.ndarray] = {}
        
    def calculate_all_metrics(
        self,
        predictions: pd.DataFrame,
        predictions_deltas: Dict[str, pd.DataFrame],
        ground_truth: pd.DataFrame,
        ground_truth_deltas: Dict[str, pd.DataFrame],
        cached_pds_all: Optional[Dict[str, Dict[str, float]]] = None,
        metrics_subset: Optional[Sequence[str]] = None,
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:

        # Ensure predictions and ground truth have the same var_names
        # Use sorted list to ensure deterministic, reproducible ordering
        pred_genes = set(predictions.columns)
        gt_genes = set(ground_truth.columns)
        common_var_names = sorted(pred_genes & gt_genes)

        if not common_var_names:
            raise ValueError("Predictions and ground truth have different var_names")

        missing_from_pred = gt_genes - pred_genes
        if missing_from_pred:
            log.warning(
                "Predictions are missing %d / %d ground-truth genes (%.1f%%). "
                "Metrics are computed on a reduced gene set. "
                "wMSE may appear artificially better when missing genes carry "
                "large DEG weights. Calibrated metrics (cwMSE, cpearson, cr2, "
                "cweighted_*) can also be distorted because baseline scores "
                "change on the restricted gene set.",
                len(missing_from_pred),
                len(gt_genes),
                100.0 * len(missing_from_pred) / len(gt_genes),
            )

        predictions = predictions[common_var_names]
        ground_truth = ground_truth[common_var_names]
        predictions_deltas = {key: df[common_var_names] for key, df in predictions_deltas.items()}
        ground_truth_deltas = {key: df[common_var_names] for key, df in ground_truth_deltas.items()}

        # Get all covariate-condition pairs from DataFrame index
        cov_condition_pairs = [(key.split('_')[0], '_'.join(key.split('_')[1:]))
                              for key in predictions.index]

        metrics_requested = set(metrics_subset) if metrics_subset is not None else None

        # Determine if PDS metrics are requested
        pds_metrics = {
            'pds', 'pds_wmse', 'cpds', 'cpds_wmse',
            'pds_pearson_deltapert', 'cpds_pearson_deltapert',
            'pds_weighted_pearson_deltapert', 'cpds_weighted_pearson_deltapert',
            'pds_r2_deltapert', 'cpds_r2_deltapert',
            'pds_weighted_r2_deltapert', 'cpds_weighted_r2_deltapert',
        }
        need_pds = metrics_requested is None or any(m in metrics_requested for m in pds_metrics)
        need_pds_wmse = metrics_requested is None or any(m in metrics_requested for m in {'pds_wmse', 'cpds_wmse'})
        need_pds_pearson = metrics_requested is None or any(
            m in metrics_requested for m in {'pds_pearson_deltapert', 'cpds_pearson_deltapert'}
        )
        need_pds_weighted_pearson = metrics_requested is None or any(
            m in metrics_requested for m in {'pds_weighted_pearson_deltapert', 'cpds_weighted_pearson_deltapert'}
        )
        need_pds_r2 = metrics_requested is None or any(
            m in metrics_requested for m in {'pds_r2_deltapert', 'cpds_r2_deltapert'}
        )
        need_pds_weighted_r2 = metrics_requested is None or any(
            m in metrics_requested for m in {'pds_weighted_r2_deltapert', 'cpds_weighted_r2_deltapert'}
        )

        _cached = cached_pds_all or {}

        # Calculate PDS scores (needs full dataset) - only if enabled
        if 'pds' in _cached:
            pds_scores = _cached['pds']
        elif need_pds:
            pds_scores = self._calculate_pds_scores(
                predictions, ground_truth
            )
        else:
            pds_scores = {key: 0.0 for key in predictions.index}

        # Calculate PDS_wMSE scores using weighted MSE distance
        if 'pds_wmse' in _cached:
            pds_wmse_scores = _cached['pds_wmse']
        elif need_pds_wmse:
            pds_wmse_scores = self._calculate_pds_scores_weighted(
                predictions, ground_truth
            )
        else:
            pds_wmse_scores = {key: 0.0 for key in predictions.index}

        # Calculate PDS_pearson_deltapert using Pearson correlation of deltas
        if 'pds_pearson_deltapert' in _cached:
            pds_pearson_deltapert_scores = _cached['pds_pearson_deltapert']
        elif need_pds_pearson:
            pds_pearson_deltapert_scores = self._calculate_pds_scores_pearson_deltapert(
                predictions_deltas['deltamean'],
                ground_truth_deltas['deltamean'],
            )
        else:
            pds_pearson_deltapert_scores = {key: 0.0 for key in predictions.index}

        # Calculate PDS_weighted_pearson_deltapert using DEG-weighted Pearson on deltas
        if 'pds_weighted_pearson_deltapert' in _cached:
            pds_weighted_pearson_deltapert_scores = _cached['pds_weighted_pearson_deltapert']
        elif need_pds_weighted_pearson:
            pds_weighted_pearson_deltapert_scores = self._calculate_pds_scores_weighted_pearson_deltapert(
                predictions_deltas['deltamean'],
                ground_truth_deltas['deltamean'],
            )
        else:
            pds_weighted_pearson_deltapert_scores = {key: 0.0 for key in predictions.index}

        # Calculate PDS_r2_deltapert using R² on deltas
        if 'pds_r2_deltapert' in _cached:
            pds_r2_deltapert_scores = _cached['pds_r2_deltapert']
        elif need_pds_r2:
            pds_r2_deltapert_scores = self._calculate_pds_scores_r2_deltapert(
                predictions_deltas['deltamean'],
                ground_truth_deltas['deltamean'],
            )
        else:
            pds_r2_deltapert_scores = {key: 0.0 for key in predictions.index}

        # Calculate PDS_weighted_r2_deltapert using DEG-weighted R² on deltas
        if 'pds_weighted_r2_deltapert' in _cached:
            pds_weighted_r2_deltapert_scores = _cached['pds_weighted_r2_deltapert']
        elif need_pds_weighted_r2:
            pds_weighted_r2_deltapert_scores = self._calculate_pds_scores_weighted_r2_deltapert(
                predictions_deltas['deltamean'],
                ground_truth_deltas['deltamean'],
            )
        else:
            pds_weighted_r2_deltapert_scores = {key: 0.0 for key in predictions.index}

        filtered_metric_names = {
            'fmse',
            'fcmse',
            'fpearson',
            'fcpearson',
            'fr2',
            'fcr2',
            'fpearson_deltactrl',
            'fcpearson_deltactrl',
            'fr2_deltactrl',
            'fcr2_deltactrl',
            'fpearson_deltapert',
            'fcpearson_deltapert',
            'fr2_deltapert',
            'fcr2_deltapert',
        }
        calibrated_metric_names = {
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
            'cpds_pearson_deltapert',
            'cpds_weighted_pearson_deltapert',
            'cpds_r2_deltapert',
            'cpds_weighted_r2_deltapert',
        }
        deltapert_calibration_names = {
            'cpearson_deltapert',
            'cr2_deltapert',
            'cweighted_pearson_deltapert',
            'fcpearson_deltapert',
            'fcr2_deltapert',
            'cweighted_r2_deltapert',
            'cpds_pearson_deltapert',
            'cpds_weighted_pearson_deltapert',
            'cpds_r2_deltapert',
            'cpds_weighted_r2_deltapert',
        }

        def should_compute(metric_name: str) -> bool:
            return metrics_requested is None or metric_name in metrics_requested

        # Calculate metrics for each covariate-condition pair
        condition_metrics: Dict[str, Dict[str, float]] = {}

        for covariate_value, condition in tqdm(cov_condition_pairs):
            cov_pert_key = f"{covariate_value}_{condition}"

            pred_expression = predictions.loc[cov_pert_key].values

            if cov_pert_key not in ground_truth.index:
                print(f"Covariate-condition pair {cov_pert_key} not found in ground truth")
                continue
            truth_expression = ground_truth.loc[cov_pert_key].values
            
            # Get pre-computed deltas
            pred_deltas_ctrl = predictions_deltas['deltactrl'].loc[cov_pert_key].values
            truth_deltas_ctrl = ground_truth_deltas['deltactrl'].loc[cov_pert_key].values
            pred_deltas_mean = predictions_deltas['deltamean'].loc[cov_pert_key].values
            truth_deltas_mean = ground_truth_deltas['deltamean'].loc[cov_pert_key].values


            # Get DEG weights and mask using covariate and perturbation
            weights = self.data_manager.get_deg_weights(covariate_value, condition, gene_order=common_var_names)
            deg_mask = self.data_manager.get_deg_mask(covariate_value, condition, gene_order=common_var_names)

            dataset_name = self.data_manager.config.get('name')
            need_filtered = metrics_requested is None or any(
                metric in metrics_requested for metric in filtered_metric_names
            )
            need_baselines = metrics_requested is None or any(
                metric in metrics_requested for metric in calibrated_metric_names
            )
            need_deltapert_baseline = metrics_requested is None or any(
                metric in metrics_requested for metric in deltapert_calibration_names
            )

            filtered_genes = None
            if need_filtered:
                filtered_genes = self._get_filtered_genes(
                    dataset_name=dataset_name,
                    perturbation=condition,
                )

            pos_baseline = None
            neg_baseline = None
            if need_baselines:
                pos_baseline = self._get_positive_baseline(
                    covariate=covariate_value,
                    perturbation=condition,
                    gene_names=common_var_names,
                    split_column="tech_dup_split",
                    current_split="first_half",
                )
                neg_baseline = self._get_negative_baseline(
                    covariate=covariate_value,
                    gene_names=common_var_names,
                )

            dataset_mean_baseline = None
            if need_deltapert_baseline:
                dataset_mean_baseline = truth_expression - truth_deltas_mean

            metric_values: Dict[str, float] = {}

            if should_compute('mse'):
                metric_values['mse'] = self._calculate_mse(pred_expression, truth_expression)
            if should_compute('wmse'):
                metric_values['wmse'] = self._calculate_wmse(pred_expression, truth_expression, weights)

            if should_compute('cmse'):
                metric_values['cmse'] = self._calculate_cmse(
                    pred_expression,
                    truth_expression,
                    covariate=covariate_value,
                    perturbation=condition,
                    gene_names=common_var_names,
                )
            if should_compute('cwmse'):
                metric_values['cwmse'] = self._calculate_cwmse(
                    pred_expression,
                    truth_expression,
                    weights,
                    covariate=covariate_value,
                    perturbation=condition,
                    gene_names=common_var_names,
                )
            if (
                should_compute('cpearson')
                and need_baselines
                and pos_baseline is not None
                and neg_baseline is not None
            ):
                metric_values['cpearson'] = self._calibrate_higher_is_better(
                    pred_score=self._calculate_pearson_direct(pred_expression, truth_expression),
                    pos_score=self._calculate_pearson_direct(pos_baseline, truth_expression),
                    neg_score=self._calculate_pearson_direct(neg_baseline, truth_expression),
                    perturbation=condition,
                    metric_name="cpearson",
                )
            if (
                should_compute('cr2')
                and need_baselines
                and pos_baseline is not None
                and neg_baseline is not None
            ):
                metric_values['cr2'] = self._calibrate_higher_is_better(
                    pred_score=self._calculate_r2_direct(truth_expression, pred_expression),
                    pos_score=self._calculate_r2_direct(truth_expression, pos_baseline),
                    neg_score=self._calculate_r2_direct(truth_expression, neg_baseline),
                    perturbation=condition,
                    metric_name="cr2",
                )
            if should_compute('fmse'):
                metric_values['fmse'] = self._calculate_fmse(
                    pred_expression,
                    truth_expression,
                    covariate=covariate_value,
                    perturbation=condition,
                    gene_names=common_var_names,
                    dataset_name=dataset_name,
                )
            if should_compute('fcmse'):
                metric_values['fcmse'] = self._calculate_fcmse(
                    pred_expression,
                    truth_expression,
                    covariate=covariate_value,
                    perturbation=condition,
                    gene_names=common_var_names,
                    dataset_name=dataset_name,
                )

            if should_compute('pearson_deltactrl'):
                metric_values['pearson_deltactrl'] = self._calculate_pearson_delta_direct(
                    pred_deltas_ctrl, truth_deltas_ctrl
                )
            if should_compute('pearson_deltactrl_degs'):
                metric_values['pearson_deltactrl_degs'] = (
                    self._calculate_pearson_delta_direct(
                        pred_deltas_ctrl[deg_mask], truth_deltas_ctrl[deg_mask]
                    )
                    if deg_mask.sum() > 2
                    else np.nan
                )
            if should_compute('r2_deltactrl'):
                metric_values['r2_deltactrl'] = self._calculate_r2_delta_direct(
                    pred_deltas_ctrl, truth_deltas_ctrl
                )
            if should_compute('r2_deltactrl_degs'):
                metric_values['r2_deltactrl_degs'] = (
                    self._calculate_r2_delta_direct(
                        pred_deltas_ctrl[deg_mask], truth_deltas_ctrl[deg_mask]
                    )
                    if deg_mask.sum() > 2
                    else np.nan
                )
            if should_compute('weighted_r2_deltactrl'):
                metric_values['weighted_r2_deltactrl'] = self._calculate_weighted_r2_delta_direct(
                    pred_deltas_ctrl, truth_deltas_ctrl, weights
                )
            if should_compute('weighted_pearson_deltactrl'):
                metric_values['weighted_pearson_deltactrl'] = self._calculate_weighted_pearson_direct(
                    pred_deltas_ctrl, truth_deltas_ctrl, weights
                )
            if (
                should_compute('cpearson_deltactrl')
                and need_baselines
                and pos_baseline is not None
                and neg_baseline is not None
            ):
                pos_delta_ctrl = pos_baseline - neg_baseline
                neg_delta_ctrl = np.zeros_like(pos_delta_ctrl)
                metric_values['cpearson_deltactrl'] = self._calibrate_higher_is_better(
                    pred_score=self._calculate_pearson_delta_direct(
                        pred_deltas_ctrl, truth_deltas_ctrl
                    ),
                    pos_score=self._calculate_pearson_delta_direct(
                        pos_delta_ctrl, truth_deltas_ctrl
                    ),
                    neg_score=self._calculate_pearson_delta_direct(
                        neg_delta_ctrl, truth_deltas_ctrl
                    ),
                    perturbation=condition,
                    metric_name="cpearson_deltactrl",
                )
            if (
                should_compute('cr2_deltactrl')
                and need_baselines
                and pos_baseline is not None
                and neg_baseline is not None
            ):
                pos_delta_ctrl = pos_baseline - neg_baseline
                neg_delta_ctrl = np.zeros_like(pos_delta_ctrl)
                metric_values['cr2_deltactrl'] = self._calibrate_higher_is_better(
                    pred_score=self._calculate_r2_delta_direct(
                        pred_deltas_ctrl, truth_deltas_ctrl
                    ),
                    pos_score=self._calculate_r2_delta_direct(
                        pos_delta_ctrl, truth_deltas_ctrl
                    ),
                    neg_score=self._calculate_r2_delta_direct(
                        neg_delta_ctrl, truth_deltas_ctrl
                    ),
                    perturbation=condition,
                    metric_name="cr2_deltactrl",
                )
            if (
                should_compute('cweighted_pearson_deltactrl')
                and need_baselines
                and pos_baseline is not None
                and neg_baseline is not None
            ):
                pos_delta_ctrl = pos_baseline - neg_baseline
                neg_delta_ctrl = np.zeros_like(pos_delta_ctrl)
                metric_values['cweighted_pearson_deltactrl'] = self._calibrate_higher_is_better(
                    pred_score=self._calculate_weighted_pearson_direct(
                        pred_deltas_ctrl, truth_deltas_ctrl, weights
                    ),
                    pos_score=self._calculate_weighted_pearson_direct(
                        pos_delta_ctrl, truth_deltas_ctrl, weights
                    ),
                    neg_score=self._calculate_weighted_pearson_direct(
                        neg_delta_ctrl, truth_deltas_ctrl, weights
                    ),
                    perturbation=condition,
                    metric_name="cweighted_pearson_deltactrl",
                )
            if (
                should_compute('cweighted_r2_deltactrl')
                and need_baselines
                and pos_baseline is not None
                and neg_baseline is not None
            ):
                pos_delta_ctrl = pos_baseline - neg_baseline
                neg_delta_ctrl = np.zeros_like(pos_delta_ctrl)
                metric_values['cweighted_r2_deltactrl'] = self._calibrate_higher_is_better(
                    pred_score=self._calculate_weighted_r2_delta_direct(
                        pred_deltas_ctrl, truth_deltas_ctrl, weights
                    ),
                    pos_score=self._calculate_weighted_r2_delta_direct(
                        pos_delta_ctrl, truth_deltas_ctrl, weights
                    ),
                    neg_score=self._calculate_weighted_r2_delta_direct(
                        neg_delta_ctrl, truth_deltas_ctrl, weights
                    ),
                    perturbation=condition,
                    metric_name="cweighted_r2_deltactrl",
                )

            if should_compute('pearson_deltapert'):
                metric_values['pearson_deltapert'] = self._calculate_pearson_delta_direct(
                    pred_deltas_mean, truth_deltas_mean
                )
            if should_compute('pearson_deltapert_degs'):
                metric_values['pearson_deltapert_degs'] = (
                    self._calculate_pearson_delta_direct(
                        pred_deltas_mean[deg_mask], truth_deltas_mean[deg_mask]
                    )
                    if deg_mask.sum() > 2
                    else np.nan
                )
            if should_compute('r2_deltapert'):
                metric_values['r2_deltapert'] = self._calculate_r2_delta_direct(
                    pred_deltas_mean, truth_deltas_mean
                )
            if should_compute('r2_deltapert_degs'):
                metric_values['r2_deltapert_degs'] = (
                    self._calculate_r2_delta_direct(
                        pred_deltas_mean[deg_mask], truth_deltas_mean[deg_mask]
                    )
                    if deg_mask.sum() > 2
                    else np.nan
                )
            if should_compute('weighted_r2_deltapert'):
                metric_values['weighted_r2_deltapert'] = self._calculate_weighted_r2_delta_direct(
                    pred_deltas_mean, truth_deltas_mean, weights
                )
            if should_compute('weighted_pearson_deltapert'):
                metric_values['weighted_pearson_deltapert'] = self._calculate_weighted_pearson_direct(
                    pred_deltas_mean, truth_deltas_mean, weights
                )
            if (
                should_compute('cpearson_deltapert')
                and need_baselines
                and dataset_mean_baseline is not None
                and pos_baseline is not None
                and neg_baseline is not None
            ):
                pos_delta_mean = pos_baseline - dataset_mean_baseline
                neg_delta_mean = neg_baseline - dataset_mean_baseline
                metric_values['cpearson_deltapert'] = self._calibrate_higher_is_better(
                    pred_score=self._calculate_pearson_delta_direct(
                        pred_deltas_mean, truth_deltas_mean
                    ),
                    pos_score=self._calculate_pearson_delta_direct(
                        pos_delta_mean, truth_deltas_mean
                    ),
                    neg_score=self._calculate_pearson_delta_direct(
                        neg_delta_mean, truth_deltas_mean
                    ),
                    perturbation=condition,
                    metric_name="cpearson_deltapert",
                )
            if (
                should_compute('cr2_deltapert')
                and need_baselines
                and dataset_mean_baseline is not None
                and pos_baseline is not None
                and neg_baseline is not None
            ):
                pos_delta_mean = pos_baseline - dataset_mean_baseline
                neg_delta_mean = neg_baseline - dataset_mean_baseline
                metric_values['cr2_deltapert'] = self._calibrate_higher_is_better(
                    pred_score=self._calculate_r2_delta_direct(
                        pred_deltas_mean, truth_deltas_mean
                    ),
                    pos_score=self._calculate_r2_delta_direct(
                        pos_delta_mean, truth_deltas_mean
                    ),
                    neg_score=self._calculate_r2_delta_direct(
                        neg_delta_mean, truth_deltas_mean
                    ),
                    perturbation=condition,
                    metric_name="cr2_deltapert",
                )
            if (
                should_compute('cweighted_pearson_deltapert')
                and need_baselines
                and dataset_mean_baseline is not None
                and pos_baseline is not None
                and neg_baseline is not None
            ):
                pos_delta_mean = pos_baseline - dataset_mean_baseline
                neg_delta_mean = neg_baseline - dataset_mean_baseline
                metric_values['cweighted_pearson_deltapert'] = self._calibrate_higher_is_better(
                    pred_score=self._calculate_weighted_pearson_direct(
                        pred_deltas_mean, truth_deltas_mean, weights
                    ),
                    pos_score=self._calculate_weighted_pearson_direct(
                        pos_delta_mean, truth_deltas_mean, weights
                    ),
                    neg_score=self._calculate_weighted_pearson_direct(
                        neg_delta_mean, truth_deltas_mean, weights
                    ),
                    perturbation=condition,
                    metric_name="cweighted_pearson_deltapert",
                )
            if (
                should_compute('cweighted_r2_deltapert')
                and need_baselines
                and dataset_mean_baseline is not None
                and pos_baseline is not None
                and neg_baseline is not None
            ):
                pos_delta_mean = pos_baseline - dataset_mean_baseline
                neg_delta_mean = neg_baseline - dataset_mean_baseline
                metric_values['cweighted_r2_deltapert'] = self._calibrate_higher_is_better(
                    pred_score=self._calculate_weighted_r2_delta_direct(
                        pred_deltas_mean, truth_deltas_mean, weights
                    ),
                    pos_score=self._calculate_weighted_r2_delta_direct(
                        pos_delta_mean, truth_deltas_mean, weights
                    ),
                    neg_score=self._calculate_weighted_r2_delta_direct(
                        neg_delta_mean, truth_deltas_mean, weights
                    ),
                    perturbation=condition,
                    metric_name="cweighted_r2_deltapert",
                )

            if filtered_genes is not None:
                filtered_pred = self._apply_gene_filter(pred_expression, common_var_names, filtered_genes)
                filtered_truth = self._apply_gene_filter(truth_expression, common_var_names, filtered_genes)
                filtered_pred_deltactrl = self._apply_gene_filter(
                    pred_deltas_ctrl, common_var_names, filtered_genes
                )
                filtered_truth_deltactrl = self._apply_gene_filter(
                    truth_deltas_ctrl, common_var_names, filtered_genes
                )
                filtered_pred_deltapert = self._apply_gene_filter(
                    pred_deltas_mean, common_var_names, filtered_genes
                )
                filtered_truth_deltapert = self._apply_gene_filter(
                    truth_deltas_mean, common_var_names, filtered_genes
                )

                if should_compute('fpearson'):
                    metric_values['fpearson'] = self._calculate_pearson_direct(
                        filtered_pred, filtered_truth
                    )
                if should_compute('fr2'):
                    metric_values['fr2'] = self._calculate_r2_direct(
                        filtered_truth, filtered_pred
                    )

                if should_compute('fpearson_deltactrl'):
                    metric_values['fpearson_deltactrl'] = self._calculate_pearson_direct(
                        filtered_pred_deltactrl, filtered_truth_deltactrl
                    )
                if should_compute('fr2_deltactrl'):
                    metric_values['fr2_deltactrl'] = self._calculate_r2_direct(
                        filtered_truth_deltactrl, filtered_pred_deltactrl
                    )
                if should_compute('fpearson_deltapert'):
                    metric_values['fpearson_deltapert'] = self._calculate_pearson_direct(
                        filtered_pred_deltapert, filtered_truth_deltapert
                    )
                if should_compute('fr2_deltapert'):
                    metric_values['fr2_deltapert'] = self._calculate_r2_direct(
                        filtered_truth_deltapert, filtered_pred_deltapert
                    )

                if need_baselines and pos_baseline is not None and neg_baseline is not None:
                    filtered_pos = self._apply_gene_filter(pos_baseline, common_var_names, filtered_genes)
                    filtered_neg = self._apply_gene_filter(neg_baseline, common_var_names, filtered_genes)

                    if should_compute('fcpearson'):
                        metric_values['fcpearson'] = self._calibrate_higher_is_better(
                            pred_score=self._calculate_pearson_direct(filtered_pred, filtered_truth),
                            pos_score=self._calculate_pearson_direct(filtered_pos, filtered_truth),
                            neg_score=self._calculate_pearson_direct(filtered_neg, filtered_truth),
                            perturbation=condition,
                            metric_name="fcpearson",
                        )
                    if should_compute('fcr2'):
                        metric_values['fcr2'] = self._calibrate_higher_is_better(
                            pred_score=self._calculate_r2_direct(filtered_truth, filtered_pred),
                            pos_score=self._calculate_r2_direct(filtered_truth, filtered_pos),
                            neg_score=self._calculate_r2_direct(filtered_truth, filtered_neg),
                            perturbation=condition,
                            metric_name="fcr2",
                        )

                    if should_compute('fcpearson_deltactrl'):
                        pos_delta_ctrl = filtered_pos - filtered_neg
                        neg_delta_ctrl = np.zeros_like(pos_delta_ctrl)
                        metric_values['fcpearson_deltactrl'] = self._calibrate_higher_is_better(
                            pred_score=self._calculate_pearson_direct(
                                filtered_pred_deltactrl, filtered_truth_deltactrl
                            ),
                            pos_score=self._calculate_pearson_direct(
                                pos_delta_ctrl, filtered_truth_deltactrl
                            ),
                            neg_score=self._calculate_pearson_direct(
                                neg_delta_ctrl, filtered_truth_deltactrl
                            ),
                            perturbation=condition,
                            metric_name="fcpearson_deltactrl",
                        )
                    if should_compute('fcr2_deltactrl'):
                        pos_delta_ctrl = filtered_pos - filtered_neg
                        neg_delta_ctrl = np.zeros_like(pos_delta_ctrl)
                        metric_values['fcr2_deltactrl'] = self._calibrate_higher_is_better(
                            pred_score=self._calculate_r2_direct(
                                filtered_truth_deltactrl, filtered_pred_deltactrl
                            ),
                            pos_score=self._calculate_r2_direct(
                                filtered_truth_deltactrl, pos_delta_ctrl
                            ),
                            neg_score=self._calculate_r2_direct(
                                filtered_truth_deltactrl, neg_delta_ctrl
                            ),
                            perturbation=condition,
                            metric_name="fcr2_deltactrl",
                        )

                    if dataset_mean_baseline is not None:
                        filtered_mean = self._apply_gene_filter(
                            dataset_mean_baseline, common_var_names, filtered_genes
                        )
                        pos_delta_mean = filtered_pos - filtered_mean
                        neg_delta_mean = filtered_neg - filtered_mean
                        if should_compute('fcpearson_deltapert'):
                            metric_values['fcpearson_deltapert'] = self._calibrate_higher_is_better(
                                pred_score=self._calculate_pearson_direct(
                                    filtered_pred_deltapert, filtered_truth_deltapert
                                ),
                                pos_score=self._calculate_pearson_direct(
                                    pos_delta_mean, filtered_truth_deltapert
                                ),
                                neg_score=self._calculate_pearson_direct(
                                    neg_delta_mean, filtered_truth_deltapert
                                ),
                                perturbation=condition,
                                metric_name="fcpearson_deltapert",
                            )
                        if should_compute('fcr2_deltapert'):
                            metric_values['fcr2_deltapert'] = self._calibrate_higher_is_better(
                                pred_score=self._calculate_r2_direct(
                                    filtered_truth_deltapert, filtered_pred_deltapert
                                ),
                                pos_score=self._calculate_r2_direct(
                                    filtered_truth_deltapert, pos_delta_mean
                                ),
                                neg_score=self._calculate_r2_direct(
                                    filtered_truth_deltapert, neg_delta_mean
                                ),
                                perturbation=condition,
                                metric_name="fcr2_deltapert",
                            )

            if should_compute('pds'):
                metric_values['pds'] = pds_scores.get(cov_pert_key, np.nan)

            if should_compute('pds_wmse'):
                metric_values['pds_wmse'] = pds_wmse_scores.get(cov_pert_key, np.nan)

            if should_compute('pds_pearson_deltapert'):
                metric_values['pds_pearson_deltapert'] = pds_pearson_deltapert_scores.get(cov_pert_key, np.nan)

            if should_compute('pds_weighted_pearson_deltapert'):
                metric_values['pds_weighted_pearson_deltapert'] = pds_weighted_pearson_deltapert_scores.get(cov_pert_key, np.nan)

            if should_compute('pds_r2_deltapert'):
                metric_values['pds_r2_deltapert'] = pds_r2_deltapert_scores.get(cov_pert_key, np.nan)

            if should_compute('pds_weighted_r2_deltapert'):
                metric_values['pds_weighted_r2_deltapert'] = pds_weighted_r2_deltapert_scores.get(cov_pert_key, np.nan)

            if should_compute('cpds_pearson_deltapert'):
                if (
                    need_baselines
                    and pos_baseline is not None
                    and neg_baseline is not None
                    and dataset_mean_baseline is not None
                    and cov_pert_key in pds_pearson_deltapert_scores
                ):
                    cov_prefix = covariate_value + '_'
                    cov_mask = ground_truth.index.str.startswith(cov_prefix)
                    ground_truth_cov = ground_truth[cov_mask]

                    if len(ground_truth_cov) < 2:
                        metric_values['cpds_pearson_deltapert'] = np.nan
                    else:
                        gt_values = ground_truth_cov.values  # (n, g)
                        cov_pert_keys_list = list(ground_truth_cov.index)
                        try:
                            idx = cov_pert_keys_list.index(cov_pert_key)
                        except ValueError:
                            idx = 0
                        n = len(cov_pert_keys_list)
                        others = [j for j in range(n) if j != idx]

                        # delta_gt[j] = gt[j] - dataset_mean
                        delta_gt = gt_values - dataset_mean_baseline[np.newaxis, :]

                        def _pearson_sim(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
                            """Pearson of vec vs each row of mat; returns array of length n."""
                            sims = np.zeros(len(mat))
                            std_v = float(np.std(vec))
                            if std_v < 1e-10:
                                return sims
                            for k in range(len(mat)):
                                std_k = float(np.std(mat[k]))
                                if std_k < 1e-10:
                                    sims[k] = 0.0
                                else:
                                    sims[k] = float(np.corrcoef(vec, mat[k])[0, 1])
                            return sims

                        # PDS for positive baseline
                        delta_pos = pos_baseline - dataset_mean_baseline
                        pos_sims = _pearson_sim(delta_pos, delta_gt)
                        pds_pearson_pos = 2.0 * np.mean([1 if pos_sims[idx] > pos_sims[j] else 0 for j in others]) - 1.0

                        # PDS for negative baseline
                        delta_neg = neg_baseline - dataset_mean_baseline
                        neg_sims = _pearson_sim(delta_neg, delta_gt)
                        pds_pearson_neg = 2.0 * np.mean([1 if neg_sims[idx] > neg_sims[j] else 0 for j in others]) - 1.0

                        metric_values['cpds_pearson_deltapert'] = self._calibrate_higher_is_better(
                            pred_score=pds_pearson_deltapert_scores[cov_pert_key],
                            pos_score=pds_pearson_pos,
                            neg_score=pds_pearson_neg,
                            perturbation=condition,
                            metric_name="cpds_pearson_deltapert",
                        )

            if should_compute('cpds') or should_compute('cpds_wmse'):
                if need_baselines and pos_baseline is not None and neg_baseline is not None:
                    cov_prefix = covariate_value + '_'
                    cov_mask = ground_truth.index.str.startswith(cov_prefix)
                    ground_truth_cov = ground_truth[cov_mask]

                    if len(ground_truth_cov) < 2:
                        metric_values['cpds'] = np.nan
                        metric_values['cpds_wmse'] = np.nan
                    else:
                        gt_values = ground_truth_cov.values  # (n_pert, n_genes)
                        cov_pert_keys_list = list(ground_truth_cov.index)
                        try:
                            idx = cov_pert_keys_list.index(cov_pert_key)
                        except ValueError:
                            idx = 0
                        n = len(cov_pert_keys_list)

                        if should_compute('cpds') and cov_pert_key in pds_scores:
                            # Euclidean distances from pos/neg baseline to every GT
                            pos_dists = np.sqrt(np.sum((pos_baseline - gt_values) ** 2, axis=1))
                            neg_dists = np.sqrt(np.sum((neg_baseline - gt_values) ** 2, axis=1))
                            others = [j for j in range(n) if j != idx]
                            pds_pos = 2.0 * np.mean([1 if pos_dists[idx] < pos_dists[j] else 0 for j in others]) - 1.0
                            pds_neg = 2.0 * np.mean([1 if neg_dists[idx] < neg_dists[j] else 0 for j in others]) - 1.0
                            metric_values['cpds'] = self._calibrate_higher_is_better(
                                pred_score=pds_scores[cov_pert_key],
                                pos_score=pds_pos,
                                neg_score=pds_neg,
                                perturbation=condition,
                                metric_name="cpds",
                            )

                        if should_compute('cpds_wmse') and cov_pert_key in pds_wmse_scores:
                            try:
                                # Use the same routing/strictness as all weighted metrics
                                _deg_source = getattr(self.data_manager, 'deg_weight_source', 'vsrest')
                                weights = np.array(self.data_manager.get_deg_weights(
                                    covariate_value, condition, common_var_names
                                ))
                                if np.sum(weights) < 1e-12:
                                    raise ValueError(
                                        f"No DEG weights for '{cov_pert_key}' in cPDS_wMSE calibration "
                                        f"(deg_weight_source='{_deg_source}')."
                                    )
                                weights_sum = np.sum(weights)

                                # Weighted distances from pos/neg baseline to every GT
                                pos_wmse = np.array([
                                    np.sum(weights * (pos_baseline - gt_values[j]) ** 2) / weights_sum
                                    for j in range(n)
                                ])
                                neg_wmse = np.array([
                                    np.sum(weights * (neg_baseline - gt_values[j]) ** 2) / weights_sum
                                    for j in range(n)
                                ])
                                others = [j for j in range(n) if j != idx]
                                pds_wmse_pos = 2.0 * np.mean([1 if pos_wmse[idx] < pos_wmse[j] else 0 for j in others]) - 1.0
                                pds_wmse_neg = 2.0 * np.mean([1 if neg_wmse[idx] < neg_wmse[j] else 0 for j in others]) - 1.0
                                metric_values['cpds_wmse'] = self._calibrate_higher_is_better(
                                    pred_score=pds_wmse_scores[cov_pert_key],
                                    pos_score=pds_wmse_pos,
                                    neg_score=pds_wmse_neg,
                                    perturbation=condition,
                                    metric_name="cpds_wmse",
                                )
                            except ValueError as e:
                                log.warning("cPDS_wMSE skipped for %s: %s", cov_pert_key, e)
                                metric_values['cpds_wmse'] = np.nan

            # --- cpds_weighted_pearson_deltapert ---
            if should_compute('cpds_weighted_pearson_deltapert'):
                if (
                    need_baselines
                    and pos_baseline is not None
                    and neg_baseline is not None
                    and dataset_mean_baseline is not None
                    and cov_pert_key in pds_weighted_pearson_deltapert_scores
                ):
                    cov_prefix = covariate_value + '_'
                    cov_mask = ground_truth.index.str.startswith(cov_prefix)
                    ground_truth_cov = ground_truth[cov_mask]

                    if len(ground_truth_cov) < 2:
                        metric_values['cpds_weighted_pearson_deltapert'] = np.nan
                    else:
                        gt_values = ground_truth_cov.values
                        cov_pert_keys_list = list(ground_truth_cov.index)
                        try:
                            idx = cov_pert_keys_list.index(cov_pert_key)
                        except ValueError:
                            idx = 0
                        n = len(cov_pert_keys_list)
                        others = [j for j in range(n) if j != idx]

                        delta_gt = gt_values - dataset_mean_baseline[np.newaxis, :]

                        try:
                            _deg_source = getattr(self.data_manager, 'deg_weight_source', 'vsrest')
                            cal_weights = np.array(self.data_manager.get_deg_weights(
                                covariate_value, condition, common_var_names
                            ))
                            if np.sum(cal_weights) < 1e-12:
                                raise ValueError(
                                    f"No DEG weights for '{cov_pert_key}' in cPDS_wPearson calibration "
                                    f"(deg_weight_source='{_deg_source}')."
                                )

                            def _wpearson_sim(vec, mat, w):
                                return np.array([
                                    self._calculate_weighted_pearson_direct(vec, mat[k], w)
                                    for k in range(len(mat))
                                ])

                            delta_pos = pos_baseline - dataset_mean_baseline
                            pos_sims = _wpearson_sim(delta_pos, delta_gt, cal_weights)
                            pds_wpearson_pos = 2.0 * np.mean([1 if pos_sims[idx] > pos_sims[j] else 0 for j in others]) - 1.0

                            delta_neg = neg_baseline - dataset_mean_baseline
                            neg_sims = _wpearson_sim(delta_neg, delta_gt, cal_weights)
                            pds_wpearson_neg = 2.0 * np.mean([1 if neg_sims[idx] > neg_sims[j] else 0 for j in others]) - 1.0

                            metric_values['cpds_weighted_pearson_deltapert'] = self._calibrate_higher_is_better(
                                pred_score=pds_weighted_pearson_deltapert_scores[cov_pert_key],
                                pos_score=pds_wpearson_pos,
                                neg_score=pds_wpearson_neg,
                                perturbation=condition,
                                metric_name="cpds_weighted_pearson_deltapert",
                            )
                        except ValueError as e:
                            log.warning("cPDS_wPearson skipped for %s: %s", cov_pert_key, e)
                            metric_values['cpds_weighted_pearson_deltapert'] = np.nan

            # --- cpds_r2_deltapert ---
            if should_compute('cpds_r2_deltapert'):
                if (
                    need_baselines
                    and pos_baseline is not None
                    and neg_baseline is not None
                    and dataset_mean_baseline is not None
                    and cov_pert_key in pds_r2_deltapert_scores
                ):
                    cov_prefix = covariate_value + '_'
                    cov_mask = ground_truth.index.str.startswith(cov_prefix)
                    ground_truth_cov = ground_truth[cov_mask]

                    if len(ground_truth_cov) < 2:
                        metric_values['cpds_r2_deltapert'] = np.nan
                    else:
                        gt_values = ground_truth_cov.values
                        cov_pert_keys_list = list(ground_truth_cov.index)
                        try:
                            idx = cov_pert_keys_list.index(cov_pert_key)
                        except ValueError:
                            idx = 0
                        n = len(cov_pert_keys_list)
                        others = [j for j in range(n) if j != idx]

                        delta_gt = gt_values - dataset_mean_baseline[np.newaxis, :]

                        def _r2_sim(vec, mat):
                            return np.array([
                                r2_score_on_deltas(mat[k], vec)
                                for k in range(len(mat))
                            ])

                        delta_pos = pos_baseline - dataset_mean_baseline
                        pos_sims = _r2_sim(delta_pos, delta_gt)
                        pds_r2_pos = 2.0 * np.mean([1 if pos_sims[idx] > pos_sims[j] else 0 for j in others]) - 1.0

                        delta_neg = neg_baseline - dataset_mean_baseline
                        neg_sims = _r2_sim(delta_neg, delta_gt)
                        pds_r2_neg = 2.0 * np.mean([1 if neg_sims[idx] > neg_sims[j] else 0 for j in others]) - 1.0

                        metric_values['cpds_r2_deltapert'] = self._calibrate_higher_is_better(
                            pred_score=pds_r2_deltapert_scores[cov_pert_key],
                            pos_score=pds_r2_pos,
                            neg_score=pds_r2_neg,
                            perturbation=condition,
                            metric_name="cpds_r2_deltapert",
                        )

            # --- cpds_weighted_r2_deltapert ---
            if should_compute('cpds_weighted_r2_deltapert'):
                if (
                    need_baselines
                    and pos_baseline is not None
                    and neg_baseline is not None
                    and dataset_mean_baseline is not None
                    and cov_pert_key in pds_weighted_r2_deltapert_scores
                ):
                    cov_prefix = covariate_value + '_'
                    cov_mask = ground_truth.index.str.startswith(cov_prefix)
                    ground_truth_cov = ground_truth[cov_mask]

                    if len(ground_truth_cov) < 2:
                        metric_values['cpds_weighted_r2_deltapert'] = np.nan
                    else:
                        gt_values = ground_truth_cov.values
                        cov_pert_keys_list = list(ground_truth_cov.index)
                        try:
                            idx = cov_pert_keys_list.index(cov_pert_key)
                        except ValueError:
                            idx = 0
                        n = len(cov_pert_keys_list)
                        others = [j for j in range(n) if j != idx]

                        delta_gt = gt_values - dataset_mean_baseline[np.newaxis, :]

                        try:
                            _deg_source = getattr(self.data_manager, 'deg_weight_source', 'vsrest')
                            cal_weights = np.array(self.data_manager.get_deg_weights(
                                covariate_value, condition, common_var_names
                            ))
                            if np.sum(cal_weights) < 1e-12:
                                raise ValueError(
                                    f"No DEG weights for '{cov_pert_key}' in cPDS_wR2 calibration "
                                    f"(deg_weight_source='{_deg_source}')."
                                )

                            def _wr2_sim(vec, mat, w):
                                return np.array([
                                    r2_score_on_deltas(mat[k], vec, w)
                                    for k in range(len(mat))
                                ])

                            delta_pos = pos_baseline - dataset_mean_baseline
                            pos_sims = _wr2_sim(delta_pos, delta_gt, cal_weights)
                            pds_wr2_pos = 2.0 * np.mean([1 if pos_sims[idx] > pos_sims[j] else 0 for j in others]) - 1.0

                            delta_neg = neg_baseline - dataset_mean_baseline
                            neg_sims = _wr2_sim(delta_neg, delta_gt, cal_weights)
                            pds_wr2_neg = 2.0 * np.mean([1 if neg_sims[idx] > neg_sims[j] else 0 for j in others]) - 1.0

                            metric_values['cpds_weighted_r2_deltapert'] = self._calibrate_higher_is_better(
                                pred_score=pds_weighted_r2_deltapert_scores[cov_pert_key],
                                pos_score=pds_wr2_pos,
                                neg_score=pds_wr2_neg,
                                perturbation=condition,
                                metric_name="cpds_weighted_r2_deltapert",
                            )
                        except ValueError as e:
                            log.warning("cPDS_wR2 skipped for %s: %s", cov_pert_key, e)
                            metric_values['cpds_weighted_r2_deltapert'] = np.nan

            condition_metrics[cov_pert_key] = metric_values

            
        # Reorganize to metric -> cov_pert_key -> score format
        organized_metrics: Dict[str, Dict[str, float]] = {}
        metric_order = [
            'mse',
            'wmse',
            'cmse',
            'cwmse',
            'fmse',
            'fcmse',
            'fpearson',
            'fcpearson',
            'cpearson',
            'fr2',
            'fcr2',
            'cr2',
            'pearson_deltactrl',
            'pearson_deltactrl_degs',
            'cpearson_deltactrl',
            'fpearson_deltactrl',
            'fcpearson_deltactrl',
            'r2_deltactrl',
            'r2_deltactrl_degs',
            'cr2_deltactrl',
            'weighted_pearson_deltactrl',
            'cweighted_pearson_deltactrl',
            'fr2_deltactrl',
            'fcr2_deltactrl',
            'weighted_r2_deltactrl',
            'cweighted_r2_deltactrl',
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
            'pds',
            'pds_wmse',
            'pds_pearson_deltapert',
            'pds_weighted_pearson_deltapert',
            'pds_r2_deltapert',
            'pds_weighted_r2_deltapert',
            'cpds',
            'cpds_wmse',
            'cpds_pearson_deltapert',
            'cpds_weighted_pearson_deltapert',
            'cpds_r2_deltapert',
            'cpds_weighted_r2_deltapert',
        ]
        for metric in metric_order:
            if metrics_requested is not None and metric not in metrics_requested:
                continue
            organized_metrics[metric] = {
                cov_pert_key: condition_metrics[cov_pert_key].get(metric, np.nan)
                for cov_pert_key in condition_metrics.keys()
            }

        if metrics_requested is not None:
            missing = metrics_requested - set(organized_metrics.keys())
            if missing:
                raise ValueError(
                    f"Requested metrics not available: {sorted(missing)}. "
                    f"Available metrics: {sorted(organized_metrics.keys())}"
                )
        
        pds_all_scores: Dict[str, Dict[str, float]] = {
            'pds': pds_scores,
            'pds_wmse': pds_wmse_scores,
            'pds_pearson_deltapert': pds_pearson_deltapert_scores,
            'pds_weighted_pearson_deltapert': pds_weighted_pearson_deltapert_scores,
            'pds_r2_deltapert': pds_r2_deltapert_scores,
            'pds_weighted_r2_deltapert': pds_weighted_r2_deltapert_scores,
        }
        return organized_metrics, pds_all_scores
    
    def _calculate_mse(self, pred: np.ndarray, truth: np.ndarray) -> float:
        """Calculate MSE following plotting.py logic.
        
        Args:
            pred: Predicted expression values.
            truth: Ground truth expression values.
            
        Returns:
            Mean squared error.
        """
        return mse(pred, truth)
    
    def _calculate_wmse(self, pred: np.ndarray, truth: np.ndarray, weights: np.ndarray) -> float:
        """Calculate weighted MSE following plotting.py logic.
        
        Args:
            pred: Predicted expression values.
            truth: Ground truth expression values.
            weights: DEG-based weights for each gene.
            
        Returns:
            Weighted mean squared error.
        """
        return wmse(pred, truth, weights)
    
    def _calculate_pearson_delta(self, pred: np.ndarray, truth: np.ndarray, 
                               control: np.ndarray) -> float:
        """Calculate Pearson correlation of deltas.
        
        Args:
            pred: Predicted expression values.
            truth: Ground truth expression values.
            control: Control/baseline expression values.
            
        Returns:
            Pearson correlation coefficient of deltas from control.
        """
        delta_pred = pred - control
        delta_truth = truth - control
        try:
            corr, _ = pearsonr(delta_pred, delta_truth)
            return corr
        except:
            return np.nan
    
    def _calculate_pearson_delta_degs(self, pred: np.ndarray, truth: np.ndarray,
                                    control: np.ndarray, deg_mask: np.ndarray) -> float:
        """Calculate Pearson correlation of deltas for DEGs only.
        
        Args:
            pred: Predicted expression values.
            truth: Ground truth expression values.
            control: Control/baseline expression values.
            deg_mask: Boolean mask indicating DEG positions.
            
        Returns:
            Pearson correlation coefficient for DEGs only.
        """
        delta_pred = pred[deg_mask] - control[deg_mask]
        delta_truth = truth[deg_mask] - control[deg_mask]
        try:
            corr, _ = pearsonr(delta_pred, delta_truth)
            return corr
        except:
            return np.nan
    
    def _calculate_r2_delta(self, pred: np.ndarray, truth: np.ndarray,
                          control: np.ndarray) -> float:
        """Calculate R² on deltas.
        
        Args:
            pred: Predicted expression values.
            truth: Ground truth expression values.
            control: Control/baseline expression values.
            
        Returns:
            R² score on delta values.
        """
        delta_pred = pred - control
        delta_truth = truth - control
        return r2_score_on_deltas(delta_truth, delta_pred)
    
    def _calculate_r2_delta_degs(self, pred: np.ndarray, truth: np.ndarray,
                               control: np.ndarray, deg_mask: np.ndarray) -> float:
        """Calculate R² on deltas for DEGs only.
        
        Args:
            pred: Predicted expression values.
            truth: Ground truth expression values.
            control: Control/baseline expression values.
            deg_mask: Boolean mask indicating DEG positions.
            
        Returns:
            R² score on delta values for DEGs only.
        """
        delta_pred = pred[deg_mask] - control[deg_mask]
        delta_truth = truth[deg_mask] - control[deg_mask]
        return r2_score_on_deltas(delta_truth, delta_pred)
    
    def _calculate_weighted_r2_delta(self, pred: np.ndarray, truth: np.ndarray,
                                   control: np.ndarray, weights: np.ndarray) -> float:
        """Calculate weighted R² on deltas.
        
        Args:
            pred: Predicted expression values.
            truth: Ground truth expression values.
            control: Control/baseline expression values.
            weights: DEG-based weights for each gene.
            
        Returns:
            Weighted R² score on delta values.
        """
        delta_pred = pred - control
        delta_truth = truth - control
        return r2_score_on_deltas(delta_truth, delta_pred, weights)
    
    def _calculate_pearson_delta_direct(self, pred_deltas: np.ndarray, truth_deltas: np.ndarray) -> float:
        """Calculate Pearson correlation on pre-computed deltas.
        
        Args:
            pred_deltas: Pre-computed predicted delta values.
            truth_deltas: Pre-computed ground truth delta values.
            
        Returns:
            Pearson correlation coefficient of deltas.
        """
        corr, _ = pearsonr(pred_deltas, truth_deltas)
        return corr
    
    def _calculate_r2_delta_direct(self, pred_deltas: np.ndarray, truth_deltas: np.ndarray) -> float:
        """Calculate R² on pre-computed deltas.
        
        Args:
            pred_deltas: Pre-computed predicted delta values.
            truth_deltas: Pre-computed ground truth delta values.
            
        Returns:
            R² score on pre-computed delta values.
        """
        return r2_score_on_deltas(truth_deltas, pred_deltas)
    
    def _calculate_weighted_r2_delta_direct(self, pred_deltas: np.ndarray, truth_deltas: np.ndarray, 
                                          weights: np.ndarray) -> float:
        """Calculate weighted R² on pre-computed deltas.
        
        Args:
            pred_deltas: Pre-computed predicted delta values.
            truth_deltas: Pre-computed ground truth delta values.
            weights: DEG-based weights for each gene.
            
        Returns:
            Weighted R² score on pre-computed delta values.
        """
        return r2_score_on_deltas(truth_deltas, pred_deltas, weights)

    def _calculate_weighted_pearson_direct(
        self,
        pred: np.ndarray,
        truth: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Calculate weighted Pearson correlation with safety checks."""
        pred = np.asarray(pred, dtype=float)
        truth = np.asarray(truth, dtype=float)
        weights = np.asarray(weights, dtype=float)

        if (
            pred.size < 2
            or truth.size < 2
            or pred.shape != truth.shape
            or weights.shape != truth.shape
        ):
            return float("nan")

        finite_mask = np.isfinite(pred) & np.isfinite(truth) & np.isfinite(weights)
        if not np.any(finite_mask):
            return float("nan")
        pred = pred[finite_mask]
        truth = truth[finite_mask]
        weights = weights[finite_mask]

        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            return float("nan")

        mean_pred = float(np.average(pred, weights=weights))
        mean_truth = float(np.average(truth, weights=weights))
        centered_pred = pred - mean_pred
        centered_truth = truth - mean_truth

        cov = float(np.average(centered_pred * centered_truth, weights=weights))
        var_pred = float(np.average(centered_pred ** 2, weights=weights))
        var_truth = float(np.average(centered_truth ** 2, weights=weights))

        if var_pred <= 0.0 or var_truth <= 0.0:
            return float("nan")

        return float(cov / np.sqrt(var_pred * var_truth))

    def _calculate_pearson_direct(self, pred: np.ndarray, truth: np.ndarray) -> float:
        """Calculate Pearson correlation with safety checks.

        Args:
            pred: Predicted values.
            truth: Ground truth values.

        Returns:
            Pearson correlation coefficient, or 0.0 if either vector is constant.
        """
        if pred.size < 2 or truth.size < 2:
            return float("nan")
        if np.std(pred) == 0 or np.std(truth) == 0:
            return float("nan")
        try:
            corr, _ = pearsonr(pred, truth)
            return float(corr)
        except Exception:
            return float("nan")

    def _calculate_r2_direct(self, truth: np.ndarray, pred: np.ndarray) -> float:
        """Calculate R² score for direct predictions.

        Args:
            truth: Ground truth values.
            pred: Predicted values.

        Returns:
            R² score, or NaN if inputs are invalid.
        """
        if truth.size < 2 or pred.size < 2 or truth.shape != pred.shape:
            return float("nan")
        try:
            return float(r2_score(truth, pred))
        except Exception:
            return float("nan")

    def _get_filtered_genes(
        self,
        dataset_name: str,
        perturbation: str,
        pds_metric: str = "Energy_Distance",
        pds_threshold: float = 0.8,
    ) -> List[str]:
        """Load filtered genes for a perturbation using PDS scores.

        Args:
            dataset_name: Dataset name for loading PDS data.
            perturbation: Perturbation identifier.
            pds_metric: PDS metric to use for filtering.
            pds_threshold: PDS threshold for gene selection.

        Returns:
            List of filtered gene names.
        """
        from cellsimbench.core.filtered_metrics import GeneFilterLoader

        if not hasattr(self, '_gene_filter_loader'):
            pds_results_dir = 'analyses/perturbation_discrimination/results'
            self._gene_filter_loader = GeneFilterLoader(pds_results_dir)

        return self._gene_filter_loader.get_filtered_genes(
            dataset_name=dataset_name,
            pds_metric=pds_metric,
            perturbation=perturbation,
            pds_threshold=pds_threshold,
        )

    def _apply_gene_filter(
        self,
        values: np.ndarray,
        gene_names: List[str],
        filter_genes: List[str],
    ) -> np.ndarray:
        """Apply gene filter to an array aligned with gene_names.

        Args:
            values: Array aligned with gene_names.
            gene_names: List of gene names for values.
            filter_genes: List of genes to keep.

        Returns:
            Filtered array containing only values for filter_genes.
        """
        from cellsimbench.core.filtered_metrics import apply_gene_filter

        return apply_gene_filter(values, gene_names, filter_genes)

    def _calibrate_higher_is_better(
        self,
        pred_score: float,
        pos_score: float,
        neg_score: float,
        perturbation: Optional[str] = None,
        metric_name: Optional[str] = None,
    ) -> float:
        """Calibrate higher-is-better metrics (Pearson/R²) using baselines.

        Args:
            pred_score: Metric score for predictions.
            pos_score: Metric score for positive baseline.
            neg_score: Metric score for negative baseline.
            perturbation: Optional perturbation name for error context.
            metric_name: Optional key for aggregated calibration diagnostics.

        Returns:
            Calibrated score in [0, 1] or NaN on invalid input.
        """
        from cellsimbench.core.calibrated_metrics import calibrate_aggregated_metric

        try:
            return calibrate_aggregated_metric(
                m_pred=pred_score,
                m_pos=pos_score,
                m_neg=neg_score,
                perturbation=perturbation,
                higher_is_better=True,
                metric_name=metric_name,
            )
        except ValueError:
            return float("nan")

    
    def _calculate_genewise_mse(
        self,
        pred: np.ndarray,
        truth: np.ndarray,
        gene_names: Optional[List[str]] = None,
    ) -> float:
        """Calculate gene-wise MSE and return average.
        
        Args:
            pred: Predicted mean expression values (n_genes,).
            truth: Ground truth mean expression values (n_genes,).
            gene_names: Optional list of gene names.
            
        Returns:
            Average MSE across all genes.
        """
        from cellsimbench.core.genewise_metrics import compute_genewise_metrics
        
        # Reshape to (1, n_genes) for compute_genewise_metrics
        pred_reshaped = pred.reshape(1, -1)
        truth_reshaped = truth.reshape(1, -1)
        
        # Compute gene-wise MSE
        genewise_values = compute_genewise_metrics(
            pred_reshaped, truth_reshaped, metric_name="mse"
        )
        
        # Return average
        return float(np.mean(genewise_values))
    
    def _calculate_cmse(
        self,
        pred: np.ndarray,
        truth: np.ndarray,
        covariate: str,
        perturbation: str,
        gene_names: Optional[List[str]] = None,
        split_column: str = "tech_dup_split",
        current_split: str = "first_half",
    ) -> float:
        """Calculate calibrated MSE (cMSE).
        
        Args:
            pred: Predicted mean expression values (n_genes,).
            truth: Ground truth mean expression values (n_genes,).
            covariate: Covariate value (e.g., donor ID).
            perturbation: Perturbation identifier.
            gene_names: Optional list of gene names.
            split_column: Column name for technical duplicate split.
            current_split: The split used for truth (to get opposite for baseline).
            
        Returns:
            Calibrated MSE value (aggregated across genes).
        """
        from cellsimbench.core.calibrated_metrics import calibrate_aggregated_metric

        pos_baseline = self._get_positive_baseline(
            covariate=covariate,
            perturbation=perturbation,
            gene_names=gene_names,
            split_column=split_column,
            current_split=current_split,
        )
        neg_baseline = self._get_negative_baseline(
            covariate=covariate,
            gene_names=gene_names,
        )

        m_pred = float(np.mean((pred - truth) ** 2))
        m_pos = float(np.mean((pos_baseline - truth) ** 2))
        m_neg = float(np.mean((neg_baseline - truth) ** 2))

        return calibrate_aggregated_metric(
            m_pred=m_pred,
            m_pos=m_pos,
            m_neg=m_neg,
            perturbation=perturbation,
            metric_name="cmse",
        )

    def _calculate_cwmse(
        self,
        pred: np.ndarray,
        truth: np.ndarray,
        weights: np.ndarray,
        covariate: str,
        perturbation: str,
        gene_names: Optional[List[str]] = None,
        split_column: str = "tech_dup_split",
        current_split: str = "first_half",
    ) -> float:
        """Calculate calibrated weighted MSE (cwMSE)."""
        from cellsimbench.core.calibrated_metrics import calibrate_aggregated_metric

        pos_baseline = self._get_positive_baseline(
            covariate=covariate,
            perturbation=perturbation,
            gene_names=gene_names,
            split_column=split_column,
            current_split=current_split,
        )
        neg_baseline = self._get_negative_baseline(
            covariate=covariate,
            gene_names=gene_names,
        )

        m_pred = self._calculate_wmse(pred, truth, weights)
        m_pos = self._calculate_wmse(pos_baseline, truth, weights)
        m_neg = self._calculate_wmse(neg_baseline, truth, weights)

        return calibrate_aggregated_metric(
            m_pred=m_pred,
            m_pos=m_pos,
            m_neg=m_neg,
            perturbation=perturbation,
            metric_name="cwmse",
        )
    
    def _calculate_fmse(
        self,
        pred: np.ndarray,
        truth: np.ndarray,
        covariate: str,
        perturbation: str,
        gene_names: List[str],
        dataset_name: str,
        pds_metric: str = "Energy_Distance",
        pds_threshold: float = 0.8,
    ) -> float:
        """Calculate filtered MSE (fMSE) - uncalibrated.
        
        Applies PDS-based gene filtering and computes MSE on selected genes.
        
        Args:
            pred: Predicted mean expression values (n_genes,).
            truth: Ground truth mean expression values (n_genes,).
            covariate: Covariate value.
            perturbation: Perturbation identifier.
            gene_names: List of gene names (required for filtering).
            dataset_name: Dataset name for loading PDS data.
            pds_metric: PDS metric to use for filtering.
            pds_threshold: PDS threshold for gene selection.
            
        Returns:
            Filtered MSE value.
        
        Raises:
            ValueError: If no genes pass filter or filter data is invalid.
        """
        from cellsimbench.core.genewise_metrics import compute_genewise_metrics
        from cellsimbench.core.filtered_metrics import (
            GeneFilterLoader, apply_gene_filter
        )
        
        # Get filtered genes for this perturbation
        if not hasattr(self, '_gene_filter_loader'):
            # Initialize filter loader (cached for reuse)
            pds_results_dir = f'analyses/perturbation_discrimination/results'
            self._gene_filter_loader = GeneFilterLoader(pds_results_dir)

        filtered_genes = self._gene_filter_loader.get_filtered_genes(
            dataset_name=dataset_name,
            pds_metric=pds_metric,
            perturbation=perturbation,
            pds_threshold=pds_threshold,
        )
        
        # Reshape for metric computation
        pred_reshaped = pred.reshape(1, -1)
        truth_reshaped = truth.reshape(1, -1)
        
        # Compute gene-wise MSE for all genes
        genewise_mse = compute_genewise_metrics(pred_reshaped, truth_reshaped, metric_name="mse")
        
        # Apply gene filter
        filtered_mse = apply_gene_filter(genewise_mse, gene_names, filtered_genes)
            
        # Return average MSE on filtered genes
        return float(np.mean(filtered_mse))

    def _calculate_fcmse(
        self,
        pred: np.ndarray,
        truth: np.ndarray,
        covariate: str,
        perturbation: str,
        gene_names: List[str],
        dataset_name: str,
        pds_metric: str = "Energy_Distance",
        pds_threshold: float = 0.8,
        split_column: str = "tech_dup_split",
        current_split: str = "first_half",
    ) -> float:
        """Calculate filtered calibrated MSE (fcMSE).
        
        Applies PDS-based gene filtering before calibration.
        
        Args:
            pred: Predicted mean expression values (n_genes,).
            truth: Ground truth mean expression values (n_genes,).
            covariate: Covariate value.
            perturbation: Perturbation identifier.
            gene_names: List of gene names (required for filtering).
            dataset_name: Dataset name for loading PDS data.
            pds_metric: PDS metric to use for filtering.
            pds_threshold: PDS threshold for gene selection.
            split_column: Column name for technical duplicate split.
            current_split: The split used for truth.
            
        Returns:
            Filtered calibrated MSE value.

        Raises:
            ValueError: If no genes pass filter or filter data is invalid.
        """
        from cellsimbench.core.calibrated_metrics import calibrate_aggregated_metric
        from cellsimbench.core.filtered_metrics import (
            GeneFilterLoader, apply_gene_filter
        )

        if not hasattr(self, '_gene_filter_loader'):
            pds_results_dir = f'analyses/perturbation_discrimination/results'
            self._gene_filter_loader = GeneFilterLoader(pds_results_dir)

        filtered_genes = self._gene_filter_loader.get_filtered_genes(
            dataset_name=dataset_name,
            pds_metric=pds_metric,
            perturbation=perturbation,
            pds_threshold=pds_threshold,
        )

        pos_baseline = self._get_positive_baseline(
            covariate=covariate,
            perturbation=perturbation,
            gene_names=gene_names,
            split_column=split_column,
            current_split=current_split,
        )
        neg_baseline = self._get_negative_baseline(
            covariate=covariate,
            gene_names=gene_names,
        )

        pred_sq = (pred - truth) ** 2
        pos_sq = (pos_baseline - truth) ** 2
        neg_sq = (neg_baseline - truth) ** 2

        filtered_pred = apply_gene_filter(pred_sq, gene_names, filtered_genes)
        filtered_pos = apply_gene_filter(pos_sq, gene_names, filtered_genes)
        filtered_neg = apply_gene_filter(neg_sq, gene_names, filtered_genes)

        m_pred = float(np.mean(filtered_pred))
        m_pos = float(np.mean(filtered_pos))
        m_neg = float(np.mean(filtered_neg))

        return calibrate_aggregated_metric(
            m_pred=m_pred,
            m_pos=m_pos,
            m_neg=m_neg,
            perturbation=perturbation,
            metric_name="fcmse",
        )
    
    def _calculate_pds_scores_pearson_deltapert(
        self,
        predictions_deltas_mean: pd.DataFrame,
        ground_truth_deltas_mean: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate PDS using Pearson correlation of perturbation deltas (delta-from-mean).

        For each perturbation i, uses Pearson correlation between the predicted
        delta (pred - dataset_mean) and each ground-truth delta as a similarity
        measure.  A perturbation is a "win" when the correct GT delta is more
        correlated with the predicted delta than all other GT deltas in the same
        covariate group.

        PDS = 2 * (win_fraction) - 1  in [-1, 1]

        Args:
            predictions_deltas_mean: DataFrame of per-perturbation prediction deltas
                (pred - dataset_mean), indexed by cov_pert_key.
            ground_truth_deltas_mean: Corresponding ground-truth deltas.

        Returns:
            Dict mapping cov_pert_key to PDS_pearson_deltapert score.
        """
        pds_scores: Dict[str, float] = {}

        covariate_groups: Dict[str, list] = {}
        for pert_key in predictions_deltas_mean.index:
            covariate = pert_key.split('_')[0]
            covariate_groups.setdefault(covariate, []).append(pert_key)

        for covariate, pert_keys in covariate_groups.items():
            valid_keys = [pk for pk in pert_keys if pk in ground_truth_deltas_mean.index]
            if len(valid_keys) < 2:
                continue

            pred_deltas = predictions_deltas_mean.loc[valid_keys].values   # (n, g)
            gt_deltas = ground_truth_deltas_mean.loc[valid_keys].values     # (n, g)
            n = len(valid_keys)

            # Pearson similarity matrix: sim[i, j] = pearson(pred_delta[i], gt_delta[j])
            sim_matrix = np.zeros((n, n))
            for i in range(n):
                pd_i = pred_deltas[i]
                std_pred = np.std(pd_i)
                for j in range(n):
                    gt_j = gt_deltas[j]
                    std_gt = np.std(gt_j)
                    if std_pred < 1e-10 or std_gt < 1e-10:
                        sim_matrix[i, j] = 0.0
                    else:
                        sim_matrix[i, j] = float(np.corrcoef(pd_i, gt_j)[0, 1])

            for i, pert_key in tqdm(
                enumerate(valid_keys),
                desc="Calculating PDS_pearson_deltapert for covariate " + covariate,
            ):
                correct_sim = sim_matrix[i, i]
                comparisons = [1 if correct_sim > sim_matrix[i, j] else 0
                               for j in range(n) if j != i]
                raw_score = np.mean(comparisons) if comparisons else 0.5
                pds_scores[pert_key] = 2.0 * (raw_score - 0.5)

        return pds_scores

    def _calculate_pds_scores_weighted_pearson_deltapert(
        self,
        predictions_deltas_mean: pd.DataFrame,
        ground_truth_deltas_mean: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate PDS using DEG-weighted Pearson correlation of perturbation deltas.

        Same win-fraction framework as ``_calculate_pds_scores_pearson_deltapert``
        but the similarity measure is weighted Pearson (using per-query DEG weights)
        instead of unweighted Pearson.

        Args:
            predictions_deltas_mean: Prediction deltas (pred - dataset_mean).
            ground_truth_deltas_mean: Ground-truth deltas.

        Returns:
            Dict mapping cov_pert_key to PDS score in [-1, 1].
        """
        pds_scores: Dict[str, float] = {}

        covariate_groups: Dict[str, list] = {}
        for pert_key in predictions_deltas_mean.index:
            covariate = pert_key.split('_')[0]
            covariate_groups.setdefault(covariate, []).append(pert_key)

        for covariate, pert_keys in covariate_groups.items():
            valid_keys = [pk for pk in pert_keys if pk in ground_truth_deltas_mean.index]
            if len(valid_keys) < 2:
                continue

            pred_deltas = predictions_deltas_mean.loc[valid_keys].values   # (n, g)
            gt_deltas = ground_truth_deltas_mean.loc[valid_keys].values     # (n, g)
            gene_names = list(predictions_deltas_mean.columns)
            n = len(valid_keys)

            deg_source = getattr(self.data_manager, 'deg_weight_source', 'vsrest')
            query_weights = []
            for i in range(n):
                pert_name = valid_keys[i].split('_', 1)[1]
                weights_i = np.array(self.data_manager.get_deg_weights(covariate, pert_name, gene_names))
                if np.sum(weights_i) < 1e-12:
                    raise ValueError(
                        f"No DEG weights found for '{valid_keys[i]}' (deg_weight_source='{deg_source}'). "
                        f"PDS_wPearson requires DEG weights."
                    )
                query_weights.append(weights_i)

            sim_matrix = np.zeros((n, n))
            for i in range(n):
                sim_matrix[i, :] = [
                    self._calculate_weighted_pearson_direct(pred_deltas[i], gt_deltas[j], query_weights[i])
                    for j in range(n)
                ]

            for i, pert_key in tqdm(
                enumerate(valid_keys),
                desc="Calculating PDS_wPearson_deltapert for covariate " + covariate,
            ):
                correct_sim = sim_matrix[i, i]
                comparisons = [1 if correct_sim > sim_matrix[i, j] else 0
                               for j in range(n) if j != i]
                raw_score = np.mean(comparisons) if comparisons else 0.5
                pds_scores[pert_key] = 2.0 * (raw_score - 0.5)

        return pds_scores

    def _calculate_pds_scores_r2_deltapert(
        self,
        predictions_deltas_mean: pd.DataFrame,
        ground_truth_deltas_mean: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate PDS using R² of perturbation deltas as the similarity measure.

        For each perturbation i, sim[i, j] = R²(gt_delta[j], pred_delta[i]).
        A perturbation "wins" when its correct GT pair has the highest R².

        Args:
            predictions_deltas_mean: Prediction deltas (pred - dataset_mean).
            ground_truth_deltas_mean: Ground-truth deltas.

        Returns:
            Dict mapping cov_pert_key to PDS score in [-1, 1].
        """
        pds_scores: Dict[str, float] = {}

        covariate_groups: Dict[str, list] = {}
        for pert_key in predictions_deltas_mean.index:
            covariate = pert_key.split('_')[0]
            covariate_groups.setdefault(covariate, []).append(pert_key)

        for covariate, pert_keys in covariate_groups.items():
            valid_keys = [pk for pk in pert_keys if pk in ground_truth_deltas_mean.index]
            if len(valid_keys) < 2:
                continue

            pred_deltas = predictions_deltas_mean.loc[valid_keys].values   # (n, g)
            gt_deltas = ground_truth_deltas_mean.loc[valid_keys].values     # (n, g)
            n = len(valid_keys)

            sim_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    sim_matrix[i, j] = r2_score_on_deltas(gt_deltas[j], pred_deltas[i])

            for i, pert_key in tqdm(
                enumerate(valid_keys),
                desc="Calculating PDS_r2_deltapert for covariate " + covariate,
            ):
                correct_sim = sim_matrix[i, i]
                comparisons = [1 if correct_sim > sim_matrix[i, j] else 0
                               for j in range(n) if j != i]
                raw_score = np.mean(comparisons) if comparisons else 0.5
                pds_scores[pert_key] = 2.0 * (raw_score - 0.5)

        return pds_scores

    def _calculate_pds_scores_weighted_r2_deltapert(
        self,
        predictions_deltas_mean: pd.DataFrame,
        ground_truth_deltas_mean: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate PDS using DEG-weighted R² of perturbation deltas.

        Same as ``_calculate_pds_scores_r2_deltapert`` but passes per-query DEG
        weights as ``sample_weight`` to ``r2_score``.

        Args:
            predictions_deltas_mean: Prediction deltas (pred - dataset_mean).
            ground_truth_deltas_mean: Ground-truth deltas.

        Returns:
            Dict mapping cov_pert_key to PDS score in [-1, 1].
        """
        pds_scores: Dict[str, float] = {}

        covariate_groups: Dict[str, list] = {}
        for pert_key in predictions_deltas_mean.index:
            covariate = pert_key.split('_')[0]
            covariate_groups.setdefault(covariate, []).append(pert_key)

        for covariate, pert_keys in covariate_groups.items():
            valid_keys = [pk for pk in pert_keys if pk in ground_truth_deltas_mean.index]
            if len(valid_keys) < 2:
                continue

            pred_deltas = predictions_deltas_mean.loc[valid_keys].values   # (n, g)
            gt_deltas = ground_truth_deltas_mean.loc[valid_keys].values     # (n, g)
            gene_names = list(predictions_deltas_mean.columns)
            n = len(valid_keys)

            deg_source = getattr(self.data_manager, 'deg_weight_source', 'vsrest')
            query_weights = []
            for i in range(n):
                pert_name = valid_keys[i].split('_', 1)[1]
                weights_i = np.array(self.data_manager.get_deg_weights(covariate, pert_name, gene_names))
                if np.sum(weights_i) < 1e-12:
                    raise ValueError(
                        f"No DEG weights found for '{valid_keys[i]}' (deg_weight_source='{deg_source}'). "
                        f"PDS_wR2 requires DEG weights."
                    )
                query_weights.append(weights_i)

            sim_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    sim_matrix[i, j] = r2_score_on_deltas(gt_deltas[j], pred_deltas[i], query_weights[i])

            for i, pert_key in tqdm(
                enumerate(valid_keys),
                desc="Calculating PDS_wR2_deltapert for covariate " + covariate,
            ):
                correct_sim = sim_matrix[i, i]
                comparisons = [1 if correct_sim > sim_matrix[i, j] else 0
                               for j in range(n) if j != i]
                raw_score = np.mean(comparisons) if comparisons else 0.5
                pds_scores[pert_key] = 2.0 * (raw_score - 0.5)

        return pds_scores

    def _calculate_pds_scores(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate PDS (Perturbation Discrimination Score) for all perturbations.

        For each perturbation, measures how much better the predicted profile is
        matched to its correct ground truth vs. other perturbations' ground truths
        WITHIN THE SAME COVARIATE GROUP.

        PDS is rescaled from the raw "win fraction" (0-1) to [-1, 1] range:
        - PDS = 1.0: Perfect discrimination (closer to correct GT than ALL others)
        - PDS = 0.0: Chance level (closer to correct GT than half of others)
        - PDS = -1.0: Always closer to some other GT than to the correct one

        Args:
            predictions: DataFrame with predicted expression profiles (cov_pert_key as index)
            ground_truth: DataFrame with ground truth expression profiles (cov_pert_key as index)

        Returns:
            Dict mapping cov_pert_key to PDS score (-1 to 1)
        """
        from scipy.spatial.distance import cdist

        pds_scores = {}

        # Group perturbations by covariate
        covariate_groups = {}
        for pert_key in predictions.index:
            covariate = pert_key.split('_')[0]
            if covariate not in covariate_groups:
                covariate_groups[covariate] = []
            covariate_groups[covariate].append(pert_key)

        # Calculate PDS within each covariate group
        for covariate, pert_keys in covariate_groups.items():

            # Filter to only perturbations present in both predictions and ground truth
            valid_pert_keys = [pk for pk in pert_keys if pk in ground_truth.index]
            missing_pert_keys = [pk for pk in pert_keys if pk not in ground_truth.index]

            if missing_pert_keys:
                print(f"Warning: {len(missing_pert_keys)} perturbations not in ground truth for covariate {covariate}, skipping those")

            if len(valid_pert_keys) < 2:
                # Need at least 2 perturbations to calculate PDS
                print(f"Skipping covariate {covariate}: only {len(valid_pert_keys)} valid perturbations (need ≥2)")
                continue

            # Get predictions and ground truths for valid perturbations only
            predictions_cov = predictions.loc[valid_pert_keys]
            ground_truth_cov = ground_truth.loc[valid_pert_keys]
            pert_keys = valid_pert_keys  # Use only valid keys for the rest of the calculation

            # Compute pairwise distance matrix for this covariate group
            distance_matrix = cdist(
                predictions_cov.values,
                ground_truth_cov.values,
                metric='euclidean'
            )

            # Calculate PDS for each perturbation in this covariate
            for i, pert_key in tqdm(enumerate(pert_keys), desc="Calculating PDS for covariate " + covariate):
                # Distance from this prediction to its correct ground truth
                correct_distance = distance_matrix[i, i]

                # Compare to all OTHER ground truths within same covariate
                comparisons = []
                for j in range(len(pert_keys)):
                    if i != j:  # Skip self-comparison
                        # Is prediction closer to correct GT than to this other GT?
                        comparisons.append(1 if correct_distance < distance_matrix[i, j] else 0)

                # Average across all comparisons (raw score in [0, 1])
                raw_score = np.mean(comparisons) if comparisons else 0.5

                # Rescale from [0, 1] to [-1, 1]: PDS = 2 * (raw_score - 0.5)
                # This maps: 1.0 → 1.0, 0.5 → 0.0, 0.0 → -1.0
                pds_scores[pert_key] = 2.0 * (raw_score - 0.5)
        return pds_scores

    def _calculate_pds_scores_weighted(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate PDS using weighted MSE distance instead of Euclidean.

        Uses DEG weights from the PREDICTED perturbation for computing distances.
        The weighted MSE distance is: sum(w * (x - y)^2) / sum(w)

        This means when comparing prediction[i] to ground_truth[j], we use
        the DEG weights of prediction[i] (the perturbation being evaluated).

        Args:
            predictions: DataFrame with predicted expression profiles (cov_pert_key as index)
            ground_truth: DataFrame with ground truth expression profiles (cov_pert_key as index)

        Returns:
            Dict mapping cov_pert_key to PDS score (-1 to 1)
        """
        pds_scores = {}

        # Group perturbations by covariate
        covariate_groups = {}
        for pert_key in predictions.index:
            covariate = pert_key.split('_')[0]
            if covariate not in covariate_groups:
                covariate_groups[covariate] = []
            covariate_groups[covariate].append(pert_key)

        # Calculate PDS within each covariate group
        for covariate, pert_keys in covariate_groups.items():
            valid_pert_keys = [pk for pk in pert_keys if pk in ground_truth.index]

            if len(valid_pert_keys) < 2:
                continue

            predictions_cov = predictions.loc[valid_pert_keys]
            ground_truth_cov = ground_truth.loc[valid_pert_keys]
            pert_keys = valid_pert_keys

            # Get gene names for weight lookup
            gene_names = list(predictions_cov.columns)

            # Pre-compute weights for each predicted perturbation (query weights).
            # Routing and strictness are handled centrally by get_deg_weights().
            deg_source = getattr(self.data_manager, 'deg_weight_source', 'vsrest')
            n = len(pert_keys)
            query_weights = []
            for i in range(n):
                pert_name = pert_keys[i].split('_', 1)[1]
                weights_i = np.array(self.data_manager.get_deg_weights(covariate, pert_name, gene_names))
                if np.sum(weights_i) < 1e-12:
                    raise ValueError(
                        f"No DEG weights found for '{pert_keys[i]}' (deg_weight_source='{deg_source}'). "
                        f"PDS_wMSE requires DEG weights — ensure the appropriate DEG CSV or "
                        f"rank_genes_groups data is available for this perturbation."
                    )
                query_weights.append(weights_i)

            # Calculate weighted MSE distance matrix for this covariate group
            # distance_matrix[i, j] = weighted MSE between prediction[i] and ground_truth[j]
            # using consistent weights from query i for all j comparisons
            distance_matrix = np.zeros((n, n))

            for i in range(n):
                pred_i = predictions_cov.iloc[i].values
                weights_i = query_weights[i]
                weights_sum = np.sum(weights_i)

                for j in range(n):
                    gt_j = ground_truth_cov.iloc[j].values
                    # Weighted MSE distance using QUERY i's weights (consistent for all comparisons)
                    diff = pred_i - gt_j
                    distance_matrix[i, j] = np.sum(weights_i * diff**2) / weights_sum

            # Calculate PDS for each perturbation
            for i, pert_key in tqdm(enumerate(pert_keys), desc="Calculating PDS_wMSE for covariate " + covariate):
                correct_distance = distance_matrix[i, i]
                comparisons = []
                for j in range(n):
                    if i != j:
                        comparisons.append(1 if correct_distance < distance_matrix[i, j] else 0)
                raw_score = np.mean(comparisons) if comparisons else 0.5
                pds_scores[pert_key] = 2.0 * (raw_score - 0.5)
        return pds_scores

    def _get_positive_baseline(
        self,
        covariate: str,
        perturbation: str,
        gene_names: Optional[List[str]],
        split_column: str,
        current_split: str,
    ) -> np.ndarray:
        """Return cached positive baseline aligned to gene_names."""
        if gene_names is None:
            raise ValueError("gene_names must be provided for baseline caching.")
        gene_key = tuple(gene_names)
        cache_key = (covariate, perturbation, split_column, current_split, gene_key)
        if cache_key not in self._pos_baseline_cache:
            self._pos_baseline_cache[cache_key] = self.data_manager.get_positive_baseline(
                covariate=covariate,
                perturbation=perturbation,
                split_column=split_column,
                current_split=current_split,
                gene_order=gene_names,
            )
        return self._pos_baseline_cache[cache_key]

    def _get_negative_baseline(
        self,
        covariate: str,
        gene_names: Optional[List[str]],
    ) -> np.ndarray:
        """Return cached negative baseline aligned to gene_names."""
        if gene_names is None:
            raise ValueError("gene_names must be provided for baseline caching.")
        gene_key = tuple(gene_names)
        cache_key = (covariate, gene_key)
        if cache_key not in self._neg_baseline_cache:
            self._neg_baseline_cache[cache_key] = self.data_manager.get_control_baseline(
                donor_id=covariate,
                gene_order=gene_names,
            )
        return self._neg_baseline_cache[cache_key]