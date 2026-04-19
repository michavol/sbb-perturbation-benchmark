"""Synthetic perturbation baseline for CellSimBench."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import scanpy as sc
import scipy.sparse as sp

from .base import BuiltinModel
from ..core.data_manager import DataManager

log = logging.getLogger(__name__)


class SyntheticPerturbationModel(BuiltinModel):
    """Synthetic perturbation baseline.

    This model uses the training split mean as a fixed offset ``b`` and fits
    a shared target scalar ``c_target`` against a generic perturbation
    reference vector:
        X_pred = b + c_target * sum(X_ref * One(target))

    The target term uses the mean expression across training perturbations.
    """

    def __init__(self, model_config: Dict[str, Any]) -> None:
        """Initialize the synthetic perturbation baseline.

        Args:
            model_config: Model configuration dictionary.
        """
        super().__init__(model_config)

    def predict(
        self,
        data_manager: DataManager,
        test_conditions: List[str],
        split_name: str,
        **kwargs: Any,
    ) -> sc.AnnData:
        """Predict condition means using the synthetic perturbation baseline.

        Args:
            data_manager: Data manager providing AnnData.
            test_conditions: Conditions to predict.
            split_name: Split column name.
            **kwargs: Unused extra arguments.

        Returns:
            AnnData with predicted expressions.
        """
        _ = kwargs
        adata = data_manager.adata if data_manager.adata is not None else data_manager.load_dataset()

        train_conditions = self._get_training_conditions(adata, split_name)
        cond_to_pert = self._get_condition_to_perturbation(adata)
        train_perturbations = [cond_to_pert.get(cond, cond) for cond in train_conditions]
        delimiter = self._infer_delimiter(train_perturbations)
        combo_only = self._should_use_combo_only(train_perturbations, delimiter)
        train_conditions = self._filter_combo_conditions(
            train_conditions, cond_to_pert, delimiter, combo_only
        )
        train_perturbations = [cond_to_pert.get(cond, cond) for cond in train_conditions]

        train_means = self._get_perturbation_means(adata, train_perturbations, split_name)
        var_names = list(adata.var_names)
        var_index = {name: idx for idx, name in enumerate(var_names)}
        target_map: Dict[str, List[int]] = {}
        for pert_label in train_means.keys():
            targets = self._targets_from_condition(pert_label, var_index, delimiter)
            if targets is None:
                continue
            target_map[pert_label] = targets

        if not target_map:
            raise ValueError("No training perturbations with valid targets found.")

        b_vec = self._get_train_dataset_mean(adata, split_name)
        target_reference_vec = self._get_train_perturbation_reference(train_means, target_map)
        c_target = self._fit_c_target(target_reference_vec, b_vec, train_means, target_map)

        predictions: Dict[str, np.ndarray] = {}
        test_conditions = self._filter_combo_conditions(
            test_conditions, cond_to_pert, delimiter, combo_only
        )
        for condition in test_conditions:
            pert_label = cond_to_pert.get(condition, condition)
            targets = self._targets_from_condition(pert_label, var_index, delimiter)
            if targets is None:
                continue
            pred = self._predict(target_reference_vec, targets, c_target, b_vec)
            predictions[condition] = pred
        if not predictions:
            raise ValueError(
                "No valid test conditions found for synthetic perturbation baseline. "
                "Check that perturbation tokens exist in adata.var_names."
            )

        pred_adata = self.create_predictions_adata(predictions, list(adata.var_names))
        pred_adata.uns["synthetic_perturbation_params"] = {
            "c_target": float(c_target),
            "b": b_vec,
            "offset_source": "train_dataset_mean",
            "target_reference_source": "train_perturbation_mean",
            "delimiter": delimiter,
            "train_conditions": sorted(train_means.keys()),
        }
        return pred_adata

    def _infer_delimiter(self, conditions: Sequence[str]) -> str:
        """Infer combo delimiter from condition labels."""
        for condition in conditions:
            if "+" in condition:
                return "+"
        if any("_" in condition for condition in conditions):
            return "_"
        return "+"

    def _get_condition_to_perturbation(self, adata: sc.AnnData) -> Dict[str, str]:
        """Map condition labels to perturbation labels if available."""
        if "perturbation" not in adata.obs:
            return {}
        mapping = {}
        pairs = adata.obs[["condition", "perturbation"]].dropna().drop_duplicates()
        for _, row in pairs.iterrows():
            mapping[str(row["condition"])] = str(row["perturbation"])
        return mapping

    def _should_use_combo_only(self, conditions: Sequence[str], delimiter: str) -> bool:
        """Decide whether to ignore single perturbations."""
        return any(delimiter in condition for condition in conditions)

    def _filter_combo_conditions(
        self,
        conditions: Sequence[str],
        cond_to_pert: Mapping[str, str],
        delimiter: str,
        combo_only: bool,
    ) -> List[str]:
        """Optionally filter to combo-only conditions."""
        if not combo_only:
            return list(conditions)
        filtered = []
        for cond in conditions:
            label = cond_to_pert.get(cond, cond)
            if delimiter in label:
                filtered.append(cond)
        return filtered

    def _get_training_conditions(self, adata: sc.AnnData, split_name: str) -> List[str]:
        """Collect train+val conditions for fitting."""
        train_mask = adata.obs[split_name].isin(["train", "val"])
        train_conditions = adata.obs.loc[train_mask, "condition"].unique().tolist()
        return sorted(train_conditions)

    def _get_perturbation_means(
        self,
        adata: sc.AnnData,
        perturbations: Sequence[str],
        split_name: str,
    ) -> Dict[str, np.ndarray]:
        """Compute perturbation mean expressions (train+val split only)."""
        if "perturbation" in adata.obs:
            pert_series = adata.obs["perturbation"].astype(str)
        else:
            pert_series = adata.obs["condition"].astype(str)
        means: Dict[str, np.ndarray] = {}
        for pert in sorted(set(perturbations)):
            mask = adata.obs[split_name].isin(["train", "val"]) & (pert_series == pert)
            mean_vec = self._get_mean_expression(adata, mask.to_numpy())
            if mean_vec is not None:
                means[pert] = mean_vec
        return means

    def _get_train_dataset_mean(self, adata: sc.AnnData, split_name: str) -> np.ndarray:
        """Compute the fixed offset as the mean over train+val cells."""
        train_mask = adata.obs[split_name].isin(["train", "val"]).to_numpy()
        train_mean = self._get_mean_expression(adata, train_mask)
        if train_mean is None:
            raise ValueError("No training samples found for synthetic perturbation baseline.")
        return train_mean

    def _get_train_perturbation_reference(
        self,
        train_means: Mapping[str, np.ndarray],
        target_map: Mapping[str, List[int]],
    ) -> np.ndarray:
        """Compute the generic perturbation response reference vector."""
        perts = [pert for pert in train_means.keys() if pert in target_map]
        if not perts:
            raise ValueError("No training perturbations found for target reference.")
        return np.stack([train_means[pert] for pert in perts], axis=0).mean(axis=0)

    def _get_mean_expression(
        self, adata: sc.AnnData, obs_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """Compute mean expression for a subset of observations."""
        if not obs_mask.any():
            return None
        x_sub = adata[obs_mask].X
        if sp.issparse(x_sub):
            return x_sub.mean(axis=0).A1
        return np.asarray(x_sub.mean(axis=0))

    def _build_target_map(
        self,
        conditions: Sequence[str],
        var_names: Sequence[str],
        delimiter: str,
    ) -> Dict[str, List[int]]:
        """Map conditions to target indices."""
        var_index = {name: idx for idx, name in enumerate(var_names)}
        target_map: Dict[str, List[int]] = {}
        for condition in conditions:
            if self._is_control_condition(condition):
                continue
            parts = [part for part in condition.split(delimiter) if part]
            indices = [var_index[part] for part in parts if part in var_index]
            if len(indices) != len(parts):
                continue
            target_map[condition] = indices
        return target_map

    def _targets_from_condition(
        self,
        condition: str,
        var_index: Mapping[str, int],
        delimiter: str,
    ) -> Optional[List[int]]:
        """Get target indices for a condition."""
        if self._is_control_condition(condition):
            return None
        parts = [part for part in condition.split(delimiter) if part]
        if not parts:
            return None
        indices = [var_index[part] for part in parts if part in var_index]
        if len(indices) != len(parts):
            return None
        return indices

    def _fit_c_target(
        self,
        target_reference_vec: np.ndarray,
        offset_vec: np.ndarray,
        train_means: Mapping[str, np.ndarray],
        target_map: Mapping[str, List[int]],
    ) -> float:
        """Fit c_target with fixed offset via least squares."""
        perts = [pert for pert in train_means.keys() if pert in target_map]
        y = np.stack([train_means[pert] for pert in perts], axis=0)

        target_mask = np.zeros_like(y, dtype=float)
        for idx, pert in enumerate(perts):
            targets = target_map[pert]
            target_mask[idx, targets] = target_reference_vec[targets]

        y_adj = y - offset_vec
        denom = float(np.sum(target_mask * target_mask))
        if abs(denom) < 1e-12:
            raise ValueError("Regression system ill-conditioned for synthetic perturbation baseline.")
        numer = float(np.sum(y_adj * target_mask))
        return float(numer / denom)

    def _predict(
        self,
        target_reference_vec: np.ndarray,
        target_indices: Sequence[int],
        c_target: float,
        b_vec: np.ndarray,
    ) -> np.ndarray:
        """Predict expression for targets."""
        pred = b_vec.copy()
        for idx in target_indices:
            pred[idx] += c_target * target_reference_vec[idx]
        return pred

    def _is_control_condition(self, condition: str) -> bool:
        """Check if a condition label corresponds to control."""
        lower = condition.lower()
        return "ctrl" in lower or "control" in lower


class SyntheticPerturbationOracleModel(SyntheticPerturbationModel):
    """Oracle variant of SyntheticPerturbationModel.

    For each test condition the non-target genes are set to the training-split
    mean (``b``), while the target genes are set to the **true test-split
    perturbation mean** (oracle: peeks at held-out test cells).  If no test
    cells are available for a given perturbation the model falls back to the
    trained ``SyntheticPerturbationModel`` prediction.
    """

    def predict(
        self,
        data_manager: "DataManager",
        test_conditions: List[str],
        split_name: str,
        **kwargs: Any,
    ) -> sc.AnnData:
        """Predict using oracle test-split means for target genes."""
        _ = kwargs
        adata = data_manager.adata if data_manager.adata is not None else data_manager.load_dataset()

        train_conditions = self._get_training_conditions(adata, split_name)
        cond_to_pert = self._get_condition_to_perturbation(adata)
        train_perturbations = [cond_to_pert.get(c, c) for c in train_conditions]
        delimiter = self._infer_delimiter(train_perturbations)
        combo_only = self._should_use_combo_only(train_perturbations, delimiter)
        train_conditions = self._filter_combo_conditions(
            train_conditions, cond_to_pert, delimiter, combo_only
        )
        train_perturbations = [cond_to_pert.get(c, c) for c in train_conditions]

        var_names = list(adata.var_names)
        var_index = {name: idx for idx, name in enumerate(var_names)}

        b_vec = self._get_train_dataset_mean(adata, split_name)

        # Fallback: fit c_target in case test mean is unavailable for some perturbation
        train_means = self._get_perturbation_means(adata, train_perturbations, split_name)
        target_map: Dict[str, List[int]] = {}
        for pert_label in train_means.keys():
            targets = self._targets_from_condition(pert_label, var_index, delimiter)
            if targets is not None:
                target_map[pert_label] = targets
        if target_map:
            target_reference_vec = self._get_train_perturbation_reference(train_means, target_map)
            c_target = self._fit_c_target(target_reference_vec, b_vec, train_means, target_map)
        else:
            target_reference_vec = b_vec.copy()
            c_target = 0.0

        # Oracle: compute test-split perturbation means
        test_conditions_filtered = self._filter_combo_conditions(
            test_conditions, cond_to_pert, delimiter, combo_only
        )
        test_perts = list({
            cond_to_pert.get(c, c) for c in test_conditions_filtered
            if not self._is_control_condition(cond_to_pert.get(c, c))
        })
        test_means = self._get_test_perturbation_means(adata, test_perts, split_name)

        predictions: Dict[str, np.ndarray] = {}
        for condition in test_conditions_filtered:
            pert_label = cond_to_pert.get(condition, condition)
            targets = self._targets_from_condition(pert_label, var_index, delimiter)
            if targets is None:
                continue
            if pert_label in test_means:
                pred = self._predict_oracle(test_means[pert_label], targets, b_vec)
            else:
                # fallback to trained prediction
                pred = self._predict(target_reference_vec, targets, c_target, b_vec)
            predictions[condition] = pred

        if not predictions:
            raise ValueError(
                "No valid test conditions found for synthetic perturbation oracle. "
                "Check that perturbation tokens exist in adata.var_names."
            )

        pred_adata = self.create_predictions_adata(predictions, var_names)
        pred_adata.uns["synthetic_perturbation_oracle_params"] = {
            "b": b_vec,
            "offset_source": "train_dataset_mean",
            "target_source": "test_perturbation_mean",
            "delimiter": delimiter,
            "n_oracle_conditions": len(test_means),
        }
        return pred_adata

    def _get_test_perturbation_means(
        self,
        adata: sc.AnnData,
        perturbations: Sequence[str],
        split_name: str,
    ) -> Dict[str, np.ndarray]:
        """Compute perturbation mean expressions from the test split."""
        if "perturbation" in adata.obs:
            pert_series = adata.obs["perturbation"].astype(str)
        else:
            pert_series = adata.obs["condition"].astype(str)
        means: Dict[str, np.ndarray] = {}
        for pert in sorted(set(perturbations)):
            mask = (adata.obs[split_name] == "test") & (pert_series == pert)
            mean_vec = self._get_mean_expression(adata, mask.to_numpy())
            if mean_vec is not None:
                means[pert] = mean_vec
        return means

    def _predict_oracle(
        self,
        test_mean: np.ndarray,
        target_indices: Sequence[int],
        b_vec: np.ndarray,
    ) -> np.ndarray:
        """Build prediction: b_vec everywhere, test_mean at target indices."""
        pred = b_vec.copy()
        for idx in target_indices:
            pred[idx] = test_mean[idx]
        return pred
