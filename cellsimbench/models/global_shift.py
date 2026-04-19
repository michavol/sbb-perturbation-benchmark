"""Global shift baseline for CellSimBench."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import scanpy as sc
import scipy.sparse as sp

from .base import BuiltinModel
from ..core.data_manager import DataManager

log = logging.getLogger(__name__)


class GlobalShiftModel(BuiltinModel):
    """Global shift baseline.

    This model fits a shared offset vector ``b`` across perturbations:
        X_pred = X_0 + b
    """

    def __init__(self, model_config: Dict[str, Any]) -> None:
        """Initialize the global shift baseline.

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
        """Predict condition means using the global shift baseline.

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

        ctrl_mean = self._get_control_mean(adata)
        train_means = self._get_perturbation_means(adata, train_perturbations, split_name)
        if not train_means:
            raise ValueError("No training perturbations found for global shift baseline.")

        b_vec = self._fit_parameters(ctrl_mean, train_means)

        predictions: Dict[str, np.ndarray] = {}
        test_conditions = self._filter_combo_conditions(
            test_conditions, cond_to_pert, delimiter, combo_only
        )
        for condition in test_conditions:
            predictions[condition] = ctrl_mean + b_vec
        if not predictions:
            raise ValueError("No valid test conditions found for global shift baseline.")

        pred_adata = self.create_predictions_adata(predictions, list(adata.var_names))
        pred_adata.uns["global_shift_params"] = {
            "b": b_vec,
            "delimiter": delimiter,
            "train_conditions": sorted(train_means.keys()),
        }
        return pred_adata

    def _fit_parameters(
        self,
        ctrl_mean: np.ndarray,
        train_means: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """Fit the shared offset vector b.

        Args:
            ctrl_mean: Control mean expression.
            train_means: Mapping from perturbation to mean expression.

        Returns:
            Learned offset vector.
        """
        y = np.stack(list(train_means.values()), axis=0)
        residuals = y - ctrl_mean
        return residuals.mean(axis=0)

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
        """Collect training conditions."""
        train_mask = adata.obs[split_name] == "train"
        train_conditions = adata.obs.loc[train_mask, "condition"].unique().tolist()
        return sorted(train_conditions)

    def _get_control_mean(self, adata: sc.AnnData) -> np.ndarray:
        """Compute control mean expression using all controls."""
        control_mask = adata.obs["condition"].str.contains("ctrl|control", case=False, regex=True)
        if not control_mask.any():
            raise ValueError("No control samples found for global shift baseline.")
        mean_vec = self._get_mean_expression(adata, control_mask.to_numpy())
        if mean_vec is None:
            raise ValueError("Control mean expression is undefined for global shift baseline.")
        return mean_vec

    def _get_perturbation_means(
        self,
        adata: sc.AnnData,
        perturbations: Sequence[str],
        split_name: str,
    ) -> Dict[str, np.ndarray]:
        """Compute perturbation mean expressions (train split only)."""
        if "perturbation" in adata.obs:
            pert_series = adata.obs["perturbation"].astype(str)
        else:
            pert_series = adata.obs["condition"].astype(str)
        means: Dict[str, np.ndarray] = {}
        for pert in sorted(set(perturbations)):
            mask = (adata.obs[split_name] == "train") & (pert_series == pert)
            mean_vec = self._get_mean_expression(adata, mask.to_numpy())
            if mean_vec is not None:
                means[pert] = mean_vec
        return means

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
