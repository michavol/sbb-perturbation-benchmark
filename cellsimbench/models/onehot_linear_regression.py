"""
One-hot linear regression baseline for CellSimBench.

Fits a linear model on full expression vectors using one-hot perturbation encodings.
Controls have a zero vector, singles have one active entry, and combos have two.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import logging

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression

from .base import BuiltinModel
from ..core.data_manager import DataManager

log = logging.getLogger(__name__)


class OneHotLinearRegressionModel(BuiltinModel):
    """One-hot linear regression baseline.

    Trains on controls, all singles, and train+val combos, then predicts test combos.
    """

    def __init__(self, model_config: Dict[str, Any]) -> None:
        """Initialize the one-hot linear regression baseline.

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
        """Predict condition means using a one-hot linear model.

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

        perturbations = self._get_unique_perturbations(adata.obs["condition"])
        pert_to_idx = {pert: idx for idx, pert in enumerate(perturbations)}

        train_conditions = self._get_training_conditions(adata, split_name)
        train_means = self._get_condition_means(adata, train_conditions, split_name)

        x_train: List[np.ndarray] = []
        y_train: List[np.ndarray] = []
        for condition, mean_vec in train_means.items():
            vec = self._condition_to_vector(condition, pert_to_idx)
            if vec is None:
                continue
            x_train.append(vec)
            y_train.append(mean_vec)

        if not x_train:
            raise ValueError("No training data available for one-hot linear regression.")

        model = LinearRegression(fit_intercept=True)
        model.fit(np.vstack(x_train), np.vstack(y_train))

        predictions: Dict[str, np.ndarray] = {}
        for condition in test_conditions:
            vec = self._condition_to_vector(condition, pert_to_idx)
            if vec is None:
                continue
            predictions[condition] = model.predict(vec.reshape(1, -1))[0]

        pred_adata = self.create_predictions_adata(predictions, list(adata.var_names))
        pred_adata.uns["onehot_linear_params"] = {
            "perturbations": perturbations,
            "coef": model.coef_,
            "intercept": model.intercept_,
        }
        return pred_adata

    def _get_unique_perturbations(self, conditions: Sequence[str]) -> List[str]:
        """Extract sorted unique perturbations from condition labels.

        Args:
            conditions: Condition labels.

        Returns:
            Sorted list of unique perturbation tokens.
        """
        pert_set: set = set()
        for condition in conditions:
            if self._is_control_condition(condition):
                continue
            parts = condition.split("+")
            for part in parts:
                if part:
                    pert_set.add(part)
        return sorted(pert_set)

    def _get_training_conditions(self, adata: sc.AnnData, split_name: str) -> List[str]:
        """Collect training conditions for the one-hot model.

        Args:
            adata: AnnData with expression values.
            split_name: Split column name.

        Returns:
            List of condition labels to use for training.
        """
        train_mask = adata.obs[split_name].isin(["train", "val"])
        train_conditions = adata.obs.loc[train_mask, "condition"].unique().tolist()

        control_conditions = [
            cond for cond in adata.obs["condition"].unique() if self._is_control_condition(cond)
        ]
        single_conditions = [
            cond
            for cond in adata.obs["condition"].unique()
            if "+" not in cond and not self._is_control_condition(cond)
        ]
        train_combos = [
            cond for cond in train_conditions if "+" in cond and not self._is_control_condition(cond)
        ]

        all_conditions = set(control_conditions + single_conditions + train_combos)
        return sorted(all_conditions)

    def _get_condition_means(
        self, adata: sc.AnnData, conditions: Sequence[str], split_name: str
    ) -> Dict[str, np.ndarray]:
        """Compute condition mean expressions.

        Controls and singles use all samples; combo conditions use train+val split only.

        Args:
            adata: AnnData with expression values.
            conditions: Condition labels.
            split_name: Split column name.

        Returns:
            Mapping from condition to mean expression vector.
        """
        means: Dict[str, np.ndarray] = {}
        for condition in conditions:
            if "+" in condition and not self._is_control_condition(condition):
                mask = adata.obs[split_name].isin(["train", "val"]) & (
                    adata.obs["condition"] == condition
                )
            else:
                mask = adata.obs["condition"] == condition
            mean_vec = self._get_mean_expression(adata, mask)
            if mean_vec is not None:
                means[condition] = mean_vec
        return means

    def _condition_to_vector(
        self, condition: str, pert_to_idx: Mapping[str, int]
    ) -> Optional[np.ndarray]:
        """Convert a condition label to a one-hot perturbation vector.

        Args:
            condition: Condition label.
            pert_to_idx: Mapping from perturbation name to index.

        Returns:
            One-hot perturbation vector, or None if unknown perturbation found.
        """
        vec = np.zeros(len(pert_to_idx), dtype=float)
        if self._is_control_condition(condition):
            return vec
        parts = condition.split("+")
        for part in parts:
            if part not in pert_to_idx:
                log.warning("Unknown perturbation '%s' in condition '%s'", part, condition)
                return None
            vec[pert_to_idx[part]] = 1.0
        return vec

    def _get_mean_expression(
        self, adata: sc.AnnData, obs_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """Compute mean expression for a subset of observations.

        Args:
            adata: AnnData with expression values.
            obs_mask: Boolean mask of observations.

        Returns:
            Mean expression vector, or None if mask is empty.
        """
        if not obs_mask.any():
            return None
        x_sub = adata[obs_mask].X
        if sp.issparse(x_sub):
            return x_sub.mean(axis=0).A1
        return np.asarray(x_sub.mean(axis=0))

    def _is_control_condition(self, condition: str) -> bool:
        """Check if a condition label corresponds to control.

        Args:
            condition: Condition label.

        Returns:
            True if the condition represents a control sample.
        """
        lower = condition.lower()
        return "ctrl" in lower or "control" in lower
