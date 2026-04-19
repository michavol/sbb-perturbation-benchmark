"""
One-hot linear regression with weighted MSE loss (oracle baseline).

Identical to OneHotLinearRegressionModel but fits per-gene weighted least
squares using the same DEG-based weights the benchmark uses for evaluation.
Because those weights are derived from ground-truth data this model is
considered an oracle.

Control conditions are excluded from training because no DEG weights exist
for them (they are the reference).
"""

from typing import Any, Dict, List

import logging

import numpy as np
import scanpy as sc
from sklearn.linear_model import LinearRegression

from .onehot_linear_regression import OneHotLinearRegressionModel
from ..core.data_manager import DataManager

log = logging.getLogger(__name__)


class OneHotLinearWMSERegressionModel(OneHotLinearRegressionModel):
    """One-hot linear regression trained with weighted MSE loss.

    Per-gene weights come from ``data_manager.get_deg_weights`` (same source
    the benchmark metrics use).  For each gene the model solves a separate
    weighted least-squares problem, giving DEG-relevant genes more influence
    on the fitted coefficients.

    Control conditions are excluded from training (no DEG weights exist for
    them).  The intercept is estimated from perturbation conditions only.
    """

    def predict(
        self,
        data_manager: DataManager,
        test_conditions: List[str],
        split_name: str,
        **kwargs: Any,
    ) -> sc.AnnData:
        _ = kwargs
        adata = data_manager.adata if data_manager.adata is not None else data_manager.load_dataset()

        perturbations = self._get_unique_perturbations(adata.obs["condition"])
        pert_to_idx = {pert: idx for idx, pert in enumerate(perturbations)}

        train_conditions = self._get_training_conditions(adata, split_name)
        train_means = self._get_condition_means(adata, train_conditions, split_name)

        x_train: List[np.ndarray] = []
        y_train: List[np.ndarray] = []
        kept_conditions: List[str] = []
        for condition, mean_vec in train_means.items():
            if self._is_control_condition(condition):
                continue
            vec = self._condition_to_vector(condition, pert_to_idx)
            if vec is None:
                continue
            x_train.append(vec)
            y_train.append(mean_vec)
            kept_conditions.append(condition)

        if not x_train:
            raise ValueError("No training data available for one-hot wMSE linear regression.")

        X = np.vstack(x_train)
        Y = np.vstack(y_train)
        n_conditions, n_genes = Y.shape
        n_features = X.shape[1]
        gene_names = list(adata.var_names)

        W = self._build_weight_matrix(data_manager, adata, kept_conditions, gene_names)

        coefs = np.zeros((n_genes, n_features))
        intercepts = np.zeros(n_genes)

        for g in range(n_genes):
            w_g = W[:, g]
            if np.sum(w_g) < 1e-12:
                raise ValueError(
                    f"All DEG weights are zero for gene '{gene_names[g]}' across "
                    f"all {n_conditions} training conditions. Cannot fit weighted "
                    f"least squares without weights (deg_weight_source="
                    f"'{data_manager.deg_weight_source}')."
                )
            lr = LinearRegression(fit_intercept=True)
            lr.fit(X, Y[:, g], sample_weight=w_g)
            coefs[g] = lr.coef_
            intercepts[g] = lr.intercept_

        predictions: Dict[str, np.ndarray] = {}
        for condition in test_conditions:
            vec = self._condition_to_vector(condition, pert_to_idx)
            if vec is None:
                continue
            predictions[condition] = vec @ coefs.T + intercepts

        pred_adata = self.create_predictions_adata(predictions, gene_names)
        pred_adata.uns["onehot_linear_params"] = {
            "perturbations": perturbations,
            "coef": coefs.T,
            "intercept": intercepts,
        }
        return pred_adata

    def _build_weight_matrix(
        self,
        data_manager: DataManager,
        adata: sc.AnnData,
        conditions: List[str],
        gene_names: List[str],
    ) -> np.ndarray:
        """Build a (n_conditions, n_genes) weight matrix from DEG weights.

        For each condition the weights are averaged across all covariates
        present in the data.  All conditions passed here must be non-control
        (controls are excluded upstream).

        Raises if no DEG weights are available for a condition.
        """
        covariate_key = data_manager.config["covariate_key"]
        all_covariates = sorted(adata.obs[covariate_key].unique().tolist())
        n_genes = len(gene_names)

        W = np.zeros((len(conditions), n_genes))

        for i, condition in enumerate(conditions):
            cov_weights: List[np.ndarray] = []
            for cov in all_covariates:
                mask = (
                    (adata.obs[covariate_key] == cov)
                    & (adata.obs["condition"] == condition)
                )
                if not mask.any():
                    continue
                try:
                    w = data_manager.get_deg_weights(cov, condition, gene_names)
                except ValueError:
                    raise
                if np.sum(w) < 1e-12:
                    raise ValueError(
                        f"DEG weights for condition '{condition}' (covariate "
                        f"'{cov}') are all zero. Cannot train weighted linear "
                        f"regression without weights (deg_weight_source="
                        f"'{data_manager.deg_weight_source}')."
                    )
                cov_weights.append(w)

            if not cov_weights:
                raise ValueError(
                    f"No covariate has DEG weights for condition '{condition}'. "
                    f"Cannot train weighted linear regression "
                    f"(deg_weight_source='{data_manager.deg_weight_source}')."
                )

            W[i] = np.mean(cov_weights, axis=0)

        return W
