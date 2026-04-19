"""
LSE Regression Model for CellSimBench.

Implements Linear Combinatorial Extrapolation (LSE) baselines as first-class models.
Supports shared-offset (perturbation-agnostic) and perturbation-specific variants
with a shared global epsilon offset.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import logging

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from sklearn.linear_model import LinearRegression

from .base import BuiltinModel
from ..core.data_manager import DataManager

log = logging.getLogger(__name__)


class LSERegressionModel(BuiltinModel):
    """LSE regression with shared epsilon offset.

    Configuration options:
        variant: "shared" (perturbation-agnostic), "specific" (pair-specific),
            or "hybrid" (train-pair-specific with agnostic fallback).
        fit_mode: "current" (legacy alternating fit) or "joint_ls" (joint least
            squares over coefficients and epsilon for shared/specific variants).
    """

    def __init__(self, model_config: Dict[str, Any]) -> None:
        """Initialize the LSE regression model.

        Args:
            model_config: Model configuration dictionary.

        Raises:
            ValueError: If the variant is not supported.
        """
        super().__init__(model_config)
        variant = model_config.get("hyperparameters", {}).get("variant", "shared")
        if variant == "universal":
            variant = "shared"
        self.variant: str = variant
        if self.variant not in {"shared", "specific", "hybrid"}:
            raise ValueError(f"Unknown variant: {self.variant}")
        fit_mode = model_config.get("hyperparameters", {}).get("fit_mode", "current")
        self.fit_mode: str = str(fit_mode)
        if self.fit_mode not in {"current", "joint_ls"}:
            raise ValueError(f"Unknown fit_mode: {self.fit_mode}")

    def predict(
        self,
        data_manager: DataManager,
        test_conditions: List[str],
        split_name: str,
        **kwargs: Any,
    ) -> sc.AnnData:
        """Generate predictions using LSE logic.

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

        train_mask = adata.obs[split_name].isin(["train", "val"])
        train_conds = adata.obs.loc[train_mask, "condition"].unique()

        ctrl_mask_all = self._get_control_mask(adata.obs["condition"])
        ctrl_mean_all = self._get_mean_expression(adata, ctrl_mask_all)
        if ctrl_mean_all is None:
            raise ValueError("Could not compute control mean from all control samples.")

        single_perts, parsed_train_pairs = self._parse_train_conditions(train_conds)
        parsed_test_pairs = self._parse_test_pairs(test_conditions)

        train_means = self._get_condition_means(
            adata,
            list(single_perts) + list(parsed_train_pairs.keys()),
            split_name,
            ("train", "val"),
        )

        d_vectors: Dict[str, np.ndarray] = {}
        for pert in single_perts:
            if pert in train_means:
                d_vectors[pert] = train_means[pert] - ctrl_mean_all

        if self.variant == "shared":
            if self.fit_mode == "joint_ls":
                c_val, epsilon = self._fit_shared_joint_ls(
                    d_vectors=d_vectors,
                    train_means=train_means,
                    parsed_pairs=parsed_train_pairs,
                    ctrl_mean_ref=ctrl_mean_all,
                    n_vars=adata.n_vars,
                )
            else:
                c_val, epsilon = self._fit_shared(
                    d_vectors,
                    train_means,
                    parsed_train_pairs,
                    ctrl_mean_all,
                    adata.n_vars,
                )
            diag = {
                "variant": "shared",
                "fit_mode": self.fit_mode,
                "c": float(c_val),
                "epsilon": epsilon,
                "epsilon_norm": float(np.linalg.norm(epsilon)),
            }
        elif self.variant == "specific":
            if self.fit_mode == "joint_ls":
                c1_vals, c2_vals, epsilon = self._fit_specific_joint_ls(
                    adata=adata,
                    d_vectors=d_vectors,
                    parsed_pairs=parsed_test_pairs,
                    test_conditions=test_conditions,
                    ctrl_mean_ref=ctrl_mean_all,
                    split_name=split_name,
                )
            else:
                c1_vals, c2_vals, epsilon = self._fit_specific(
                    adata=adata,
                    d_vectors=d_vectors,
                    parsed_pairs=parsed_test_pairs,
                    test_conditions=test_conditions,
                    ctrl_mean_ref=ctrl_mean_all,
                    split_name=split_name,
                )
            c_sum_abs = [
                float(abs(c1_vals[key] + c2_vals[key])) for key in c1_vals.keys()
            ]
            diag = {
                "variant": "specific",
                "fit_mode": self.fit_mode,
                "c_sum_abs": np.array(c_sum_abs, dtype=float),
                "epsilon": epsilon,
                "epsilon_norm": float(np.linalg.norm(epsilon)),
            }
        else:
            c_shared, epsilon_shared = self._fit_shared(
                d_vectors,
                train_means,
                parsed_train_pairs,
                ctrl_mean_all,
                adata.n_vars,
            )
            c1_vals, c2_vals, epsilon = self._fit_specific_train(
                adata=adata,
                d_vectors=d_vectors,
                parsed_pairs=parsed_train_pairs,
                train_conditions=train_conds,
                ctrl_mean_ref=ctrl_mean_all,
                split_name=split_name,
            )
            c1_by_pert, c2_by_pert = self._aggregate_pair_coefficients(
                parsed_train_pairs, c1_vals, c2_vals
            )
            c_sum_abs = [
                float(abs(c1_vals[key] + c2_vals[key])) for key in c1_vals.keys()
            ]
            diag = {
                "variant": "hybrid",
                "c_shared": float(c_shared),
                "c_sum_abs": np.array(c_sum_abs, dtype=float),
                "epsilon": epsilon_shared,
                "epsilon_norm": float(np.linalg.norm(epsilon_shared)),
            }

        predictions: Dict[str, np.ndarray] = {}
        for condition, pair in parsed_test_pairs.items():
            p1, p2 = pair
            if p1 not in d_vectors or p2 not in d_vectors:
                continue
            d1, d2 = d_vectors[p1], d_vectors[p2]
            if self.variant == "shared":
                pred = c_val * (d1 + d2) + ctrl_mean_all + epsilon
            elif self.variant == "specific":
                if condition in c1_vals:
                    pred = (
                        c1_vals[condition] * d1
                        + c2_vals[condition] * d2
                        + ctrl_mean_all
                        + epsilon
                    )
                else:
                    pred = d1 + d2 + ctrl_mean_all + epsilon
            else:
                if condition in c1_vals:
                    c1_use = c1_vals[condition]
                    c2_use = c2_vals[condition]
                else:
                    c1_use = c1_by_pert.get(p1)
                    c2_use = c2_by_pert.get(p2)
                    if c1_use is None:
                        c1_use = c_shared
                    if c2_use is None:
                        c2_use = c_shared
                pred = c1_use * d1 + c2_use * d2 + ctrl_mean_all + epsilon_shared
            predictions[condition] = pred

        pred_adata = self.create_predictions_adata(predictions, list(adata.var_names))
        pred_adata.uns["lse_diagnostics"] = diag
        return pred_adata

    def _parse_train_conditions(
        self, train_conditions: Sequence[str]
    ) -> Tuple[set, Dict[str, Tuple[str, str]]]:
        """Parse train conditions into single perturbations and pairs.

        Args:
            train_conditions: Training condition labels.

        Returns:
            Tuple of (single_perts, parsed_pairs).
        """
        single_perts: set = set()
        parsed_pairs: Dict[str, Tuple[str, str]] = {}
        for condition in train_conditions:
            if self._is_control_condition(condition):
                continue
            parts = condition.split("+")
            if len(parts) == 1:
                single_perts.add(condition)
                continue
            if len(parts) == 2:
                p1, p2 = parts
                if self._is_control_condition(p1):
                    single_perts.add(p2)
                elif self._is_control_condition(p2):
                    single_perts.add(p1)
                else:
                    parsed_pairs[condition] = (p1, p2)
        return single_perts, parsed_pairs

    def _parse_test_pairs(
        self, test_conditions: Sequence[str]
    ) -> Dict[str, Tuple[str, str]]:
        """Parse test conditions into perturbation pairs.

        Args:
            test_conditions: Conditions to predict.

        Returns:
            Mapping from condition to (p1, p2) pair.
        """
        parsed_pairs: Dict[str, Tuple[str, str]] = {}
        for condition in test_conditions:
            parts = condition.split("+")
            if len(parts) == 2:
                parsed_pairs[condition] = (parts[0], parts[1])
        return parsed_pairs

    def _fit_shared(
        self,
        d_vectors: Mapping[str, np.ndarray],
        train_means: Mapping[str, np.ndarray],
        parsed_pairs: Mapping[str, Tuple[str, str]],
        ctrl_mean_ref: np.ndarray,
        n_vars: int,
    ) -> Tuple[float, np.ndarray]:
        """Fit shared scalar and epsilon using training pairs.

        Args:
            d_vectors: Mapping of single perturbation D-vectors.
            train_means: Mapping of condition means.
            parsed_pairs: Mapping of train pair conditions.
            ctrl_mean_ref: Control mean used for delta calculation.
            n_vars: Number of genes.

        Returns:
            Tuple of (c_value, epsilon_vector).
        """
        x_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        for cond, (p1, p2) in parsed_pairs.items():
            if cond in train_means and p1 in d_vectors and p2 in d_vectors:
                d_sum = d_vectors[p1] + d_vectors[p2]
                target = train_means[cond] - ctrl_mean_ref
                if np.isfinite(d_sum).all() and np.isfinite(target).all():
                    x_list.append(d_sum)
                    y_list.append(target)

        c_final = 1.0
        eps_final = np.zeros(n_vars)
        if x_list:
            x_flat = np.array(x_list).reshape(-1, 1)
            y_flat = np.array(y_list).flatten()
            mask = np.isfinite(x_flat).flatten() & np.isfinite(y_flat)
            if int(mask.sum()) > 0:
                c_final = (
                    LinearRegression(fit_intercept=False)
                    .fit(x_flat[mask], y_flat[mask])
                    .coef_[0]
                )
                preds = c_final * np.array(x_list)
                resids = np.array(y_list) - preds
                eps_final = np.nan_to_num(np.nanmean(resids, axis=0))

        return float(c_final), eps_final

    def _fit_shared_joint_ls(
        self,
        d_vectors: Mapping[str, np.ndarray],
        train_means: Mapping[str, np.ndarray],
        parsed_pairs: Mapping[str, Tuple[str, str]],
        ctrl_mean_ref: np.ndarray,
        n_vars: int,
    ) -> Tuple[float, np.ndarray]:
        """Joint least-squares fit for shared c and global epsilon.

        Solves y_{k,g} = c * dsum_{k,g} + epsilon_g for all train pairs k, genes g.
        """
        dsum_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        for cond, (p1, p2) in parsed_pairs.items():
            if cond in train_means and p1 in d_vectors and p2 in d_vectors:
                dsum = d_vectors[p1] + d_vectors[p2]
                target = train_means[cond] - ctrl_mean_ref
                if np.isfinite(dsum).all() and np.isfinite(target).all():
                    dsum_list.append(dsum)
                    y_list.append(target)

        if not dsum_list:
            return 1.0, np.zeros(n_vars, dtype=float)

        dsum_mat = np.asarray(dsum_list, dtype=float)
        y_mat = np.asarray(y_list, dtype=float)
        n_pairs, n_genes = dsum_mat.shape
        n_obs = n_pairs * n_genes

        row = np.empty(2 * n_obs, dtype=np.int64)
        col = np.empty(2 * n_obs, dtype=np.int64)
        data = np.empty(2 * n_obs, dtype=float)
        y_vec = y_mat.reshape(-1)
        obs_idx = np.arange(n_obs, dtype=np.int64)
        gene_idx = np.tile(np.arange(n_genes, dtype=np.int64), n_pairs)

        row[:n_obs] = obs_idx
        col[:n_obs] = 0
        data[:n_obs] = dsum_mat.reshape(-1)

        row[n_obs:] = obs_idx
        col[n_obs:] = 1 + gene_idx
        data[n_obs:] = 1.0

        design = sp.coo_matrix((data, (row, col)), shape=(n_obs, 1 + n_genes)).tocsr()
        sol = lsqr(design, y_vec, atol=1e-10, btol=1e-10, iter_lim=2000)[0]
        c_final = float(sol[0])
        eps_final = np.asarray(sol[1:], dtype=float)
        return c_final, eps_final

    def _fit_specific(
        self,
        adata: sc.AnnData,
        d_vectors: Mapping[str, np.ndarray],
        parsed_pairs: Mapping[str, Tuple[str, str]],
        test_conditions: Sequence[str],
        ctrl_mean_ref: np.ndarray,
        split_name: str,
    ) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray]:
        """Fit pair-specific coefficients and shared epsilon.

        Args:
            adata: AnnData with expression values.
            d_vectors: Mapping of single perturbation D-vectors.
            parsed_pairs: Mapping of test pair conditions.
            test_conditions: Conditions to predict.
            ctrl_mean_ref: Control mean used for delta calculation.
            split_name: Split column name.

        Returns:
            Tuple of (c1_vals, c2_vals, epsilon).
        """
        has_tech_splits = "tech_dup_split" in adata.obs
        rep_b = "second_half" if has_tech_splits else None
        test_means = self._get_condition_means(
            adata, list(test_conditions), split_name, "test", tech_split=rep_b
        )

        c1_vals: Dict[str, float] = {}
        c2_vals: Dict[str, float] = {}
        resids: List[np.ndarray] = []
        valid_pairs: List[str] = []

        for cond, (p1, p2) in parsed_pairs.items():
            if cond in test_means and p1 in d_vectors and p2 in d_vectors:
                y_vec = test_means[cond] - ctrl_mean_ref
                d1, d2 = d_vectors[p1], d_vectors[p2]
                params = self._fit_pair(d1, d2, y_vec)
                if params is None:
                    continue
                c1, c2 = params
                c1_vals[cond] = c1
                c2_vals[cond] = c2
                resids.append(y_vec - (c1 * d1 + c2 * d2))
                valid_pairs.append(cond)

        eps_final = np.mean(resids, axis=0) if resids else np.zeros(adata.n_vars)

        for cond in valid_pairs:
            y_vec = test_means[cond] - ctrl_mean_ref - eps_final
            p1, p2 = parsed_pairs[cond]
            params = self._fit_pair(d_vectors[p1], d_vectors[p2], y_vec)
            if params is not None:
                c1_vals[cond], c2_vals[cond] = params

        return c1_vals, c2_vals, eps_final

    def _fit_specific_joint_ls(
        self,
        adata: sc.AnnData,
        d_vectors: Mapping[str, np.ndarray],
        parsed_pairs: Mapping[str, Tuple[str, str]],
        test_conditions: Sequence[str],
        ctrl_mean_ref: np.ndarray,
        split_name: str,
    ) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray]:
        """Joint least-squares fit for pair-specific c1/c2 and shared epsilon."""
        has_tech_splits = "tech_dup_split" in adata.obs
        rep_b = "second_half" if has_tech_splits else None
        test_means = self._get_condition_means(
            adata, list(test_conditions), split_name, "test", tech_split=rep_b
        )

        pair_keys: List[str] = []
        d1_list: List[np.ndarray] = []
        d2_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        for cond, (p1, p2) in parsed_pairs.items():
            if cond in test_means and p1 in d_vectors and p2 in d_vectors:
                y_vec = test_means[cond] - ctrl_mean_ref
                d1, d2 = d_vectors[p1], d_vectors[p2]
                if np.isfinite(y_vec).all() and np.isfinite(d1).all() and np.isfinite(d2).all():
                    pair_keys.append(cond)
                    d1_list.append(d1)
                    d2_list.append(d2)
                    y_list.append(y_vec)

        if not pair_keys:
            return {}, {}, np.zeros(adata.n_vars, dtype=float)

        d1_mat = np.asarray(d1_list, dtype=float)
        d2_mat = np.asarray(d2_list, dtype=float)
        y_mat = np.asarray(y_list, dtype=float)
        n_pairs, n_genes = d1_mat.shape
        n_obs = n_pairs * n_genes
        n_params = 2 * n_pairs + n_genes

        row = np.empty(3 * n_obs, dtype=np.int64)
        col = np.empty(3 * n_obs, dtype=np.int64)
        data = np.empty(3 * n_obs, dtype=float)
        y_vec = y_mat.reshape(-1)
        obs_idx = np.arange(n_obs, dtype=np.int64)
        pair_idx = np.repeat(np.arange(n_pairs, dtype=np.int64), n_genes)
        gene_idx = np.tile(np.arange(n_genes, dtype=np.int64), n_pairs)

        row[:n_obs] = obs_idx
        col[:n_obs] = pair_idx
        data[:n_obs] = d1_mat.reshape(-1)

        row[n_obs:2 * n_obs] = obs_idx
        col[n_obs:2 * n_obs] = n_pairs + pair_idx
        data[n_obs:2 * n_obs] = d2_mat.reshape(-1)

        row[2 * n_obs:] = obs_idx
        col[2 * n_obs:] = 2 * n_pairs + gene_idx
        data[2 * n_obs:] = 1.0

        design = sp.coo_matrix((data, (row, col)), shape=(n_obs, n_params)).tocsr()
        sol = lsqr(design, y_vec, atol=1e-10, btol=1e-10, iter_lim=4000)[0]

        c1_raw = np.asarray(sol[:n_pairs], dtype=float)
        c2_raw = np.asarray(sol[n_pairs:2 * n_pairs], dtype=float)
        eps_final = np.asarray(sol[2 * n_pairs:], dtype=float)
        c1_vals = {pair_keys[i]: float(c1_raw[i]) for i in range(n_pairs)}
        c2_vals = {pair_keys[i]: float(c2_raw[i]) for i in range(n_pairs)}
        return c1_vals, c2_vals, eps_final

    def _fit_specific_train(
        self,
        adata: sc.AnnData,
        d_vectors: Mapping[str, np.ndarray],
        parsed_pairs: Mapping[str, Tuple[str, str]],
        train_conditions: Sequence[str],
        ctrl_mean_ref: np.ndarray,
        split_name: str,
    ) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray]:
        """Fit pair-specific coefficients and shared epsilon using training pairs.

        Args:
            adata: AnnData with expression values.
            d_vectors: Mapping of single perturbation D-vectors.
            parsed_pairs: Mapping of train pair conditions.
            train_conditions: Training condition labels.
            ctrl_mean_ref: Control mean used for delta calculation.
            split_name: Split column name.

        Returns:
            Tuple of (c1_vals, c2_vals, epsilon).
        """
        train_means = self._get_condition_means(
            adata, list(train_conditions), split_name, ("train", "val")
        )

        c1_vals: Dict[str, float] = {}
        c2_vals: Dict[str, float] = {}
        resids: List[np.ndarray] = []
        valid_pairs: List[str] = []

        for cond, (p1, p2) in parsed_pairs.items():
            if cond in train_means and p1 in d_vectors and p2 in d_vectors:
                y_vec = train_means[cond] - ctrl_mean_ref
                d1, d2 = d_vectors[p1], d_vectors[p2]
                params = self._fit_pair(d1, d2, y_vec)
                if params is None:
                    continue
                c1, c2 = params
                c1_vals[cond] = c1
                c2_vals[cond] = c2
                resids.append(y_vec - (c1 * d1 + c2 * d2))
                valid_pairs.append(cond)

        eps_final = np.mean(resids, axis=0) if resids else np.zeros(adata.n_vars)

        for cond in valid_pairs:
            y_vec = train_means[cond] - ctrl_mean_ref - eps_final
            p1, p2 = parsed_pairs[cond]
            params = self._fit_pair(d_vectors[p1], d_vectors[p2], y_vec)
            if params is not None:
                c1_vals[cond], c2_vals[cond] = params

        return c1_vals, c2_vals, eps_final

    def _aggregate_pair_coefficients(
        self,
        parsed_pairs: Mapping[str, Tuple[str, str]],
        c1_vals: Mapping[str, float],
        c2_vals: Mapping[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Aggregate per-perturbation coefficients from pair-specific fits.

        Args:
            parsed_pairs: Mapping from condition to (p1, p2) pair.
            c1_vals: Pair-specific coefficients for p1.
            c2_vals: Pair-specific coefficients for p2.

        Returns:
            Tuple of (c1_by_pert, c2_by_pert).
        """
        c1_buckets: Dict[str, List[float]] = {}
        c2_buckets: Dict[str, List[float]] = {}

        for cond, (p1, p2) in parsed_pairs.items():
            if cond in c1_vals:
                c1_buckets.setdefault(p1, []).append(c1_vals[cond])
            if cond in c2_vals:
                c2_buckets.setdefault(p2, []).append(c2_vals[cond])

        c1_by_pert = {pert: float(np.mean(vals)) for pert, vals in c1_buckets.items()}
        c2_by_pert = {pert: float(np.mean(vals)) for pert, vals in c2_buckets.items()}
        return c1_by_pert, c2_by_pert

    def _fit_pair(
        self, d1: np.ndarray, d2: np.ndarray, y: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Fit pair-specific coefficients for a single condition.

        Args:
            d1: First perturbation D-vector.
            d2: Second perturbation D-vector.
            y: Target delta vector.

        Returns:
            Tuple of (c1, c2) or None if fitting fails.
        """
        if not (np.isfinite(y).all() and np.isfinite(d1).all() and np.isfinite(d2).all()):
            return None
        x = np.column_stack([d1, d2])
        if x.shape[0] == 0:
            return None
        try:
            model = LinearRegression(fit_intercept=False).fit(x, y)
            return float(model.coef_[0]), float(model.coef_[1])
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to fit pair coefficients: %s", exc)
            return None

    def _get_control_mask(self, conditions: Sequence[str]) -> np.ndarray:
        """Return a boolean mask for control conditions.

        Args:
            conditions: Condition labels.

        Returns:
            Boolean mask for control rows.
        """
        cond_series = pd.Series(list(conditions), dtype="string")
        mask = cond_series.str.contains("ctrl", case=False, na=False) | cond_series.str.contains(
            "control", case=False, na=False
        )
        return mask.to_numpy(dtype=bool)

    def _is_control_condition(self, condition: str) -> bool:
        """Check if a condition label corresponds to control.

        Args:
            condition: Condition label.

        Returns:
            True if the condition represents a control sample.
        """
        lower = condition.lower()
        return "ctrl" in lower or "control" in lower

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

    def _get_condition_means(
        self,
        adata: sc.AnnData,
        conditions: Sequence[str],
        split_col: str,
        split_val: Sequence[str] | str,
        tech_split: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Get mean expression for requested conditions within a split.

        Args:
            adata: AnnData with expression values.
            conditions: Target conditions.
            split_col: Split column name.
            split_val: Split value(s) to filter.
            tech_split: Optional technical split value.

        Returns:
            Mapping from condition to mean expression vector.
        """
        means: Dict[str, np.ndarray] = {}
        obs = adata.obs
        if isinstance(split_val, str):
            base_mask = obs[split_col] == split_val
        else:
            base_mask = obs[split_col].isin(list(split_val))
        if tech_split is not None:
            base_mask &= obs["tech_dup_split"] == tech_split

        present_conds = obs.loc[base_mask, "condition"].unique()
        target_conds = set(conditions).intersection(present_conds)
        for cond in target_conds:
            mask = base_mask & (obs["condition"] == cond)
            res = self._get_mean_expression(adata, mask)
            if res is not None:
                means[cond] = res
        return means
