"""
GenePT-embedding nearest-neighbor baseline.

Builds condition embeddings from GenePT gene embeddings in adata.uns,
finds nearest train+val perturbation for each test perturbation, and
returns the neighbor's mean expression profile.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from .base import BuiltinModel
from ..core.data_manager import DataManager


class GeneptEmbeddingNearestNeighborModel(BuiltinModel):
    """Nearest-neighbor baseline using GenePT embeddings."""

    def predict(
        self,
        data_manager: DataManager,
        test_conditions: List[str],
        split_name: str,
        **kwargs: Any,
    ) -> sc.AnnData:
        """Predict test conditions from nearest train+val condition embedding."""
        _ = kwargs
        adata = data_manager.adata if data_manager.adata is not None else data_manager.load_dataset()

        hyper = self.config.get("hyperparameters", {})
        distance_metric = str(hyper.get("distance_metric", "cosine")).lower()
        requested_dim = hyper.get("embedding_dim")
        if requested_dim is not None:
            requested_dim = int(requested_dim)

        if distance_metric not in {"cosine", "euclidean"}:
            raise ValueError(
                f"Unsupported distance_metric '{distance_metric}'. Expected 'cosine' or 'euclidean'."
            )

        split_conditions = data_manager.get_perturbation_conditions(split_name)
        candidate_conditions = sorted(
            {
                cond
                for cond in split_conditions["train"] + split_conditions["val"]
                if not self._is_control_condition(cond)
            }
        )
        if not candidate_conditions:
            raise ValueError("No train+val perturbation conditions available for nearest-neighbor lookup.")

        candidate_means = self._get_condition_means(adata, candidate_conditions, split_name)
        if not candidate_means:
            raise ValueError("No train+val candidate means available for nearest-neighbor predictions.")

        genept_embeddings = self._load_genept_embeddings_from_adata_uns(adata)
        gene_names = [str(g) for g in adata.var_names]
        gene_embedding_matrix, embedding_dim = self._build_gene_embedding_matrix(
            genept_embeddings, gene_names, requested_dim
        )
        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}

        candidate_keys: List[str] = []
        candidate_vectors: List[np.ndarray] = []
        candidate_missing: Dict[str, List[str]] = {}
        candidate_order_to_indices: Dict[int, List[int]] = {}
        for condition in sorted(candidate_means.keys()):
            emb, missing = self._condition_embedding(condition, gene_to_idx, gene_embedding_matrix)
            candidate_missing[condition] = missing
            if emb is None:
                continue
            order = self._perturbation_order(condition)
            if order <= 0:
                continue
            candidate_idx = len(candidate_keys)
            candidate_keys.append(condition)
            candidate_vectors.append(emb)
            candidate_order_to_indices.setdefault(order, []).append(candidate_idx)

        if not candidate_vectors:
            raise ValueError("No candidate conditions had valid GenePT embeddings.")

        candidate_matrix = np.vstack(candidate_vectors)
        global_fallback = np.mean(np.vstack([candidate_means[c] for c in candidate_keys]), axis=0)

        predictions: Dict[str, np.ndarray] = {}
        nearest_map: Dict[str, Dict[str, Any]] = {}
        for condition in test_conditions:
            query_emb, missing = self._condition_embedding(condition, gene_to_idx, gene_embedding_matrix)
            if query_emb is None:
                predictions[condition] = global_fallback
                nearest_map[condition] = {
                    "neighbor_condition": None,
                    "similarity": None,
                    "distance_metric": distance_metric,
                    "missing_tokens": missing,
                    "fallback": "global_candidate_mean",
                }
                continue

            query_order = self._perturbation_order(condition)
            candidate_subset = candidate_order_to_indices.get(query_order, [])
            if candidate_subset:
                sub_matrix = candidate_matrix[candidate_subset]
                best_sub_idx, score = self._nearest_neighbor(query_emb, sub_matrix, distance_metric)
                best_idx = candidate_subset[best_sub_idx]
            else:
                best_idx, score = self._nearest_neighbor(query_emb, candidate_matrix, distance_metric)
            neighbor_condition = candidate_keys[best_idx]
            predictions[condition] = candidate_means[neighbor_condition]
            nearest_map[condition] = {
                "neighbor_condition": neighbor_condition,
                "similarity": float(score),
                "distance_metric": distance_metric,
                "query_order": int(query_order),
                "candidate_pool_size": int(len(candidate_subset) if candidate_subset else len(candidate_keys)),
                "missing_tokens": missing,
                "fallback": None if candidate_subset else "all_orders",
            }

        pred_adata = self.create_predictions_adata(predictions, gene_names)
        pred_adata.uns["genept_embedding_nn_diagnostics"] = {
            "embedding_source": "adata.uns.embeddings_genept",
            "distance_metric": distance_metric,
            "embedding_dim": int(embedding_dim),
            "n_candidate_conditions": len(candidate_keys),
            "n_test_conditions": len(test_conditions),
            "candidate_missing_tokens": candidate_missing,
            "candidate_order_counts": {
                str(order): int(len(indices)) for order, indices in sorted(candidate_order_to_indices.items())
            },
            "nearest_neighbors": nearest_map,
        }
        return pred_adata

    def _load_genept_embeddings_from_adata_uns(self, adata: sc.AnnData) -> pd.DataFrame:
        """Load GenePT embeddings from adata.uns['embeddings_genept'] only."""
        key = "embeddings_genept"
        if key not in adata.uns:
            raise ValueError(
                "Required embedding key 'embeddings_genept' not found in adata.uns. "
                "Please provide datasets with GenePT embeddings."
            )
        raw = adata.uns[key]
        if isinstance(raw, pd.DataFrame):
            df = raw.copy()
        elif isinstance(raw, pd.Series):
            df = raw.to_frame(name="value")
        elif isinstance(raw, dict):
            df = pd.DataFrame.from_dict(raw, orient="index")
        else:
            raise ValueError(
                "adata.uns['embeddings_genept'] has unsupported type. "
                f"Expected DataFrame/Series/dict, got {type(raw).__name__}."
            )

        if df.empty:
            raise ValueError("adata.uns['embeddings_genept'] is empty.")

        df.index = df.index.astype(str)
        df.columns = [f"{key}__{str(col)}" for col in df.columns]
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=1, how="all")
        if df.shape[1] == 0:
            raise ValueError("adata.uns['embeddings_genept'] has no numeric columns.")
        return df

    def _build_gene_embedding_matrix(
        self,
        embeddings: pd.DataFrame,
        gene_names: Sequence[str],
        requested_dim: Optional[int],
    ) -> Tuple[np.ndarray, int]:
        """Align embeddings to adata genes and return dense matrix."""
        aligned = embeddings.reindex(gene_names).fillna(0.0)
        available_dim = aligned.shape[1]
        if available_dim == 0:
            raise ValueError("GenePT embeddings have zero usable dimensions after loading.")

        if requested_dim is None:
            use_dim = available_dim
        else:
            if requested_dim <= 0:
                raise ValueError(f"embedding_dim must be positive when provided, got {requested_dim}.")
            if requested_dim > available_dim:
                raise ValueError(
                    f"Requested embedding_dim={requested_dim} exceeds available dimensions={available_dim}."
                )
            use_dim = requested_dim

        matrix = aligned.iloc[:, :use_dim].to_numpy(dtype=float, copy=False)
        return matrix, use_dim

    def _condition_embedding(
        self,
        condition: str,
        gene_to_idx: Dict[str, int],
        gene_embedding_matrix: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Build a condition embedding as mean of available perturbation gene vectors."""
        if self._is_control_condition(condition):
            return None, []

        tokens = [token.strip() for token in condition.split("+") if token.strip()]
        vectors: List[np.ndarray] = []
        missing: List[str] = []
        for token in tokens:
            idx = gene_to_idx.get(token)
            if idx is None:
                missing.append(token)
                continue
            vectors.append(gene_embedding_matrix[idx])

        if not vectors:
            return None, missing
        return np.mean(np.vstack(vectors), axis=0), missing

    def _nearest_neighbor(
        self,
        query: np.ndarray,
        candidates: np.ndarray,
        metric: str,
    ) -> Tuple[int, float]:
        """Return best candidate index and similarity score."""
        if metric == "cosine":
            sims = cosine_similarity(query.reshape(1, -1), candidates).ravel()
            best = int(np.argmax(sims))
            return best, float(sims[best])

        dists = np.linalg.norm(candidates - query.reshape(1, -1), axis=1)
        best = int(np.argmin(dists))
        return best, float(-dists[best])

    def _get_condition_means(
        self, adata: sc.AnnData, conditions: Sequence[str], split_name: str
    ) -> Dict[str, np.ndarray]:
        """Compute condition means with same split policy as onehot_linear."""
        means: Dict[str, np.ndarray] = {}
        for condition in conditions:
            if "+" in condition and not self._is_control_condition(condition):
                mask = adata.obs[split_name].isin(["train", "val"]) & (
                    adata.obs["condition"] == condition
                )
            else:
                mask = adata.obs["condition"] == condition
            mean_vec = self._get_mean_expression(adata, mask.to_numpy())
            if mean_vec is not None:
                means[condition] = mean_vec
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

    def _is_control_condition(self, condition: str) -> bool:
        """Return whether a condition is a control condition."""
        lower = condition.lower()
        return "ctrl" in lower or "control" in lower

    def _perturbation_order(self, condition: str) -> int:
        """Return number of perturbation tokens in a condition."""
        if self._is_control_condition(condition):
            return 0
        tokens = [token.strip() for token in condition.split("+") if token.strip()]
        return len(tokens)
