"""
Built-in models for CellSimBench.

Note: Most baseline functionality is handled directly by BaselineRunner.
This module is kept for any future non-baseline built-in models.
"""

from typing import Dict, List
import numpy as np
import scanpy as sc
from .base import BuiltinModel
from .genept_embedding_nn import GeneptEmbeddingNearestNeighborModel
from .lse_regression import LSERegressionModel
from .onehot_linear_regression import OneHotLinearRegressionModel
from .onehot_linear_wmse_regression import OneHotLinearWMSERegressionModel
from .synthetic_perturbation import SyntheticPerturbationModel, SyntheticPerturbationOracleModel
from .global_shift import GlobalShiftModel

# Registry of built-in models
BUILTIN_MODELS = {
    'lse_regression': LSERegressionModel,
    'genept_embedding_nn': GeneptEmbeddingNearestNeighborModel,
    'onehot_linear': OneHotLinearRegressionModel,
    'onehot_linear_wmse': OneHotLinearWMSERegressionModel,
    'synthetic_perturbation': SyntheticPerturbationModel,
    'synthetic_perturbation_oracle': SyntheticPerturbationOracleModel,
    'global_shift': GlobalShiftModel,
}

def get_builtin_model(model_name: str):
    """Get a built-in model by name."""
    if model_name not in BUILTIN_MODELS:
        raise ValueError(f"Unknown built-in model: {model_name}. Available: {list(BUILTIN_MODELS.keys())}")
    return BUILTIN_MODELS[model_name] 