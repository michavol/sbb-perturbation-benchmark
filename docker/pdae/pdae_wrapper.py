"""
pDAE model wrapper for CellSimBench integration.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from cellsimbench.core.data_manager import DataManager
from cellsimbench.utils.utils import PathEncoder
from PerturbationExtrapolation.pdae.api import PDAEConfig, PDAEData, PDAEModel

log = logging.getLogger(__name__)


class PDAEWrapper:
    """Wrapper implementing CellSimBench train/predict contract for pDAE."""

    def __init__(self, config: Dict):
        self.config = config
        self.hyperparams = self.config["hyperparameters"]
        self.data_manager = DataManager(self.config)
        self.device = self._resolve_device()
        self.pert_sep = self.hyperparams.get("pert_sep", "+")
        self.pert_ctrl = self.hyperparams.get("pert_ctrl", "ctrl")

    def _resolve_device(self) -> str:
        requested = self.hyperparams.get("device")
        if requested:
            return requested
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _build_pdae_config(self) -> PDAEConfig:
        return PDAEConfig(
            seed=int(self.hyperparams.get("seed", 42)),
            model_name=str(self.hyperparams.get("model_name", "pdae")),
            batch_size=int(self.hyperparams.get("batch_size", 64)),
            learning_rate=float(self.hyperparams.get("learning_rate", 1e-4)),
            dim_z_model=int(self.hyperparams.get("dim_z_model", 5000)),
            dim_noise_model=int(self.hyperparams.get("dim_noise_model", 6)),
            sigma_noise_model=float(self.hyperparams.get("sigma_noise_model", 0.1)),
            decoder_layer_shapes=list(self.hyperparams.get("decoder_layer_shapes", [])),
            encoder_layer_shapes=list(self.hyperparams.get("encoder_layer_shapes", [])),
            weight_reconstruction_loss=float(self.hyperparams.get("weight_reconstruction_loss", 0.01)),
            weight_perturbation_energy_loss=float(self.hyperparams.get("weight_perturbation_energy_loss", 1.0)),
            weight_marginal_prior_energy_loss=float(self.hyperparams.get("weight_marginal_prior_energy_loss", 0.01)),
            weight_l21=float(self.hyperparams.get("weight_l21", 0.0)),
            update_encoder_on_reconstruction_loss=bool(
                self.hyperparams.get("update_encoder_on_reconstruction_loss", False)
            ),
            normalize_energy_loss=bool(self.hyperparams.get("normalize_energy_loss", False)),
            beta=float(self.hyperparams.get("beta", 1.0)),
            use_bias_in_perturbation_matrix=bool(self.hyperparams.get("use_bias_in_perturbation_matrix", False)),
            use_softplus_after_decoder=bool(self.hyperparams.get("use_softplus_after_decoder", True)),
            device=self.device,
        )

    def _resolve_split_labels(self) -> Tuple[str, Optional[str], str]:
        train_split = "train"
        val_split = "val" if len(self.config.get("val_conditions", [])) > 0 else None
        test_split = "test"
        return train_split, val_split, test_split

    def train(self) -> None:
        log.info("Starting pDAE training")
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        split_name = self.config["split_name"]
        train_split, val_split, test_split = self._resolve_split_labels()
        log.info(
            "Preparing pDAE tensors from adata (split_col=%s, train=%s, val=%s, test=%s)",
            split_name,
            train_split,
            val_split,
            test_split,
        )
        prep_start = time.time()
        pdae_data = PDAEData.from_adata(
            data_dir=str(output_dir),
            adata_path=self.config["data_path"],
            pert_col=self.hyperparams.get("pert_col", "condition"),
            pert_sep=self.pert_sep,
            pert_ctrl=self.pert_ctrl,
            split_col=split_name,
            train_split=train_split,
            train_holdout_split=test_split,
            val_split=val_split,
            test_split=test_split,
            data_name=f"{self.hyperparams.get('model_name', 'pdae')}_{split_name}",
        )
        log.info(
            "Prepared pDAE data in %.1fs (obs=%s, pert=%s)",
            time.time() - prep_start,
            tuple(pdae_data.observations.shape),
            tuple(pdae_data.perturbations.shape),
        )

        pdae_config = self._build_pdae_config()
        log.info("Initializing pDAE model on device=%s", self.device)
        model = PDAEModel(
            device=self.device,
            pdae_data=pdae_data,
            pdae_config=pdae_config,
            save_dir=str(output_dir),
        )

        num_epochs = int(self.hyperparams.get("num_epochs", 50))
        n_epochs_loss_print = self.hyperparams.get("n_epochs_loss_print")
        if n_epochs_loss_print is None:
            n_epochs_loss_print = 20
        print_tqdm = bool(self.hyperparams.get("print_tqdm", True))
        log.info(
            "Launching pDAE train loop (epochs=%d, loss_log_every=%s, tqdm=%s)",
            num_epochs,
            n_epochs_loss_print,
            print_tqdm,
        )
        train_start = time.time()
        model.train(
            num_epochs=num_epochs,
            n_epochs_model_checkpoint=self.hyperparams.get("n_epochs_model_checkpoint"),
            n_epochs_plot_losses=self.hyperparams.get("n_epochs_plot_losses"),
            n_epochs_loss_print=n_epochs_loss_print,
            print_tqdm=print_tqdm,
        )
        log.info(
            "Finished pDAE train loop in %.1fs (epochs_trained=%s)",
            time.time() - train_start,
            model.num_epochs_trained,
        )

        self._save_training_metadata(output_dir, model)
        log.info("pDAE training complete")

    def _save_training_metadata(self, output_dir: Path, model: PDAEModel) -> None:
        checkpoint = self._find_checkpoint(output_dir, self.hyperparams.get("model_name", "pdae"))
        data_file_name = f"pdae_data_{model.pdae_data.data_name}.pth"
        config_file_name = f"pdae_config_{model.pdae_config.model_name}.json"
        metadata = {
            "model_type": "pdae",
            "config": self.config,
            "device": self.device,
            "trained_epochs": model.num_epochs_trained,
            "checkpoint_file_name": checkpoint.name if checkpoint else None,
            "data_file_name": data_file_name,
            "config_file_name": config_file_name,
        }
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, cls=PathEncoder)

    def _find_checkpoint(self, model_dir: Path, model_name: str) -> Optional[Path]:
        candidates = list(model_dir.glob(f"pdae_model_{model_name}_epoch_*.pth"))
        if not candidates:
            candidates = list(model_dir.glob("pdae_model_*_epoch_*.pth"))
        if not candidates:
            return None

        def extract_epoch(path: Path) -> int:
            stem = path.stem
            if "_epoch_" not in stem:
                return -1
            return int(stem.split("_epoch_")[-1])

        return sorted(candidates, key=extract_epoch)[-1]

    def _load_single_pert_index(self, model_dir: Path) -> Dict[str, int]:
        mapping_path = model_dir / "dict_single_pert_ix.json"
        if not mapping_path.exists():
            raise FileNotFoundError(f"Missing perturbation mapping: {mapping_path}")
        with open(mapping_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _condition_to_tensor(self, condition: str, single_pert_ix: Dict[str, int]) -> torch.Tensor:
        parts = condition.split(self.pert_sep)
        vec = np.zeros(len(single_pert_ix), dtype=np.float32)
        for part in parts:
            if part == self.pert_ctrl:
                continue
            if part not in single_pert_ix:
                raise KeyError(f"Perturbation '{part}' is not in training perturbation index")
            vec[single_pert_ix[part]] = 1.0
        return torch.tensor(vec, dtype=torch.float32)

    def predict(self) -> None:
        log.info("Starting pDAE prediction")
        model_dir = Path(self.config["model_path"])
        checkpoint = self._find_checkpoint(model_dir, self.hyperparams.get("model_name", "pdae"))
        if checkpoint is None:
            raise FileNotFoundError(f"No pDAE checkpoint found in {model_dir}")

        metadata_path = model_dir / "metadata.json"
        data_path = None
        config_path = None
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            data_file_name = metadata.get("data_file_name")
            config_file_name = metadata.get("config_file_name")
            if data_file_name:
                candidate = model_dir / data_file_name
                if candidate.exists():
                    data_path = str(candidate)
            if config_file_name:
                candidate = model_dir / config_file_name
                if candidate.exists():
                    config_path = str(candidate)

        model = PDAEModel(
            device=self.device,
            pretrained_model_path=str(checkpoint),
            data_path=data_path,
            config_path=config_path,
        )
        x_tr = model.x_tr.to(self.device)
        pert_tr = model.pert_tr.to(self.device)
        single_pert_ix = self._load_single_pert_index(model_dir)

        adata = self.data_manager.load_dataset()
        predictions_adata = self._predict_conditions(
            model=model,
            x_tr=x_tr,
            pert_tr=pert_tr,
            single_pert_ix=single_pert_ix,
            original_adata=adata,
        )
        output_path = self.config["output_path"]
        predictions_adata.write_h5ad(output_path)
        log.info("Saved pDAE predictions to %s", output_path)

    def _predict_conditions(
        self,
        model: PDAEModel,
        x_tr: torch.Tensor,
        pert_tr: torch.Tensor,
        single_pert_ix: Dict[str, int],
        original_adata: sc.AnnData,
    ) -> sc.AnnData:
        prediction_rows: List[np.ndarray] = []
        conditions: List[str] = []

        for condition in self.config["test_conditions"]:
            if "*" in condition:
                continue
            try:
                pert_te = self._condition_to_tensor(condition, single_pert_ix).to(self.device)
            except KeyError as exc:
                log.warning("Skipping condition %s: %s", condition, exc)
                continue

            with torch.no_grad():
                x_pred, _ = model.predict(x_tr=x_tr, pert_tr=pert_tr, pert_te=pert_te)
            prediction_rows.append(x_pred.mean(dim=0).detach().cpu().numpy())
            conditions.append(condition)

        if not prediction_rows:
            raise ValueError("No valid pDAE predictions were generated for test conditions")

        prediction_matrix = np.vstack(prediction_rows)
        obs = pd.DataFrame(
            {
                "condition": conditions,
                "covariate": ["none"] * len(conditions),
                "pair_key": [f"none_{cond}" for cond in conditions],
            }
        )
        pred_adata = sc.AnnData(X=prediction_matrix, obs=obs)
        pred_adata.var_names = original_adata.var_names
        return pred_adata
