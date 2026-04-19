import os
import json
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Callable, List, Literal
from dataclasses import dataclass, asdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from PerturbationExtrapolation.pdae.dataloader import PDAEDataset
from PerturbationExtrapolation.pdae.mmd_check import get_mmdagg_test_results, make_mmdagg_distribution_plot, run_mmdagg_test
from PerturbationExtrapolation.pdae.training import train_pdae
from PerturbationExtrapolation.pdae.baselines import BaselineModel, OracleHalfModel
from PerturbationExtrapolation.pdae.metrics import Metrics
from PerturbationExtrapolation.pdae.extrapolation_check import check_extrapolation_guarantees
from PerturbationExtrapolation.pdae.utils import plot_mmdagg_test_results, plot_pdae_gt_vs_predicted_distributions, set_random_seed, show_loss_curves
from PerturbationExtrapolation.pdae.model import PerturbationAutoencoder
from PerturbationExtrapolation.pdae.adata_to_tensor import map_adata_split_to_tensors


def get_items_by_split(data: torch.Tensor, splits: List[str], target_split: str):
    mask = [split_item == target_split for split_item in splits]
    return data[mask]


@dataclass
class PDAEData:
    """
    Data class to hold the data for the Perturbation Extrapolation Autoencoder (PDAE).

    BEWARE: The control condition must be encoded as a zero vector in the perturbations tensor.

    Attributes:
        data_name (str): Name of the dataset.
        save_dir (str): Directory to save the data.
        observations (torch.Tensor): 2D tensor with shape (n_samples, dim_x) containing the observations.
        perturbations (torch.Tensor): 2D tensor with shape (n_samples, n_perturbations) containing the perturbations.
        pert_ctrl (str): The name of the control condition to be encoded as a zero vector.
        perturbation_annotations (List[str], optional): List of strings containing the perturbation annotations.
        splits (List[str]): List of strings indicating the split for each sample. Possible values are "train", "train_holdout", "val", and "test".
            "train" is obligatory
            "train_holdout" has the same perturbations as "train" but different observations. It is used for validaton during training and for evaluation after training.
            "val" is used for validation during training and for evaluation after training.
            "test" is used for evaluation after training.
    """
    data_name: str
    save_dir: str
    observations: torch.Tensor 
    perturbations: torch.Tensor 
    pert_ctrl: str
    splits: List[str]
    train_split: str = "train"
    train_holdout_split: str = "train_holdout"
    val_split: str = "val"
    test_split: str = "test"
    perturbation_annotations: List[str] = None
    
    def __post_init__(self):
        # validate data types
        if not isinstance(self.observations, torch.Tensor):
            raise TypeError("observations must be a torch.Tensor")
        if not isinstance(self.perturbations, torch.Tensor):
            raise TypeError("perturbations must be a torch.Tensor")
        if self.perturbation_annotations is not None and (not isinstance(self.perturbation_annotations, list) or not all(isinstance(p, str) for p in self.perturbation_annotations)):
            raise TypeError("perturbation_annotations must be a list of strings")
        if not isinstance(self.splits, list) or not all(isinstance(s, str) for s in self.splits):
            raise TypeError("splits must be a list of strings")
        # validate dimensions
        if self.observations.ndim != 2:
            raise ValueError("observations must be a 2D tensor with shape (n_samples, dim_x)")
        if self.perturbations.ndim != 2:
            raise ValueError("perturbations must be a 2D tensor with shape (n_samples, n_perturbations)")
        # validate shapes
        if self.observations.shape[0] != self.perturbations.shape[0]:
            raise ValueError("observations and perturbations must have the same number of samples")
        if self.observations.shape[0] != len(self.splits):
            raise ValueError("splits must have the same length as the number of samples")
        # validate splits
        if self.train_split is None:
            raise ValueError("train_split must be provided")
        if not all(s in [self.train_split, self.train_holdout_split, self.val_split, self.test_split] for s in self.splits if s is not None):
            raise ValueError("the provided split names must be contained in splits")
        
        # set the reference perturbation
        self.is_reference_perturbation = self.perturbations.sum(dim=1) == 0

        if self.perturbation_annotations is not None:
            assert self.is_reference_perturbation.sum() == sum([ann==self.pert_ctrl for ann in self.perturbation_annotations])

        # Save data to the specified directory
        if self.save_dir:
            save_file_path = os.path.join(self.save_dir, f"pdae_data_{self.data_name}.pth")
            torch.save({
                "data_name": self.data_name,
                "save_dir": self.save_dir,
                "observations": self.observations,
                "perturbations": self.perturbations,
                "pert_ctrl": self.pert_ctrl,
                "perturbation_annotations": self.perturbation_annotations,
                "splits": self.splits,
                "train_split": self.train_split,
                "train_holdout_split": self.train_holdout_split,
                "val_split": self.val_split,
                "test_split": self.test_split,
                "is_reference_perturbation": self.is_reference_perturbation
            }, save_file_path)
    
    
    def check_extrapolation_guarantees(self, extrapolation_splits: List[str] = None, print_progress=False) -> dict:
        train_perturbations = self.perturbations[["train" in s for s in self.splits], :]
        if extrapolation_splits is not None:
            test_perturbations = self.perturbations[[s in extrapolation_splits for s in self.splits], :]
        else:
            test_perturbations = self.perturbations[["train" not in s for s in self.splits], :]
        return check_extrapolation_guarantees(
            train_perturbations=train_perturbations,
            test_perturbations=test_perturbations,
            print_progress=print_progress
        )


    @classmethod
    def load(cls, file_path: str) -> "PDAEData":
        """Load a PDAEData instance from a saved .pth file."""
        data_dict = torch.load(file_path, map_location="cpu", weights_only=False)

        return cls(
            data_name=data_dict["data_name"],
            # Avoid writing back to the source directory during inference loads.
            save_dir=None,
            observations=data_dict["observations"],
            perturbations=data_dict["perturbations"],
            pert_ctrl=data_dict["pert_ctrl"],
            splits=data_dict["splits"],
            perturbation_annotations=data_dict.get("perturbation_annotations", None),
            train_split=data_dict["train_split"],
            train_holdout_split=data_dict["train_holdout_split"],
            val_split=data_dict["val_split"],
            test_split=data_dict["test_split"],
        )

    
    @classmethod
    def from_adata(
        cls,
        data_dir: str,
        adata_path: str,
        pert_col: str,
        pert_sep: str, 
        pert_ctrl: str,
        split_col: int, 
        train_split: str = "train",
        train_holdout_split: str = "train_holdout",
        val_split: str = "val",
        test_split: str = "test",
        data_name: str = "default_dataset"
    ) -> "PDAEData":
        """Create a PDAEData instance from an AnnData object."""

        data_dict = map_adata_split_to_tensors(
            adata_path=adata_path,
            pert_col=pert_col,
            pert_sep=pert_sep,
            pert_ctrl=pert_ctrl,
            split_col=split_col,
            train_split=train_split,
            train_holdout_split=train_holdout_split,
            val_split=val_split,
            test_split=test_split,
            data_dir=data_dir
        )

        return cls(
            data_name=data_name,
            save_dir=data_dir,
            observations=data_dict["observations"],
            perturbations=data_dict["perturbations"],
            pert_ctrl=data_dict["pert_ctrl"],
            splits=data_dict["splits"],
            perturbation_annotations=data_dict["perturbation_annotations"],
            train_split=train_split,
            train_holdout_split=train_holdout_split,
            val_split=val_split,
            test_split=test_split,
        )
        

@dataclass
class PDAEConfig:
    seed: int
    model_name: str
    batch_size: int
    learning_rate: float
    dim_z_model: int
    dim_noise_model: int
    sigma_noise_model: float
    decoder_layer_shapes: List[int]
    encoder_layer_shapes: List[int]
    weight_reconstruction_loss: float
    weight_perturbation_energy_loss: float
    weight_marginal_prior_energy_loss: float
    weight_l21: float
    update_encoder_on_reconstruction_loss: bool
    normalize_energy_loss: bool
    beta: float
    use_bias_in_perturbation_matrix: bool
    use_softplus_after_decoder: bool
    device: str = None # this argument is not enforced, not used and only here for backward compatibility

    dict = asdict


class PDAEModel:
    pdae_data: PDAEData
    pdae_config: PDAEConfig 
    save_dir: str 
    pretrained_model_path: str 
    data_path: str 
    config_path: str 
    device: str
    pdae_model: PerturbationAutoencoder = None
    num_epochs_trained: int = None
    losses: dict = None
    mmd_stats: dict = None
    df_evals: pd.DataFrame = None

    def __init__ (
        self, 
        device: str,
        pdae_data: PDAEData = None, 
        pdae_config: PDAEConfig = None,
        save_dir: str = None,
        pretrained_model_path: str = None,
        data_path: str = None,
        config_path: str = None
    ):  
        self.device = device

        if pretrained_model_path:
            self._load_pretrained_model(pretrained_model_path, data_path, config_path)
        else:
            assert pdae_data is not None, "pdae_data must be provided if pretrained_model_path is not given."
            assert pdae_config is not None, "pdae_config must be provided if pretrained_model_path is not given."
            assert save_dir is not None, "save_dir must be provided if pretrained_model_path is not given."
    
            self.pdae_data = pdae_data
            self.pdae_config = pdae_config
            self.save_dir = save_dir
    
            # Save configuration to the specified directory
            with open(os.path.join(save_dir, f"pdae_config_{self.pdae_config.model_name}.json"), "w") as f:
                json.dump(self.pdae_config.dict(), f, indent=4)
        
            self._prepare_data()

    def _load_pretrained_model(self, pretrained_model_path: str, data_path: str, config_path: str):
        model_dir = os.path.dirname(pretrained_model_path)
        model_name = os.path.basename(pretrained_model_path).split("pdae_model_")[1].split("_epoch_")[0]
        epoch = int(os.path.basename(pretrained_model_path).split("pdae_model_")[1].split("_epoch_")[1].replace(".pth", ""))

        if data_path is None:
            data_path = os.path.join(model_dir, f"pdae_data_{model_name}.pth")
        if config_path is None:
            config_path = os.path.join(model_dir, f"pdae_config_{model_name}.json")

        if not os.path.exists(pretrained_model_path):
            raise FileNotFoundError(f"Pretrained model not found at {pretrained_model_path}. Please check the path and try again.")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}. Please check the path and try again.")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}. Please check the path and try again.")

        self.save_dir = model_dir
        self.pdae_data = PDAEData.load(data_path)
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        self.pdae_config = PDAEConfig(**config_dict)

        self.pdae_model = PerturbationAutoencoder(
            dim_z=self.pdae_config.dim_z_model,
            dim_noise=self.pdae_config.dim_noise_model,
            sigma_noise=self.pdae_config.sigma_noise_model,
            dim_x=self.pdae_data.observations.shape[-1],
            n_perturbations=self.pdae_data.perturbations.shape[-1],
            encoder_layer_shapes=self.pdae_config.encoder_layer_shapes,
            decoder_layer_shapes=self.pdae_config.decoder_layer_shapes,
            update_encoder_on_reconstruction_loss=self.pdae_config.update_encoder_on_reconstruction_loss,
            use_bias_in_perturbation_matrix=self.pdae_config.use_bias_in_perturbation_matrix,
            use_softplus_after_decoder=self.pdae_config.use_softplus_after_decoder,
        )
        self.pdae_model.load_state_dict(torch.load(pretrained_model_path, map_location=self.device))
        self.pdae_model.to(self.device)
        self.num_epochs_trained = epoch
        self.losses = torch.load(os.path.join(self.save_dir, f'pdae_loss_dict_{self.pdae_config.model_name}.pth'), weights_only=False)
        self.mmd_stats = torch.load(os.path.join(self.save_dir, f'pdae_mmd_stats_dict_{self.pdae_config.model_name}.pth'), weights_only=False)

        self._prepare_data()

    def _prepare_data(self):
        self.x_tr = get_items_by_split(self.pdae_data.observations, self.pdae_data.splits, self.pdae_data.train_split)
        self.pert_tr = get_items_by_split(self.pdae_data.perturbations, self.pdae_data.splits, self.pdae_data.train_split)
        
        self.x_tr_holdout = (
            get_items_by_split(self.pdae_data.observations, self.pdae_data.splits, self.pdae_data.train_holdout_split)
            if self.pdae_data.train_holdout_split else None
        )
        self.pert_tr_holdout = (
            get_items_by_split(self.pdae_data.perturbations, self.pdae_data.splits, self.pdae_data.train_holdout_split)
            if self.pdae_data.train_holdout_split else None
        )

        self.x_val = (get_items_by_split(self.pdae_data.observations, self.pdae_data.splits, self.pdae_data.val_split) if self.pdae_data.val_split else None)
        self.pert_val = (get_items_by_split(self.pdae_data.perturbations, self.pdae_data.splits, self.pdae_data.val_split) if self.pdae_data.val_split else None)
        
        self.x_te = (get_items_by_split(self.pdae_data.observations, self.pdae_data.splits, self.pdae_data.test_split) if self.pdae_data.test_split else None)
        self.pert_te = (get_items_by_split(self.pdae_data.perturbations, self.pdae_data.splits, self.pdae_data.test_split) if self.pdae_data.test_split else None)
    

    def train(self, num_epochs: int, **kwargs):
        config_dict = self.pdae_config.dict()
        config_dict["dim_x"] = self.pdae_data.observations.shape[-1]
        config_dict["n_perturbations"] = self.pdae_data.perturbations.shape[-1]
        
        if self.pdae_model is not None:
            pdae_model = self.pdae_model
            num_epochs_trained = self.num_epochs_trained
            losses = self.losses
            mmd_stats = self.mmd_stats
            print("Using existing PDAE model already trained for {} epochs.".format(self.num_epochs_trained))
        else:
            pdae_model = None
            num_epochs_trained = 0
            losses = None
            mmd_stats = None
            print("No existing PDAE model found, training a new one.")

        if self.pdae_data.train_holdout_split is None or self.pdae_data.val_split is None:
            if "n_epochs_run_eval" in kwargs:
                raise ValueError("n_epochs_run_eval can only be used if train_holdout_split and val_split are set in the PDAEConfig. Set n_epochs_run_eval to None to disable this feature.")

        model, num_epochs_trained_new, losses, mmd_stats  = train_pdae(
            num_epochs=num_epochs,
            x_tr = self.x_tr,
            pert_tr = self.pert_tr,
            x_tr_holdout = self.x_tr_holdout,
            pert_tr_holdout = self.pert_tr_holdout,
            x_val = self.x_val,
            pert_val = self.pert_val,
            x_te = self.x_te, 
            pert_te = self.pert_te,
            settings_dict=config_dict,
            save_dir=self.save_dir,
            model_name=self.pdae_config.model_name,
            device=self.device,
            return_model=True,
            pdae_model=pdae_model,
            num_epochs_trained=num_epochs_trained,
            losses=losses,
            mmd_stats=mmd_stats,
            **kwargs
        )
        self.pdae_model = model
        self.num_epochs_trained = num_epochs_trained_new
        self.losses = losses
        self.mmd_stats = mmd_stats
    

    def predict(self, x_tr: torch.Tensor, pert_tr: torch.Tensor, pert_te: torch.Tensor):
        """
        Predicts the observations for a given perturbation using the trained PDAE model.
        """
        if not hasattr(self, 'pdae_model'):
            raise RuntimeError("PDAE model has not been trained yet. Please call train() before predict().")
        if not isinstance(pert_te, torch.Tensor):
            raise TypeError("pert_te must be a torch.Tensor")
        x_pred, z_pred = self.pdae_model.predict(x_tr, pert_tr, pert_te)
        assert x_pred.isnan().sum() == 0, "Predicted observations contain NaN values. Please check the model and input data."
        assert z_pred.isnan().sum() == 0, "Predicted latent representations contain NaN values. Please check the model and input data."
        return x_pred, z_pred


    def run_eval(
            self, 
            models: List[BaselineModel],
            metrics: List[Metrics],
            metrics_device: str = "cpu",
            use_train_holdout: bool = False,
            ratio_subsample_train: float = 1.0,
            ratio_subsample_val_test: float = 1.0,
            run_eval_batched: bool = False,
            run_eval_for_train: bool = False,
            return_df: bool = False,
            eval_pdae: bool = True,
            gene_selector: Callable = None,
        ):

        set_random_seed(self.pdae_config.seed)

        if self.pdae_model is None:
            raise RuntimeError("PDAE model has not been trained yet. Please call train() before run_eval().")

        if (self.x_val is None or self.pert_val is None) and (self.x_te is None and self.pert_te is None):
            raise RuntimeError("Validation or test data must be provided in the PDAEData object.")

        if use_train_holdout and (self.x_tr_holdout is None or self.pert_tr_holdout is None):
            raise RuntimeError("Train holdout data must be provided in the PDAEData object if use_train_holdout is True.")

        assert all(isinstance(model, BaselineModel) for model in models), "All models must be instances of BaselineModel"
        assert isinstance(metrics, list), "metrics must be a list of Metrics instances"
        assert all(isinstance(metric, Metrics) for metric in metrics), "All metrics must be instances of Metrics"

        if use_train_holdout:
            x_tr = self.x_tr_holdout
            pert_tr = self.pert_tr_holdout
        elif ratio_subsample_train < 1.0:
            n_tr_samples = int(self.x_tr.shape[0] * ratio_subsample_train)
            tr_ixs = torch.randperm(self.x_tr.shape[0])[:n_tr_samples]
            x_tr = self.x_tr[tr_ixs]
            pert_tr = self.pert_tr[tr_ixs]
        else:
            x_tr = self.x_tr
            pert_tr = self.pert_tr
        
        x_tr = x_tr.to(self.device)
        pert_tr = pert_tr.to(self.device)
        print("Using {} training samples to make predictions.".format(x_tr.shape[0]))

        # initialize models with right data
        models = [model.fit(x_tr=x_tr, pert_tr=pert_tr) for model in models]
        
        x_val = self.x_val
        pert_val = self.pert_val
        x_te = self.x_te
        pert_te = self.pert_te

        if run_eval_for_train:
            x_val_train = x_tr.cpu()
            pert_val_train = pert_tr.cpu()
        else:
            x_val_train = torch.tensor([])
            pert_val_train = torch.tensor([])
        
        if ratio_subsample_val_test < 1.0:
            if self.x_val is not None:
                n_val_samples = int(self.x_val.shape[0] * ratio_subsample_val_test)
                val_ixs = torch.randperm(self.x_val.shape[0])[:n_val_samples]
                x_val = self.x_val[val_ixs]
                pert_val = self.pert_val[val_ixs] 
            if self.x_te is not None:
                n_te_samples = int(self.x_te.shape[0] * ratio_subsample_val_test)
                te_ixs = torch.randperm(self.x_te.shape[0])[:n_te_samples]
                x_te = self.x_te[te_ixs]
                pert_te = self.pert_te[te_ixs]
            if run_eval_for_train and x_tr is not None:
                n_train_samples = int(x_val_train.shape[0] * ratio_subsample_val_test)
                train_ixs = torch.randperm(x_val_train.shape[0])[:n_train_samples]
                x_val_train = x_val_train[train_ixs]
                pert_val_train = pert_val_train[train_ixs]

        if x_val is None or pert_val is None:
            x_val = torch.tensor([])
            pert_val = torch.tensor([])

        if x_te is None or pert_te is None:
            x_te = torch.tensor([])
            pert_te = torch.tensor([])

        x_val_test = torch.cat([x_val, x_te, x_val_train], dim=0)
        pert_val_test = torch.cat([pert_val, pert_te, pert_val_train], dim=0)
        split_val_test = np.array([self.pdae_data.val_split] * pert_val.shape[0] + [self.pdae_data.test_split] * pert_te.shape[0] + [self.pdae_data.train_split] * x_val_train.shape[0])
        
        print("Running evaluation on {} samples in the training set.".format(x_val_train.shape[0]))
        print("Running evaluation on {} samples in the validation set.".format(x_val.shape[0]))
        print("Running evaluation on {} samples in the test set.".format(x_te.shape[0]))


        if run_eval_batched == False:
            x_val_test = x_val_test.to(self.device)
            pert_val_test = pert_val_test.to(self.device)
            pert_val_test_unique = torch.unique(pert_val_test, dim=0)

        df_evals_baseline = []
        df_evals_pdae = []

        if run_eval_batched:
            batch_counter = 0
            val_test_dataset = PDAEDataset(x_val_test, pert_val_test)
            val_test_dataloader = DataLoader(val_test_dataset, batch_size=self.pdae_config.batch_size, shuffle=False, num_workers=0, pin_memory=False)
            for i, (x_val_test_batch, pert_val_test_batch) in enumerate(val_test_dataloader):
                split_batch = split_val_test[i * self.pdae_config.batch_size:(i + 1) * self.pdae_config.batch_size]
                batch_counter += 1
                x_val_test_batch = x_val_test_batch.to(self.device)
                pert_val_test_batch = pert_val_test_batch.to(self.device)
                pert_val_test_batch_unique_envs = torch.unique(pert_val_test_batch, dim=0)
                for pert_env in pert_val_test_batch_unique_envs:
                    pert_mask = (pert_val_test_batch == pert_env).all(dim=1)
                    x_true = x_val_test_batch[pert_mask]
                    split = split_batch[pert_mask.cpu().numpy()][0]
                    if gene_selector is not None:
                        genes = gene_selector(x_true, pert_env)
                    else:
                        genes = list(range(x_true.shape[1]))
                    for metric in metrics:
                        metric_name = metric.name
                        # baselines
                        for model in models:    
                            model_name = model.name
                            
                            # Special handling for OracleHalfModel
                            if isinstance(model, OracleHalfModel):
                                model.set_test_data(x_true)
                                x_pred = model.predict(pert_env)
                                # For oracle model, only evaluate on second half
                                x_true_oracle = x_true[x_true.shape[0] // 2:]
                                metric_value = metric.compute(x_true_oracle[:,genes].to(metrics_device), x_pred[:,genes].to(metrics_device)).item()
                            else:
                                x_pred = model.predict(pert_env)
                                metric_value = metric.compute(x_true[:,genes].to(metrics_device), x_pred[:,genes].to(metrics_device)).item()
                            
                            df_env = pd.DataFrame({"Split": [split], "Perturbation": [str(pert_env.cpu().numpy().tolist())], "Batch": [batch_counter], "Method": [model_name], "Metric": [metric_name], "Value": [metric_value]})
                            df_evals_baseline.append(df_env)
                        # pdae
                        if eval_pdae:
                            x_pred_pdae, z_pred_pdae = self.predict(x_tr, pert_tr, pert_te=pert_env)
                            metric_value_pdae = metric.compute(x_true[:,genes].to(metrics_device), x_pred_pdae[:,genes].to(metrics_device)).item()
                            df_env_pdae = pd.DataFrame({"Split": [split], "Perturbation": [str(pert_env.cpu().numpy().tolist())], "Batch": [batch_counter],  "Method": ["pdae"], "Metric": [metric_name], "Value": [metric_value_pdae]})
                            df_evals_pdae.append(df_env_pdae)
        else:
            for pert_env in pert_val_test_unique:
                pert_mask = (pert_val_test == pert_env).all(dim=1)
                x_true = x_val_test[pert_mask]
                split = split_val_test[pert_mask.cpu().numpy()][0]

                if gene_selector is not None:
                    genes = gene_selector(x_true, pert_env)
                else:
                    genes = list(range(x_true.shape[1]))

                for metric in metrics:
                    metric_name = metric.name
                    # baselines
                    for model in models:    
                        model_name = model.name  
                        
                        # Special handling for OracleHalfModel
                        if isinstance(model, OracleHalfModel):
                            model.set_test_data(x_true)
                            x_pred = model.predict(pert_env)
                            # For oracle model, only evaluate on second half
                            x_true_oracle = x_true[x_true.shape[0] // 2:]
                            metric_value = metric.compute(x_true_oracle[:,genes].to(metrics_device), x_pred[:,genes].to(metrics_device)).item()
                        else:
                            x_pred = model.predict(pert_env).to(self.device)
                            metric_value = metric.compute(x_true[:,genes].to(metrics_device), x_pred[:,genes].to(metrics_device)).item()
                        
                        df_env = pd.DataFrame({"Split": [split], "Perturbation": [str(pert_env.cpu().numpy().tolist())], "Method": [model_name], "Metric": [metric_name], "Value": [metric_value]})
                        df_evals_baseline.append(df_env)
                    # pdae
                    if eval_pdae:
                        x_pred_pdae, z_pred_pdae = self.predict(x_tr, pert_tr, pert_te=pert_env)
                        metric_value_pdae = metric.compute(x_true[:,genes].to(metrics_device), x_pred_pdae[:,genes].to(metrics_device)).item()
                        df_env_pdae = pd.DataFrame({"Split": [split], "Perturbation": [str(pert_env.cpu().numpy().tolist())], "Method": ["pdae"], "Metric": [metric_name], "Value": [metric_value_pdae]})
                        df_evals_pdae.append(df_env_pdae)


        
        self.df_evals = pd.concat(df_evals_baseline + df_evals_pdae, ignore_index=True)
        self.df_evals.to_csv(os.path.join(self.save_dir, f"pdae_eval_results_{self.pdae_config.model_name}_epoch_{self.num_epochs_trained}.csv"), index=False)
        
        if return_df:
            return self.df_evals


    def get_agg_eval_results(self, layout="long", drop: List[str] = None):
        print(f"Evaluation results after training the PDAE model '{self.pdae_config.model_name}' for {self.num_epochs_trained} epochs:")
        if self.df_evals is None:
            raise RuntimeError("No evaluation results found. Please run run_eval() first.")
        df_evals = self.df_evals.copy()
        for col in ["Batch", "Perturbation"]:
            if col in df_evals.columns:
                df_evals = df_evals.drop(columns=[col])
        df_evals_agg = df_evals.groupby(["Split", "Method", "Metric"]).agg(["mean", "std"]).reset_index()
        df_evals_agg["Vals"] = (
            df_evals_agg[("Value", "mean")].round(3).astype(str)
            + " ± "
            + df_evals_agg[("Value", "std")].round(3).astype(str)
        )
        df_evals_agg = df_evals_agg.drop(columns=[("Value", "mean"), ("Value", "std")])
        if layout == "long":
            return df_evals_agg
        elif layout == "wide":
            df_evals_agg_pivot = df_evals_agg.pivot(index=["Split", "Method"], columns="Metric", values="Vals").reset_index()
            df_wide = []
            for split in df_evals_agg_pivot["Split"].unique():
                df_wide.append(
                    df_evals_agg_pivot[df_evals_agg_pivot["Split"] == split].reset_index(drop=True)
                )
            return pd.concat(df_wide, axis=1)


    def boxplot_metrics(self, metric: str):
        assert hasattr(self, 'df_evals'), "No evaluation results found. Please run run_eval() first."
        assert metric in self.df_evals["Metric"].unique(), f"Metric '{metric}' not found in evaluation results. Available metrics: {self.df_evals['Metric'].unique()}"
        df = self.df_evals.drop(columns=["Split", "Perturbation", "Batch"]).copy()
        df = df[df["Metric"] == metric]
        df = df.rename(columns={"Value": metric})
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x="Method", y=metric)
        plt.show()

    def plot_losses(self, losses_to_plot: List[str] = [
        "tr_reconstruction_losses",
        "tr_marginal_prior_energy_losses",
        "tr_perturbation_energy_losses",
        "tr_l21_losses",
        "tr_mmd_p_median_mean",
        "val_energy_losses",
        "val_mean_diffs",
        "val_mmd_p_median_mean"
    ], rename_dict: dict = None):
        show_loss_curves(self.save_dir, self.pdae_config.model_name, losses=self.losses.copy(), losses_to_plot=losses_to_plot, mmd_stats=self.mmd_stats.copy(), rename_dict=rename_dict)
        print(f"Loss curves saved to {self.save_dir} for model {self.pdae_config.model_name}.")
    
    def plot_distributions(
            self,
            plot_distribution_ratio_sample,
            z_tr_holdout,
            z_val,
            z_te,
        ):
        plot_pdae_gt_vs_predicted_distributions(
            model=self.pdae_model,
            model_name=self.pdae_config.model_name,
            epoch=self.num_epochs_trained,
            save_dir=self.save_dir,
            device=self.device,
            x_tr_holdout=self.x_tr_holdout.to(self.device),
            pert_tr_holdout=self.pert_tr_holdout.to(self.device),
            x_val=self.x_val.to(self.device),
            pert_val=self.pert_val.to(self.device),
            x_te=self.x_te.to(self.device),
            pert_te=self.pert_te.to(self.device),
            z_tr_holdout=z_tr_holdout.to(self.device),  # Assuming z_tr_holdout is not defined in the class
            z_val=z_val.to(self.device),  # Assuming z_val is not defined in the class
            z_te=z_te.to(self.device),  # Assuming z_te is not defined in the class
            plot_distribution_ratio_sample=plot_distribution_ratio_sample,  # Assuming plot_distribution_ratio_sample is not defined in the class
        )


    def run_mmd_test(self, alpha, n_samples_max: int = None):
        perturbations = []
        rejects = []
        ratios_rejects = []
        p_mins = []
        p_medians = []
        p_maxs = []
        for i, pert_env in enumerate(tqdm(torch.unique(self.pert_tr, dim=0))):
            # print(f"Running MMD Test for Training domain {i} with perturbation environment {pert_env.cpu().numpy().tolist()}")
            pert_mask = (self.pert_tr_holdout == pert_env).all(dim=1)
            x_true = self.x_tr_holdout[pert_mask].to(self.device)
            pert_tr_subset = self.pert_tr_holdout[pert_mask].to(self.device)
            x_pred, z_pred = self.predict(x_tr=x_true, pert_tr=pert_tr_subset, pert_te=pert_env.to(self.device))
            if n_samples_max is not None:
                x_true = x_true[:n_samples_max]
                x_pred = x_pred[:n_samples_max]
            mmd_dict = get_mmdagg_test_results(x_true, x_pred, alpha=alpha)
            
            perturbations.append(pert_env.cpu().numpy().tolist())
            rejects.append(mmd_dict["reject"])
            ratios_rejects.append(mmd_dict["ratio_rejects"])
            p_mins.append(mmd_dict["p_min"])
            p_medians.append(mmd_dict["p_median"])
            p_maxs.append(mmd_dict["p_max"])
            # print(f"MMD Test for Training domain {i}: Reject = {mmd_dict['reject']}, Ratio Rejects = {mmd_dict['ratio_rejects']}, p_min = {mmd_dict['p_min']}, p_median = {mmd_dict['p_median']}, p_max = {mmd_dict['p_max']}")
        return {
            "perturbations": perturbations,
            "rejects": rejects,
            "ratios_rejects": ratios_rejects,
            "p_mins": p_mins,
            "p_medians": p_medians,
            "p_maxs": p_maxs,
        }


    def plot_mmd_test(self, alpha):
        for i, pert_env in enumerate(torch.unique(self.pert_tr, dim=0)):
            pert_mask = (self.pert_tr_holdout == pert_env).all(dim=1)
            x_true = self.x_tr_holdout[pert_mask].to(self.device)
            x_pred = self.predict(pert_te=pert_env, use_train_holdout=True)[0]
            make_mmdagg_distribution_plot(x_true, x_pred, alpha, self.save_dir, "mmd_test_train_environments", f"MMD Test for Training domain {i}", xlabel="x_true", ylabel="x_pred")


    def plot_mmd_stats(self):
        plot_mmdagg_test_results(self.mmd_stats["tr_mmd_test_results"])
