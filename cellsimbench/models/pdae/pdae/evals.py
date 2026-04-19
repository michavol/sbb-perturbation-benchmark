import gc
import random
import pathlib
from typing import List

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import display
from cpa.plotting import CPAHistory
from torch.utils.data import DataLoader

import cpa

from PerturbationExtrapolation.pdae_nested.model_nested import PerturbationAutoencoder
from PerturbationExtrapolation.pdae_nested.losses_nested import get_L2_difference_in_mean, get_energy_distance

from PerturbationExtrapolation.pdae.model import PerturbationAutoencoder
from PerturbationExtrapolation.pdae.metrics import get_mean_difference
from PerturbationExtrapolation.mmd import MMDLoss
from PerturbationExtrapolation.pdae_nested.dataloader_nested import PDAEDataset, get_train_dataset_dataloader, get_train_tensors, get_data_dict_new_pdae

from mt.utils import get_ix_pert_tr_ctrl_simulation, read_dict, get_pseudo_bulk, predict_from_pooled_training, get_prediction_data, get_r2_score, get_ix_pert_tr_ctrl_norman, get_perturbation_strings, get_device


def prepare_eval_data(data_path, data_dict, dataloader_mode, device, batch_size, n_samples_max, n_samples_max_holdout, data_type, subsample_mode):
    
    # prepare the data
    x_tr = data_dict["x_tr"]
    x_tr_holdout = data_dict["x_tr_holdout"]
    x_val = data_dict["x_val"]
    x_te = data_dict["x_te"]
    pert_tr = data_dict["pert_tr"]
    pert_val = data_dict["pert_val"]
    pert_te = data_dict["pert_te"]
    
    pert_tr = pert_tr.to(device)
    pert_val = pert_val.to(device)
    pert_te = pert_te.to(device)

    train_dataset, train_dataloader = get_train_dataset_dataloader(x_tr, dataloader_mode, batch_size, n_samples_max=n_samples_max)
    train_holdout_dataset, train_holdout_dataloader = get_train_dataset_dataloader(x_tr_holdout, dataloader_mode, batch_size, n_samples_max=n_samples_max_holdout)

    if isinstance(x_tr, list):
        list_mode = True
        x_tr = get_train_tensors(train_dataset, train_dataloader, dataloader_mode, device) 
        x_tr_holdout = get_train_tensors(train_holdout_dataset, train_holdout_dataloader, dataloader_mode, device) 
        # val and test
        x_ood = x_val + x_te
        x_ood = [x.to(device) for x in x_ood]
        if len(x_ood[0].shape) == 2:
            x_ood = [xe.unsqueeze(0) for xe in x_ood]
    else:
        list_mode = False
        x_val = x_val.to(device)
        x_te = x_te.to(device)
        x_ood = torch.cat([x_val, x_te], dim=0)
    
    pert_ood = torch.cat([pert_val, pert_te], dim=0)

    if data_type == "simulation" and list_mode == False:
        labels_id_ood = ["id"] * x_val.shape[0] + ["ood"] * x_te.shape[0]
    else:
        labels_id_ood = None
    
    if list_mode:
        _env_range = range(len(x_ood))
    else:
        _env_range = range(x_ood.shape[0])
    
    dim_x = x_tr.shape[-1]
    n_perturbations = pert_val.shape[-1]

    if data_type == "simulation":
        ix_control = get_ix_pert_tr_ctrl_simulation(pert_tr, device)
    elif data_type == "norman":
        ix_control = get_ix_pert_tr_ctrl_norman(data_path=data_path)

    if subsample_mode:
        print(f"Using x_tr_holdout due to {subsample_mode=}.")
        x_tr = x_tr_holdout.to(device)
        train_dataloader = train_holdout_dataloader
    else:
        x_tr = x_tr.to(device)
    
    return x_tr, pert_tr, train_dataloader, x_ood, pert_ood, labels_id_ood, _env_range, dim_x, n_perturbations, ix_control


def get_cpa_pert_ood_str(data_type, data_path, pert_ood, control_encoding):
    def get_norman_perturbation_string(pert_ohe, inv_dict_single_pert_ix):
        return "+".join([inv_dict_single_pert_ix[k] for k in np.where(pert_ohe)[0]])

    if data_type == "norman":
        dict_single_pert_ix = read_dict(data_path, file_stem="dict_single_pert_ix")
        inv_dict_single_pert_ix = {v: k for k, v in dict_single_pert_ix.items()}
        pert_ood_str = [get_norman_perturbation_string(p, inv_dict_single_pert_ix) for p in pert_ood.cpu()]
    elif data_type == "simulation":
        pert_ood_str = get_perturbation_strings(pert_ood.cpu(), control_encoding)
    return pert_ood_str


def get_cpa_adata(data_type, data_path):
    if data_type == "norman":
        adata = sc.read(data_path / "adata_pdae_norman.h5ad")
    elif data_type == "simulation":
        adata = sc.read(data_path / "cpa_simulation_data_processed.h5ad")
    return adata


def get_cpa_cov_info(data_type):
    if data_type == "norman":
        cov_key = "cell_type"
        cov_val = "A549"
    elif data_type == "simulation":
        cov_key = "dummy_cov"
        cov_val = "XYZ"
    return cov_key, cov_val


def get_df_evals_cpa(cpa_api, train_dataloader, x_tr, x_ood, pert_ood, pert_ood_str, labels_id_ood, ix_control, dim_x, device, batch_size, _env_range, cpa_model_name, cov_key, cov_val, mmd, oracle_latents=None, ):
    df_evals_cpa = pd.DataFrame()
    for e in tqdm(_env_range):
        xe = x_ood[e]
        
        if len(xe.shape) == 2:
            xe = xe.unsqueeze(0)
        
        pe = pert_ood[e]
        pe_str = [pert_ood_str[e]]
        if labels_id_ood is not None:
            label_id_ood = labels_id_ood[e]
        else:
            label_id_ood = "Undefined"

        # predictions
        prediction_data_e = get_prediction_data(pert_str_list=pe_str, pert_tensor=pe, cov_key=cov_key, cov_val=cov_val)
        
        if batch_size is None:
            x_e_hat, _, _, _, _ = predict_from_pooled_training(
                cpa_api=cpa_api,
                genes=x_tr[ix_control].reshape(-1, dim_x),   
                # genes=x_tr.reshape(-1, dim_x), # pooling all training domains works better
                # genes=x_tr.reshape(-1, dim_x),   
                oracle_latents=oracle_latents,
                cov=prediction_data_e["cov"],  
                pert=prediction_data_e["conditions"],
                dose=prediction_data_e["dose"], 
                uncertainty=False,  
                return_anndata=False,  
                sample=False,  
                n_samples=1
            )
        else:   
            x_e_hat = []
            for x_tr_batch in train_dataloader:
                x_tr_batch = x_tr_batch.transpose(0,1).to(device)
                x_e_hat_batch, _, _, _, _ = predict_from_pooled_training(
                    cpa_api=cpa_api,
                    genes=x_tr[ix_control].reshape(-1, dim_x),   
                    # genes=x_tr_batch.reshape(-1, dim_x), # pooling all training domains works better
                    # genes=x_tr.reshape(-1, dim_x),   
                    oracle_latents=oracle_latents,
                    cov=prediction_data_e["cov"],  
                    pert=prediction_data_e["conditions"],
                    dose=prediction_data_e["dose"], 
                    uncertainty=False,  
                    return_anndata=False,  
                    sample=False,  
                    n_samples=1
                )
                x_e_hat.append(torch.Tensor(x_e_hat_batch))
            x_e_hat = torch.cat(x_e_hat, dim=0)

        # pooled
        x_e_hat = torch.Tensor(x_e_hat).reshape(len(pe_str), -1, dim_x).detach() # (1, n_samples_pooled, dim_x)
        # pooled mean
        # x_e_hat = torch.Tensor(x_e_hat).reshape(x_tr.shape).mean(dim=0).reshape(len(pe_str), -1, dim_x).detach()

        # move to cpu for metrics
        xe = xe.to(device)
        x_e_hat = x_e_hat.to(device)

        x_e_hat_mean = torch.mean(x_e_hat, dim=[0,1])
        
        # L2 mean diff
        cpa_mean_diff_e = get_mean_difference(xe, x_e_hat_mean)

        # R²
        cpa_r2_e = get_r2_score(xe.mean(dim=[0,1]), x_e_hat_mean, dim_x)
        
        # Energy Distance    
        try:
            cpa_energy_loss_e = get_energy_distance(xe, x_e_hat)
        except RuntimeError as e:
            if str(e).startswith('CUDA out of memory.'):
                print("Switching to CPU due to", e)
                cpa_energy_loss_e = get_energy_distance(xe.cpu(), x_e_hat.cpu())
            else:
                raise e
        
        # MMD
        try:
            cpa_mmd_e = mmd(xe.view(-1, dim_x), x_e_hat.view(-1, dim_x))
        except RuntimeError as e:
            if str(e).startswith('CUDA out of memory.') or str(e).startswith('CUDA error: an illegal memory access was encountered'):
                print("Switching to CPU due to", e)
                mmd = MMDLoss(device="cpu")
                cpa_mmd_e = mmd(xe.cpu().view(-1, dim_x), x_e_hat.cpu().view(-1, dim_x))
            else:
                raise e

        df_evals_cpa_e = pd.DataFrame({
            "Method": [f"CPA/{cpa_model_name}"],
            "Env type": [label_id_ood],
            "Energy Distance": [cpa_energy_loss_e.item()],
            "MMD": [cpa_mmd_e.item()],
            "Mean Difference": [cpa_mean_diff_e.item()],
            "R²": [cpa_r2_e]
        })

        df_evals_cpa = pd.concat([df_evals_cpa, df_evals_cpa_e], axis=0)
    
    return df_evals_cpa


def get_df_evals_pdae(pdae_model, train_dataloader, x_tr, pert_tr, x_ood, pert_ood, labels_id_ood, dim_x, device, batch_size, _env_range, pdae_name, mmd):
    
    df_evals_pdae = pd.DataFrame()
    for e in tqdm(_env_range):
        xe = x_ood[e]
        
        if len(xe.shape) == 2:
            xe = xe.unsqueeze(0)
        
        pe = pert_ood[e]
        
        if labels_id_ood is not None:
            label_id_ood = labels_id_ood[e]
        else:   
            label_id_ood = "Undefined"
    
        # predict
        if batch_size is None:
            x_e_hat, _ = pdae_model.predict(x_tr, pert_tr, pe)
        else:
            x_e_hat = []
            for x_tr_batch in train_dataloader:
                x_tr_batch = x_tr_batch.transpose(0,1).to(device)
                x_e_hat_batch, _ = pdae_model.predict(x_tr_batch, pert_tr, pe)
                x_e_hat.append(x_e_hat_batch)
            x_e_hat = torch.cat(x_e_hat, dim=0)

        x_e_hat = x_e_hat.detach().unsqueeze(0) # (1, n_samples_pooled, dim_x)

        # move to cpu for metrics
        xe = xe.to(device)
        x_e_hat = x_e_hat.to(device)

        x_e_hat_mean = torch.mean(x_e_hat, dim=[0,1])
        # L2 mean diff
        pdae_mean_diff_e = get_mean_difference(xe, x_e_hat_mean)
        # R²
        pdae_r2_e = get_r2_score(xe.mean(dim=[0,1]), x_e_hat_mean, dim_x)
        
        # Energy Distance 
        try:
            pdae_energy_loss_e = get_energy_distance(xe, x_e_hat)
        except RuntimeError as e:
            if str(e).startswith('CUDA out of memory.'):
                print("Switching to CPU due to", e)
                pdae_energy_loss_e = get_energy_distance(xe.cpu(), x_e_hat.cpu())
            else:
                raise e
            
        # MMD
        try:
            pdae_mmd_e = mmd(xe.view(-1, dim_x), x_e_hat.view(-1, dim_x))
        except RuntimeError as e:
            if str(e).startswith('CUDA out of memory.'):
                print("Switching to CPU due to", e)
                mmd = MMDLoss(device="cpu")
                pdae_mmd_e = mmd(xe.cpu().view(-1, dim_x), x_e_hat.cpu().view(-1, dim_x))
            else:
                raise e
        
        df_evals_pdae_e = pd.DataFrame({
            "Method": [f"PDAE/{pdae_name}"],
            "Env type": [label_id_ood],
            "Energy Distance": [pdae_energy_loss_e.item()],
            "MMD": [pdae_mmd_e.item()],
            "Mean Difference": [pdae_mean_diff_e.item()],
            "R²": [pdae_r2_e],
        })

        df_evals_pdae = pd.concat([df_evals_pdae, df_evals_pdae_e], axis=0)
    
    return df_evals_pdae


def get_df_evals_new_pdae(pdae_model, train_dataloader, x_tr, pert_tr, x_ood, pert_ood, labels_id_ood, dim_x, device, batch_size, _env_range, pdae_name, mmd):
    
    df_evals_pdae = pd.DataFrame()

    pert_ood_unique = torch.unique(pert_ood, dim=0)

    for pe in tqdm(pert_ood_unique):
        
        xe = x_ood[pert_ood.eq(pe).all(dim=1)]

        if labels_id_ood is not None:
            label_id_ood = np.array(labels_id_ood)[pert_ood.eq(pe).all(dim=1)][0]
        else:   
            label_id_ood = "Undefined"

        pe = pe.to(device)
        xe = xe.to(device)
    
        # predict
        if batch_size is None:
            x_e_hat, _ = pdae_model.predict(x_tr, pert_tr, pe)
        else:
            x_e_hat = []
            for x_tr_batch, pert_tr_batch in train_dataloader:
                x_tr_batch = x_tr_batch.to(device)
                pert_tr_batch = pert_tr_batch.to(device)
                x_e_hat_batch, _ = pdae_model.predict(x_tr_batch, pert_tr_batch, pe)
                x_e_hat.append(x_e_hat_batch)
            x_e_hat = torch.cat(x_e_hat, dim=0)

        x_e_hat = x_e_hat.detach()
        x_e_hat_mean = torch.mean(x_e_hat, dim=0)
        
        # L2 mean diff
        pdae_mean_diff_e = get_mean_difference(xe, x_e_hat_mean)
        # R²
        pdae_r2_e = get_r2_score(xe.mean(dim=0), x_e_hat_mean, dim_x)
        
        # Energy Distance 
        try:
            pdae_energy_loss_e = get_energy_distance(xe, x_e_hat)
        except RuntimeError as e:
            if str(e).startswith('CUDA out of memory.'):
                print("Switching to CPU for energy distance due to", e)
                pdae_energy_loss_e = get_energy_distance(xe.cpu(), x_e_hat.cpu())
            else:
                raise e
            
        # MMD
        try:
            pdae_mmd_e = mmd(xe.view(-1, dim_x), x_e_hat.view(-1, dim_x))
        except RuntimeError as e:
            if str(e).startswith('CUDA out of memory.'):
                print("Switching to CPU for MMD due to", e)
                mmd = MMDLoss(device="cpu")
                pdae_mmd_e = mmd(xe.cpu().view(-1, dim_x), x_e_hat.cpu().view(-1, dim_x))
            else:
                raise e
        
        df_evals_pdae_e = pd.DataFrame({
            "Method": [f"PDAE/{pdae_name}"],
            "Env type": [label_id_ood],
            "Energy Distance": [pdae_energy_loss_e.item()],
            "MMD": [pdae_mmd_e.item()],
            "Mean Difference": [pdae_mean_diff_e.item()],
            "R²": [pdae_r2_e],
        })

        df_evals_pdae = pd.concat([df_evals_pdae, df_evals_pdae_e], axis=0)
    
    return df_evals_pdae


def get_df_evals_new_pdae_batched(pdae_model, x_tr, pert_tr, train_dataloader, x_ood, pert_ood, ood_dataloaders, labels_id_ood, dim_x, device, pdae_name, mmd, subsample_dist_metrics=True):
    
    x_tr = x_tr.to(device)
    pert_tr = pert_tr.to(device)

    df_evals_pdae = pd.DataFrame()
    for e, ood_dataloader in enumerate(ood_dataloaders):
        
        if labels_id_ood is not None:
            label_id_ood = labels_id_ood[e]
        else:   
            label_id_ood = "Undefined"

        # predict
        for x_ood_batch, pert_ood_batch in tqdm(ood_dataloader, desc=f"Evaluating PDAE on {label_id_ood} data", leave=False):
            x_ood_batch = x_ood_batch.to(device)
            pert_ood_batch = pert_ood_batch.to(device)
            x_e_hat_batch, _ = pdae_model.predict(x_tr, pert_tr, pert_ood_batch)
            x_e_hat_batch = x_e_hat_batch.detach() # (n_samples_pooled, dim_x)

            x_e_hat_batch_mean = torch.mean(x_e_hat_batch, dim=0)
            # L2 mean diff
            pdae_mean_diff_e = get_mean_difference(x_ood_batch, x_e_hat_batch_mean)
            # R²
            pdae_r2_e = get_r2_score(x_ood_batch.mean(dim=0), x_e_hat_batch_mean, dim_x)
            
            if subsample_dist_metrics:
                x_e_hat_batch = x_e_hat_batch[torch.randint(low=0, high=x_e_hat_batch.shape[0], size=(x_ood_batch.shape[0],))]

            # Energy Distance 
            try:
                pdae_energy_loss_e = get_energy_distance(x_ood_batch, x_e_hat_batch)
            except RuntimeError as e:
                if str(e).startswith('CUDA out of memory.'):
                    print("Switching to CPU for energy distance due to", e)
                    pdae_energy_loss_e = get_energy_distance(x_ood_batch.cpu(), x_e_hat_batch.cpu())
                else:
                    raise e
                
            # MMD
            try:
                pdae_mmd_e = mmd(x_ood_batch.view(-1, dim_x), x_e_hat_batch.view(-1, dim_x))
            except RuntimeError as e:
                if str(e).startswith('CUDA out of memory.'):
                    print("Switching to CPU for MMD due to", e)
                    mmd = MMDLoss(device="cpu")
                    pdae_mmd_e = mmd(x_ood_batch.cpu().view(-1, dim_x), x_e_hat_batch.cpu().view(-1, dim_x))
                else:
                    raise e
        
            df_evals_pdae_e = pd.DataFrame({
                "Method": [f"PDAE/{pdae_name}"],
                "Env type": [label_id_ood],
                "Energy Distance": [pdae_energy_loss_e.item()],
                "MMD": [pdae_mmd_e.item()],
                "Mean Difference": [pdae_mean_diff_e.item()],
                "R²": [pdae_r2_e],
            })

            df_evals_pdae = pd.concat([df_evals_pdae, df_evals_pdae_e], axis=0)
    
    return df_evals_pdae



def get_evals(
    data_dict: dict,
    data_path: pathlib.Path,
    cpa_model_paths: List[pathlib.Path],
    pdae_model_paths: List[pathlib.Path],
    new_pdae_model_paths: List[pathlib.Path],
    data_type: str,
    seed: int,
    cpa_model_aliases=[],
    pdae_model_aliases=[],
    dataloader_mode="default_index_batches",
    settings_dict_name=None,
    batch_size=1, # TODO: choose batch_size according to config of pdae and cpa
    subsample_mode=False, 
    old_baseline_mode=False,
    list_mode=False,
    n_samples_max=None,
    n_samples_max_holdout=None,
    gpu_id=None,
    device=None,
    control_encoding="triple control",
):

    if isinstance(cpa_model_paths, pathlib.Path):
        cpa_model_paths = [cpa_model_paths]
    if isinstance(pdae_model_paths, pathlib.Path):
        pdae_model_paths = [pdae_model_paths]

    # set device
    if device is None:
        device = get_device(gpu_id)

    # prepare data
    x_tr, pert_tr, train_dataloader, x_ood, pert_ood, labels_id_ood, _env_range, dim_x, n_perturbations, ix_control = prepare_eval_data(data_path, data_dict, dataloader_mode, device, batch_size, n_samples_max, n_samples_max_holdout, data_type, subsample_mode)

    device_orig = device
    device = "cpu" # baseline data is too large for GPU
    mmd = MMDLoss(device=device)

    if subsample_mode:
        name_baseline_evals = "df_evals_baseline_holdout.csv"
    else:
        name_baseline_evals = "df_evals_baseline.csv"
    path_df_evals_baseline = data_path / name_baseline_evals

    if not path_df_evals_baseline.exists():

        df_evals_baseline = pd.DataFrame()

        print("Processing Baseline models")

        for e in tqdm(_env_range):
            xe = x_ood[e].to(device)
            if len(xe.shape) == 2:
                xe = xe.unsqueeze(0)

            pe = pert_ood[e].to(device)
            
            if labels_id_ood is not None:
                label_id_ood = labels_id_ood[e]
            else:
                label_id_ood = "Undefined"

            # Baseline 1: pool all training data and predict the mean (i.e. )
            x_tr_pooled_mean = x_tr.to(device).mean(dim=[0,1])
            pooled_mean_mean_diff_e = get_mean_difference(xe, x_tr_pooled_mean)
            pooled_mean_r2_e = get_r2_score(xe.mean(dim=[0,1]), x_tr_pooled_mean, dim_x)
            if old_baseline_mode:
                pooled_mean_energy_loss_e = get_energy_distance(xe, x_tr_pooled_mean.reshape(1,dim_x))
                pooled_mean_mmd_e = mmd(xe.squeeze(), x_tr_pooled_mean.reshape(1,dim_x))
            else:
                pooled_mean_energy_loss_e = get_energy_distance(xe, x_tr.to(device))
                pooled_mean_mmd_e = mmd(xe.squeeze(), x_tr.to(device).reshape(-1, dim_x))

            # Baseline 1.5: pseudobulked mean
            pseudobulk_x_tr_e = get_pseudo_bulk(x_tr.to(device), pert_tr.to(device), pe.squeeze(), ix_control)
            pseudobulked_mean_x_tr_e = pseudobulk_x_tr_e.mean(dim=[0,1])
            pseudobulked_mean_mean_diff_e = get_mean_difference(xe, pseudobulked_mean_x_tr_e)
            pseudobulked_mean_r2_e = get_r2_score(xe.mean(dim=[0,1]), pseudobulked_mean_x_tr_e, dim_x)
            if old_baseline_mode:
                pseudobulked_mean_energy_loss_e = get_energy_distance(xe, pseudobulked_mean_x_tr_e.reshape(1,dim_x))
                pseudobulked_mean_mmd_e = mmd(xe.squeeze(), pseudobulked_mean_x_tr_e.reshape(-1,dim_x))
            else:
                pseudobulked_mean_energy_loss_e = get_energy_distance(xe, pseudobulk_x_tr_e)
                pseudobulked_mean_mmd_e = mmd(xe.squeeze(), pseudobulk_x_tr_e.reshape(-1,dim_x))

            # Baseline 2: regress the means in observation space on the perturbation labels and use the OLS solution to predict the test mean
            x_tr_means = x_tr.to(device).mean(dim=1) # (n_env, dim_x)
            x_tr_means_centered = x_tr_means - x_tr_means[ix_control]
            beta_hat_ols = torch.linalg.lstsq(pert_tr.to(device), x_tr_means_centered).solution
            xe_mean_hat_ols = pe @ beta_hat_ols + x_tr_means[ix_control]
            ols_mean_diff_e = get_mean_difference(xe, mu_hat=xe_mean_hat_ols)
            ols_r2_e = get_r2_score(xe.mean(dim=[0,1]), xe_mean_hat_ols, dim_x)
            if old_baseline_mode:
                ols_energy_loss_e = get_energy_distance(xe, xe_mean_hat_ols.reshape(1,dim_x))
                ols_mmd_e = mmd(xe.squeeze(), xe_mean_hat_ols.reshape(1,dim_x))
            else:
                x_tr_ctrl = x_tr.to(device)[ix_control].unsqueeze(0)
                x_tr_ctrl_mean = x_tr_ctrl.mean(dim=1)
                x_tr_ctrl_e = x_tr_ctrl - x_tr_ctrl_mean + xe_mean_hat_ols
                ols_energy_loss_e = get_energy_distance(xe, x_tr_ctrl_e)
                ols_mmd_e = mmd(xe.squeeze(), x_tr_ctrl_e.squeeze())

            df_evals_baseline_e = pd.DataFrame({
                "Method": ['Pooled Mean', 'Pseudobulked Mean', 'Linear Regression'],
                "Env type": [label_id_ood] * 3,
                "Energy Distance": [pooled_mean_energy_loss_e.item(), pseudobulked_mean_energy_loss_e.item(), ols_energy_loss_e.item()],
                "MMD": [pooled_mean_mmd_e.item(), pseudobulked_mean_mmd_e.item(), ols_mmd_e.item()],
                "Mean Difference": [pooled_mean_mean_diff_e.item(), pseudobulked_mean_mean_diff_e.item(), ols_mean_diff_e.item()],
                "R²": [pooled_mean_r2_e.item(), pseudobulked_mean_r2_e.item(), ols_r2_e.item()]
            })

            df_evals_baseline = pd.concat([df_evals_baseline, df_evals_baseline_e], axis=0)
            df_evals_baseline.to_csv(path_df_evals_baseline, index=False)
    else:
        df_evals_baseline = pd.read_csv(path_df_evals_baseline)
    
    display(df_evals_baseline.groupby(["Method", "Env type"]).agg(["mean", "std"]))
    
    device = device_orig
    mmd = MMDLoss(device=device)

    # # need string format for cpa
    if len(cpa_model_paths) > 0:
        pert_ood_str = get_cpa_pert_ood_str(data_type, data_path, pert_ood, control_encoding)

    ####
    # CPA
    ####
    df_evals_cpa_all = pd.DataFrame()
    for cpa_model_path in cpa_model_paths:
        print("Processing CPA model", cpa_model_path)

        name_cpa_evals = f"df_evals_{cpa_model_path.stem}"
        if subsample_mode:
            name_cpa_evals += "_holdout.csv"
        else:
            name_cpa_evals += ".csv"
        path_df_evals_cpa = cpa_model_path.parent / name_cpa_evals

        if path_df_evals_cpa.exists():
            df_evals_cpa = pd.read_csv(path_df_evals_cpa)
            df_evals_cpa_all = pd.concat([df_evals_cpa_all, df_evals_cpa], axis=0)
            continue
        
        # load model
        adata = get_cpa_adata(data_type, data_path)
        cov_key, cov_val = get_cpa_cov_info(data_type)
        cpa_api = cpa.api.API(
                adata,
                pretrained=cpa_model_path,
                device=device,
                covariate_keys=[cov_key],
                seed=seed,
        )  

        cpa_history = CPAHistory(cpa_api)
        cpa_model_name = cpa_model_path.stem
        cpa_history.plot_losses(filename=str(cpa_model_path.parents[0] / f"losses_{cpa_model_name}.png"))

        # get evals
        oracle_latents = None
        df_evals_cpa = get_df_evals_cpa(cpa_api, train_dataloader, x_tr, x_ood, pert_ood, pert_ood_str, labels_id_ood, ix_control, dim_x, device, batch_size, _env_range, cpa_model_name, cov_key, cov_val, mmd=mmd, oracle_latents=oracle_latents)
        ###########

        df_evals_cpa.to_csv(path_df_evals_cpa, index=False)

        df_evals_cpa_all = pd.concat([df_evals_cpa_all, df_evals_cpa], axis=0)
        
    if not df_evals_cpa_all.empty:
        display(df_evals_cpa_all.groupby(["Method", "Env type"]).agg(["mean", "std"]))

    ######
    # PDAE
    ######
    df_evals_pdae_all = pd.DataFrame()
    for i, pdae_path in enumerate(pdae_model_paths):
        print("Processing PDAE model", pdae_path)

        name_pdae_evals = f"df_evals_{pdae_path.stem}"
        if subsample_mode:
            name_pdae_evals += "_holdout.csv"
        else:
            name_pdae_evals += ".csv"
        path_df_evals_pdae = pdae_path.parent / name_pdae_evals
        
        if path_df_evals_pdae.exists():
            df_evals_pdae = pd.read_csv(path_df_evals_pdae)
            df_evals_pdae_all = pd.concat([df_evals_pdae_all, df_evals_pdae], axis=0)
            continue

        # load model    
        if settings_dict_name is None:
            settings_name = "pdae_config" + str(pdae_path).split("pdae_model")[1].replace(".pth", "")
        elif settings_dict_name is not None:
            settings_name = settings_dict_name
        settings = read_dict(path=pdae_path.parents[0], file_stem=settings_name)
        dim_z_model = settings["dim_z_model"]
        dim_noise_model = settings["dim_noise_model"]
        sigma_noise_model = settings["sigma_noise_model"]
        decoder_layer_shapes = settings["decoder_layer_shapes"]
        encoder_layer_shapes = settings["encoder_layer_shapes"]
        batch_size = settings["batch_size"]
        update_encoder_on_reconstruction_loss = settings["update_encoder_on_reconstruction_loss"]
        use_OLS_for_perturbation_matrix = settings["use_OLS_for_perturbation_matrix"]

        # set random seed from training
        if "seed" in settings.keys():
            seed = settings["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        pdae_model = PerturbationAutoencoder(dim_z_model, dim_noise_model, sigma_noise_model, dim_x, n_perturbations, encoder_layer_shapes, decoder_layer_shapes, use_OLS_for_perturbation_matrix, ix_control, update_encoder_on_reconstruction_loss)
        pdae_model.load_state_dict(torch.load(pdae_path))
        pdae_model.to(device)
        pdae_model.eval()
        pdae_model.update_perturbation_matrix_via_OLS(pdae_model.encoder(x_tr), pert_tr) 
        
        df_evals_pdae = get_df_evals_pdae(pdae_model, train_dataloader, x_tr, pert_tr, x_ood, pert_ood, labels_id_ood, dim_x, device, batch_size, _env_range, pdae_path, mmd)
        df_evals_pdae.to_csv(path_df_evals_pdae, index=False)

        df_evals_pdae_all = pd.concat([df_evals_pdae_all, df_evals_pdae], axis=0)
    if not df_evals_pdae_all.empty:
        display(df_evals_pdae_all.groupby(["Method", "Env type"]).agg(["mean", "std"]))


    ######
    # PDAE NEW
    ######
    df_evals_new_pdae_all = pd.DataFrame()
    for i, pdae_path in enumerate(new_pdae_model_paths):
        print("Processing PDAE model", pdae_path)

        name_pdae_evals = f"df_evals_{pdae_path.stem}"
        if subsample_mode:
            name_pdae_evals += "_holdout.csv"
        else:
            name_pdae_evals += ".csv"
        path_df_evals_pdae = pdae_path.parent / name_pdae_evals
        
        if path_df_evals_pdae.exists():
            df_evals_pdae = pd.read_csv(path_df_evals_pdae)
            df_evals_new_pdae_all = pd.concat([df_evals_new_pdae_all, df_evals_pdae], axis=0)
            continue

        # load model    
        if settings_dict_name is None:
            settings_name = "pdae_config" + str(pdae_path).split("pdae_model")[1].replace(".pth", "")
        elif settings_dict_name is not None:
            settings_name = settings_dict_name
        else:
            raise NotImplementedError
        settings = read_dict(path=pdae_path.parents[0], file_stem=settings_name)
        dim_z_model = settings["dim_z_model"]
        dim_noise_model = settings["dim_noise_model"]
        sigma_noise_model = settings["sigma_noise_model"]
        decoder_layer_shapes = settings["decoder_layer_shapes"]
        encoder_layer_shapes = settings["encoder_layer_shapes"]
        batch_size = settings["batch_size"]
        update_encoder_on_reconstruction_loss = settings["update_encoder_on_reconstruction_loss"]
        use_OLS_for_perturbation_matrix = settings.get("use_OLS_for_perturbation_matrix")

        # set random seed from training
        if "seed" in settings.keys():
            seed = settings["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        pdae_model = PerturbationAutoencoder(dim_z_model, dim_noise_model, sigma_noise_model, dim_x, n_perturbations, encoder_layer_shapes, decoder_layer_shapes, use_OLS_for_perturbation_matrix, ix_control, update_encoder_on_reconstruction_loss)
        if use_OLS_for_perturbation_matrix:
            pdae_model.update_perturbation_matrix_via_OLS_old(x_tr, pert_tr)
        pdae_model.load_state_dict(torch.load(pdae_path))
        pdae_model.to(device)
        pdae_model.eval()

        # get data
        data_dict_new = get_data_dict_new_pdae(data_dict)
        print(data_dict_new.keys())
        if subsample_mode:
            x_tr_new = data_dict_new["tr_holdout"]["observations"]
            pert_tr_new = data_dict_new["tr_holdout"]["perturbation_labels"]
            train_dataset = PDAEDataset(x_tr_new, pert_tr_new)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        else:
            x_tr_new = data_dict_new["tr"]["observations"]
            pert_tr_new = data_dict_new["tr"]["perturbation_labels"]
            train_dataset = PDAEDataset(x_tr_new, pert_tr_new)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        
        ood_dataloaders = []
        if data_dict_new["val"]["observations"].numel() != 0:
            val_dataset = PDAEDataset(data_dict_new["val"]["observations"], data_dict_new["val"]["perturbation_labels"])
            val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False)
            ood_dataloaders.append(val_dataloader)
        if data_dict_new["te"]["observations"].numel() != 0:
            test_dataset = PDAEDataset(data_dict_new["te"]["observations"], data_dict_new["te"]["perturbation_labels"])
            test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False)
            ood_dataloaders.append(test_dataloader)

        x_ood = torch.cat([data_dict_new["val"]["observations"], data_dict_new["te"]["observations"]], dim=0)
        pert_ood = torch.cat([data_dict_new["val"]["perturbation_labels"], data_dict_new["te"]["perturbation_labels"]], dim=0)
        labels_id_ood = ["id"] * data_dict_new["val"]["observations"].shape[0] + ["ood"] * data_dict_new["te"]["observations"].shape[0]

        pdae_name = pdae_path.stem
        # df_evals_pdae = get_df_evals_new_pdae_batched(pdae_model, x_tr_new, pert_tr_new, train_dataloader, x_ood, pert_ood, ood_dataloaders, labels_id_ood, dim_x, device, pdae_name, mmd)
        df_evals_pdae = get_df_evals_new_pdae(pdae_model, train_dataloader, x_tr_new, pert_tr_new, x_ood, pert_ood, labels_id_ood, dim_x, device, batch_size, _env_range, pdae_name, mmd)
        df_evals_pdae.to_csv(path_df_evals_pdae, index=False)

        df_evals_new_pdae_all = pd.concat([df_evals_new_pdae_all, df_evals_pdae], axis=0)
    if not df_evals_new_pdae_all.empty:
        display(df_evals_new_pdae_all.groupby(["Method", "Env type"]).agg(["mean", "std"]))


    # final table
    df_evals_full = pd.DataFrame()
    for df in [df_evals_baseline, df_evals_cpa_all, df_evals_pdae_all, df_evals_new_pdae_all]:
        if not df.empty:
            df_evals_full = pd.concat([df_evals_full, df], axis=0)
    
    if len(pdae_model_aliases) > 0:
        for pdae_model_path, pdae_alias in zip(pdae_model_paths, pdae_model_aliases):
            df_evals_full.loc[df_evals_full["Method"] == f"PDAE/{pdae_model_path}", "Method"] = pdae_alias
    if len(cpa_model_aliases) > 0:
        for cpa_model_path, cpa_alias in zip(cpa_model_paths, cpa_model_aliases):
            df_evals_full.loc[df_evals_full["Method"] == f"CPA/{cpa_model_path}", "Method"] = cpa_alias
    
    df_evals = df_evals_full.groupby(["Method", "Env type"]).agg(["mean", "std"])
    display(df_evals)



    # df_evals_full.to_csv(data_path / "df_evals_full.csv", index=False)
    
    # # R² plot
    # df_r2 = df_evals_full.groupby("Method")["R²"].apply(list)
    # df_r2 = df_r2.to_frame()
    # df_r2["r2_median"] = df_r2.apply(lambda x: np.median(x["R²"]), axis=1).sort_values()
    # df_r2 = df_r2.sort_values("r2_median")
    # xlabs = df_r2.index
    # xlabs = [x.replace(" ", "\n") for x in xlabs]

    # plt.boxplot(df_r2["R²"])
    # plt.xticks(list(range(1, len(xlabs)+1)), xlabs)
    # plt.title("R² scores of regressing predicted on true gene-wise means")
    # plt.xlabel("(methods sorted by ascending median R² score)")
    # plt.show()

    return df_evals_full