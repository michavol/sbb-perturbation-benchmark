import os
from pathlib import Path

import scanpy as sc
from tqdm import tqdm	
import torch
from sklearn.model_selection import ParameterGrid

import optuna
from optuna.samplers import TPESampler, RandomSampler

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.train.torch import get_device

from PerturbationExtrapolation.pdae_training import train_and_eval_pdae
from mt.cpa_utils.cpa_training import train_cpa, train_and_eval_cpa
from mt.experiments_simulation.simulation_config import get_cpa_simulation_config
from mt.utils import param_dict_to_str, round_sig


# Define the trainable wrapper
def pdae_objective(config, static_config, save_dir, eval_arg_dict, eval_mode):

    param_str = param_dict_to_str(config, sig=3)

    train_config = static_config.copy()
    for k in config.keys():
        train_config[k] = config[k]
    
    device = get_device()
    if eval_mode == "random_search":
        (
            energy_dist, 
            mmd, 
            mean_diff) = train_and_eval_pdae(train_config, save_dir, device, eval_arg_dict, pdae_name=param_str, eval_mode=eval_mode)
        loss_dict = {
            "eval_energy_dist": energy_dist,
            "eval_mmd": mmd,
            "eval_mean_diff": mean_diff,
        }
    
    tune.report(loss_dict)

    return loss_dict


def run_pdae_optuna_ray_search(static_config, initial_params, param_space, save_dir, eval_arg_dict, num_samples, time_budget_h, resource_dict, max_concurrent_trials, eval_mode, seed):
    
    if isinstance(initial_params, dict):
        initial_params = [initial_params]

    if eval_mode == "random_search":
        optuna_search = OptunaSearch(
            metric=[
                "eval_energy_dist",
                "eval_mmd",
                "eval_mean_diff",
            ],
            mode=["min"]*3,
            points_to_evaluate=initial_params,
            sampler=RandomSampler(seed=seed),
        )

    trainable_with_params = tune.with_parameters(pdae_objective, static_config=static_config, save_dir=save_dir, eval_arg_dict=eval_arg_dict, eval_mode=eval_mode)
    trainable_with_resources  = tune.with_resources(trainable_with_params, resource_dict)
    tuner = tune.Tuner(
        trainable=trainable_with_resources,
        run_config=ray.air.RunConfig(
            storage_path=str(save_dir / "ray_results"),
        ),
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials,
            time_budget_s=int(time_budget_h*60*60),
            search_alg=optuna_search
        ),
        param_space=param_space,
    )

    results = tuner.fit()

    df = results.get_dataframe()
    df.to_csv(save_dir / "tune_results.csv", index=False)
