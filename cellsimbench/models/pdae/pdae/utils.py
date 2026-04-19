import os
from dataclasses import dataclass
import json
import math
import os
import pathlib
from typing import List, Union
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

from PerturbationExtrapolation.pdae.model import PerturbationAutoencoder

def set_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def plot_mmdagg_test_results(tr_mmd_test_results: dict):
    for env in tr_mmd_test_results.keys():
        fig, ax = plt.subplots(figsize=(10, 6))
        num_steps = len(tr_mmd_test_results[env]["reject"])
        sns.lineplot(x=range(num_steps), y=tr_mmd_test_results[env]["reject"], label="Rejection")
        sns.lineplot(x=range(num_steps), y=tr_mmd_test_results[env]["p_min"], label="Min p-value")
        sns.lineplot(x=range(num_steps), y=tr_mmd_test_results[env]["p_median"], label="Median p-value")
        sns.lineplot(x=range(num_steps), y=tr_mmd_test_results[env]["p_max"], label="Max p-value")
        sns.lineplot(x=range(num_steps), y=np.median(tr_mmd_test_results[env]["thresholds"], axis=1), label="Median p-value threshold")
        plt.title(f"MMDAgg test results for train environment {env+1} (20 tests)")
        plt.xlabel("MMD Test step")
        plt.legend()
        plt.show()
        

def plot_pdae_gt_vs_predicted_distributions(
    model: PerturbationAutoencoder, 
    model_name: str, 
    epoch: int, 
    save_dir: str, 
    device: torch.device, 
    x_tr_holdout: torch.Tensor, 
    z_tr_holdout: torch.Tensor, 
    pert_tr_holdout: torch.Tensor, 
    x_val: torch.Tensor, 
    z_val: torch.Tensor, 
    pert_val: torch.Tensor, 
    x_te: torch.Tensor, 
    z_te: torch.Tensor, 
    pert_te: torch.Tensor, 
    plot_distribution_ratio_sample: Union[float, int]
) -> None:
    # assert x_val.shape[-1] == 2, "Plotting distributions is only supported for 2D data."
    assert z_tr_holdout is not None and z_val is not None, "z_tr and z_val must be provided for plotting distributions."
    # tr data
    pert_tr_unique_envs = torch.unique(pert_tr_holdout, dim=0)
    x_tr_hat_full = torch.tensor([]).to(device)
    z_tr_hat_full = torch.tensor([]).to(device)
    pert_tr_full = np.repeat(pert_tr_unique_envs.cpu().numpy(), x_tr_holdout.shape[0], axis=0)
    for pert_env in pert_tr_unique_envs:
        pert_env = pert_env.to(device)
        x_tr_hat_full = torch.cat([x_tr_hat_full, model.predict(x_tr_holdout, pert_tr_holdout, pert_env)[0]], dim=0)
        z_tr_hat_full = torch.cat([z_tr_hat_full, model.predict(x_tr_holdout, pert_tr_holdout, pert_env)[1]], dim=0)
    assert x_tr_hat_full.shape[0] == pert_tr_full.shape[0]
    
    pert_val_unique_envs = torch.unique(pert_val, dim=0)
    x_val_hat_full = torch.tensor([]).to(device)
    z_val_hat_full = torch.tensor([]).to(device)
    pert_val_full = np.repeat(pert_val_unique_envs.cpu().numpy(), x_tr_holdout.shape[0], axis=0)
    for pert_env in pert_val_unique_envs:
        pert_env = pert_env.to(device)
        x_val_hat_full = torch.cat([x_val_hat_full, model.predict(x_tr_holdout, pert_tr_holdout, pert_env)[0]], dim=0)
        z_val_hat_full = torch.cat([z_val_hat_full, model.predict(x_tr_holdout, pert_tr_holdout, pert_env)[1]], dim=0)
    assert x_val_hat_full.shape[0] == pert_val_full.shape[0]

    pert_te_unique_envs = torch.unique(pert_te, dim=0)#
    x_te_hat_full = torch.tensor([]).to(device)
    z_te_hat_full = torch.tensor([]).to(device)
    pert_te_full = np.repeat(pert_te_unique_envs.cpu().numpy(), x_tr_holdout.shape[0], axis=0)
    for pert_env in pert_te_unique_envs:
        pert_env = pert_env.to(device)
        x_te_hat_full = torch.cat([x_te_hat_full, model.predict(x_tr_holdout, pert_tr_holdout, pert_env)[0]], dim=0)
        z_te_hat_full = torch.cat([z_te_hat_full, model.predict(x_tr_holdout, pert_tr_holdout, pert_env)[1]], dim=0)
    assert x_te_hat_full.shape[0] == pert_te_full.shape[0]

    # create density plot data
    x_gt = torch.cat([x_tr_holdout.cpu(), x_val.cpu(), x_te.cpu()], dim=0).numpy()
    z_gt = torch.cat([z_tr_holdout.cpu(), z_val.cpu(), z_te.cpu()], dim=0).numpy()
    pert_gt = np.concatenate([pert_tr_holdout.cpu().numpy(), pert_val.cpu().numpy(), pert_te.cpu().numpy()], axis=0)
    x_pred = torch.cat([x_tr_hat_full.cpu(), x_val_hat_full.cpu(), x_te_hat_full.cpu()], dim=0).numpy()
    z_pred = torch.cat([z_tr_hat_full.cpu(), z_val_hat_full.cpu(), z_te_hat_full.cpu()], dim=0).numpy()
    pert_pred = np.concatenate([pert_tr_full, pert_val_full, pert_te_full], axis=0)

    gt_plot_data = DensityPlotData2D(
        title='Ground Truth',
        observations=x_gt[:,:2],
        latents=z_gt[:,:2],
        perturbations=pert_gt,
        colors=["black"]*x_tr_holdout.shape[0] + ["blue"]*x_val.shape[0] + ["orange"]*x_te.shape[0],
        labels=["train"]*x_tr_holdout.shape[0] + ["val"]*x_val.shape[0] + ["test"]*x_te.shape[0],
        is_ground_truth=True
    )

    pdae_plot_data = DensityPlotData2D(
        title='PDAE',
        observations=x_pred[:,:2],
        latents=z_pred[:,:2],
        perturbations=pert_pred,
        colors=["black"]*x_tr_hat_full.shape[0] + ["blue"]*x_val_hat_full.shape[0] + ["orange"]*x_te_hat_full.shape[0],
        labels=["train"]*x_tr_hat_full.shape[0] + ["val"]*x_val_hat_full.shape[0] + ["test"]*x_te_hat_full.shape[0],
        is_ground_truth=False
    )
    
    make_density_plot_flat(
        [gt_plot_data, pdae_plot_data],
        title=f"Density Plot {model_name} Epoch {epoch}",
        save_path=os.path.join(save_dir, f"density_plot_{model_name}_epoch_{epoch}.png"),
        ratio_sample=plot_distribution_ratio_sample,
        fontsize=16
    )


def save_dict(name: str, _dict: dict, path: pathlib.Path):
    with open(str(path / f'{name}.json'), 'w') as f:
        json.dump(_dict, f, indent=4, default=str)

@dataclass
class DensityPlotData2D():
    title: str = ""
    observations: np.ndarray = None
    latents: np.ndarray = None
    perturbations: np.ndarray = None
    colors: list[str] = None
    labels: list[str] = None
    is_ground_truth: bool = False


def make_density_plot_flat(list_plot_data: List[DensityPlotData2D], title, save_path, ratio_sample=0.5, fontsize=16):
    sns.set_style("darkgrid")
    n_cols = max(len(list_plot_data), 2)
    fig, axs = plt.subplots(2, n_cols, figsize=(6*n_cols, 5*2))
    
    for i, data in enumerate(list_plot_data):
        
        data.colors = np.array(data.colors)
        data.labels = np.array(data.labels)

        assert data.observations.shape[0] == data.latents.shape[0] == data.perturbations.shape[0] == data.colors.shape[0], "Observations, latents, perturbations and colors must have the same number of samples."

        if ratio_sample < 1:
            ixs = np.random.choice(range(data.observations.shape[0]), size=math.ceil(data.observations.shape[0]*ratio_sample))
            data.observations = data.observations[ixs]
            data.latents = data.latents[ixs]
            data.perturbations = data.perturbations[ixs]
            data.colors = data.colors[ixs]
            data.labels = data.labels[ixs]

        axs[0,i].set_title(data.title, weight='bold', fontsize=fontsize)

        pert_unique = np.unique(data.perturbations, axis=0)
        
        # latents
        if data.is_ground_truth:
            axs[0,i].set_xlabel('$Z_1$', weight='bold', fontsize=fontsize)
            axs[0,i].set_ylabel('$Z_2$', weight='bold', fontsize=fontsize)
        else:
            axs[0,i].set_xlabel('$\widehat{Z}_1$', weight='bold', fontsize=fontsize)
            axs[0,i].set_ylabel('$\widehat{Z}_2$', weight='bold', fontsize=fontsize)
        colors = []
        labels = []
        for j, pert_env in enumerate(pert_unique):
            pert_mask = (data.perturbations == pert_env).all(axis=1)
            color = data.colors[pert_mask][0]
            label = data.labels[pert_mask][0]
            colors.append(color)
            labels.append(label)
            sns.kdeplot(x=data.latents[pert_mask, 0], y=data.latents[pert_mask, 1], ax=axs[0,i], color=color, linestyles='-', linewidths=0.5)
            axs[0,i].plot(data.latents[pert_mask, 0].mean(), data.latents[pert_mask, 1].mean(), marker="X", color=color)

        ## observations
        if data.is_ground_truth:
            axs[1,i].set_xlabel('$X_1$', weight='bold', fontsize=fontsize)
            axs[1,i].set_ylabel('$X_2$', weight='bold', fontsize=fontsize)
        else:
            axs[1,i].set_xlim(axs[1,0].get_xlim())
            axs[1,i].set_ylim(axs[1,0].get_ylim())
            axs[1,i].set_xlabel('$\widehat{X}_1$', weight='bold', fontsize=fontsize)
            axs[1,i].set_ylabel('$\widehat{X}_2$', weight='bold', fontsize=fontsize)
        for j, pert_env in enumerate(pert_unique):
            pert_mask = (data.perturbations == pert_env).all(axis=1)
            color = colors[j]
            label = labels[j]
            sns.kdeplot(x=data.observations[pert_mask, 0], y=data.observations[pert_mask, 1], ax=axs[1,i], color=color, linestyles='-', linewidths=0.5)
            axs[1,i].plot(data.observations[pert_mask, 0].mean(), data.observations[pert_mask, 1].mean(), marker="X", color=color)
    # Create legend handles
    colors = list(dict.fromkeys(data.colors))  
    labels = list(dict.fromkeys(data.labels))
    legend_handles = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]

    ncol = len(legend_handles)

    # Add horizontal legend
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05 / 16 * fontsize),  # Position below the plot
        ncol=ncol,  # Horizontal layout
        fontsize=fontsize
    )
    plt.suptitle(title, weight='bold', fontsize=fontsize)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def show_mixing(true_mixing):
    # Define the range for the gridlines
    x = np.linspace(-.5, 2.5, 50)
    y = np.linspace(-.5, 2.5, 50)

    # Create a meshgrid
    X, Y = np.meshgrid(x, y)

    # Transform the grid points using the fixed mixing function
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)
    mixed_points = true_mixing(grid_points_tensor).detach().numpy()

    # Reshape the transformed points to match the grid shape
    X1 = mixed_points[:, 0].reshape(X.shape)
    Y1 = mixed_points[:, 1].reshape(Y.shape)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the original gridlines
    ax1.plot(X, Y, color='gray', linestyle='-', linewidth=0.5)
    ax1.plot(X.T, Y.T, color='gray', linestyle='-', linewidth=0.5)
    ax1.set_title('Cartesian Gridlines')
    ax1.set_xlabel('$Z_1$')
    ax1.set_ylabel('$Z_2$')

    # Plot the transformed gridlines
    ax2.plot(X1, Y1, color='blue', linestyle='-', linewidth=0.5)
    ax2.plot(X1.T, Y1.T, color='blue', linestyle='-', linewidth=0.5)
    ax2.set_title('Transformation of Gridlines by Fixed Mixing')
    ax2.set_xlabel('$X_1$')
    ax2.set_ylabel('$X_2$')
    plt.tight_layout()
    # plt.savefig(os.path.join(current_time, 'mixing_function.pdf'), format='pdf', bbox_inches='tight')
    # plt.savefig(os.path.join(current_time, 'mixing_function.png'), format='png', bbox_inches='tight')
    plt.show()

def show_GT_data(true_z_tr, true_z_val, true_z_te, x_tr, x_val, x_te, save_dir, ratio_sample=1.0):
    _, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Subsample data if ratio_sample < 1
    if ratio_sample < 1.0:
        def subsample(data):
            assert data.ndim == 3, "Data must be 3D (n_envs, n_samples, dim_x)"
            n_samples = math.ceil(data.shape[1] * ratio_sample)
            indices = np.random.choice(data.shape[1], n_samples, replace=False)
            return data[:, indices, :]
        
        # print(true_z_tr.shape, true_z_val.shape, true_z_te.shape, x_tr.shape, x_val.shape, x_te.shape)
        true_z_tr = subsample(true_z_tr)
        true_z_val = subsample(true_z_val)
        true_z_te = subsample(true_z_te)
        x_tr = subsample(x_tr)
        x_val = subsample(x_val)
        x_te = subsample(x_te)
        # print(true_z_tr.shape, true_z_val.shape, true_z_te.shape, x_tr.shape, x_val.shape, x_te.shape)

    # GT latents
    axs[0].set_aspect('equal')
    axs[0].set_title('Latent Space')
    axs[0].set_xlabel('$Z_1$')
    axs[0].set_ylabel('$Z_2$')
    for i in range(true_z_tr.shape[0]):
        sns.kdeplot(x=true_z_tr[i, :, 0], y=true_z_tr[i, :, 1], ax=axs[0], label=f'Perturbation {i}', color='black', linestyles='--', linewidths=0.5)
    for i in range(true_z_val.shape[0]):
        sns.kdeplot(x=true_z_val[i, :, 0], y=true_z_val[i, :, 1], ax=axs[0], label='Validation', color='green', linestyles='-', linewidths=1)
    for i in range(true_z_te.shape[0]):
        sns.kdeplot(x=true_z_te[i, :, 0], y=true_z_te[i, :, 1], ax=axs[0], label='Test', color='red', linestyles='-', linewidths=1)

    # GT observations
    axs[1].set_title('Observation Space')
    axs[1].set_xlabel('$X_1$')
    axs[1].set_ylabel('$X_2$')
    for i in range(x_tr.shape[0]):
        sns.kdeplot(x=x_tr[i, :, 0], y=x_tr[i, :, 1], ax=axs[1], label=f'Perturbation {i}', color='black', linestyles='--', linewidths=0.5)

    for i in range(x_val.shape[0]):
        sns.kdeplot(x=x_val[i, :, 0], y=x_val[i, :, 1], ax=axs[1], label='Validation', color='green', linestyles='-', linewidths=1)
    for i in range(x_te.shape[0]):
        sns.kdeplot(x=x_te[i, :, 0], y=x_te[i, :, 1], ax=axs[1], label='Test', color='red', linestyles='-', linewidths=1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'GT_data.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'GT_data.png'), format='png', bbox_inches='tight')
    plt.show()


def show_loss_curves(model_path, model_name, losses: dict, losses_to_plot: list, mmd_stats: dict = None, use_log_scale: bool = True, rename_dict: dict = None):
    plt.figure(figsize=(10, 5))

    if mmd_stats is not None and len(mmd_stats['mmd_epochs']) > 0:
        mmd_p_median_all_envs = []
        for env in mmd_stats["tr_mmd_test_results"].keys():
            mmd_p_median_all_envs.append(torch.tensor(mmd_stats["tr_mmd_test_results"][env]["p_median"]).unsqueeze(0))
        mmd_p_median_all_envs = torch.cat(mmd_p_median_all_envs, dim=0)
        mmd_p_median_mean = mmd_p_median_all_envs.mean(dim=0)
        losses["tr_mmd_p_median_mean"] = mmd_p_median_mean

        mmd_p_median_all_envs = []
        for env in mmd_stats["val_mmd_test_results"].keys():
            mmd_p_median_all_envs.append(torch.tensor(mmd_stats["val_mmd_test_results"][env]["p_median"]).unsqueeze(0))
        mmd_p_median_all_envs = torch.cat(mmd_p_median_all_envs, dim=0)
        mmd_p_median_mean = mmd_p_median_all_envs.mean(dim=0)
        losses["val_mmd_p_median_mean"] = mmd_p_median_mean

        losses["mmd_epochs"] = mmd_stats["mmd_epochs"]

    for loss in losses.keys():
        if loss not in losses_to_plot:
            continue
        if "epochs" in loss:
            continue
        if rename_dict is not None and loss in rename_dict:
            loss_plot_name = rename_dict[loss]
        else:
            loss_plot_name = loss
        loss_values = losses[loss]
        
        if "tr" in loss and "val" not in loss and "mmd" not in loss:
            plt.plot(losses["tr_epochs"], loss_values, label=loss_plot_name)
        elif "val" in loss and "mmd" not in loss:
            plt.plot(losses["val_epochs"], loss_values, label=loss_plot_name)
        elif "mmd" in loss:
            plt.plot(losses["mmd_epochs"], loss_values, label=loss_plot_name)

    plt.xlabel('Epoch')
    plt.yscale('log')
    if use_log_scale:
        plt.ylabel('Value (log-scale)')
    else:
        plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(str(model_path), f'training_curves_{model_name}.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(str(model_path), f'training_curves_{model_name}.png'), format='png', bbox_inches='tight')
    plt.show()


def flatten_arr(arr):
    return arr.reshape(-1, arr.shape[-1]) # (n_envs, n_samples, dim) -> (n_envs * n_samples, dim)


def show_results(path, true_z_tr, true_z_val, true_z_te, x_tr, x_val, x_te, z_tr_hat, z_val_hat, z_te_hat, x_tr_hat, x_val_hat, x_te_hat):
    _, axs = plt.subplots(2, 2, figsize=(10, 10))

    # GT latents
    axs[0,0].set_aspect('equal')
    axs[0,0].set_title('Ground Truth')
    axs[0,0].set_xlabel('$Z_1$')
    axs[0,0].set_ylabel('$Z_2$')
    for i in range(true_z_tr.shape[0]):
        sns.kdeplot(x=true_z_tr[i, :, 0], y=true_z_tr[i, :, 1], ax=axs[0,0], label=f'Perturbation {i}', color='black', linestyles='--', linewidths=0.5)
    
    sns.kdeplot(x=flatten_arr(true_z_val)[:, 0], y=flatten_arr(true_z_val)[:, 1], ax=axs[0,0], label='Validation', color='green', linestyles='-', linewidths=1)
    sns.kdeplot(x=flatten_arr(true_z_te)[:, 0], y=flatten_arr(true_z_te)[:, 1], ax=axs[0,0], label='Test', color='red', linestyles='-', linewidths=1)

    # GT observations
    axs[1,0].set_xlabel('$X_1$')
    axs[1,0].set_ylabel('$X_2$')
    for i in range(x_tr.shape[0]):
        sns.kdeplot(x=x_tr[i, :, 0], y=x_tr[i, :, 1], ax=axs[1,0], label=f'Perturbation {i}', color='black', linestyles='--', linewidths=0.5)
    
    sns.kdeplot(x=flatten_arr(x_val)[:, 0], y=flatten_arr(x_val)[:, 1], ax=axs[1,0], label='Validation', color='green', linestyles='-', linewidths=1)
    sns.kdeplot(x=flatten_arr(x_te)[:, 0], y=flatten_arr(x_te)[:, 1], ax=axs[1,0], label='Test', color='red', linestyles='-', linewidths=1)

    # Predicted latents from held out data
    axs[0,1].set_title('Predicted from held out data')
    axs[0,1].set_xlabel('$\widehat{Z}_1$')
    axs[0,1].set_ylabel('$\widehat{Z}_2$')
    for i in range(z_tr_hat.shape[0]):
        sns.kdeplot(x=z_tr_hat[i, :, 0], y=z_tr_hat[i, :, 1], ax=axs[0,1], label=f'Perturbation {i}', color='black', linestyles='--', linewidths=0.5)
    
    sns.kdeplot(x=flatten_arr(z_val_hat)[:, 0], y=flatten_arr(z_val_hat)[:, 1], ax=axs[0,1], label='Val', color='green', linestyles='-', linewidths=1)
    sns.kdeplot(x=flatten_arr(z_te_hat)[:, 0], y=flatten_arr(z_te_hat)[:, 1], ax=axs[0,1], label='Test', color='red', linestyles='-', linewidths=1)

    
    # Predicted observations from held out data
    axs[1,1].set_xlim(axs[1,0].get_xlim())
    axs[1,1].set_ylim(axs[1,0].get_ylim())
    axs[1,1].set_xlabel('$\widehat{X}_1$')
    axs[1,1].set_ylabel('$\widehat{X}_2$')
    for i in range(z_tr_hat.shape[0]):
        sns.kdeplot(x=x_tr_hat[i, :, 0], y=x_tr_hat[i, :, 1], ax=axs[1,1], label=f'Perturbation {i}', color='black', linestyles='--', linewidths=0.5)
    
    sns.kdeplot(x=flatten_arr(x_val_hat)[:, 0], y=flatten_arr(x_val_hat)[:, 1], ax=axs[1,1], label='Val', color='green', linestyles='-', linewidths=1)
    sns.kdeplot(x=flatten_arr(x_te_hat)[:, 0], y=flatten_arr(x_te_hat)[:, 1], ax=axs[1,1], label='Test', color='red', linestyles='-', linewidths=1)

    # # Predicted latents from training data
    # axs[0,2].set_title('Predicted from training data')
    # axs[0,2].set_xlabel('$\widehat{Z}_1$')
    # axs[0,2].set_ylabel('$\widehat{Z}_2$')
    # for i in range(z_tr_hat.shape[0]):
    #     sns.kdeplot(x=z_tr_hat[i, :, 0], y=z_tr_hat[i, :, 1], ax=axs[0,2], label=f'Perturbation {i}', color='black', linestyles='--', linewidths=0.5)
    
    # sns.kdeplot(x=z_val_hat_full[:, 0], y=z_val_hat_full[:, 1], ax=axs[0,2], label='Val', color='green', linestyles='-', linewidths=1)
    # sns.kdeplot(x=z_te_hat_full[:, 0], y=z_te_hat_full[:, 1], ax=axs[0,2], label='Test', color='red', linestyles='-', linewidths=1)
    
    # # Predicted observations from training data
    # axs[1,2].set_xlim(axs[1,0].get_xlim())
    # axs[1,2].set_ylim(axs[1,0].get_ylim())
    # axs[1,2].set_xlabel('$\widehat{X}_1$')
    # axs[1,2].set_ylabel('$\widehat{X}_2$')
    # for i in range(z_tr_hat.shape[0]):
    #     sns.kdeplot(x=x_tr_hat[i, :, 0], y=x_tr_hat[i, :, 1], ax=axs[1,2], label=f'Perturbation {i}', color='black', linestyles='--', linewidths=0.5)
    
    # sns.kdeplot(x=x_val_hat_full[:, 0], y=x_val_hat_full[:, 1], ax=axs[1,2], label='Val', color='green', linestyles='-', linewidths=1)
    # sns.kdeplot(x=x_te_hat_full[:, 0], y=x_te_hat_full[:, 1], ax=axs[1,2], label='Test', color='red', linestyles='-', linewidths=1)

    plt.tight_layout()
    plt.savefig(path / 'GT_vs_prediction.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(path / 'GT_vs_prediction.png', format='png', bbox_inches='tight')
    plt.show()


def show_results_alternative(true_z_tr, true_z_val, true_z_te, x_tr, x_val, x_te, z_tr_hat, z_val_hat, z_te_hat, x_tr_hat, x_val_hat, x_te_hat):
    # plot the GT and predicted distributions
    _, axs = plt.subplots(1, 4, figsize=(16, 4.25))
    sns.color_palette("dark")

    # GT latents
    axs[0].set_title('Truth (Latent Space)', weight='bold', fontsize=16)
    axs[0].set_xlabel('$Z_1$')
    axs[0].set_ylabel('$Z_2$')

    for i in range(true_z_tr.shape[0]):
        sns.kdeplot(x=true_z_tr[i, :, 0], y=true_z_tr[i, :, 1], label="Train", ax=axs[0],  color='black', linestyles='--', linewidths=0.5)

    sns.kdeplot(x=true_z_val[:, 0], y=true_z_val[:, 1], label="Test 1", ax=axs[0], linestyles='-', linewidths=1)
    sns.kdeplot(x=true_z_te[:, 0], y=true_z_te[:, 1], label="Test 2", ax=axs[0], linestyles='-', linewidths=1)
    # axs[0].legend()

    # GT observations
    axs[1].set_title('Truth (Observation Space)', weight='bold', fontsize=16)
    axs[1].set_xlabel('$X_1$')
    axs[1].set_ylabel('$X_2$')
    for i in range(x_tr.shape[0]):
        sns.kdeplot(x=x_tr[i, :, 0], y=x_tr[i, :, 1], ax=axs[1], label=f'Perturbation {i}', color='black', linestyles='--', linewidths=0.5)

    sns.kdeplot(x=x_val[:, 0], y=x_val[:, 1], ax=axs[1], label='Validation', linestyles='-', linewidths=1)
    sns.kdeplot(x=x_te[:, 0], y=x_te[:, 1], ax=axs[1], label='Test', linestyles='-', linewidths=1)

    # Predicted latents from held out data
    axs[2].set_title('Inferred Latents', weight='bold', fontsize=16)
    axs[2].set_xlabel('$\widehat{Z}_1$')
    axs[2].set_ylabel('$\widehat{Z}_2$')
    for i in range(z_tr_hat.shape[0]):
        sns.kdeplot(x=z_tr_hat[i, :, 0], y=z_tr_hat[i, :, 1], ax=axs[2], label=f'Perturbation {i}', color='black', linestyles='--', linewidths=0.5)

    sns.kdeplot(x=z_val_hat[:, 0], y=z_val_hat[:, 1], ax=axs[2], label='Val', linestyles='-', linewidths=1)
    sns.kdeplot(x=z_te_hat[:, 0], y=z_te_hat[:, 1], ax=axs[2], label='Test', linestyles='-', linewidths=1)


    # Predicted observations from held out data
    axs[3].set_title('Predicted Observations', weight='bold', fontsize=16)
    axs[3].set_xlim(axs[1].get_xlim())
    axs[3].set_ylim(axs[1].get_ylim())
    axs[3].set_xlabel('$\widehat{X}_1$')
    axs[3].set_ylabel('$\widehat{X}_2$')
    for i in range(z_tr_hat.shape[0]):
        sns.kdeplot(x=x_tr_hat[i, :, 0], y=x_tr_hat[i, :, 1], ax=axs[3], label=f'Perturbation {i}', color='black', linestyles='--', linewidths=0.5)

    sns.kdeplot(x=x_val_hat[:, 0], y=x_val_hat[:, 1], ax=axs[3], label='Val', linestyles='-', linewidths=1)
    sns.kdeplot(x=x_te_hat[:, 0], y=x_te_hat[:, 1], ax=axs[3], label='Test', linestyles='-', linewidths=1)


    plt.tight_layout()
    # plt.savefig(os.path.join(current_time, 'rs.pdf'), format='pdf', bbox_inches='tight')
    # plt.savefig(os.path.join(current_time, 'rs.png'), format='png', bbox_inches='tight')
    plt.show()