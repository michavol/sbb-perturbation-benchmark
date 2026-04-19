import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader

from PerturbationExtrapolation.pdae.utils import plot_pdae_gt_vs_predicted_distributions, set_random_seed, show_loss_curves, save_dict
from PerturbationExtrapolation.pdae.losses import get_perturbation_energy_loss, get_reconstruction_loss, get_l21_loss, get_marginal_prior_energy_loss
from PerturbationExtrapolation.pdae.metrics import EnergyLossMetric, MeanDifferenceMetric
from PerturbationExtrapolation.pdae.model import PerturbationAutoencoder
from PerturbationExtrapolation.pdae.mmd_check import get_mmdagg_test_results
from PerturbationExtrapolation.pdae.dataloader import PDAEDataset

energy_loss = EnergyLossMetric()
mean_difference = MeanDifferenceMetric()

def train_pdae(
    num_epochs,
    x_tr,
    pert_tr,
    x_val,
    pert_val,
    settings_dict,
    save_dir,
    x_tr_holdout=None,
    z_tr_holdout=None,
    z_val=None,
    x_te=None,
    z_te=None,
    pert_te=None,
    pert_tr_holdout=None,
    model_name="",
    device=None,
    # print
    n_epochs_plot_losses=None,
    n_epochs_plot_distributions=None,
    n_epochs_loss_print=None,
    n_epochs_model_checkpoint=None,
    n_epochs_run_eval=None,
    n_epochs_mmd_test=None,
    print_tqdm=True,
    losses_to_plot: list[str] = [
        "tr_reconstruction_losses",
        "tr_marginal_prior_energy_losses",
        "tr_perturbation_energy_losses",
        "tr_l21_losses",
        "tr_mmd_p_median_mean",
        "val_energy_losses",
        "val_mean_diffs",
        "val_mmd_p_median_mean"
    ],
    wandb_logging=False,
    # return
    return_metrics=False,
    return_model=False,
    # advanced settings
    run_eval_batched=False,
    remove_marginal=True,
    plot_distribution_ratio_sample=1.0,
    update_perturbation_matrix_via_OLS=False,
    pdae_model=None,
    num_epochs_trained=0,
    losses=None,
    mmd_stats=None,
    ratio_mmd_test_val_data=1.0
):
    print("Prepare PDAE training", flush=True)

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    # get settings
    dim_x = settings_dict["dim_x"]
    dim_z_model = settings_dict["dim_z_model"]
    dim_noise_model = settings_dict["dim_noise_model"]
    sigma_noise_model = settings_dict["sigma_noise_model"]
    decoder_layer_shapes = settings_dict["decoder_layer_shapes"]
    encoder_layer_shapes = settings_dict["encoder_layer_shapes"]
    learning_rate = settings_dict["learning_rate"]
    batch_size = settings_dict["batch_size"]
    weight_reconstruction_loss = settings_dict["weight_reconstruction_loss"]
    update_encoder_on_reconstruction_loss = settings_dict["update_encoder_on_reconstruction_loss"]
    weight_marginal_prior_energy_loss = settings_dict["weight_marginal_prior_energy_loss"]
    weight_perturbation_energy_loss = settings_dict["weight_perturbation_energy_loss"]
    weight_l21 = settings_dict["weight_l21"]
    normalize_energy_loss = settings_dict["normalize_energy_loss"]
    beta = settings_dict["beta"]
    use_bias_in_perturbation_matrix = settings_dict["use_bias_in_perturbation_matrix"]
    use_softplus_after_decoder = settings_dict["use_softplus_after_decoder"]

    # set seed
    seed = settings_dict["seed"]
    set_random_seed(seed)
    
    n_perturbations = pert_tr.shape[1] # length of ohe / number of perturbation dimensions / cols in perturbation_matrix
    settings_dict["n_perturbations"] = n_perturbations
    
    # get data
    train_dataset = PDAEDataset(x_tr, pert_tr)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    if n_epochs_run_eval is not None or update_perturbation_matrix_via_OLS:
        x_tr_holdout = x_tr_holdout.to(device)
        pert_tr_holdout = pert_tr_holdout.to(device)
    if n_epochs_run_eval is not None:
        val_dataset = PDAEDataset(x_val, pert_val)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    print(
        "PDAE setup complete: "
        f"train_samples={len(train_dataset)}, "
        f"train_batches={len(train_dataloader)}, "
        f"batch_size={batch_size}, "
        f"device={device}"
    , flush=True)
    if n_epochs_run_eval is not None:
        print(
            "PDAE eval setup: "
            f"val_samples={len(val_dataset)}, "
            f"val_batches={len(val_dataloader)}"
        , flush=True)

    # log training
    if wandb_logging:
        import wandb
        wandb.init(
            project=os.path.basename(__file__),
            config=settings_dict
        )

    # model and losses
    if pdae_model is None:
        model = PerturbationAutoencoder(
            dim_z_model,
            dim_noise_model,
            sigma_noise_model,
            dim_x,
            n_perturbations,
            encoder_layer_shapes,
            decoder_layer_shapes,
            use_softplus_after_decoder,
            update_encoder_on_reconstruction_loss,
            use_bias_in_perturbation_matrix
        )
        
        # losses
        tr_reconstruction_losses, tr_marginal_prior_energy_losses, tr_perturbation_energy_losses, tr_l21_losses = [], [], [], []
        val_energy_losses, val_mean_diffs = [], []
        # mmd test results
        tr_rejection_rates = []
        tr_min_pvals = []
        tr_median_pvals = []
        tr_max_pvals = []  
        val_rejection_rates = []
        val_min_pvals = []
        val_median_pvals = []
        val_max_pvals = []
        # mmd test results per perturbation
        if n_epochs_mmd_test is not None:
            tr_mmd_test_results = {i:{
                "pert": pert,
                "p_min": [], 
                "p_median": [],
                "p_max": [],
                "reject": [],
                "num_rejects": [],
                "num_tests": [],
                "ratio_rejects": [],
                "pvals": [],
                "thresholds": [],
            } for i, pert in enumerate(torch.unique(pert_tr_holdout, dim=0))}
            val_mmd_test_results = {i:{
                "pert": pert,
                "p_min": [],
                "p_median": [],
                "p_max": [],
                "reject": [],
                "num_rejects": [],
                "num_tests": [],
                "ratio_rejects": [],
                "pvals": [],
                "thresholds": [],
            } for i, pert in enumerate(torch.unique(pert_val, dim=0))}
        # epochs
        tr_epochs = []
        val_epochs = []
        mmd_epochs = []
    else:
        model = pdae_model
        tr_reconstruction_losses = losses["tr_reconstruction_losses"]
        tr_marginal_prior_energy_losses = losses["tr_marginal_prior_energy_losses"]
        tr_perturbation_energy_losses = losses["tr_perturbation_energy_losses"]
        tr_l21_losses = losses["tr_l21_losses"]
        val_energy_losses = losses["val_energy_losses"]
        val_mean_diffs = losses["val_mean_diffs"]
        tr_rejection_rates = mmd_stats["tr_rejection_rates"]
        tr_min_pvals = mmd_stats["tr_min_pvals"]
        tr_median_pvals = mmd_stats["tr_median_pvals"]
        tr_max_pvals = mmd_stats["tr_max_pvals"]
        if "val_rejection_rates" in mmd_stats:
            val_rejection_rates = mmd_stats["val_rejection_rates"] 
            val_min_pvals = mmd_stats["val_min_pvals"]
            val_median_pvals = mmd_stats["val_median_pvals"]
            val_max_pvals = mmd_stats["val_max_pvals"]
        else:
            val_rejection_rates = []
            val_min_pvals = []
            val_median_pvals = []
            val_max_pvals = []
        if n_epochs_mmd_test is not None:
            tr_mmd_test_results = mmd_stats["tr_mmd_test_results"]
            val_mmd_test_results = mmd_stats["val_mmd_test_results"]
        tr_epochs = losses["tr_epochs"] if "tr_epochs" in losses else []
        val_epochs = losses["val_epochs"] if "val_epochs" in losses else []
        mmd_epochs = mmd_stats["mmd_epochs"] if "mmd_epochs" in mmd_stats else []
    model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    num_epochs_start = 1 + num_epochs_trained
    num_epochs_end = num_epochs_start + num_epochs
    epoch_total = num_epochs_end - 1
    heartbeat_every = n_epochs_loss_print if n_epochs_loss_print is not None else 20
    heartbeat_every = max(1, int(heartbeat_every))
    print(
        "PDAE epoch loop starting: "
        f"epochs={num_epochs_start}-{epoch_total}, "
        f"heartbeat_every={heartbeat_every}"
    , flush=True)

    use_tqdm = bool(print_tqdm and sys.stderr.isatty())
    if print_tqdm and not use_tqdm:
        print("PDAE tqdm disabled in non-interactive logs; using print heartbeats instead.", flush=True)

    for epoch in tqdm(range(num_epochs_start, num_epochs_end), desc="Run PDAE training", disable=not use_tqdm, initial=num_epochs_start-1, total=num_epochs_end-1, leave=True):
        if epoch == num_epochs_start or epoch % heartbeat_every == 0:
            print(f"PDAE epoch {epoch}/{epoch_total} started", flush=True)
        
        # initialize batch counter and losses for this epoch 
        batch_idx = 0
        epoch_tr_reconstruction_loss, epoch_tr_marginal_energy_loss, epoch_tr_perturbation_energy_loss, epoch_tr_l21_loss = 0., 0., 0., 0.
        epoch_val_energy_loss, epoch_val_mean_diff = 0., 0.
        
        for x, pert in train_dataloader:

            batch_idx += 1
            
            x = x.to(device)
            pert = pert.to(device)
            
            # Forward pass
            x_rec, x_perturbed, z_base = model(x, pert)

            # reconstruction loss = distributional/pointwise reconstruction of each data point by stochastic/deterministic
            if weight_reconstruction_loss > 0 or "tr_reconstruction_losses" in losses_to_plot:
                reconstruction_loss = get_reconstruction_loss(x, x_rec, x_perturbed, beta)
            else:
                reconstruction_loss = torch.tensor(0.0)

            # perturbation energy (distribution matching) loss
            if weight_perturbation_energy_loss > 0 or "tr_perturbation_energy_losses" in losses_to_plot:
                perturbation_energy_loss = get_perturbation_energy_loss(x, x_perturbed, beta, remove_marginal=remove_marginal)
            else:
                perturbation_energy_loss = torch.tensor(0.0)

            # marginal prior energy loss
            if weight_marginal_prior_energy_loss > 0 or "tr_marginal_prior_energy_losses" in losses_to_plot:
                prior_noise = torch.randn(z_base.shape).to(x.device)
                marginal_prior_energy_loss = get_marginal_prior_energy_loss(prior_noise, z_base, beta)
            else:
                marginal_prior_energy_loss = torch.tensor(0.0)
            
            if weight_l21 > 0 or "tr_l21_losses" in losses_to_plot:
                l21_loss = get_l21_loss(model.perturbation_matrix.weight)
            else:
                l21_loss = torch.tensor(0.0)

            # Backward and optimize
            optimizer.zero_grad()
            total_loss = (
                weight_perturbation_energy_loss * perturbation_energy_loss 
                + weight_reconstruction_loss * reconstruction_loss
                + weight_marginal_prior_energy_loss * marginal_prior_energy_loss
                + weight_l21 * l21_loss
            )
            total_loss.backward()
            optimizer.step()

            # record train losses for this batch
            epoch_tr_reconstruction_loss += reconstruction_loss.item()
            epoch_tr_marginal_energy_loss += marginal_prior_energy_loss.item()
            epoch_tr_perturbation_energy_loss += perturbation_energy_loss.item()
            epoch_tr_l21_loss += l21_loss.item()

        # get average training losses for this epoch
        epoch_tr_reconstruction_loss /= batch_idx
        epoch_tr_marginal_energy_loss /= batch_idx
        epoch_tr_perturbation_energy_loss /= batch_idx
        epoch_tr_l21_loss /= batch_idx

        # store the losses
        tr_epochs.append(epoch)
        tr_reconstruction_losses.append(epoch_tr_reconstruction_loss)
        tr_marginal_prior_energy_losses.append(epoch_tr_marginal_energy_loss)
        tr_perturbation_energy_losses.append(epoch_tr_perturbation_energy_loss)
        tr_l21_losses.append(epoch_tr_l21_loss)

        # update the perturbation matrix via OLS
        if update_perturbation_matrix_via_OLS:
            model.update_perturbation_matrix_via_OLS(model.encoder(x_tr_holdout), pert_tr_holdout)

        if n_epochs_mmd_test is not None and epoch % n_epochs_mmd_test == 0:
            mmd_epochs.append(epoch)
            with torch.no_grad():
                n_envs = 0
                n_rejects = 0
                n_tests = 0
                min_pvals = []
                median_pvals = []
                max_pvals = []
                pvals = []
                for e, pert_env in enumerate(torch.unique(pert_tr_holdout, dim=0)):
                    # print(pert_env)
                    n_envs += 1
                    env_mask = pert_tr_holdout.eq(pert_env).all(dim=1)
                    x_env_gt = x_tr_holdout[env_mask].to(device)
                    x_env_hat, _ = model.predict(x_tr_holdout, pert_tr_holdout, pert_env.to(device))
                    # run mmd test
                    _mmd_test_results = get_mmdagg_test_results(x_env_gt, x_env_hat)

                    n_rejects += _mmd_test_results["num_rejects"]
                    n_tests += _mmd_test_results["num_tests"]

                    min_pvals.append(_mmd_test_results["p_min"])
                    median_pvals.append(_mmd_test_results["p_median"])
                    max_pvals.append(_mmd_test_results["p_max"])
                    pvals.append(_mmd_test_results["pvals"])

                    # for each perturbation
                    tr_mmd_test_results[e]["p_min"].append(_mmd_test_results["p_min"])
                    tr_mmd_test_results[e]["p_median"].append(_mmd_test_results["p_median"])
                    tr_mmd_test_results[e]["p_max"].append(_mmd_test_results["p_max"])
                    tr_mmd_test_results[e]["reject"].append(_mmd_test_results["reject"])
                    tr_mmd_test_results[e]["num_rejects"].append(_mmd_test_results["num_rejects"])
                    tr_mmd_test_results[e]["num_tests"].append(_mmd_test_results["num_tests"])
                    tr_mmd_test_results[e]["ratio_rejects"].append(_mmd_test_results["ratio_rejects"])
                    tr_mmd_test_results[e]["pvals"].append(_mmd_test_results["pvals"])
                    tr_mmd_test_results[e]["thresholds"].append(_mmd_test_results["thresholds"])
                    # print(f"Epoch {epoch}, TRAIN Perturbation {e}: MMD test rejection: {_mmd_test_results['reject']}, p_min: {_mmd_test_results['p_min']:.3f}, p_median: {_mmd_test_results['p_median']:.3f}, p_max: {_mmd_test_results['p_max']:.3f}, ratio_reject: {_mmd_test_results['num_rejects']}/{_mmd_test_results['num_tests']}")

                tr_epoch_rejection_rate = n_rejects / n_tests if n_tests > 0 else 0
                tr_epoch_min_pval = np.min(pvals)
                tr_epoch_median_pval = np.median(pvals)
                tr_epoch_max_pval = np.max(pvals)

                print(f"Epoch {epoch}, TRAIN: MMD test rejection rate: {tr_epoch_rejection_rate:.3f} ({n_rejects}/{n_tests}), min p-value: {tr_epoch_min_pval:.3f}, median p-value: {tr_epoch_median_pval:.3f}, max p-value: {tr_epoch_max_pval:.3f}", flush=True)

                tr_rejection_rates.append(tr_epoch_rejection_rate)
                tr_min_pvals.append(tr_epoch_min_pval)
                tr_median_pvals.append(tr_epoch_median_pval)
                tr_max_pvals.append(tr_epoch_max_pval)
            
                n_envs = 0
                n_rejects = 0
                n_tests = 0
                min_pvals = []
                median_pvals = []
                max_pvals = []
                pvals = []
                for e, pert_env in enumerate(torch.unique(pert_val, dim=0)):
                    n_envs += 1
                    subsample_indices = torch.randperm(pert_val.size(0))[:int(ratio_mmd_test_val_data * pert_val.size(0))]
                    pert_val_subsampled = pert_val[subsample_indices]
                    x_val_subsampled = x_val[subsample_indices]
                    env_mask = pert_val_subsampled.eq(pert_env).all(dim=1)
                    x_env_gt = x_val_subsampled[env_mask].to(device)
                    x_env_hat, z_env_hat = model.predict(x_tr_holdout, pert_tr_holdout, pert_env.to(device))
                    # run mmd test
                    _mmd_test_results = get_mmdagg_test_results(x_env_gt, x_env_hat)

                    n_rejects += _mmd_test_results["num_rejects"]
                    n_tests += _mmd_test_results["num_tests"]

                    min_pvals.append(_mmd_test_results["p_min"])
                    median_pvals.append(_mmd_test_results["p_median"])
                    max_pvals.append(_mmd_test_results["p_max"])
                    pvals.append(_mmd_test_results["pvals"])

                    # for each perturbation
                    val_mmd_test_results[e]["p_min"].append(_mmd_test_results["p_min"])
                    val_mmd_test_results[e]["p_median"].append(_mmd_test_results["p_median"])
                    val_mmd_test_results[e]["p_max"].append(_mmd_test_results["p_max"])
                    val_mmd_test_results[e]["reject"].append(_mmd_test_results["reject"])
                    val_mmd_test_results[e]["num_rejects"].append(_mmd_test_results["num_rejects"])
                    val_mmd_test_results[e]["num_tests"].append(_mmd_test_results["num_tests"])
                    val_mmd_test_results[e]["ratio_rejects"].append(_mmd_test_results["ratio_rejects"])
                    val_mmd_test_results[e]["pvals"].append(_mmd_test_results["pvals"])
                    val_mmd_test_results[e]["thresholds"].append(_mmd_test_results["thresholds"])
                    # print(f"Epoch {epoch}, VAL Perturbation {e}: MMD test rejection: {_mmd_test_results['reject']}, p_min: {_mmd_test_results['p_min']:.3f}, p_median: {_mmd_test_results['p_median']:.3f}, p_max: {_mmd_test_results['p_max']:.3f}, ratio_reject: {_mmd_test_results['num_rejects']}/{_mmd_test_results['num_tests']}")

                val_epoch_rejection_rate = n_rejects / n_tests if n_tests > 0 else 0
                val_epoch_min_pval = np.min(pvals)
                val_epoch_median_pval = np.median(pvals)
                val_epoch_max_pval = np.max(pvals)

                print(f"Epoch {epoch}, VAL: MMD test rejection rate: {val_epoch_rejection_rate:.3f} ({n_rejects}/{n_tests}), min p-value: {val_epoch_min_pval:.3f}, median p-value: {val_epoch_median_pval:.3f}, max p-value: {val_epoch_max_pval:.3f}", flush=True)

                val_rejection_rates.append(val_epoch_rejection_rate)
                val_min_pvals.append(val_epoch_min_pval)
                val_median_pvals.append(val_epoch_median_pval)
                val_max_pvals.append(val_epoch_max_pval)
            
                
        # get validation loss for this epoch
        if n_epochs_run_eval is not None and epoch % n_epochs_run_eval == 0:
            val_epochs.append(epoch)
            with torch.no_grad():
                sum_epoch_val_energy_loss = 0
                sum_epoch_val_mean_diff = 0  
                
                n_pert_val = 0
                if run_eval_batched:
                    for x_val_batch, pert_val_batch in val_dataloader:
                        x_val_batch = x_val_batch.to(device)
                        pert_val_batch = pert_val_batch.to(device)
                        pert_val_batch_unique_envs = torch.unique(pert_val_batch, dim=0)
                        for pert_env in pert_val_batch_unique_envs:
                            n_pert_val += 1
                            x_val_env = x_val_batch[(pert_val_batch == pert_env).all(dim=1)].to(device)
                            x_val_hat, z_val_hat = model.predict(x_tr_holdout, pert_tr_holdout, pert_env)
                            epoch_val_energy_loss_e = energy_loss.compute(x_val_env, x_val_hat, normalize_energy_loss, beta).cpu()
                            epoch_val_mean_diff_e = mean_difference.compute(x_val_env, x_val_hat).cpu()
                            sum_epoch_val_energy_loss += epoch_val_energy_loss_e.item()
                            sum_epoch_val_mean_diff += epoch_val_mean_diff_e.item()
                else:
                    x_val = x_val.to(device)
                    pert_val = pert_val.to(device)   
                    pert_val_unique_envs = torch.unique(pert_val, dim=0)
                    for pert_env in pert_val_unique_envs:
                        n_pert_val += 1
                        x_val_env = x_val[(pert_val == pert_env).all(dim=1)].to(device)
                        x_val_hat, z_val_hat = model.predict(x_tr_holdout, pert_tr_holdout, pert_env)
                        epoch_val_energy_loss_e = energy_loss.compute(x_val_env, x_val_hat, normalize_energy_loss, beta).cpu()
                        epoch_val_mean_diff_e = mean_difference.compute(x_val_env, x_val_hat).cpu()
                        sum_epoch_val_energy_loss += epoch_val_energy_loss_e.item()
                        sum_epoch_val_mean_diff += epoch_val_mean_diff_e.item()
                # compute average losses per env for this epoch
                epoch_val_energy_loss = sum_epoch_val_energy_loss / n_pert_val
                epoch_val_mean_diff = sum_epoch_val_mean_diff / n_pert_val
                # store the losses for this epoch
                val_energy_losses.append(epoch_val_energy_loss)
                val_mean_diffs.append(epoch_val_mean_diff)
 
        if n_epochs_plot_distributions is not None and epoch % n_epochs_plot_distributions == 0:
            plot_pdae_gt_vs_predicted_distributions(
                model=model,
                model_name=model_name,
                epoch=epoch,
                save_dir=save_dir,
                device=device,
                x_tr_holdout=x_tr_holdout,
                z_tr_holdout=z_tr_holdout,
                pert_tr_holdout=pert_tr_holdout,
                x_val=x_val,
                z_val=z_val,
                pert_val=pert_val,
                x_te=x_te,
                z_te=z_te,
                pert_te=pert_te,
                plot_distribution_ratio_sample=plot_distribution_ratio_sample,
            )

        losses = {
            "tr_epochs": tr_epochs,
            "tr_reconstruction_losses": tr_reconstruction_losses,
            "tr_marginal_prior_energy_losses": tr_marginal_prior_energy_losses,
            "tr_perturbation_energy_losses": tr_perturbation_energy_losses,
            "tr_l21_losses": tr_l21_losses,
            "val_epochs": val_epochs,
            "val_energy_losses": val_energy_losses,
            "val_mean_diffs": val_mean_diffs,
        }

        mmd_stats = {
            "mmd_epochs": mmd_epochs,
            "tr_rejection_rates": tr_rejection_rates,
            "tr_min_pvals": tr_min_pvals,
            "tr_median_pvals": tr_median_pvals,
            "tr_max_pvals": tr_max_pvals,
            "tr_mmd_test_results": tr_mmd_test_results if n_epochs_mmd_test is not None else None,
            "val_rejection_rates": val_rejection_rates,
            "val_min_pvals": val_min_pvals,
            "val_median_pvals": val_median_pvals,
            "val_max_pvals": val_max_pvals,
            "val_mmd_test_results": val_mmd_test_results if n_epochs_mmd_test is not None else None,
        }
        
        # print the losses
        if n_epochs_loss_print is not None and epoch % n_epochs_loss_print == 0:
            if n_epochs_run_eval is not None:
                print(f'Epoch [{epoch}/{num_epochs}], Rec.Loss: {epoch_tr_reconstruction_loss:.3}, Marg.En.Loss: {epoch_tr_marginal_energy_loss:.3}, Pert.En.Loss: {epoch_tr_perturbation_energy_loss:.3}, Tot.Loss: {epoch_tr_reconstruction_loss + epoch_tr_marginal_energy_loss + epoch_tr_perturbation_energy_loss:.3}, Val.En.Loss: {epoch_val_energy_loss:.3}, Val.MeanDiff: {epoch_val_mean_diff:.3}')
            else:
                print(f'Epoch [{epoch}/{num_epochs}], Rec.Loss: {epoch_tr_reconstruction_loss:.3}, Marg.En.Loss: {epoch_tr_marginal_energy_loss:.3}, Pert.En.Loss: {epoch_tr_perturbation_energy_loss:.3}, Tot.Loss: {epoch_tr_reconstruction_loss + epoch_tr_marginal_energy_loss + epoch_tr_perturbation_energy_loss:.3}', flush=True)

        if n_epochs_model_checkpoint is not None and epoch % n_epochs_model_checkpoint == 0 and epoch > 0 and epoch < num_epochs_end-1:
            torch.save(model.state_dict(), os.path.join(save_dir, f'pdae_model_{model_name}_epoch_{epoch}.pth'))
            torch.save(losses, os.path.join(save_dir, f'pdae_loss_dict_{model_name}.pth'))
            torch.save(mmd_stats, os.path.join(save_dir, f'pdae_mmd_stats_dict_{model_name}.pth'))
        
        if n_epochs_plot_losses is not None and epoch % n_epochs_plot_losses == 0 and epoch > 0:
            show_loss_curves(save_dir, model_name, losses=losses, losses_to_plot=losses_to_plot, mmd_stats=mmd_stats if n_epochs_mmd_test is not None else None)

    # Save the trained model
    model_path = save_dir / f'pdae_model_{model_name}_epoch_{epoch}.pth'
    torch.save(model.state_dict(), model_path)
    torch.save(losses, save_dir / f'pdae_loss_dict_{model_name}.pth')
    torch.save(mmd_stats, save_dir / f'pdae_mmd_stats_dict_{model_name}.pth')
    
    return model, epoch, losses, mmd_stats
    
