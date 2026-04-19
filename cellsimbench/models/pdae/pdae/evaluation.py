from typing import List

from PerturbationExtrapolation.pdae.baselines import BaselineModel
from PerturbationExtrapolation.pdae.metrics import get_mean_difference


def run_eval(
        self, 
        models: List[BaselineModel], 
        metrics: List[callable],
        use_train_holdout: bool = False,
        ratio_subsample_val_test: float = 1.0,
        run_eval_batched: bool = False,
        return_df: bool = False,
        return_baselines: bool = False
    ):
    assert isinstance(baseline_names, list) and all(isinstance(name, str) for name in baseline_names), "baseline_names must be a list of strings"
    assert isinstance(metric_names, list) and all(isinstance(name, str) for name in metric_names), "metric_names must be a list of strings"
    assert len(baseline_names) > 0, "baseline_names must contain at least one baseline model"
    assert len(metric_names) > 0, "metric_names must contain at least one metric"
    
    set_random_seed(self.pdae_config.seed)

    if self.pdae_model is None:
        raise RuntimeError("PDAE model has not been trained yet. Please call train() before run_eval().")
    
    if (self.x_val is None or self.pert_val is None) and (self.x_te is None and self.pert_te is None):
        raise RuntimeError("Validation or test data must be provided in the PDAEData object.")
    
    if use_train_holdout and (self.x_tr_holdout is None or self.pert_tr_holdout is None):
        raise RuntimeError("Train holdout data must be provided in the PDAEData object if use_train_holdout is True.")

    if use_train_holdout:
        x_tr = self.x_tr_holdout.to(self.device)
        pert_tr = self.pert_tr_holdout.to(self.device)
    else:
        x_tr = self.x_tr.to(self.device)
        pert_tr = self.pert_tr.to(self.device)
    
    x_val = self.x_val
    pert_val = self.pert_val
    x_te = self.x_te
    pert_te = self.pert_te
    
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
    

    if x_val is None or pert_val is None:
        x_val = torch.tensor([])
        pert_val = torch.tensor([])

    if x_te is None or pert_te is None:
        x_te = torch.tensor([])
        pert_te = torch.tensor([])

    x_val_test = torch.cat([x_val, x_te], dim=0)
    pert_val_test = torch.cat([pert_val, pert_te], dim=0)
    split_val_test = np.array([self.pdae_config.val_split] * pert_val.shape[0] + [self.pdae_config.test_split] * pert_te.shape[0])
    
    print("Running evaluation on {} samples in the validation and test set.".format(x_val_test.shape[0]))

    if run_eval_batched == False:
        x_val_test = x_val_test.to(self.device)
        pert_val_test = pert_val_test.to(self.device)
        pert_val_test_unique = torch.unique(pert_val_test, dim=0)

    baseline_models = []
    if "pool_all" in baseline_names:
        pool_all_model = PoolAllModel(x_tr, pert_tr)
        baseline_models.append(pool_all_model)
    if "pseudo_bulking" in baseline_names:
        pseudo_bulking_model = PseudoBulkingModel(x_tr, pert_tr)
        baseline_models.append(pseudo_bulking_model)
    if "linear_regression" in baseline_names:
        linear_regression_model = LinearRegressionModel(x_tr, pert_tr, return_distribution=True)
        baseline_models.append(linear_regression_model)
    if "linear_regression_agg" in baseline_names:
        linear_regression_agg_model = LinearRegressionModelAgg(x_tr, pert_tr, return_distribution=True)
        baseline_models.append(linear_regression_agg_model)
    
    metric_functions = []
    if "energy_distance" in metric_names:
        metric_functions.append(get_energy_distance)
    if "mmd" in metric_names:
        metric_functions.append(get_mmd_loss)
    if "mean_difference" in metric_names:
        metric_functions.append(get_mean_difference)
    if "r2_score" in metric_names:
        metric_functions.append(get_r2_score)

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
                x_true = x_val_test_batch[pert_mask].to(self.device)
                split = split_batch[pert_mask.cpu().numpy()][0]
                for metric_function, metric_name in zip(metric_functions, metric_names):
                    # baselines
                    for baseline_model, baseline_name in zip(baseline_models, baseline_names):    
                        x_pred = baseline_model.predict(pert_env)
                        metric_value = metric_function(x_true, x_pred).item()
                        df_env = pd.DataFrame({"Split": [split], "Perturbation": [str(pert_env.cpu().numpy().tolist())], "Batch": [batch_counter], "Method": [baseline_name], "Metric": [metric_name], "Value": [metric_value]})
                        df_evals_baseline.append(df_env)
                    # pdae
                    x_pred_pdae, z_pred_pdae = self.predict(pert_env, use_train_holdout=use_train_holdout)
                    metric_value_pdae = metric_function(x_true, x_pred_pdae).item()
                    df_env_pdae = pd.DataFrame({"Split": [split], "Perturbation": [str(pert_env.cpu().numpy().tolist())], "Batch": [batch_counter],  "Method": ["pdae"], "Metric": [metric_name], "Value": [metric_value_pdae]})
                    df_evals_pdae.append(df_env_pdae)

    else:
        for pert_env in pert_val_test_unique:
            pert_mask = (pert_val_test == pert_env).all(dim=1)
            x_true = x_val_test[pert_mask]
            split = split_val_test[pert_mask.cpu().numpy()][0]

            for metric_function, metric_name in zip(metric_functions, metric_names):
                # baselines
                for baseline_model, baseline_name in zip(baseline_models, baseline_names):    
                    x_pred = baseline_model.predict(pert_env)
                    metric_value = metric_function(x_true, x_pred).item()
                    df_env = pd.DataFrame({"Split": [split], "Perturbation": [str(pert_env.cpu().numpy().tolist())], "Method": [baseline_name], "Metric": [metric_name], "Value": [metric_value]})
                    df_evals_baseline.append(df_env)
                # pdae
                x_pred_pdae, z_pred_pdae = self.predict(pert_env, use_train_holdout=use_train_holdout)
                metric_value_pdae = metric_function(x_true, x_pred_pdae).item()
                df_env_pdae = pd.DataFrame({"Split": [split], "Perturbation": [str(pert_env.cpu().numpy().tolist())], "Method": ["pdae"], "Metric": [metric_name], "Value": [metric_value_pdae]})
                df_evals_pdae.append(df_env_pdae)


    self.df_evals = pd.concat(df_evals_baseline + df_evals_pdae, ignore_index=True)
    self.df_evals.to_csv(os.path.join(self.save_dir, f"pdae_eval_results_{self.pdae_config.model_name}_epoch_{self.num_epochs_trained}.csv"), index=False)
    if return_baselines:
        return {baseline_name: baseline_model for baseline_name, baseline_model in zip(baseline_names, baseline_models)}
    elif return_df:
        return self.df_evals