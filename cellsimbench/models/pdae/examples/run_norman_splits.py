#!/usr/bin/env python
"""
Run PDAE experiments on Norman et al. (2019) data using 5 different train/test/ood splits.
Saves results for each split for later analysis and comparison.
"""

import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import gc

# Setup paths
sys.path.append("../")
sys.path.append("../../")

from PerturbationExtrapolation.pdae.api import PDAEData, PDAEConfig, PDAEModel
from PerturbationExtrapolation.pdae.baselines import get_baselines
from PerturbationExtrapolation.pdae.metrics import MeanDifferenceMetric

# Configuration
SEED = 42
N_EPOCHS = 1500
N_SPLITS = 5
STORAGE_PATH = Path("../data")
NORMAN_ADATA_TOP1K_PATH = STORAGE_PATH / "Norman2019_top1k.h5ad"

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def run_split_experiment(split_idx: int):
    """
    Run a single experiment for a given split.
    
    Args:
        split_idx: Index of the split (0-4 for 5 splits)
    """
    split_name = f'splitAE{split_idx + 1}'
    print(f"\n{'='*80}")
    print(f"Running experiment for {split_name} (split {split_idx + 1}/{N_SPLITS})")
    print(f"{'='*80}\n")
    
    # Create save directory with timestamp
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    pdae_save_dir = STORAGE_PATH / f"{current_time}_norman_split{split_idx + 1}"
    pdae_save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved to: {pdae_save_dir}\n")
    
    # Load data with current split
    print(f"Loading data with {split_name}...")
    data_name = f"Norman_AE(top1k)_{split_name}"
    pdae_data_path = pdae_save_dir / f"pdae_data_{data_name}.pth"
    
    pdae_data = PDAEData.from_adata(
        adata_path=NORMAN_ADATA_TOP1K_PATH,
        pert_col="condition",
        pert_sep="+",
        pert_ctrl="ctrl",
        split_col=split_name,
        train_split="train",
        train_holdout_split="test",
        val_split=None,
        test_split="ood",
        data_dir=pdae_save_dir,
        data_name=data_name
    )
    
    # Create config
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    pdae_config = PDAEConfig(**{
        'device': device,
        'seed': SEED,
        'model_name': f'norman_split{split_idx + 1}',
        'batch_size': 128,
        'learning_rate': 1e-4,
        'dim_z_model': 101,
        'dim_noise_model': 27,
        'sigma_noise_model': 0.1,
        'decoder_layer_shapes': [128, 256, 256, 512],
        'encoder_layer_shapes': [512, 256, 256, 128],
        'weight_reconstruction_loss': 0.1,
        'weight_perturbation_energy_loss': 10.0,
        'weight_marginal_prior_energy_loss': 0.0001,
        'weight_l21': 0.0001,
        'update_encoder_on_reconstruction_loss': False,
        'normalize_energy_loss': False,
        'beta': 1.0,
        'use_bias_in_perturbation_matrix': False,
        'use_softplus_after_decoder': True,
    })
    
    # Create and train model
    print(f"Creating PDAE model...")
    pdae = PDAEModel(device, pdae_data, pdae_config, save_dir=pdae_save_dir)
    
    print(f"Training model for {N_EPOCHS} epochs...\n")
    pdae.train(
        num_epochs=N_EPOCHS,
        n_epochs_model_checkpoint=N_EPOCHS,
        n_epochs_plot_losses=N_EPOCHS
    )
    
    print(f"\nTraining complete. Evaluating model...\n")
    
    # Evaluate model
    eval_results = pdae.run_eval(
        models=get_baselines(),
        metrics=[MeanDifferenceMetric()],
        metrics_device="cuda:0",
        use_train_holdout=False,
        ratio_subsample_train=1,
        ratio_subsample_val_test=1,
        run_eval_batched=False,
        return_df=True,
    )

    # Save raw, per-perturbation results (63 rows expected per split)
    raw_results_path = pdae_save_dir / f"eval_results_raw_{split_name}.csv"
    eval_results.to_csv(raw_results_path, index=False)
    print(f"Raw evaluation results saved to: {raw_results_path}\n")
    
    # Get aggregated results
    df_evals_agg = pdae.get_agg_eval_results(layout="wide")
    
    # Save results
    results_path = pdae_save_dir / f"eval_results_{split_name}.csv"
    df_evals_agg.to_csv(results_path)
    print(f"Evaluation results saved to: {results_path}\n")
    
    print(f"Evaluation Results for {split_name}:")
    print(df_evals_agg)
    print()
    
    # Clean up
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'split': split_name,
        'split_idx': split_idx + 1,
        'save_dir': str(pdae_save_dir),
        'results': df_evals_agg,
        'raw_results': eval_results,
    }


def main():
    """Run all 5 split experiments."""
    print(f"\n{'='*80}")
    print(f"PDAE Experiments on Norman et al. (2019) - {N_SPLITS} Random Splits")
    print(f"{'='*80}\n")
    
    all_results = []
    all_raw_results = []
    
    for split_idx in range(N_SPLITS):
        try:
            result = run_split_experiment(split_idx)
            all_results.append(result)
            all_raw_results.append(result['raw_results'].assign(split=result['split']))
        except Exception as e:
            print(f"\nError running experiment for split {split_idx + 1}:")
            print(f"{type(e).__name__}: {e}\n")
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Summary of All Experiments")
    print(f"{'='*80}\n")
    
    for result in all_results:
        print(f"Split: {result['split']}")
        print(f"Save directory: {result['save_dir']}")
        print(result['results'])
        print()
    
    # Save combined results
    summary_path = STORAGE_PATH / f"norman_splits_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
    
    # Combine all results into a single dataframe
    combined_results = []
    for result in all_results:
        df = result['results'].copy()
        df['split'] = result['split']
        combined_results.append(df)
    
    if combined_results:
        df_combined = pd.concat(combined_results, axis=0)
        df_combined.to_csv(summary_path)
        print(f"Combined results saved to: {summary_path}\n")

    # Combine raw per-test-case results and compute mean/std across all 315 cases
    if all_raw_results:
        df_raw_all = pd.concat(all_raw_results, axis=0)
        raw_summary_path = STORAGE_PATH / f"norman_splits_raw_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
        df_raw_all.to_csv(raw_summary_path, index=False)
        print(f"Concatenated raw results saved to: {raw_summary_path}\n")

        df_values = df_raw_all.copy()
        if 'Value' in df_values.columns:
            df_values['Value'] = pd.to_numeric(df_values['Value'], errors='coerce')
        df_values = df_values.drop(columns=[col for col in ['Batch', 'Perturbation'] if col in df_values.columns])
        df_global_agg = (
            df_values
            .groupby(['Method', 'Metric'])['Value']
            .agg(['mean', 'std'])
            .reset_index()
        )
        df_global_agg['Vals'] = (
            df_global_agg['mean'].round(3).astype(str)
            + " ± "
            + df_global_agg['std'].round(3).astype(str)
        )
        df_global_agg = df_global_agg.drop(columns=['mean', 'std'])
        global_summary_path = STORAGE_PATH / f"norman_splits_global_mean_std_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
        df_global_agg.to_csv(global_summary_path, index=False)
        print("Global mean/std across all 315 test cases:")
        print(df_global_agg)
        print(f"Global summary saved to: {global_summary_path}\n")
    
    print(f"Experiment completed successfully!")
    print(f"Total splits completed: {len(all_results)}/{N_SPLITS}\n")


if __name__ == "__main__":
    main()