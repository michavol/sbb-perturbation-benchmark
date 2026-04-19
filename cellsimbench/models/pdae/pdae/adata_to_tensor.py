import json
from pathlib import Path
import numpy as np
import anndata

import pandas as pd
from tqdm import tqdm
import torch


def _get_tensor(adata, pert_labels_unique: list, pert_col: str):
    """
    Create a tensor from the AnnData object based on filtering by perturbation labels.
    """
    tensor_list = []
    for pert in pert_labels_unique:
        x_pert = adata[adata.obs[pert_col]==pert]    
        tensor_list.append(torch.Tensor(x_pert.X.toarray()))
    return torch.cat(tensor_list, dim=0)


def _get_perturbation_ohe(pert_labels: list, dict_single_pert_ix: dict, reference_perturbation: str, pert_sep: str) -> torch.Tensor:
    """
    Get a tensor of multi-hot encodings for perturbations.
    Each single perturbation is encoded as a one-hot vector and the reference perturbation is encoded as a zero vector.
    """
    # get dictionary{pert: few_hot_encoding}
    pert_labels_fhe = []
    for pert in pert_labels:
        single_pert_list = pert.split(pert_sep)
        fhe = np.zeros(len(dict_single_pert_ix), dtype=np.float32)
        for spert in single_pert_list:
            if spert == reference_perturbation:
                pass
            else:
                fhe[dict_single_pert_ix[spert]] = 1.0
        pert_labels_fhe.append(torch.tensor(fhe).unsqueeze(0))  # add batch dimension
    # get tensor of perturbation labels (few hot encodings of single and double perturbations)
    pert_labels_fhe_tensor = torch.cat(pert_labels_fhe, dim=0)
    return pert_labels_fhe_tensor
    

def map_adata_split_to_tensors(
        adata_path: str,
        pert_col: str,
        pert_sep: str, 
        pert_ctrl: str,
        split_col: int, 
        data_dir: str,
        train_split: str = None,
        train_holdout_split: str = None,
        val_split: str = None,
        test_split: str = None,
    ) -> dict:
    """
    Maps an AnnData object to tensors for different splits and perturbations.
    Args:
        adata_path (str): Path to the AnnData object.
        pert_col (str): Column name in adata.obs that contains perturbation labels.
        pert_sep (str): Separator used in perturbation labels to encode combinatorial perturbations.
        pert_ctrl (str): The name of the control condition to be encoded as a zero vector.
        split_col (str): Column name in adata.obs that contains split information.
        train_split (str): Name of the training split.
        train_holdout_split (str): Name of the training holdout split.
        val_split (str): Name of the validation split.
        test_split (str): Name of the test split.
        data_dir (str): Directory to save the tensors and mappings.
    Returns:
        dict: A dictionary containing tensors for observations, perturbations, split and perturbation string information.
    """

    adata = anndata.read_h5ad(adata_path)

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        
    assert train_split is not None, "train_split must be specified"
    assert split_col in adata.obs.columns, f"Split column '{split_col}' not found in adata.obs"
    assert pert_col in adata.obs.columns, f"Perturbation column '{pert_col}' not found in adata.obs"

    unique_single_perts = set(adata.obs[pert_col].str.split(pert_sep).explode().unique())
    unique_single_perts = unique_single_perts - {pert_ctrl} # the reference condition is encoded as zero vector
    dict_single_pert_ix = dict(zip(unique_single_perts, list(range(len(unique_single_perts)))))
    with open(data_dir / "dict_single_pert_ix.json", "w") as f:
        json.dump(dict_single_pert_ix, f)

    splits_unique = [train_split] + [
        split for split in [train_holdout_split, val_split, test_split] if split is not None
    ]

    assert all(split in adata.obs[split_col].unique() for split in splits_unique), \
        f"Not all splits {splits_unique} are present in adata.obs[{split_col}]"

    observations = []
    perturbations = []
    splits = []
    perturbation_annotations = []

    for split in splits_unique:
        print(f"Processing split: {split}")
        pert_str = adata.obs.loc[adata.obs[split_col] == split, pert_col]
        pert_ohe = _get_perturbation_ohe(pert_str.tolist(), dict_single_pert_ix, reference_perturbation=pert_ctrl, pert_sep=pert_sep)
        x = _get_tensor(adata=adata[adata.obs[split_col] == split], pert_labels_unique=pert_str.drop_duplicates().tolist(), pert_col=pert_col)
        print(f"Observations shape: {x.shape[0]}, Perturbation encoding shape: {pert_ohe.shape}, Number of unique perturbations: {pert_str.nunique()}")
        assert pert_str.shape[0] == pert_ohe.shape[0], \
            f"Number of perturbations ({pert_str.shape[0]}) does not match number of perturbation encodings ({pert_ohe.shape[0]}) for split '{split}'"
        assert pert_str.shape[0] == x.shape[0], \
            f"Number of perturbations ({pert_str.shape[0]}) does not match number of observations ({x.shape[0]}) for split '{split}'"
        
        observations.append(x)
        perturbations.append(pert_ohe)
        perturbation_annotations += pert_str.tolist()
        splits += [split] * x.shape[0]
    
    observations = torch.cat(observations, dim=0)
    perturbations = torch.cat(perturbations, dim=0)

    pert_ohe_str_mapping = {k: v for v, k in zip(perturbations.tolist(), perturbation_annotations)}
    with open(data_dir / "pert_ohe_str_mapping.json", "w") as f:
        json.dump(pert_ohe_str_mapping, f)

    return {
        "observations": observations,
        "perturbations": perturbations,
        "pert_ctrl": pert_ctrl,
        "splits": splits,
        "perturbation_annotations": perturbation_annotations,   
    }
