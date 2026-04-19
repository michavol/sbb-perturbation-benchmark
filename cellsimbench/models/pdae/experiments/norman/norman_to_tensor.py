import json
from pathlib import Path
import numpy as np
import anndata

import pandas as pd
from tqdm import tqdm
import torch


def _get_tensor(adata, pert_labels: list):
    tensor_list = []
    for pert in tqdm(pert_labels, desc="Creating tensors from adata"):
        x_pert = adata[adata.obs["condition"]==pert]    
        tensor_list.append(torch.Tensor(x_pert.X.toarray()))
    return torch.cat(tensor_list, dim=0)


def _get_perturbation_ohe(pert_labels: list, dict_single_pert_ix: dict, reference_condition: str) -> torch.Tensor:
    # get dictionary{pert: few_hot_encoding}
    pert_labels_fhe = []
    for pert in tqdm(pert_labels, desc="Creating perturbation encodings"):
        single_pert_list = pert.split("+")
        fhe = np.zeros(len(dict_single_pert_ix), dtype=np.float32)
        for spert in single_pert_list:
            if spert == reference_condition:
                pass
            else:
                fhe[dict_single_pert_ix[spert]] = 1.0
        pert_labels_fhe.append(torch.tensor(fhe).unsqueeze(0))  # add batch dimension
    # get tensor of perturbation labels (few hot encodings of single and double perturbations)
    pert_labels_fhe_tensor = torch.cat(pert_labels_fhe, dim=0)
    return pert_labels_fhe_tensor
    

def map_norman_adata_split_to_tensors(
        norman_adata_path: str, 
        split: int, 
        data_dir: str = None,
        return_data: bool = False,
        return_mapping: bool = False
    ) -> str:

    adata = anndata.read_h5ad(norman_adata_path)

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    
    if not (data_dir / "x_tr.pt").exists():
        data_dir.mkdir(exist_ok=True)

        pert_tr_str = adata.obs.loc[adata.obs[f"split{split}"] == "train", "condition"]
        pert_tr_holdout_str = adata.obs.loc[adata.obs[f"split{split}"] == "test", "condition"]
        pert_te_str = adata.obs.loc[adata.obs[f"split{split}"] == "ood", "condition"]

        x_tr = _get_tensor(adata[adata.obs[f"split{split}"] == "train"], pert_tr_str.drop_duplicates().tolist())
        x_tr_holdout = _get_tensor(adata[adata.obs[f"split{split}"] == "test"], pert_tr_holdout_str.drop_duplicates().tolist())
        x_te = _get_tensor(adata[adata.obs[f"split{split}"] == "ood"], pert_te_str.drop_duplicates().tolist())

        # save expression tensor to disk
        for x, name in zip([x_tr, x_tr_holdout, x_te], ["x_tr", "x_tr_holdout", "x_te"]):
            torch.save(x, data_dir / f"{name}.pt")

        unique_single_perts = set(adata.obs["condition"].str.split("+").explode().unique()) - {"ctrl"}
        dict_single_pert_ix = dict(zip(unique_single_perts, list(range(len(unique_single_perts)))))
        # save dict to data_dir
        with open(data_dir / "dict_single_pert_ix.json", "w") as f:
            json.dump(dict_single_pert_ix, f)

        # get a one hot encoding of perturbations
        pert_tr = _get_perturbation_ohe(pert_tr_str.tolist(), dict_single_pert_ix, reference_condition="ctrl")
        pert_tr_holdout = _get_perturbation_ohe(pert_tr_holdout_str.tolist(), dict_single_pert_ix, reference_condition="ctrl")
        pert_te = _get_perturbation_ohe(pert_te_str.tolist(), dict_single_pert_ix, reference_condition="ctrl")

        # save perturbation encodings tensor to disk
        for p, name in zip([pert_tr, pert_tr_holdout, pert_te], ["tr", "tr_holdout", "te"]):
            torch.save(p, data_dir / f"perturbation_labels_{name}.pt")

        # create dict mapping from perturbation strings to perturbation one-hot (or few hot) encodings: {pert_str: pert_ohe}
        pert_ohe_str_mapping = {k: v for v, k in zip(torch.cat([pert_tr, pert_tr_holdout, pert_te], dim=0).tolist(), pd.concat([pert_tr_str, pert_tr_holdout_str, pert_te_str], axis=0).tolist())}
        with open(data_dir / "pert_ohe_str_mapping.json", "w") as f:
            json.dump(pert_ohe_str_mapping, f)
    
    else:
        print(f"Data for split {split} already exists at {data_dir}. Skipping data preparation.")
        x_tr = torch.load(data_dir / "x_tr.pt").to(torch.float32)
        x_tr_holdout = torch.load(data_dir / "x_tr_holdout.pt").to(torch.float32)
        x_te = torch.load(data_dir / "x_te.pt").to(torch.float32)
        pert_tr = torch.load(data_dir / "perturbation_labels_tr.pt").to(torch.float32)
        pert_tr_holdout = torch.load(data_dir / "perturbation_labels_tr_holdout.pt").to(torch.float32)
        pert_te = torch.load(data_dir / "perturbation_labels_te.pt").to(torch.float32)
        with open(data_dir / "dict_single_pert_ix.json", "r") as f:
            dict_single_pert_ix = json.load(f)
    
    assert x_tr.shape[0] == pert_tr.shape[0]
    assert x_tr_holdout.shape[0] == pert_tr_holdout.shape[0]
    assert x_te.shape[0] == pert_te.shape[0]
    assert x_tr.shape[1] == x_tr_holdout.shape[1] == x_te.shape[1] 
    assert pert_tr.shape[1] == pert_tr_holdout.shape[1] == pert_te.shape[1]


    if return_data:
        return x_tr, x_tr_holdout, x_te, pert_tr, pert_tr_holdout, pert_te
    elif return_mapping:
        return dict_single_pert_ix
    else:
        return data_dir
