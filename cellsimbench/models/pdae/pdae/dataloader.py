import numpy as np
import torch
from torch.utils.data import Dataset


class PDAEDataset(Dataset):
    def __init__(self, x, pert):
        self.x = x
        self.pert = pert

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        x = self.x[idx]
        pert = self.pert[idx]
        return x, pert
    

def flatten_observations_perturbations(x, pert):
    assert x.numel() != 0, "x should not be empty!"
    assert pert.numel() != 0, "pert should not be empty!"
    assert len(x.shape) == 3, "x should have shape (n_envs, n_samples, dim_x)!"
    assert len(pert.shape) == 2, "pert should have shape (n_envs, n_perturbations)!"
    dim_x = x.shape[-1]
    x_new = x.reshape(-1, dim_x)
    pert_new = np.repeat(pert, x.shape[1], axis=0)
    assert x_new.shape[0] == pert_new.shape[0]
    return x_new, pert_new


def _process_obs_perts(data_dict, key):
    x = data_dict[f"x_{key}"]
    pert = data_dict[f"pert_{key}"]
    if isinstance(x, list):
        assert len(x) == pert.shape[0]
        if len(x) == 0:
            return torch.Tensor(), torch.Tensor()
        x_list = []
        pert_list = []
        for e in range(len(x)):
            x_list.append(x[e])
            pert_list.append(np.repeat(pert[e].unsqueeze(0), x[e].shape[0], axis=0))
        x_new = torch.vstack(x_list)
        pert_new = torch.vstack(pert_list)
    else:
        x_new, pert_new = flatten_observations_perturbations(x, pert)
    assert x_new.shape[0] == pert_new.shape[0]
    return x_new, pert_new


def get_data_dict_new_pdae(pdae_data_dict):
    data_keys = {key.split("x_")[1] for key in pdae_data_dict.keys() if key.startswith("x")}
    data_dict_new = {}
    for key in data_keys:
        obs, pert = _process_obs_perts(pdae_data_dict, key)
        data_dict_new[key] = {
            "observations": obs,
            "perturbation_labels": pert
        }
    return data_dict_new
