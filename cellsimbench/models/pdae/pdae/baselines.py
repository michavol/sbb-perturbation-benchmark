import json
from typing import List, Literal

import anndata
import torch
import scanpy as sc

def check_tr_data(x_tr: torch.Tensor, pert_tr: torch.Tensor):
    if x_tr.ndim != 2 or pert_tr.ndim != 2:
        raise ValueError("Input tensors must be 2-dimensional.")
    if x_tr.shape[0] != pert_tr.shape[0]:
        raise ValueError("Input tensors must have the same number of rows (samples).")
    if x_tr.shape[1] == 0 or pert_tr.shape[1] == 0:
        raise ValueError("Input tensors must have non-zero feature dimensions.")


def check_te_pert(pert_te: torch.Tensor):
    if pert_te.ndim != 1:
        raise ValueError("pert_te must be a 1-dimensional tensor.")
    if pert_te.shape[0] == 0:
        raise ValueError("pert_te must have non-zero feature dimensions.")


class BaselineModel:
    name: str
    x_tr: torch.Tensor
    pert_tr: torch.Tensor
    pert_te: torch.Tensor


class PoolAllModel(BaselineModel):
    def __init__(self):
        self.name = "pool_all"

    def fit(self, x_tr: torch.Tensor, pert_tr: torch.Tensor):
        self.x_tr = x_tr
        self.pert_tr = pert_tr
        check_tr_data(x_tr, pert_tr)
        return self

    def predict(self, pert_te: torch.Tensor = None):
        return self.x_tr


def get_pseudo_bulk(x_tr, pert_tr, pert_te):
    """
    Pool only data arising from perturbations involved in the combination to be predicted (“Pseudobulking”)
    e.g. return train data beloning to perturbations [1,0,0] and [0,1,0] if pert_te == [1,1,0]
    """
    check_tr_data(x_tr, pert_tr)
    check_te_pert(pert_te)

    # if pert_te is seen during training, return set of matching observations
    matches = (pert_tr == pert_te).all(dim=1)
    if matches.any():
        return x_tr[matches]

    pert_tr = (pert_tr != 0.).to(int) # convert perturbation numbers different from zero to integers, e.g. [0.25, 0., 1.0] -> [1,0,1]
    pert_to_predict = (pert_te != 0.).to(int).unsqueeze(0) # add batch dimension, e.g. [1., 1., 0.] -> [[1, 1, 0]]
    n_ones_diff = ((pert_to_predict - pert_tr) == 1).sum(dim=1) # if ohe match, 1-1=0
    n_ones = (pert_to_predict == 1).sum(dim=1)
    is_match = n_ones_diff < n_ones
    assert is_match.sum() > 0
    return x_tr[is_match,:]
    

class PseudoBulkingModel(BaselineModel):
    def __init__(self):
        self.name = "pseudo_bulking"
    
    def fit(self, x_tr: torch.Tensor, pert_tr: torch.Tensor):
        self.x_tr = x_tr
        self.pert_tr = pert_tr
        check_tr_data(x_tr, pert_tr)
        return self
    
    def predict(self, pert_te):
        check_te_pert(pert_te)
        return get_pseudo_bulk(
            self.x_tr, 
            self.pert_tr, 
            pert_te, 
        )


class ControlModel(BaselineModel):
    def __init__(self):
        self.name = "control"
    
    def fit(self, x_tr: torch.Tensor, pert_tr: torch.Tensor):
        self.x_tr = x_tr
        self.pert_tr = pert_tr
        check_tr_data(x_tr, pert_tr)
        # Store control cells
        self.is_control = (pert_tr == 0.).all(dim=1)
        assert self.is_control.sum() > 0, "No control cells found in training data."
        return self
    
    def predict(self, pert_te: torch.Tensor = None):
        return self.x_tr[self.is_control]


class AdditiveModel(BaselineModel):
    def __init__(self):
        self.name = "additive"
    
    def fit(self, x_tr: torch.Tensor, pert_tr: torch.Tensor):
        self.x_tr = x_tr
        self.pert_tr = pert_tr
        check_tr_data(x_tr, pert_tr)
        # Store control cells and mean
        self.is_control = (pert_tr == 0.).all(dim=1)
        assert self.is_control.sum() > 0, "No control cells found in training data."
        self.x_control_mean = self.x_tr[self.is_control].mean(dim=0)
        return self
    
    def predict(self, pert_te: torch.Tensor):
        check_te_pert(pert_te)
        
        # if pert_te is seen during training, return set of matching observations
        matches = (self.pert_tr == pert_te).all(dim=1)
        if matches.any():
            return self.x_tr[matches]

        # Check if pert_te is a double perturbation (has exactly two non-zero entries)
        non_zero_indices = (pert_te != 0.).nonzero(as_tuple=False).squeeze()
        
        if non_zero_indices.dim() == 0:
            # Single element, make it 1D
            non_zero_indices = non_zero_indices.unsqueeze(0)
        
        if len(non_zero_indices) != 2:
            raise ValueError(f"AdditiveModel only works for double perturbations. Got {len(non_zero_indices)} non-zero entries.")
        
        # Create single perturbation vectors
        pert_A = torch.zeros_like(pert_te)
        pert_A[non_zero_indices[0]] = pert_te[non_zero_indices[0]]
        
        pert_B = torch.zeros_like(pert_te)
        pert_B[non_zero_indices[1]] = pert_te[non_zero_indices[1]]
        
        # Find training cells matching these single perturbations
        mask_A = (self.pert_tr == pert_A).all(dim=1)
        mask_B = (self.pert_tr == pert_B).all(dim=1)
        
        if not mask_A.any():
            raise ValueError(f"No training data found for single perturbation A: {pert_A}")
        if not mask_B.any():
            raise ValueError(f"No training data found for single perturbation B: {pert_B}")
        
        # Calculate means
        mean_A = self.x_tr[mask_A].mean(dim=0)
        mean_B = self.x_tr[mask_B].mean(dim=0)
        
        # Calculate additive shift: (mean_A - mean_CTRL) + (mean_B - mean_CTRL)
        shift = (mean_A - self.x_control_mean) + (mean_B - self.x_control_mean)
        
        # Apply shift to all control cells
        x_control = self.x_tr[self.is_control]
        return x_control + shift.unsqueeze(0)


class OracleHalfModel(BaselineModel):
    def __init__(self):
        self.name = "oracle_half"
        self.x_te_true = None
        self.second_half_indices = None
    
    def fit(self, x_tr: torch.Tensor, pert_tr: torch.Tensor):
        """
        Fit the oracle model. For this oracle, training data is not used.
        Instead, x_te and x_te_indices will be set during predict.
        """
        self.x_tr = x_tr
        self.pert_tr = pert_tr
        check_tr_data(x_tr, pert_tr)
        return self
    
    def set_test_data(self, x_te_true: torch.Tensor, x_te_indices: torch.Tensor = None):
        """
        Set the true test data. This is called during evaluation to provide
        access to the ground truth test observations.
        x_te_true: tensor of shape (n_samples, n_features) - all test observations
        x_te_indices: tensor of indices that correspond to the current perturbation
        """
        self.x_te_true = x_te_true
        self.x_te_indices = x_te_indices
    
    def predict(self, pert_te: torch.Tensor = None):
        """
        Split test observations in half and return the first half as prediction.
        Evaluation should only be done on the second half.
        """
        if self.x_te_true is None:
            raise RuntimeError(
                "OracleHalfModel requires x_te_true to be set via set_test_data() "
                "before calling predict(). This baseline cannot work with standard evaluation."
            )
        
        x_te_matched = self.x_te_true
        if x_te_matched.shape[0] == 0:
            raise ValueError("No test observations available for the given perturbation.")
        
        n_samples = x_te_matched.shape[0]
        split_point = n_samples // 2
        
        # Return first half as prediction
        return x_te_matched[:split_point]


class LinearRegressionModel(BaselineModel):
    def __init__(self, return_distribution: bool = True, remove_col_ix: int = None):
        """
        remove_col_ix: this allows fixing the bug in norman_to_tensor
        """
        self.name = "linear_regression"
        self.return_distribution = return_distribution
        self.remove_col_ix = remove_col_ix
    
    def fit(self, x_tr: torch.Tensor, pert_tr: torch.Tensor):
        check_tr_data(x_tr, pert_tr)
        if self.remove_col_ix is not None:
            pert_tr = pert_tr[:,1:]
        self.x_tr = x_tr
        self.pert_tr = pert_tr
        self.is_reference_tr = (pert_tr == 0.).all(dim=1)
        assert self.is_reference_tr.sum() > 0, "No reference perturbation found in x_tr."
        self.x_tr_reference_mean = self.x_tr[self.is_reference_tr].mean(dim=0).unsqueeze(0)
        x_tr_centered = self.x_tr - self.x_tr_reference_mean
        self.beta_hat_ols = torch.linalg.lstsq(self.pert_tr, x_tr_centered).solution
        return self
    
    def predict(self, pert_te):
        check_te_pert(pert_te)
        if self.remove_col_ix is not None:
            pert_te = pert_te[1:]
        pert_te = pert_te.unsqueeze(0) 
        x_te_pred_mean = pert_te @ self.beta_hat_ols + self.x_tr_reference_mean
        if self.return_distribution:
            return self.x_tr[self.is_reference_tr] - self.x_tr_reference_mean + x_te_pred_mean
        else:
            return x_te_pred_mean


class LinearRegressionModelAgg(BaselineModel):
    def __init__(self, return_distribution: bool = True):
        self.name = "linear_regression_agg"
        self.return_distribution = return_distribution

    def fit(self, x_tr: torch.Tensor, pert_tr: torch.Tensor):
        self.x_tr = x_tr
        self.pert_tr = pert_tr
        self.is_reference_tr = (pert_tr == 0.).all(dim=1)
        assert self.is_reference_tr.sum() > 0, "No reference perturbation found in x_tr."
        check_tr_data(x_tr, pert_tr)
        self.x_tr_reference_mean = self.x_tr[self.is_reference_tr].mean(dim=0).unsqueeze(0)
        x_tr_centered = self.x_tr - self.x_tr_reference_mean
        # aggregate per domain
        x_tr_centered_means = []
        pert_tr_unique = torch.unique(self.pert_tr, dim=0)
        for pert in pert_tr_unique:
            pert_mask = (self.pert_tr == pert).all(dim=1)
            x_tr_centered_means.append(x_tr_centered[pert_mask].mean(dim=0))
        x_tr_centered_means = torch.stack(x_tr_centered_means, dim=0)
        self.beta_hat_ols = torch.linalg.lstsq(pert_tr_unique, x_tr_centered_means).solution
        return self
 
    def predict(self, pert_te):
        check_te_pert(pert_te)
        pert_te = pert_te.unsqueeze(0) 
        x_te_pred_mean = pert_te @ self.beta_hat_ols + self.x_tr_reference_mean
        if self.return_distribution:
            return self.x_tr[self.is_reference_tr] - self.x_tr_reference_mean + x_te_pred_mean
        else:
            return x_te_pred_mean


def get_cpa_cov_info(data_type):
    if data_type == "norman":
        cov_key = "cell_type"
        cov_val = "A549"
    elif data_type == "simulation":
        cov_key = "dummy_cov"
        cov_val = "XYZ"
    return cov_key, cov_val


def get_cpa_adata(data_type, data_path):
    if data_type == "norman":
        adata = sc.read(data_path / "adata_pdae_norman.h5ad")
    elif data_type == "simulation":
        adata = sc.read(data_path / "cpa_simulation_data_processed.h5ad")
    return adata


def get_prediction_data(pert_str_list: list, pert_tensor: torch.Tensor, adata=None, oracle_latents_type=None, cov_key="dummy_cov", cov_val="XYZ"):
    assert isinstance(pert_str_list, list), "pert_str_list must be a list of perturbation strings."
    doses = get_dosages(pert_str_list, pert_tensor)
    
    cov = {cov_key: [cov_val]*len(pert_str_list)}

    if adata is not None:
        all_genes = []
        all_oracle_latents = []
        for ps in pert_str_list:
            all_genes.append(adata[adata.obs["condition"]==ps].X.copy())
            if oracle_latents_type is None:
                all_oracle_latents = None
            else:
                try:
                    all_oracle_latents.append(adata[adata.obs["condition"]==ps].obsm[oracle_latents_type].copy())
                except KeyError:
                    all_oracle_latents.append(adata[adata.obs["condition"]==ps].obs[oracle_latents_type].copy())
                
                
        return {
            "conditions": pert_str_list, 
            "cov": cov, 
            "dose": doses, 
            "all_genes": all_genes,
            "all_oracle_latents": all_oracle_latents
        }
    else:
        return {
            "conditions": pert_str_list, 
            "cov": cov, 
            "dose": doses
        }
    

def get_dosages(pert_str_list: list, pert_tensor: torch.Tensor) -> List:
    if not isinstance(pert_str_list, list):
        pert_str_list = [pert_str_list]
    if not isinstance(pert_tensor, torch.Tensor):
        pert_tensor = torch.Tensor(pert_tensor)
    if len(pert_tensor.shape) == 1:
        pert_tensor = [pert_tensor] # make list
    doses = []
    for ps, p in zip(pert_str_list, pert_tensor):
        if (p==1).sum() != (p!=0).sum(): # float perturbation with values other than 0 and 1
            dose = get_continuous_dosage(ps, p)
        else:
            dose = get_binary_dosage(ps)
        doses.append(dose)
    return doses


def get_binary_dosage(pert: str):
    return "".join(["1+"]*(len(pert.split("+"))))[:-1]


def get_continuous_dosage(ps: str, p: torch.Tensor):
    if isinstance(p, torch.Tensor):
        p = p.cpu().numpy()
    if ps == "ctrl":
        return "1"
    else:
        ps_tok = ps.split("+")
        dims = [int(tok.split("dim")[1])-1 for tok in ps_tok if "dim" in tok]
        return "1.0+" + "+".join(p[[dims]][0].astype(str).tolist())


class CPAModel(BaselineModel):
    def __init__(self, 
        seed: int, 
        device: str,
        pert_ohe_str_mapping_path: str, # mapping {pert_str: pert_ohe}
        cpa_model_path: str,
        adata: anndata.AnnData, 
        data_type: List[Literal["norman", "simulation"]], 
        name: str = "cpa"
    ):
        self.name = name
        self.cpa_model_path = cpa_model_path
        self.data_type = data_type    
        self.seed = seed
        self.device = device

        # set up model and data
        self.cov_key, self.cov_val = get_cpa_cov_info(self.data_type)
        try:
            from cpa.api import API as CPA_API
            self.cpa_api = CPA_API(
                adata,
                pretrained=self.cpa_model_path,
                device=self.device,
                covariate_keys=[self.cov_key],
                seed=self.seed,
            )  
            self.genes_control = self.cpa_api.datasets["test"].subset_condition(control=True).genes
        except ImportError as e:
            raise ImportError("Please install CPA: pip install git+https://github.com/facebookresearch/CPA.git") from e

        with open(pert_ohe_str_mapping_path, "r") as f:
            pert_ohe_str_mapping = json.load(f)
        self.pert_ohe_str_mapping = pert_ohe_str_mapping
        

    def fit(self, x_tr: torch.Tensor = None, pert_tr: torch.Tensor = None):
        return self

    def pert_ohe_to_str(self, pert_te):
        filtered_dict = {k: v for k, v in self.pert_ohe_str_mapping.items() if v == pert_te.tolist()}
        assert len(filtered_dict) == 1, f"Expected exactly one perturbation string for pert_te {pert_te.tolist()}, but found {len(filtered_dict)}."
        pert_te_str = list(filtered_dict.keys())[0]
        return pert_te_str

    def predict(self, pert_te):
        pert_te_str = self.pert_ohe_to_str(pert_te)
        prediction_data_e = get_prediction_data(pert_str_list=[pert_te_str], pert_tensor=pert_te, cov_key=self.cov_key, cov_val=self.cov_val)
        gene_means, gene_vars, df_obs = self.cpa_api.predict(
            genes=self.genes_control,
            cov=prediction_data_e["cov"],  
            pert=prediction_data_e["conditions"],
            dose=prediction_data_e["dose"], 
            uncertainty=False,
            return_anndata=False,
            sample=False,
            n_samples=1,
        )
        return torch.tensor(gene_means)


def get_baselines():
    pool_all_model = PoolAllModel()
    pseudo_bulking_model = PseudoBulkingModel()
    control_model = ControlModel()
    additive_model = AdditiveModel()
    linear_regression_model = LinearRegressionModel(return_distribution=True)
    oracle_half_model = OracleHalfModel()
    return [control_model, pool_all_model, pseudo_bulking_model, additive_model, linear_regression_model, oracle_half_model]