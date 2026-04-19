import pytest
import anndata
import numpy as np
import torch
from pathlib import Path
from PerturbationExtrapolation.experiments.norman.norman_to_tensor import _get_tensor, _get_perturbation_ohe, map_norman_adata_split_to_tensors


@pytest.fixture
def mock_adata():
    obs = {
        "condition": ["ctrl", "ctrl", "pert1", "pert2", "pert1+pert2", "ctrl+pert1", "pert2"],
        "split1": ["train", "train", "train", "test", "ood", "ood", "ood"]
    }
    X = np.random.rand(7, 10)
    return anndata.AnnData(X=X, obs=obs)


@pytest.fixture
def mock_save_dir(tmp_path):
    return tmp_path / "norman_tensors"

@pytest.fixture
def mock_adata_path(mock_adata, tmp_path):
    dir = tmp_path / "adata"
    dir.mkdir(exist_ok=True)
    adata_path = dir / "mock_norman.h5ad"
    mock_adata.write(adata_path)
    return adata_path


def test__get_tensor(mock_adata):
    pert_labels = ["ctrl", "pert1", "pert2"]
    tensor = _get_tensor(mock_adata, pert_labels)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[0] == sum(mock_adata.obs["condition"].isin(pert_labels))
    assert tensor.shape[1] == mock_adata.shape[1]
    assert (tensor[pert_labels=="ctrl"] == 0).all()  # Check if ctrl is correctly represented as zeros


def test__get_perturbation_ohe():
    pert_labels = ["ctrl", "pert1", "pert2", "ctrl+pert1", "pert1+pert2"]
    dict_single_pert_ix = {"pert1": 0, "pert2": 1} # we encode ctrl with the zero vector, so it doesn't encode a 1
    reference_condition = "ctrl"
    ohe_tensor = _get_perturbation_ohe(pert_labels, dict_single_pert_ix, reference_condition)
    assert isinstance(ohe_tensor, torch.Tensor)
    assert ohe_tensor.shape[0] == len(pert_labels)
    assert ohe_tensor.shape[1] == len(dict_single_pert_ix)
    assert (ohe_tensor == torch.tensor([[0, 0], [1, 0], [0, 1], [1,0], [1, 1]])).all()  # Check if the one-hot encoding is correct


def test_map_norman_adata_split_to_tensors(mock_adata_path, mock_save_dir):
    split = 1

    # Call the function
    x_tr, x_tr_holdout, x_te, pert_tr, pert_tr_holdout, pert_te = map_norman_adata_split_to_tensors(mock_adata_path, split, mock_save_dir, return_data=True)

    # Assert that the function skips data preparation
    # assert result_dir == data_dir
    result_dir = mock_save_dir
    assert Path(result_dir / "x_tr.pt").exists()
    assert Path(result_dir / "x_tr_holdout.pt").exists()
    assert Path(result_dir / "x_te.pt").exists()
    assert Path(result_dir / "perturbation_labels_tr.pt").exists()
    assert Path(result_dir / "perturbation_labels_tr_holdout.pt").exists()
    assert Path(result_dir / "perturbation_labels_te.pt").exists()

    # Assert that the tensors have the expected shapes
    assert x_tr.shape[0] == pert_tr.shape[0]
    assert x_tr_holdout.shape[0] == pert_tr_holdout.shape[0]
    assert x_te.shape[0] == pert_te.shape[0]
    assert x_tr.shape[1] == x_tr_holdout.shape[1] == x_te.shape[1]
    assert pert_tr.shape[1] == pert_tr_holdout.shape[1] == pert_te.shape[1]

    # Assert that the tensors are not empty
    assert x_tr.numel() > 0
    assert x_tr_holdout.numel() > 0
    assert x_te.numel() > 0
    assert pert_tr.numel() > 0
    assert pert_tr_holdout.numel() > 0
    assert pert_te.numel() > 0

    pert = torch.cat([pert_tr, pert_tr_holdout, pert_te], dim=0).to(torch.float32)
    assert (pert.sum(dim=0) > 0).all() # make sure that every column is at least used once; otherwise the one-hot encoding is not correct
    assert (pert == torch.tensor([
        [0., 0.],
        [0., 0.],
        [1., 0.],
        [0., 1.],
        [1., 1.],
        [1., 0.],
        [0., 1.]])).all()
    

def test_map_norman_adata_split_to_tensors_mapping(mock_adata_path, mock_save_dir, mock_adata):
    split = 1

    # check the index dict
    dict_single_pert_ix = map_norman_adata_split_to_tensors(mock_adata_path, split, mock_save_dir, return_mapping=True)
    assert isinstance(dict_single_pert_ix, dict)
    assert len(dict_single_pert_ix) > 0
    assert all(isinstance(k, str) for k in dict_single_pert_ix.keys())
    assert all(isinstance(v, int) for v in dict_single_pert_ix.values())
    assert "ctrl" not in dict_single_pert_ix # make sure that 



