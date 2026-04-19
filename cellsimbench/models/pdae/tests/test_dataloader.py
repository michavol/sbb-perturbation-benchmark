import pytest
import torch

from PerturbationExtrapolation.pdae.dataloader import _process_obs_perts, PDAEDataset, flatten_observations_perturbations


def test__process_obs_perts():
    pdae_data_dict = {
        "x_tr": [torch.rand((100, 50)), torch.rand((80, 50)), torch.rand((1, 50))],
        "x_val": torch.rand((1, 100, 50)),
        "x_te": [torch.rand((100, 50)), torch.rand((10, 50))],
        "x_empty": [],
        "pert_tr": torch.rand((3, 3)),
        "pert_val": torch.rand((1, 3)),
        "pert_te": torch.rand((2, 3)),
        "pert_empty": torch.Tensor(),
    }

    x_new, pert_new = _process_obs_perts(pdae_data_dict, "tr")
    assert x_new.shape == (181, 50)
    assert pert_new.shape == (181, 3)

    x_new, pert_new = _process_obs_perts(pdae_data_dict, "val")
    assert x_new.shape == (100, 50)
    assert pert_new.shape == (100, 3)

    x_new, pert_new = _process_obs_perts(pdae_data_dict, "te")
    assert x_new.shape == (110, 50)
    assert pert_new.shape == (110, 3)

    x_new, pert_new = _process_obs_perts(pdae_data_dict, "empty")
    assert x_new.shape == torch.Size([0])
    assert pert_new.shape == torch.Size([0])


def test_PDAEDataset():
    # Test with valid data
    x = torch.rand((10, 5))
    pert = torch.rand((10, 3))
    dataset = PDAEDataset(x, pert)

    assert len(dataset) == 10, "Dataset length should match the number of samples in x"
    for i in range(len(dataset)):
        sample_x, sample_pert = dataset[i]
        assert torch.equal(sample_x, x[i]), f"Sample x at index {i} does not match"
        assert torch.equal(sample_pert, pert[i]), f"Sample pert at index {i} does not match"

    # Test with empty data
    x_empty = torch.Tensor()
    pert_empty = torch.Tensor()
    dataset_empty = PDAEDataset(x_empty, pert_empty)

    assert len(dataset_empty) == 0, "Dataset length should be 0 for empty data"
    try:
        dataset_empty[0]
        assert False, "Accessing an empty dataset should raise an error"
    except IndexError:
        pass  # Expected behavior

    
def test_flatten_observations_perturbations():
    # Test with valid input
    x = torch.rand((3, 4, 5))  # (n_envs, n_samples, dim_x)
    pert = torch.rand((3, 2))  # (n_envs, n_perturbations)
    x_new, pert_new = flatten_observations_perturbations(x, pert)
    assert x_new.shape == (12, 5), "Flattened x shape is incorrect"
    assert pert_new.shape == (12, 2), "Flattened pert shape is incorrect"

    # Test with empty x
    x_empty = torch.Tensor()
    pert = torch.rand((3, 2))
    with pytest.raises(AssertionError, match="x should not be empty!"):
        flatten_observations_perturbations(x_empty, pert)

    # Test with empty pert
    x = torch.rand((3, 4, 5))
    pert_empty = torch.Tensor()
    with pytest.raises(AssertionError, match="pert should not be empty!"):
        flatten_observations_perturbations(x, pert_empty)

    # Test with incorrect x shape
    x_invalid = torch.rand((3, 4))  # Missing dim_x
    pert = torch.rand((3, 2))
    with pytest.raises(AssertionError, match="x should have shape \\(n_envs, n_samples, dim_x\\)!"):
        flatten_observations_perturbations(x_invalid, pert)

    # Test with incorrect pert shape
    x = torch.rand((3, 4, 5))
    pert_invalid = torch.rand((3, 2, 1))  # Extra dimension
    with pytest.raises(AssertionError, match="pert should have shape \\(n_envs, n_perturbations\\)!"):
        flatten_observations_perturbations(x, pert_invalid)

    # Test with mismatched shapes
    x = torch.rand((3, 4, 5))
    pert_mismatched = torch.rand((2, 2))  # n_envs mismatch
    with pytest.raises(AssertionError):
        flatten_observations_perturbations(x, pert_mismatched)

