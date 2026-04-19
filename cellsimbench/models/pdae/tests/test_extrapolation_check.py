import torch
import pytest
from PerturbationExtrapolation.pdae.extrapolation_check import check_extrapolation_guarantees


def check_equal_dicts(dict1, dict2):
    """
    Helper function to check if two dictionaries have the same keys and values.
    """
    print(set(dict1.keys()), set(dict2.keys()))
    return set(dict1.keys()) == set(dict2.keys()) and all(dict1[k] == dict2[k] for k in dict1)


def test_check_extrapolation_guarantees():
    # Case 1: Test perturbations lie within the span of train perturbations
    train_perturbations = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    test_perturbations = torch.tensor([[0.5, 0.5], [0.0, 0.0]])
    result = check_extrapolation_guarantees(train_perturbations, test_perturbations)
    expected = {test_perturbations[0]: True, test_perturbations[1]: True}
    check_equal_dicts(result, expected)

    # Case 2: Test perturbations do not lie within the span of train perturbations
    train_perturbations = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    test_perturbations = torch.tensor([[2.0, 2.0]])
    result = check_extrapolation_guarantees(train_perturbations, test_perturbations)
    expected = {test_perturbations[0]: False}
    check_equal_dicts(result, expected)

    # Case 3: Edge case with empty train and test perturbations
    train_perturbations = torch.tensor([])
    test_perturbations = torch.tensor([])
    with pytest.raises(IndexError):  # Expecting an error due to empty tensors
        check_extrapolation_guarantees(train_perturbations, test_perturbations)

    # Case 4: Test perturbations identical to train perturbations
    train_perturbations = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    test_perturbations = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    result = check_extrapolation_guarantees(train_perturbations, test_perturbations)
    expected = {test_perturbations[0]: True, test_perturbations[1]: True}
    check_equal_dicts(result, expected)