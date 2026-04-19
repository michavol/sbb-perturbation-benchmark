import torch
from tqdm import tqdm

def check_extrapolation_guarantees(train_perturbations: torch.Tensor, test_perturbations: torch.Tensor, print_progress=False) -> dict:
    """
    Verifies if the test perturbations (relative to the reference) lie within the span of the train perturbations (relative to the reference).
    This ensures the test effects are uniquely identifiable, as per Theorem 4.6 in https://arxiv.org/abs/2504.18522.

    The method compares the rank of the matrix of differenced train perturbations with and without each test perturbation.

    Args:
        train_perturbations (torch.Tensor): Tensor of training perturbations.
        ref_indices (torch.Tensor): Indices of the reference perturbations in the training set.
        test_perturbations (torch.Tensor): Tensor of test perturbations.

    Returns:
        dict: A dictionary mapping each test perturbation to a boolean indicating whether it lies in the span of the train perturbations.
    """
    train_perturbations_unique = torch.unique(train_perturbations, dim=0)
    test_perturbations_unique = torch.unique(test_perturbations, dim=0)

    is_reference = (train_perturbations_unique == 0.0).all(dim=1)

    # compute differences relative to the reference for train and test perturbations
    train_diff = (train_perturbations_unique - train_perturbations_unique[is_reference]).to(torch.float)
    train_diff = train_diff[~is_reference]
    test_diff = (test_perturbations_unique - train_perturbations_unique[is_reference]).to(torch.float)
    
    # check if each test perturbation lies in the span of the train perturbations
    train_rank = torch.linalg.matrix_rank(train_diff)
    in_span = []
    for test in tqdm(test_diff, desc="Checking extrapolation guarantees for test perturbations", disable=not print_progress):
        augmented_matrix = torch.vstack([train_diff, test])
        augmented_rank = torch.linalg.matrix_rank(augmented_matrix)
        in_span.append((train_rank == augmented_rank).item())
    
    return dict(zip(test_perturbations_unique, in_span))
