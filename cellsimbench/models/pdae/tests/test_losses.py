import torch
from PerturbationExtrapolation.pdae.losses import get_reconstruction_loss
from PerturbationExtrapolation.pdae.losses import get_perturbation_energy_loss

def test_get_reconstruction_loss_deterministic_decoder():
    # Test case for deterministic decoder where x_rec_prime == x_rec
    batch_size, dim_x = 4, 2
    beta = 1.0

    x = torch.rand(batch_size, dim_x)
    x_rec = torch.rand(batch_size, dim_x)
    x_perturbed = torch.zeros(batch_size, batch_size, dim_x)
    for i in range(batch_size):
        x_perturbed[i, i] = x_rec[i]

    loss = get_reconstruction_loss(x, x_rec, x_perturbed, beta)

    # For deterministic decoder: r1 = r2 and r3 = 0, so loss = r1
    r1 = torch.linalg.vector_norm(x - x_rec, dim=1).pow(beta).mean()
    expected_loss = 0.5 * (r1 + r1 - 0)  # r2 = r1, r3 = 0

    assert torch.isclose(loss, expected_loss), f"Expected {0.5 * r1}, got {loss}"


def test_get_reconstruction_loss_stochastic_decoder():
    # Test case for stochastic decoder where x_rec_prime != x_rec
    batch_size, dim_x = 4, 2
    beta = 1.0

    x = torch.rand(batch_size, dim_x)
    x_rec = torch.rand(batch_size, dim_x)
    x_perturbed = torch.rand(batch_size, batch_size, dim_x)

    loss = get_reconstruction_loss(x, x_rec, x_perturbed, beta)

    x_rec_prime = x_perturbed.diagonal().permute(1, 0)
    r1 = torch.linalg.vector_norm(x - x_rec, dim=1).pow(beta).mean()
    r2 = torch.linalg.vector_norm(x - x_rec_prime, dim=1).pow(beta).mean()
    r3 = torch.linalg.vector_norm(x_rec - x_rec_prime, dim=1).pow(beta).mean()
    expected_loss = 0.5 * (r1 + r2 - r3)

    assert torch.isclose(loss, expected_loss), f"Expected {expected_loss}, got {loss}"


def test_get_reconstruction_loss_zero_loss():
    # Test case where x == x_rec and x_rec_prime == x_rec
    batch_size, dim_x = 4, 2
    beta = 1.0

    x = torch.rand(batch_size, dim_x)
    x_rec = x.clone()
    x_perturbed = torch.zeros(batch_size, batch_size, dim_x)
    for i in range(batch_size):
        x_perturbed[i, i] = x_rec[i]

    loss = get_reconstruction_loss(x, x_rec, x_perturbed, beta)

    # All terms should cancel out, resulting in zero loss
    assert torch.isclose(loss, torch.tensor(0.0)), f"Expected 0.0, got {loss}"

def test_get_perturbation_energy_loss_basic():
    # Basic test case with random inputs
    batch_size, dim_x = 4, 2
    beta = 1.0

    x = torch.rand(batch_size, dim_x)
    x_perturbed = torch.rand(batch_size, batch_size, dim_x)

    loss = get_perturbation_energy_loss(x, x_perturbed, beta)

    # Ensure the loss is a scalar
    assert loss.dim() == 0, f"Expected scalar loss, got {loss.dim()}-dimensional tensor"


def test_get_perturbation_energy_loss_zero_loss():
    # Test case where x_perturbed matches x, leading to zero similarities and diversities
    batch_size, dim_x = 4, 2
    beta = 1.0

    x = torch.rand(batch_size, dim_x)
    x_perturbed = x.unsqueeze(1).expand(batch_size, batch_size, dim_x).transpose(0,1)

    loss = get_perturbation_energy_loss(x, x_perturbed, beta)

    # Loss should be zero
    assert torch.isclose(loss, torch.tensor(0.0)), f"Expected 0.0, got {loss}"


def test_get_perturbation_energy_loss_high_beta():
    # Test case with a high beta value to emphasize differences
    batch_size, dim_x = 4, 2
    beta = 10.0

    x = torch.rand(batch_size, dim_x)
    x_perturbed = torch.rand(batch_size, batch_size, dim_x)

    loss = get_perturbation_energy_loss(x, x_perturbed, beta)

    # Ensure the loss is computed without errors
    assert loss is not None, "Loss computation failed for high beta value"


def test_get_peturbation_energy_loss_zero_losses():
    batch_size = 128
    dim_x = 2
    beta = 1

    x = torch.zeros((batch_size, dim_x))
    x_pert = torch.zeros((batch_size, batch_size, dim_x))
    
    loss, similarities, diversities = get_perturbation_energy_loss(x, x_pert, beta=beta, return_components=True, remove_marginal=True)
    
    assert loss == 0.0, f"Expected loss to be 0.0, got {loss}"
    assert similarities.sum() == 0.0, f"Expected similarities sum to be 0.0, got {similarities.sum()}"
    assert diversities.sum() == 0.0, f"Expected diversities sum to be 0.0, got {diversities.sum()}"
    assert similarities.shape == (batch_size, batch_size, 1), f"Expected similarities shape to be {(batch_size, batch_size, 1)}, got {similarities.shape}"
    assert diversities.shape == (batch_size, batch_size, batch_size), f"Expected diversities shape to be {(batch_size, batch_size, batch_size)}, got {diversities.shape}"

def test_get_perturbation_energy_loss_similarities():
    batch_size = 128
    dim_x = 2
    beta = 1

    x = torch.rand((batch_size, dim_x))
    x_pert = torch.rand((batch_size, batch_size, dim_x))
    
    expected_similarities_sum = 0
    elements = 0
    for i in range(batch_size):
        for j in range(batch_size):
            if i != j:
                elements += 1
                expected_similarities_sum += torch.linalg.vector_norm(x_pert[j,i] - x[i])
    
    _, similarities, _ = get_perturbation_energy_loss(x, x_pert, beta=beta, return_components=True, remove_marginal=True)
    actual_similarities_sum = similarities.sum()
    print(f"{(expected_similarities_sum - actual_similarities_sum).abs()=}")
    assert (expected_similarities_sum - actual_similarities_sum).abs() < 5e-2


def test_get_perturbation_energy_loss_diversities():
    batch_size = 128
    dim_x = 2
    beta = 1

    x = torch.rand((batch_size, dim_x))
    x_pert = torch.rand((batch_size, batch_size, dim_x))

    expected_diversities_sum = 0
    for i in range(batch_size):
        for j in range(batch_size):
            for j2 in range(batch_size):
                if i != j and i != j2 and j != j2:
                    expected_diversities_sum += torch.linalg.vector_norm(x_pert[j,i] - x_pert[j2,i])

    _, _, diversities = get_perturbation_energy_loss(x, x_pert, beta=beta, return_components=True, remove_marginal=True)
    actual_diversities_sum = diversities.sum()
    print(f"{(expected_diversities_sum - actual_diversities_sum).abs()=}")
    assert (expected_diversities_sum - actual_diversities_sum).abs() < expected_diversities_sum * 1e-4


def test_get_perturbation_energy_loss_diversities_with_marginal():
    batch_size = 128
    dim_x = 2
    beta = 1

    x = torch.rand((batch_size, dim_x))
    x_pert = torch.rand((batch_size, batch_size, dim_x))

    expected_diversities_sum = 0
    for i in range(batch_size):
        for j in range(batch_size):
            for j2 in range(batch_size):
                expected_diversities_sum += torch.linalg.vector_norm(x_pert[j,i] - x_pert[j2,i])

    _, _, diversities = get_perturbation_energy_loss(x, x_pert, beta=beta, return_components=True, remove_marginal=False)
    actual_diversities_sum = diversities.sum()
    print(f"{(expected_diversities_sum - actual_diversities_sum).abs()=}")
    assert (expected_diversities_sum - actual_diversities_sum).abs() < expected_diversities_sum * 1e-4