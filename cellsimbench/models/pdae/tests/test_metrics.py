import torch
import pytest

from PerturbationExtrapolation.pdae.metrics import MeanDifferenceMetric, EnergyLossMetric, MMDLossMetric, R2ScoreMetric, check_metrics_input_shapes

def test_check_metrics_input_shapes():
    x_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_pred = torch.tensor([[1.5], [2.5]])  # Different number of features
    with pytest.raises(AssertionError, match="x and x_pred must have the same number of features"):
        check_metrics_input_shapes(x_true, x_pred)

def test_check_metrics_input_shapes():
    x_true = torch.tensor([1.0, 2.0])
    x_pred = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
    with pytest.raises(AssertionError, match="x_true must be two-dimensional"):
        check_metrics_input_shapes(x_true, x_pred)

def test_check_metrics_input_shapes():
    x_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_pred = torch.tensor([1.5, 2.5])
    with pytest.raises(AssertionError, match="x_pred must be two-dimensional"):
        check_metrics_input_shapes(x_true, x_pred)

def test_check_metrics_input_shapes_empty_tensors():
    x_true = torch.tensor([])
    x_pred = torch.tensor([])
    with pytest.raises(AssertionError, match="x_true must not be empty"):
        check_metrics_input_shapes(x_true, x_pred)

def test_check_metrics_input_shapes_non_tensor_inputs():
    x_true = [[1.0, 2.0], [3.0, 4.0]]  # Not a torch.Tensor
    x_pred = [[1.5, 2.5], [3.5, 4.5]]  # Not a torch.Tensor
    with pytest.raises(AssertionError, match="x_true must be a torch.Tensor"):
        check_metrics_input_shapes(x_true, x_pred)

def test_check_metrics_input_shapes_valid_input():
    x_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_pred = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
    try:
        check_metrics_input_shapes(x_true, x_pred)
    except AssertionError:
        pytest.fail("check_metrics_input_shapes raised AssertionError unexpectedly!")


def test_mean_difference_metric_valid_input():
    metric = MeanDifferenceMetric()
    x_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_pred = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
    result = metric.compute(x_true, x_pred)
    expected = torch.linalg.vector_norm(torch.tensor([0.5, 0.5]))
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_mean_difference_metric_zero_difference():
    metric = MeanDifferenceMetric()
    x_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = metric.compute(x_true, x_pred)
    expected = torch.tensor(0.0)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_energy_loss_metric_valid_input():
    metric = EnergyLossMetric()
    x_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_pred = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
    result = metric.compute(x_true, x_pred, normalize_energy_loss=False, beta=1.0)
    expected = torch.cdist(x_true, x_pred).pow(1.0).mean() - 0.5 * torch.cdist(x_pred, x_pred).pow(1.0).mean()
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_energy_loss_metric_normalized():
    metric = EnergyLossMetric()
    x_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_pred = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
    result = metric.compute(x_true, x_pred, normalize_energy_loss=True, beta=1.0)
    expected = (
        torch.cdist(x_true, x_pred).pow(1.0).mean()
        - 0.5 * torch.cdist(x_pred, x_pred).pow(1.0).mean()
        - 0.5 * torch.cdist(x_true, x_true).pow(1.0).mean()
    )
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_energy_loss_metric_beta_parameter():
    metric = EnergyLossMetric()
    x_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_pred = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
    beta = 2.0
    result = metric.compute(x_true, x_pred, normalize_energy_loss=False, beta=beta)
    expected = torch.cdist(x_true, x_pred).pow(beta).mean() - 0.5 * torch.cdist(x_pred, x_pred).pow(beta).mean()
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_energy_loss_zero_difference():
    metric = EnergyLossMetric()
    x_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = metric.compute(x_true, x_pred, normalize_energy_loss=True, beta=1.0)
    expected = torch.tensor(0.0)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_mmd_loss_metric_zero_difference():
    metric = MMDLossMetric()
    x_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = metric.compute(x_true, x_pred)
    expected = torch.tensor(0.0)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_r2_score_metric_perfect_prediction():
    metric = R2ScoreMetric()
    x_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = metric.compute(x_true, x_pred)
    expected = torch.tensor(1.0)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_r2_score_metric_zero_variance():
    metric = R2ScoreMetric()
    x_true = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    x_pred = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    expected = torch.tensor(1.0)
    result = metric.compute(x_true, x_pred)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_r2_score_metric_zero_mean():
    metric = R2ScoreMetric()
    x_true = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    x_pred = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    result = metric.compute(x_true, x_pred)
    expected = torch.tensor(1.0)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_r2_score_metric_negative_r2():
    metric = R2ScoreMetric()
    x_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_pred = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    result = metric.compute(x_true, x_pred)
    assert result < 0, f"Expected negative R^2 score, but got {result}"
