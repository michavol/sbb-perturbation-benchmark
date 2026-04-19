import torch
import pytest
from PerturbationExtrapolation.pdae.baselines import CPAModel, PoolAllModel, PseudoBulkingModel, LinearRegressionModel, get_pseudo_bulk

def test_pool_all_model():
    x_tr = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pert_tr = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    model = PoolAllModel()
    model.fit(x_tr, pert_tr)
    prediction = model.predict()
    assert torch.equal(prediction, x_tr), "PoolAllModel prediction failed."


@pytest.mark.parametrize("x_tr, pert_tr, pert_te, result",[(
    torch.ones((7,1)),
    torch.Tensor([
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.]
    ]), 
    torch.Tensor(
        [1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
    torch.ones((2,1))
    ),
    (
    torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
    torch.tensor([
        [0.0, 0.0, 0.0], 
        [0.0, 1.0, 0.0], 
        [1.0, 0.0, 1.0], 
        [0.0, 1.0, 0.0]
    ]), 
    torch.tensor([1.0, 0.0, 0.0]),
    torch.tensor([[5.0, 6.0]])
    )
    ])
def test_get_pseudo_bulk(x_tr, pert_tr, pert_te, result):
    actual_result = get_pseudo_bulk(x_tr, pert_tr, pert_te)
    assert actual_result.shape == result.shape, "Shape mismatch in get_pseudo_bulk output."
    assert torch.equal(actual_result, result), "get_pseudo_bulk output does not match expected result."


def test_pseudo_bulking_model():
    x_tr = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    pert_tr = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.25, 0.0, 1.0], [0.0, 1.0, 0.0]])
    pert_te = torch.tensor([1.0, 1.0, 0.0])
    model = PseudoBulkingModel()
    model.fit(x_tr, pert_tr)
    prediction = model.predict(pert_te)
    expected_prediction = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    assert prediction.shape == expected_prediction.shape, "PseudoBulkingModel prediction shape mismatch."
    assert torch.equal(prediction, expected_prediction), "PseudoBulkingModel prediction failed."


def test_linear_regression_model_with_reference():
    x_tr = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    pert_tr = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    pert_te = torch.tensor([0.5, 0.5])
    model = LinearRegressionModel(return_distribution=False)
    model.fit(x_tr, pert_tr)
    prediction = model.predict(pert_te)
    expected_prediction = torch.tensor([[4.0, 5.0]])
    assert torch.allclose(prediction, expected_prediction, atol=1e-5), "LinearRegressionModel prediction failed."


def test_linear_regression_model_with_reference_return_distribution():
    x_tr = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    pert_tr = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    pert_te = torch.tensor([0.5, 0.5])
    model = LinearRegressionModel(return_distribution=True)
    model.fit(x_tr, pert_tr)
    prediction = model.predict(pert_te)
    expected_prediction = torch.tensor([[4.0, 5.0]])
    assert torch.allclose(prediction, expected_prediction, atol=1e-5), "LinearRegressionModel prediction failed."


def test_cpa_model(missing_modules):
    seed = 0
    device = "cpu"
    pert_ohe_str_mapping_path = "tmp/pert_ohe_str_mapping.json"
    cpa_model_path = "tmp/cpa_model.pt"
    with missing_modules("cpa"):
        with pytest.raises(ImportError):
            CPAModel(seed=seed, device=device, pert_ohe_str_mapping_path=pert_ohe_str_mapping_path, cpa_model_path=cpa_model_path)