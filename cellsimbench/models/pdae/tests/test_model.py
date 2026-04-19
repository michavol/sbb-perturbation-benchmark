import pytest
import torch
from PerturbationExtrapolation.pdae.model import Encoder, Decoder, PerturbationAutoencoder

@pytest.fixture
def encoder():
    return Encoder(dim_x=10, dim_z=2, layer_shapes=[20, 15])

@pytest.fixture
def decoder():
    return Decoder(dim_z=2, dim_noise=2, sigma_noise=0.1, dim_x=10, layer_shapes=[15, 20])

@pytest.fixture
def autoencoder():
    return PerturbationAutoencoder(
        dim_z=2,
        dim_noise=0,
        sigma_noise=0.1,
        dim_x=10,
        n_perturbations=3,
        encoder_layer_shapes=[20, 20],
        decoder_layer_shapes=[20, 20],
        update_encoder_on_reconstruction_loss=False,
        use_bias_in_perturbation_matrix=False,
    )

def test_encoder_forward(encoder):
    x = torch.randn(32, 10)  # batch_size=32, dim_x=10
    z = encoder(x)
    assert z.shape == (32, 2)  # batch_size=32, dim_z=2

def test_decoder_forward(decoder):
    z = torch.randn(32, 2)  # batch_size=32, dim_z=2
    x_rec = decoder(z)
    assert x_rec.shape == (32, 10)  # batch_size=32, dim_x=10

def test_autoencoder_forward(autoencoder):
    x = torch.randn(32, 10)  # batch_size=32, dim_x=10
    perturbation_labels = torch.randint(0, 2, (32, 3)).to(torch.float32)  # batch_size=32, n_perturbations=3
    x_rec, x_perturbed, z_base = autoencoder(x, perturbation_labels)
    assert x_rec.shape == (32, 10)  # batch_size=32, dim_x=10
    assert x_perturbed.shape == (32, 32, 10)  # batch_size=32, batch_size=32, dim_x=10
    assert z_base.shape == (32, 2)  # batch_size=32, dim_z=2

def test_autoencoder_predict(autoencoder):
    x_src = torch.randn(32, 10)  # batch_size=32, dim_x=10
    perturbation_labels_src = torch.randint(0, 2, (32, 3)).to(torch.float32)  # batch_size=32, n_perturbations=3
    perturbation_labels_trg = torch.randint(0, 2, (32, 3)).to(torch.float32)  # batch_size=32, n_perturbations=3
    x_trg, z_trg = autoencoder.predict(x_src, perturbation_labels_src, perturbation_labels_trg)
    assert x_trg.shape == (32 * 32, 10)  # batch_size * batch_size, dim_x=10
    assert z_trg.shape == (32 * 32, 2)  # batch_size * batch_size, dim_z=2

@pytest.mark.parametrize("z, perturbation_labels, perturbation_matrix", [
    (torch.tensor([[2, 1], [2, 1], [2, 1], [2, 1], [2, 1]], dtype=torch.float32),
     torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=torch.float32),
     torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)),
    (torch.tensor([[0.18465004, 0.04531688],
       [0.59959955, 0.48785261],
       [0.37410318, 0.07178675],
       [0.07587651, 0.95420308]], dtype=torch.float64),
    torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=torch.float64),
    torch.tensor([
        [0.41494951, 0.18945315, -0.10877352],
        [0.44253573, 0.02646987,  0.9088862 ]], dtype=torch.float64)
    ),
    (torch.tensor([
       [0.18465004, 0.04531688],
       [0.18465004, 0.04531688],
       [0.59959955, 0.48785261],
       [0.59959955, 0.48785261],
       [0.37410318, 0.07178675],
       [0.37410318, 0.07178675],
       [0.07587651, 0.95420308],
       [0.07587651, 0.95420308],
    ], dtype=torch.float64),
    torch.tensor([
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ], dtype=torch.float64),
    torch.tensor([
        [0.41494951, 0.18945315, -0.10877352],
        [0.44253573, 0.02646987,  0.9088862 ]], dtype=torch.float64)
    )
])
def test_update_perturbation_matrix_via_OLS(autoencoder, z, perturbation_labels, perturbation_matrix):
    """
    Here we test the function code in general and whether the OLS is performed correctly in pytorch, in particular w.r.t. the intercept.
    These test cases were generated using the following code:
    ```python
        import numpy as np
        from sklearn.linear_model import LinearRegression
        z = np.random.rand(6, 2)
        perturbation_labels = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        reg = LinearRegression(fit_intercept=True).fit(perturbation_labels, z)
        print(reg.coef_)
    ```
    """
    # Call the method to update the perturbation matrix
    autoencoder.update_perturbation_matrix_via_OLS(z, perturbation_labels)
    # Check that the perturbation matrix has the correct shape
    assert autoencoder.perturbation_matrix.weight.shape == perturbation_matrix.shape
    assert autoencoder.perturbation_matrix.weight.data == pytest.approx(perturbation_matrix, abs=1e-8)


def test_autoencoder_forward_with_components(autoencoder):
    x = torch.randn(32, 10)  # batch_size=32, dim_x=10
    perturbation_labels = torch.randint(0, 2, (32, 3)).to(torch.float32)  # batch_size=32, n_perturbations=3
    result = autoencoder(x, perturbation_labels, return_components=True)
    x_rec, x_perturbed, z_base, z, z_perturbed, perturbation_shifts, perturbation_diff = result

    # Check the shapes of the returned components
    assert x_rec.shape == (32, 10)  # batch_size=32, dim_x=10
    assert x_perturbed.shape == (32, 32, 10)  # batch_size=32, batch_size=32, dim_x=10
    assert z_base.shape == (32, 2)  # batch_size=32, dim_z=2
    assert z_perturbed.shape == (32, 32, 2)  # batch_size=32, batch_size=32, dim_z=2
    assert perturbation_shifts.shape == (32, 2)  # batch_size=32, dim_z=2
    assert perturbation_diff.shape == (32, 32, 2)  # batch_size=32, batch_size=32, dim_z=2

    # Check the computations
    assert torch.isclose(z_perturbed[1,3], (z[1] - perturbation_shifts[1] + perturbation_shifts[3])).all()
    assert torch.isclose(z_perturbed[1,3], z[1] + perturbation_diff[1,3]).all()
    assert torch.isclose(z_perturbed[1,3], z_base[1] + perturbation_shifts[3]).all()
    assert torch.isclose(x_perturbed[1,3], autoencoder.decoder(z_perturbed[1,3])).all()


def test_predict_with_components(autoencoder):
    x_src = torch.randn(32, 10)  # batch_size=32, dim_x=10
    perturbation_labels_src = torch.randint(0, 2, (32, 3)).to(torch.float32)  # batch_size=32, n_perturbations=3
    perturbation_labels_trg = torch.randint(0, 2, (32, 3)).to(torch.float32)  # batch_size=32, n_perturbations=3

    result = autoencoder.predict(x_src, perturbation_labels_src, perturbation_labels_trg, return_components=True)
    x_trg, z_trg, z_base, shifts_src, shifts_trg = result

    # Check the shapes of the returned components
    assert x_trg.shape == (32 * 32, 10)  # batch_size * batch_size, dim_x=10
    assert z_trg.shape == (32 * 32, 2)  # batch_size * batch_size, dim_z=2
    assert z_base.shape == (32, 1, 2)  # batch_size, 1, dim_z
    assert shifts_src.shape == (32, 2)  # batch_size, dim_z
    assert shifts_trg.shape == (1, 32, 2)  # batch_size, dim_z

    # Check the computations
    assert (z_trg[1 * 32 + 3] == z_base[1] + shifts_trg[0,3]).all()
    assert z_base[1].squeeze() == pytest.approx((autoencoder.encoder(x_src)[1] - shifts_src[1]).detach(), abs=1e-5)
    assert x_trg[1 * 32 + 3] == pytest.approx(autoencoder.decoder(z_trg[1 * 32 + 3]).detach(), abs=1e-5)
