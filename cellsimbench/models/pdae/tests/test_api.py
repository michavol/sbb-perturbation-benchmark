import os
import pytest
import torch
from PerturbationExtrapolation.pdae.api import PDAEData, PDAEConfig, PDAEModel
from PerturbationExtrapolation.pdae.baselines import PoolAllModel, get_baselines
from PerturbationExtrapolation.pdae.metrics import EnergyDistanceMetric, get_metrics


@pytest.fixture
def sample_pdae_data():
    # simulated data for testing
    observations = torch.tensor([[ 0.9997,  0.4042],
        [ 0.4492,  1.3532],
        [ 1.4445,  0.1535],
        [-0.8044,  1.4797],
        [ 1.2002, -0.4335],
        [ 1.3971,  1.9065],
        [ 0.9776,  0.3468],
        [ 1.2897,  0.1901],
        [ 1.3730, -0.3514],
        [ 1.0546,  1.2565],
        [ 0.8697, -0.0588],
        [ 1.0625,  1.2484],
        [ 1.0009, -0.0454],
        [ 0.6400,  0.9752],
        [ 2.0877,  0.5601],
        [ 2.1990,  0.1546],
        [ 1.1444,  0.0444],
        [ 1.6986,  0.7587],
        [ 1.8318,  0.1838],
        [ 0.9631, -0.0817],
        [-0.1913,  0.9991],
        [ 1.1575,  1.1319],
        [ 0.9966, -0.4473],
        [ 1.2225,  0.1734],
        [ 1.4965,  0.5354],
        [ 1.5285, -0.1326],
        [ 0.3567,  1.7564],
        [ 0.7594,  1.9524],
        [ 1.3325, -0.1494],
        [ 0.5523,  0.8037],
        [ 1.9979,  2.0030],
        [ 1.0027, -0.5002],
        [ 0.6942,  0.6799],
        [ 1.2191, -0.1672],
        [ 1.5024, -0.2642],
        [-0.7494,  0.8240],
        [ 0.6244,  0.8213],
        [ 1.1116,  0.5444],
        [ 2.5417, -0.5160],
        [ 1.6689, -0.3834],
        [ 1.0671, -0.0892],
        [ 0.7506,  0.4555],
        [-0.3612,  0.7693],
        [ 0.9221,  0.6240],
        [ 0.3207,  0.7108],
        [ 0.8597,  0.4935],
        [ 1.4082, -0.3916],
        [ 3.6891, -0.4329],
        [ 0.8941,  0.3929],
        [ 0.9219,  0.8079],
        [ 1.4008,  1.0708],
        [ 2.9368,  0.9058],
        [ 0.2725,  1.7847],
        [ 0.9794,  0.3785],
        [ 0.8515, -0.2498],
        [ 0.6570,  1.0475],
        [-0.2268,  1.7431],
        [ 0.9250,  0.4664],
        [ 0.5875,  1.4371],
        [ 1.1761,  0.9006],
        [ 0.8802,  0.7274],
        [ 0.2997,  1.5641],
        [ 0.5651,  0.8017],
        [ 0.9637,  0.0126],
        [ 0.2831,  1.5431],
        [ 0.5920,  0.7950],
        [ 1.0063,  0.0891],
        [ 1.5856, -0.2388],
        [-0.0622,  1.3002],
        [-0.5469,  1.3422],
        [ 1.0618,  0.2676],
        [ 1.5523, -0.1085],
        [ 1.1781, -0.1766],
        [-0.0353,  1.4711],
        [ 0.4170,  0.8220],
        [ 1.1432,  1.6139],
        [-0.7611,  0.6778],
        [ 0.2767,  0.8889],
        [ 0.8403,  1.6690],
        [ 0.8615,  0.5829],
        [ 1.3116, -0.2148],
        [ 1.7171, -0.6074],
        [ 1.4603,  1.0224],
        [-0.6400,  1.1421],
        [ 0.6690,  0.6130],
        [ 1.3616, -0.6172],
        [ 0.9202, -0.6206],
        [ 1.3848,  1.4680],
        [ 1.0716,  0.0588],
        [ 0.8520,  1.0055],
        [ 0.2213,  0.9284],
        [-0.3230,  0.9627],
        [ 0.3567,  0.8153],
        [ 1.4538, -0.0407],
        [ 1.4541,  0.6024],
        [ 0.4172,  0.7821],
        [ 1.4876, -0.1188],
        [ 0.3853,  0.8802],
        [ 1.4639,  0.2670],
        [ 1.0474, -0.3820]])
    
    perturbations = torch.tensor([[0.1000, 0.2500, 0.0000],
        [0.5000, 1.5000, 0.0000],
        [0.5000, 0.0000, 0.0000],
        [0.0000, 1.0000, 1.0000],
        [0.5000, 0.0000, 0.0000],
        [1.5000, 0.5000, 0.0000],
        [0.0000, 0.5000, 0.0000],
        [0.5000, 0.0000, 0.0000],
        [0.5000, 0.0000, 0.0000],
        [0.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, 0.0000],
        [0.0000, 1.0000, 0.0000],
        [1.5000, 0.5000, 0.0000],
        [2.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 1.0000],
        [1.0000, 0.0000, 0.0000],
        [0.1000, 0.2500, 0.0000],
        [0.0000, 2.0000, 0.0000],
        [1.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [0.5000, 0.0000, 0.0000],
        [1.0000, 0.0000, 0.0000],
        [1.0000, 0.0000, 0.0000],
        [1.0000, 1.0000, 0.0000],
        [1.0000, 0.0000, 1.0000],
        [1.0000, 0.0000, 0.0000],
        [0.0000, 1.0000, 0.0000],
        [1.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, 0.0000],
        [0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [1.0000, 0.0000, 0.0000],
        [0.0000, 2.0000, 0.0000],
        [0.0000, 1.0000, 0.0000],
        [0.5000, 0.7500, 0.0000],
        [2.0000, 0.0000, 0.0000],
        [1.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [0.0000, 1.0000, 0.0000],
        [0.0000, 0.5000, 0.0000],
        [0.0000, 1.0000, 0.0000],
        [0.0000, 0.5000, 0.0000],
        [0.5000, 0.0000, 0.0000],
        [2.0000, 0.0000, 0.0000],
        [0.0000, 0.5000, 0.0000],
        [0.5000, 0.7500, 0.0000],
        [0.0000, 0.0000, 1.0000],
        [1.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, 1.0000],
        [0.5000, 0.7500, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [0.5000, 0.7500, 0.0000],
        [0.0000, 1.0000, 1.0000],
        [0.5000, 0.5000, 0.0000],
        [0.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, 1.0000],
        [0.5000, 0.7500, 0.0000],
        [0.0000, 0.0000, 1.0000],
        [0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [1.0000, 1.0000, 0.0000],
        [0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [1.0000, 0.0000, 0.0000],
        [0.5000, 1.5000, 0.0000],
        [0.0000, 1.0000, 1.0000],
        [0.0000, 0.0000, 0.0000],
        [1.0000, 0.0000, 0.0000],
        [0.5000, 0.0000, 0.0000],
        [0.5000, 1.5000, 0.0000],
        [0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 1.0000],
        [0.0000, 2.0000, 0.0000],
        [0.0000, 2.0000, 0.0000],
        [1.0000, 1.0000, 0.0000],
        [0.0000, 1.0000, 0.0000],
        [1.0000, 0.0000, 0.0000],
        [1.0000, 0.0000, 0.0000],
        [1.5000, 0.5000, 0.0000],
        [0.0000, 1.0000, 1.0000],
        [0.0000, 1.0000, 0.0000],
        [1.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 1.0000],
        [0.5000, 1.5000, 0.0000],
        [0.0000, 2.0000, 0.0000],
        [0.0000, 1.0000, 0.0000],
        [1.0000, 0.0000, 0.0000],
        [1.0000, 0.0000, 0.0000],
        [0.0000, 1.0000, 0.0000],
        [1.0000, 0.0000, 0.0000],
        [0.0000, 1.0000, 0.0000],
        [0.5000, 0.0000, 0.0000],
        [0.5000, 0.0000, 0.0000]])
    
    split = ['val', 'test', 'val', 'test', 'val', 'test', 'val', 'val', 'val', 'train', 'train', 'train', 'train', 'train', 'test', 'test', 'train', 'train', 'train', 'val', 'test', 'train_holdout', 'train', 'val', 'train', 'train', 'train_holdout', 'test', 'train', 'train', 'val', 'train', 'train', 'train', 'train', 'test', 'train', 'val', 'test', 'train', 'train', 'train', 'train', 'val', 'train', 'val', 'val', 'test', 'val', 'val', 'train', 'val', 'train', 'val', 'train', 'val', 'test', 'val', 'train', 'train', 'val', 'train', 'train', 'train', 'val', 'train', 'train', 'train', 'test', 'test', 'train', 'train', 'val', 'test', 'train', 'train', 'test', 'test', 'val', 'train', 'train', 'train', 'test', 'test', 'train', 'train', 'train', 'train', 'train', 'train', 'test', 'test', 'train', 'train', 'train', 'train', 'train', 'train', 'val', 'val']

    is_reference_perturbation = (perturbations.sum(dim=1) == 0).bool()

    return PDAEData(
        data_name="test_data",
        save_dir=None,
        observations=observations,
        perturbations=perturbations,
        is_reference_perturbation=is_reference_perturbation,
        splits=split,
        train_split="train",
        train_holdout_split="train_holdout",
        val_split="val",
        test_split="test",
    )

@pytest.fixture
def sample_pdae_config():
    return PDAEConfig(
        seed=42,
        model_name="test_model",
        batch_size=10,  # Increased batch size for more stable training
        learning_rate=0.01,  # Increased learning rate for faster convergence
        dim_z_model=2,  # Reduced latent dimension
        dim_noise_model=0,  # Removed noise to make model deterministic
        sigma_noise_model=0.0,  # No noise
        decoder_layer_shapes=[2, 2],  # Smaller, simpler architecture
        encoder_layer_shapes=[2, 2],  # Smaller, simpler architecture
        weight_reconstruction_loss=0.0,
        weight_perturbation_energy_loss=1.0,
        weight_marginal_prior_energy_loss=0.0,
        weight_l21=0.0,
        update_encoder_on_reconstruction_loss=False,  # Allow encoder updates
        normalize_energy_loss=False,
        beta=1,
        use_bias_in_perturbation_matrix=False,
    )

def test_pdae_data_validation(sample_pdae_data):
    with pytest.raises(ValueError):
        sample_pdae_data.perturbations[sample_pdae_data.is_reference_perturbation] = torch.tensor([[1.0, 1.0, 1.0]])
        sample_pdae_data.__post_init__()


def test_pdae_model_initialization(sample_pdae_data, sample_pdae_config, tmp_path):
    device = "cpu"
    save_dir = tmp_path
    model = PDAEModel(
        device=device,
        pdae_data=sample_pdae_data,
        pdae_config=sample_pdae_config,
        save_dir=save_dir,
    )
    assert model.device == device
    assert model.pdae_data == sample_pdae_data
    assert model.pdae_config == sample_pdae_config
    assert model.save_dir == save_dir


def test_pdae_model_train(sample_pdae_data, sample_pdae_config, tmp_path):
    device = "cpu"
    save_dir = tmp_path
    model = PDAEModel(
        device=device,
        pdae_data=sample_pdae_data,
        pdae_config=sample_pdae_config,
        save_dir=save_dir,
    )
    model.train(num_epochs=1, n_epochs_run_eval=1, n_epochs_model_checkpoint=1, n_epochs_plot_losses=1, losses_to_plot=["tr_reconstruction_losses"])
    assert model.num_epochs_trained == 1
    assert model.pdae_model is not None
    assert model.losses is not None
    assert model.mmd_stats is not None

def test_pdae_model_predict(sample_pdae_data, sample_pdae_config, tmp_path):
    device = "cpu"
    save_dir = tmp_path
    model = PDAEModel(
        device=device,
        pdae_data=sample_pdae_data,
        pdae_config=sample_pdae_config,
        save_dir=save_dir,
    )
    model.train(num_epochs=10)  # Increased epochs for better convergence
    pert_te = torch.tensor([[1.0, 0.0, 0.0]])  # Use simpler perturbation
    predictions = model.predict(pert_te=pert_te)
    assert predictions is not None
    assert isinstance(predictions, tuple)
    assert len(predictions) == 2

def test_pdae_model_run_eval(sample_pdae_data, sample_pdae_config, tmp_path):
    device = "cpu"
    save_dir = tmp_path
    model = PDAEModel(
        device=device,
        pdae_data=sample_pdae_data,
        pdae_config=sample_pdae_config,
        save_dir=save_dir,
    )
    model.train(num_epochs=10)  # Increased epochs for better convergence
    eval_results = model.run_eval(
        models=get_baselines(),
        metrics=get_metrics(),
        run_eval_batched=True,
        return_df=True
    )
    assert eval_results is not None
    assert not eval_results.empty

def test_pdae_model_get_agg_eval_results(sample_pdae_data, sample_pdae_config, tmp_path):
    device = "cpu"
    save_dir = tmp_path
    model = PDAEModel(
        device=device,
        pdae_data=sample_pdae_data,
        pdae_config=sample_pdae_config,
        save_dir=save_dir,
    )
    model.train(num_epochs=10)  # Increased epochs for better convergence
    pool_all = PoolAllModel()
    energy_distance = EnergyDistanceMetric()
    model.run_eval(
        models=[pool_all],
        metrics=[energy_distance],
        run_eval_batched=False
    )
    agg_results = model.get_agg_eval_results(layout="long")
    assert agg_results is not None
    assert not agg_results.empty

def test_pdae_model_plot_losses(sample_pdae_data, sample_pdae_config, tmp_path):
    device = "cpu"
    save_dir = tmp_path
    model = PDAEModel(
        device=device,
        pdae_data=sample_pdae_data,
        pdae_config=sample_pdae_config,
        save_dir=save_dir,
    )
    model.train(num_epochs=1)
    model.plot_losses(losses_to_plot=["tr_reconstruction_losses"])
    assert os.path.exists(save_dir)
