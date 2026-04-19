import torch
from PerturbationExtrapolation.pdae import mmd
from sklearn.metrics import r2_score as sklearn_r2_score

def check_metrics_input_shapes(x_true: torch.Tensor, x_pred: torch.Tensor) -> None:
    '''
    Check if the input tensors have the correct shapes for the metrics.
    
    Args:
        x_true: true observations (n_x_true, dim_x)
        x_pred: predicted observations (n_x_pred, dim_x)
    
    Raises:
        AssertionError: if the input tensors do not have the correct shapes
    '''
    assert isinstance(x_true, torch.Tensor), "x_true must be a torch.Tensor"
    assert isinstance(x_pred, torch.Tensor), "x_pred must be a torch.Tensor"
    assert x_true.numel() > 0, "x_true must not be empty"
    assert x_pred.numel() > 0, "x_pred must not be empty"
    assert len(x_true.shape) == 2, "x_true must be two-dimensional"
    assert len(x_pred.shape) == 2, "x_pred must be two-dimensional"
    assert x_true.shape[1] == x_pred.shape[1], "x_true and x_pred must have the same number of features (dim_x)"


class Metrics:
    name: str


class MeanDifferenceMetric(Metrics):
    def __init__(self):
        self.name = "mean_difference"

    def compute(self, x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x_true: true observations (n_x_true, dim_x)
            x_pred: predicted observations (n_x_pred, dim_x)
        Returns:
            L2 norm between empirical mean of observations and empirical mean of predictions
        '''
        check_metrics_input_shapes(x_true, x_pred)
        mu_true = torch.mean(x_true, dim=[0])
        mu_pred = torch.mean(x_pred, dim=[0])
        return torch.linalg.vector_norm(mu_true - mu_pred)


class EnergyLossMetric(Metrics):
    def __init__(self):
        self.name = "energy_loss"

    def compute(self, x_true: torch.Tensor, x_pred: torch.Tensor, normalize_energy_loss: bool = False, beta: float = 1.0) -> torch.Tensor:
        '''
        Args:
            x_true: true observations (n_x_true, dim_x)
            x_pred: predicted observations (n_x_pred, dim_x)
            normalize_energy_loss: if True, normalize the energy loss s.t. get_energy_loss(x, x) == 0
        Returns:
            energy loss between true and predicted observations
        '''
        loss = torch.cdist(x_true, x_pred).pow(beta).mean() - 0.5 * torch.cdist(x_pred, x_pred).pow(beta).mean()
        if normalize_energy_loss:
            loss -= 0.5 * torch.cdist(x_true, x_true).pow(beta).mean()
        return loss


class EnergyDistanceMetric(Metrics):
    def __init__(self):
        self.name = "energy_distance"
        self.energy_loss = EnergyLossMetric()

    def compute(self, x_true: torch.Tensor, x_pred: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        check_metrics_input_shapes(x_true, x_pred)
        return 2 * self.energy_loss.compute(x_true, x_pred, normalize_energy_loss=True, beta=beta)


class MMDLossMetric(Metrics):
    def __init__(self):
        self.name = "mmd_loss"
        
    def compute(self, x_true: torch.Tensor, x_pred: torch.Tensor, rbf_bandwidth: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x_true: true observations (n_x_true, dim_x)
            x_pred: predicted observations (n_x_pred, dim_x)
            rbf_bandwidth: bandwidth for the RBF kernel
        Returns:
            Maximum Mean Discrepancy (MMD) between true and predicted observations
        """
        check_metrics_input_shapes(x_true, x_pred)
        mmd_loss = mmd.MMDLoss(device=x_true.device, rbf_bandwidth=rbf_bandwidth)
        return mmd_loss(x_true, x_pred)


class R2ScoreMetric(Metrics):
    def __init__(self):
        self.name = "r2_score"

    def compute(self, x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_true: true observations (n_x_true, dim_x)
            x_pred: predicted observations (n_x_pred, dim_x)
        Returns:
            R^2 score between true and predicted observations
        """
        check_metrics_input_shapes(x_true, x_pred)
        return torch.tensor(sklearn_r2_score(x_true.mean(dim=0).cpu().numpy(), x_pred.mean(dim=0).cpu().numpy()), dtype=torch.float32, device=x_true.device)


def get_metrics():
    """
    Returns a list of metric instances.
    """
    return [
        MeanDifferenceMetric(),
        EnergyDistanceMetric(),
        MMDLossMetric(),
        R2ScoreMetric()
    ]