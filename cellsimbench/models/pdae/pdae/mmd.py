# from https://github.com/yiftachbeer/mmd_loss_pytorch/tree/master

import torch
from torch import nn


class RBF(nn.Module):

    def __init__(self, device, n_kernels=2, mul_factor=2.0, rbf_bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = (mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)).to(device)
        self.bandwidth = rbf_bandwidth
        self.device = device

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            with torch.no_grad():
                # mean heuristic: compute average L2 distance across 
                n_samples = L2_distances.shape[0]
                return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        L2_distances = (torch.cdist(X, X) ** 2).to(self.device)
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, device, rbf_bandwidth=None):
        super().__init__()
        self.kernel = RBF(device=device, rbf_bandwidth=rbf_bandwidth)

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y])) # the kernel K is a quadratic matrix of shape (X.shape[0]+Y.shape[0], X.shape[0]+Y.shape[0])
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY