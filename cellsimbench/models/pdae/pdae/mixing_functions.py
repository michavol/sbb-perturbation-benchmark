import torch
import torch.nn as nn

# deterministic mixing functions from R^2 to R^2
class SphereInversion(nn.Module):
    def __init__(self, loc):
        super(SphereInversion, self).__init__()
        self.loc = loc

    def forward(self, x):
        x1 = x[..., 0]
        x2 = x[..., 1]
        b1 = self.loc[0]
        b2 = self.loc[1]
        inv = (x1 - b1) ** 2 + (x2 - b2) ** 2
        y1 = (x1 - b1) / inv + b1
        y2 = (x2 - b2) / inv + b2
        return torch.stack((y1, y2), dim=-1)
    

class PolarTransformation(nn.Module):
    def __init__(self):
        super(PolarTransformation, self).__init__()

    def forward(self, x):
        x1 = x[..., 0]
        x2 = x[..., 1]
        y1 = torch.sqrt(x1 ** 2 + x2  ** 2)
        y2 = torch.atan2(x2, x1)
        return torch.stack((y1, y2), dim=-1)
    

class ComplexSquare(nn.Module):
    def __init__(self, loc):
        super(ComplexSquare, self).__init__()
        self.loc = loc

    def forward(self, x):
        x1 = x[..., 0]
        x2 = x[..., 1]
        b1 = self.loc[0]
        b2 = self.loc[1]
        y1 = 0.5 * ((x1 - b1) ** 2 - (x2 - b2)  ** 2)
        y2 = (x1 - b1) * (x2 - b2)
        return torch.stack((y1, y2), dim=-1)
    

class ComplexSine(nn.Module):
    def __init__(self):
        super(ComplexSine, self).__init__()

    def forward(self, x):
        x1 = x[..., 0]
        x2 = x[..., 1]
        y1 = torch.sin(x1) * torch.cosh(x2)
        y2 = torch.cos(x1) * torch.sinh(x2)
        return torch.stack((y1, y2), dim=-1)
    

class ComplexCosine(nn.Module):
    def __init__(self):
        super(ComplexCosine, self).__init__()

    def forward(self, x):
        x1 = x[..., 0]
        x2 = x[..., 1]
        y1 = torch.cos(x1) * torch.cosh(x2)
        y2 = torch.sin(x1) * torch.sinh(x2)
        return torch.stack((y1, y2), dim=-1)
    

class ComplexExponential(nn.Module):
    def __init__(self, scale=1.0):
        super(ComplexExponential, self).__init__()
        self.scale = scale

    def forward(self, x):
        x1 = x[..., 0]
        x2 = x[..., 1]
        y1 = torch.exp(self.scale * x1) * torch.cos(x2)
        y2 = torch.exp(self.scale * x1) * torch.sin(x2)
        return torch.stack((y1, y2), dim=-1)
    

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    

# stochastic mixing functions (constructed by using a deterministic mixing and concatenating isotropic Gaussian noise as additional dimensions)
class NoisyIdentity(nn.Module):
    def __init__(self, sigma_noise, dim_noise):
        super(NoisyIdentity, self).__init__()
        self.sigma_noise = sigma_noise
        self.dim_noise = dim_noise

    def forward(self, x):
        noise = torch.randn(x.shape[:-1] + (self.dim_noise,)) * self.sigma_noise
        return torch.concatenate((x, noise), dim=-1)


class NoisyComplexExponential(nn.Module):
    def __init__(self, sigma_noise, dim_noise, scale=1.0):
        super(NoisyComplexExponential, self).__init__()
        self.scale = scale
        self.sigma_noise = sigma_noise
        self.dim_noise = dim_noise

    def forward(self, x):
        x1 = x[..., 0]
        x2 = x[..., 1]
        y1 = torch.exp(self.scale * x1) * torch.cos(x2)
        y2 = torch.exp(self.scale * x1) * torch.sin(x2)
        y = torch.stack((y1, y2), dim=-1)
        # if self.dim_noise == 0, noise will be empty
        noise = torch.randn((x.shape[:-1] + (self.dim_noise,)), device=x.device) * self.sigma_noise
        return torch.concatenate((y, noise), dim=-1) # concat at last dim, i.e. (:,:,2) -> (:,:,2+dim_noise)