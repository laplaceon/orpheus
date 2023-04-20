import torch
from torch import nn
import torch.nn.functional as F

class ChannelRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        normed = F.normalize(x, dim = 1)
        # print(normed.shape, self.gamma.shape)
        scaled = normed * self.scale * self.gamma
        return scaled

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        scaled = normed * self.scale * self.gamma
        return scaled

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x, mask=None):
        residual = x
        if mask is not None:
            x = x * (1. - mask)
        x = x.transpose(1, 2)
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        normed = self.gamma * (x * Nx) + self.beta + residual.transpose(1, 2)
        return normed.transpose(1, 2)