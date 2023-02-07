import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import SqueezeExcitation
# from einops.layers.torch import Rearrange, Reduce

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        padding=1,
        bias=False
    ):
        super().__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, dilation=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class PointwiseConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=False
    ):
        super().__init__()

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, dilation=1, bias=bias)

    def forward(self, x):
        return self.pointwise(x)

class DepthwiseConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size,
        stride=1,
        dilation=1,
        padding=1,
        bias=False
    ):
        super().__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, groups=in_channels, bias=bias)

    def forward(self, x):
        return self.depthwise(x)

# class SqueezeExcitation(nn.Module):
#     def __init__(self, dim, shrinkage_rate = 0.25):
#         super().__init__()
#         hidden_dim = int(dim * shrinkage_rate)

#         self.gate = nn.Sequential(
#             Reduce('b c h w -> b c', 'mean'),
#             nn.Linear(dim, hidden_dim, bias = False),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, dim, bias = False),
#             nn.Sigmoid(),
#             Rearrange('b c -> b c 1 1')
#         )

#     def forward(self, x):
#         return x * self.gate(x)

class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        # self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        # out = self.dropsample(out)
        return out + x

class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    kernel_size = 3,
    padding = 1,
    dilation = 1,
    downsample_factor = None,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    squeeze_channels = int(hidden_dim * shrinkage_rate)
    stride = downsample_factor if downsample_factor is not None else 1

    net = nn.Sequential(
        PointwiseConv2d(dim_in, hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        DepthwiseConv2d(hidden_dim, kernel_size, stride=stride, padding=padding, dilation=dilation),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, squeeze_channels=squeeze_channels),
        PointwiseConv2d(hidden_dim, dim_out),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and downsample_factor is None:
        net = MBConvResidual(net, dropout = dropout)

    return net