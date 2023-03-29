from torch import nn, functional as F

from .conv import CausalConv1d
from .norm import GRN

class DBlockV2_R(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        stride = 1,
        padding = 1,
        dilation = 1,
        bias = False,
        num_groups = 8,
        activation = nn.GELU(),
        causal = False
    ):
        super().__init__()

        self.net = nn.Sequential(
            activation,
            nn.GroupNorm(num_groups, channels),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias) if not causal else 
            CausalConv1d(channels, channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias),
            activation,
            GRN(channels),
            nn.Conv1d(channels, channels, kernel_size=1, padding=padding, bias=bias)
        )
    
    def forward(self, x):
        return x + self.net(x)

class DBlockV2_DS(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        stride = 1,
        padding = 1,
        dilation = 1,
        bias = False,
        num_groups = 8,
        expansion_factor=1.25,
        activation = nn.GELU(),
        causal = False
    ):
        super().__init__()

        hidden_channels = int(channels * expansion_factor)

        self.net = nn.Sequential(
            activation,
            nn.GroupNorm(num_groups),
            nn.Conv1d(channels, hidden_channels, kernel_size=1, padding=padding, bias=bias),
            activation,
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_channels, bias=bias) if not causal else
            CausalConv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, dilation=dilation, groups=hidden_channels, bias=bias),
            activation,
            GRN(hidden_channels),
            nn.Conv1d(hidden_channels, channels, kernel_size=1, padding=padding, bias=bias)
        )
    
    def forward(self, x):
        return x + self.net(x)

class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="linear"):
        super().__init__()
        
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode="linear")