from torch import nn

from .norm import GRN
from .odconv import ODConv1d as DyConv1d

class EBlock_R(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
        num_groups=8,
        activation=nn.GELU(),
        dynamic=False
    ):
        super().__init__()

        self.net = nn.Sequential(
            activation,
            nn.GroupNorm(num_groups, channels),
            DyConv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation) if dynamic else 
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias),
            activation,
            GRN(channels),
            nn.Conv1d(channels, channels, kernel_size=1, bias=bias)
        )
    
    def forward(self, x):
        return x + self.net(x)

class EBlock_DS(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
        num_groups=8,
        expansion_factor=2,
        activation=nn.GELU(),
        dynamic=False
    ):
        super().__init__()

        hidden_channels = int(channels * expansion_factor)

        self.net = nn.Sequential(
            activation,
            nn.GroupNorm(num_groups, channels),
            nn.Conv1d(channels, hidden_channels, kernel_size=1, bias=bias),
            activation,
            DyConv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_channels) if dynamic else
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_channels, bias=bias),
            activation,
            GRN(hidden_channels),
            nn.Conv1d(hidden_channels, channels, kernel_size=1, bias=bias)
        )
    
    def forward(self, x):
        return x + self.net(x)

class EBlock_DOWN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale,
        bias=False,
        num_groups=8,
        expansion_factor=2,
        activation=nn.GELU(),
        dynamic=False
    ):
        super().__init__()

        hidden_channels = int(out_channels * expansion_factor)

        self.net = nn.Sequential(
            activation,
            nn.GroupNorm(num_groups, in_channels),
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=bias),
            activation,
            DyConv1d(hidden_channels, hidden_channels, kernel_size=scale*2, padding=scale//2, groups=hidden_channels) if dynamic else 
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=scale*2, stride=scale, padding=scale//2, groups=hidden_channels, bias=bias),
            activation,
            GRN(hidden_channels),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=1, bias=bias)
        )
    
    def forward(self, x):
        return self.net(x)