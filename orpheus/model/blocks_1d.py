import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
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

        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, dilation=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class PointwiseConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=False
    ):
        super().__init__()

        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, dilation=1, bias=bias)

    def forward(self, x):
        return self.pointwise(x)

class DepthwiseConv(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        stride=1,
        dilation=1,
        padding=1,
        bias=False
    ):
        super().__init__()

        self.depthwise = nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, groups=channels, bias=bias)

    def forward(self, x):
        return self.depthwise(x)

class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="linear"):
        super().__init__()
        
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode="linear")

class SqueezeExcite(nn.Module):
    def __init__(
        self,
        channels,
        rd_ratio=0.25
    ):
        super().__init__()

        rd_channels = int(channels * rd_ratio)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, rd_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(rd_channels, channels, 1),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.gate(x)

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=3,
        dilation=1,
        groups=8,
        activation=nn.SiLU()
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=kernel, dilation=dilation, padding="same"),
            nn.BatchNorm1d(out_channels),
            activation
        )
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=kernel, dilation=dilation, padding="same"),
            nn.BatchNorm1d(out_channels)
        )

        self.conv_res = nn.Conv1d(in_channels, out_channels, kernel_size=1, dilation=dilation, padding="same")

        self.activation = activation

    def forward(self, x):
        residual = x
        h = self.conv1(x)
        h = self.conv2(h)

        return self.activation(h + residual)

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=3,
        padding="same",
        dilation=1,
        groups=8,
        bias=False,
        activation=nn.ReLU()
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel, dilation=dilation, padding=padding, bias=bias),
            nn.BatchNorm1d(out_channels),
            activation
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel, dilation=dilation, padding=padding, bias=bias),
            nn.BatchNorm1d(out_channels)
        )

        self.conv_res = nn.Conv1d(in_channels, out_channels, kernel_size=1, dilation=dilation, padding=padding, bias=bias)

        self.activation = activation

    def forward(self, x):
        residual = x
        h = self.conv1(x)
        h = self.conv2(h)

        return self.activation(h + residual)

class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
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
    expansion_rate = 1,
    se_ratio = 0.25,
    dropout = 0.,
    bias=False,
    activation=nn.ReLU()
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = downsample_factor if downsample_factor is not None else 1

    net = nn.Sequential(
        PointwiseConv(dim_in, hidden_dim, bias=bias),
        nn.BatchNorm1d(hidden_dim),
        activation,
        DepthwiseConv(hidden_dim, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
        nn.BatchNorm1d(hidden_dim),
        activation,
        SqueezeExcite(hidden_dim, rd_ratio=se_ratio) if se_ratio is not None else nn.Identity(),
        PointwiseConv(hidden_dim, dim_out, bias=bias),
        nn.BatchNorm1d(dim_out)
    )

    if dim_in == dim_out and downsample_factor is None:
        net = MBConvResidual(net, dropout = dropout)

    return net