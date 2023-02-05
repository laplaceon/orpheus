import torch
import torch.nn as nn
import torch.nn.functional as F

# from attention import window_partition, window_reverse

class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        padding=1,
        bias=False
    ):
        super().__init__()

        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, groups=in_channels, bias=bias)
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
        kernel_size,
        bias=False
    ):
        super().__init__()

        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, dilation=1, bias=bias)

    def forward(self, x):
        return self.pointwise(x)

class DepthwiseConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        padding=1,
        bias=False
    ):
        super().__init__()

        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, groups=in_channels, bias=bias)

    def forward(self, x):
        return self.depthwise(x)

class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_channels,
        rd_ratio=0.25
    ):
        super().__init__()

        rd_channels = int(in_channels * rd_ratio)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, rd_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(rd_channels, in_channels, 1),
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

        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=kernel, dilation=dilation, padding="same")
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=kernel, dilation=dilation, padding="same")

        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)

        self.conv_res = nn.Conv1d(in_channels, out_channels, kernel_size=1, dilation=dilation, padding="same")

        self.activation = activation

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))

        return self.activation(h + self.conv_res(x))

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=3,
        dilation=1,
        groups=8,
        activation=nn.ReLU()
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

class MBConv(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel=3,
        dilation=1,
        groups=8,
        se_ratio=None,
        activation=nn.SiLU()
    ):
        super().__init__()

        self.conv = nn.Sequential(
            PointwiseConv(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            activation,
            DepthwiseConv(hidden_channels, hidden_channels, kernel_size=kernel, dilation=dilation, padding="same"),
            nn.BatchNorm1d(hidden_channels),
            activation,
            SqueezeExcite(hidden_channels, rd_ratio=se_ratio) if se_ratio is not None else nn.Identity(),
            PointwiseConv(hidden_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        residual = x

        return self.conv(x) + residual

# class AttentionLayer(nn.Module):
#     def __init__(
#         self,
#         dim,
#         max_seq_len,
#         dim_head = 64,
#         heads = 8,
#         causal = False,
#         dropout = 0.0,
#         window_len = 32
#     ):
#         super().__init__()

#         self.attn = AttentionLayers(dim, 1, rel_pos_bias=True)

#         self.attn_window_len = window_len

#     def forward(self, x):
#         seq_len = x.shape[-1]

#         x = window_partition(x, window_size=self.attn_window_len)
#         x = self.attn(x)
#         x = window_reverse(x, seq_len, window_size=self.attn_window_len)

#         return x
