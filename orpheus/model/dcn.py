from dc1d.nn import DeformConv1d
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class DeformableConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size = 3,
        stride = 1,
        padding = 1,
        dilation = 1,
        groups = 1,
        bias = False
    ):
        super().__init__()

        self.deform_conv = DeformConv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.dw_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, padding=(kernel_size-1)//2, groups=in_channels),
            Rearrange('b c l -> b l c'),
            nn.LayerNorm(in_channels),
            Rearrange('b l c -> b c l'),
            nn.ReLU(inplace=True)
        )

        self.offset = nn.Linear(in_channels, groups * kernel_size)

    def forward(self, x):
        out = self.dw_conv(x)
        offset = self.offset(out.transpose(1, 2))

        return self.deform_conv(x, offset.unsqueeze(1))

# deform_conv = DeformableConv1d(16, 32, 3)
# x = torch.randn(8, 16, 64)

# print(x.shape, deform_conv(x).shape)