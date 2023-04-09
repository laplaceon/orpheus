from torch import nn

from .norm import GRN
from .odconv import ODConv1d as DyConv1d
from timm.models.layers import DropPath

class EBlock_R(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        stride = 1,
        padding = 1,
        dilation = 1,
        bias = False,
        num_groups = 4,
        activation = nn.GELU(),
        dynamic = False,
        drop_path = 0.
    ):
        super().__init__()

        self.act = activation
        self.norm = nn.GroupNorm(num_groups, channels)
        self.dw = DyConv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation) if dynamic else \
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias)
        self.grn = GRN(channels)
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.dynamic = dynamic
    
    def forward(self, x, mask=None):
        # print(mask.shape)
        residual = x
        x = self.act(x)
        x = self.norm(x)
        if mask is not None:
            x = x * (1. - mask)
        
        if self.dynamic:
            x = self.dw(x, mask)
        else:
            x = self.dw(x)

        if mask is not None:
            x = x * (1. - mask)
        x = self.act(x)
        x = self.grn(x, mask=mask)
        x = self.pw(x)
        
        return residual + self.drop_path(x)

class EBlock_DS(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        stride = 1,
        padding = 1,
        dilation = 1,
        bias = False,
        num_groups = 4,
        expansion_factor = 2.,
        activation = nn.GELU(),
        dynamic = False,
        drop_path=0.
    ):
        super().__init__()

        hidden_channels = int(channels * expansion_factor)

        self.act = activation
        self.norm = nn.GroupNorm(num_groups, channels)
        self.pw1 = nn.Conv1d(channels, hidden_channels, kernel_size=1, bias=bias)
        self.dw = DyConv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_channels) if dynamic else \
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_channels, bias=bias)
        self.grn = GRN(hidden_channels)
        self.pw2 = nn.Conv1d(hidden_channels, channels, kernel_size=1, bias=bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.dynamic = dynamic
    
    def forward(self, x, mask=None):
        residual = x
        x = self.act(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)

        if mask is not None:
            x = x * (1. - mask)

        if self.dynamic:
            x = self.dw(x, mask)
        else:
            x = self.dw(x)

        if mask is not None:
            x = x * (1. - mask)
            
        x = self.act(x)
        x = self.grn(x, mask=mask)
        x = self.pw2(x)
        
        return residual + self.drop_path(x)

# class EBlock_DOWN(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         scale,
#         bias = False,
#         num_groups = 4,
#         expansion_factor = 2,
#         activation = nn.GELU()
#     ):
#         super().__init__()

#         hidden_channels = int(out_channels * expansion_factor)

#         self.act = activation
#         self.norm = nn.GroupNorm(num_groups, in_channels)
#         self.dw = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=scale*2, stride=scale, padding=scale//2, groups=hidden_channels, bias=bias)
#         self.pw1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=bias)
#         self.pw2 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1, bias=bias)
#         self.grn = GRN(hidden_channels)
    
#     def forward(self, x, masks=None):
#         x = self.act(x)
#         x = self.norm(x)
#         x = self.pw1(x)
#         x = self.act(x)
#         if masks is not None:
#             x = x * (1. - masks[0])
#         x = self.dw(x)
#         if masks is not None:
#             x = x * (1. - masks[1])
#         x = self.act(x)
#         x = self.grn(x, mask=masks[1] if masks is not None else None)
#         x = self.pw2(x)
#         return x

class EBlock_DOWN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale,
        bias = False,
        num_groups = 4,
        expansion_factor = 2.,
        activation = nn.GELU()
    ):
        super().__init__()

        self.act = activation
        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.dw = nn.Conv1d(in_channels, out_channels, kernel_size=scale*2, stride=scale, padding=scale//2, bias=bias)
    
    def forward(self, x, masks=None):
        x = self.act(x)
        x = self.norm(x)
        if masks is not None:
            x = x * (1. - masks[0])
        x = self.dw(x)
        if masks is not None:
            x = x * (1. - masks[1])
        return x