import torch
import torch.nn as nn

from torch.nn.utils import weight_norm

from .blocks.odconv import ODConv1d as DyConv1d
from .blocks.enc import EBlock_R, EBlock_DS, EBlock_DOWN
from .blocks.attn import AttnBlock

from .mask import upsample_mask

from einops.layers.torch import Rearrange

class Encoder(nn.Module):
    def __init__(
        self,
        h_dims,
        latent_dim,
        scales,
        attns,
        blocks_per_stages,
        layers_per_blocks,
        drop_path = 0.
    ):
        super().__init__()

        mask_scales = [1]
        for i in scales[::-1]:
            if i is not None:
                mask_scales.append(mask_scales[-1] * i)
            else:
                mask_scales.append(mask_scales[-1])

        mask_scales = mask_scales[1:][::-1]

        self.scales = scales

        stages = []
        for i in range(len(h_dims)-1):
            in_channels, out_channels = h_dims[i], h_dims[i+1]

            encoder_stage = EncoderStage(in_channels, out_channels, scales[i], attns[i], blocks_per_stages[i], layers_per_blocks[i], (mask_scales[i], mask_scales[i+1]), drop_path=drop_path)
            stages.append(encoder_stage)

        self.pre_d_norm = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.GroupNorm(8, h_dims[-1])
        )

        self.down = nn.Conv1d(h_dims[-1], h_dims[-1] * 2, kernel_size=4, stride=2, padding=1)
        
        self.pre_latent_act = nn.LeakyReLU(0.2)
        # nn.GroupNorm(8, h_dims[-1] * 2),
        self.to_latent = weight_norm(nn.Conv1d(h_dims[-1] * 2, latent_dim, kernel_size=3, padding=1))

        self.stages = nn.ModuleList(stages)

    def forward(self, x, mask=None):
        for stage in self.stages:
            x = stage(x, mask)
            
        x = self.pre_d_norm(x)

        if mask is not None:
            x = x * (1. - upsample_mask(mask, self.scales[-1]).unsqueeze(1))
        x = self.down(x)
        if mask is not None:
            x = x * (1. - mask.unsqueeze(1))

        x = self.pre_latent_act(x)
        if mask is not None:
            x = x * (1. - mask.unsqueeze(1))
        x = self.to_latent(x)
        if mask is not None:
            x = x * (1. - mask.unsqueeze(1))

        return torch.tanh(x)

class EncoderStage(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale,
        attns,
        num_blocks,
        layers_per_block,
        mask_scale,
        drop_path = 0.
    ):
        super().__init__()

        blocks = []

        if scale is None:
            self.expand = nn.Conv1d(in_channels, out_channels, 7, padding=3, bias=False)

            for _ in range(num_blocks):
                blocks.append(
                    EncoderBlock(
                        out_channels,
                        3,
                        layers_per_block,
                        drop_path = drop_path
                    )
                )
        else:
            self.downscale = EBlock_DOWN(in_channels, out_channels, scale, expansion_factor=1.5, activation=nn.LeakyReLU(0.2))

            for i in range(num_blocks):
                blocks.append(
                    EncoderBlock(
                        out_channels,
                        3,
                        layers_per_block,
                        ds = True,
                        attn = attns[i],
                        drop_path = drop_path
                    )
                )
            
        self.scale = scale
        self.blocks = nn.ModuleList(blocks)
        self.mask_scale = mask_scale

    def forward(self, x, mask=None):
        if self.scale is None:
            x = self.expand(x)

            if mask is not None:
                x = x * (1. - upsample_mask(mask, self.mask_scale[1]).unsqueeze(1))
        else:
            x = self.downscale(x, (upsample_mask(mask, self.mask_scale[0]).unsqueeze(1), upsample_mask(mask, self.mask_scale[1]).unsqueeze(1)) if mask is not None else None)

        for block in self.blocks:
            x = block(x, upsample_mask(mask, self.mask_scale[1]).unsqueeze(1) if mask is not None else None)
        
        return x

class EncoderBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel,
        num_layers,
        dilation_factor = 2,
        ds = False,
        attn = False,
        drop_path = 0.
    ):
        super().__init__()

        conv = []

        for i in range(num_layers):
            dilation = dilation_factor ** i

            conv.append(
                EBlock_DS(
                    channels,
                    kernel,
                    dilation = dilation,
                    bias = False,
                    expansion_factor = 1.5,
                    activation = nn.LeakyReLU(0.2),
                    dynamic = True,
                    drop_path = drop_path
                ) if ds else

                EBlock_R(
                    channels,
                    kernel,
                    dilation = dilation,
                    bias = False,
                    activation = nn.LeakyReLU(0.2),
                    drop_path = drop_path
                )   
            )

            if attn:
                conv.append(AttnBlock(channels, bucket=2048, expansion_factor_feedforward=1., dropout=0.1))

        self.conv = nn.ModuleList(conv)

    def forward(self, x, mask=None):
        for conv in self.conv:
            x = conv(x, mask)
        return x