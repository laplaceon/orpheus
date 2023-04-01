import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from torch.nn.utils import weight_norm

from .blocks.dec import CausalDBlock, CausalConv1d
from .blocks.conv import CausalConvTranspose1d

class PredictiveDecoder(nn.Module):
    def __init__(
        self,
        h_dims,
        latent_dim,
        scales,
        blocks_per_stage,
        layers_per_blocks,
        drop_path = 0.
    ):
        super().__init__()

        self.from_latent = weight_norm(CausalConv1d(latent_dim, h_dims[0] * 2, 3))

        stages = []
        h_dims_new = [h_dims[0] * 2] + h_dims[:-1]
        for i in range(len(h_dims)-1):
            in_channels, out_channels = h_dims_new[i], h_dims_new[i+1]

            decoder_stage = DecoderStage(in_channels, out_channels, scales[i], blocks_per_stage[i], layers_per_blocks[i], i+1 == len(h_dims) - 1, drop_path=drop_path)
            stages.append(decoder_stage)

        final_conv = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.GroupNorm(16, h_dims[-2]),
            CausalConv1d(h_dims[-2], h_dims[-1], 7)
        )

        stages.append(final_conv)

        self.conv = nn.Sequential(*stages)

    def forward(self, z):
        out = self.from_latent(z)
        return self.conv(out)

class DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale,
        num_blocks,
        layers_per_block,
        last_stage = False,
        drop_path = 0.
    ):
        super().__init__()

        blocks = []

        blocks.append(
            nn.Sequential(
                nn.LeakyReLU(0.2),
                CausalConvTranspose1d(in_channels, out_channels, scale * 2, scale, padding=scale//2, bias=False)
            )
        )

        for _ in range(num_blocks):
            blocks.append(
                DecoderBlock(
                    out_channels,
                    3,
                    layers_per_block,
                    drop_path = drop_path
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class DecoderBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel,
        num_layers=2,
        dilation_factor=2,
        drop_path = 0.
    ):
        super().__init__()

        conv = []

        for i in range(num_layers):
            dilation = dilation_factor ** i
            conv.append(
                CausalDBlock(
                    channels,
                    kernel,
                    padding = "same",
                    dilation = dilation,
                    bias = False,
                    activation = nn.LeakyReLU(0.2),
                    drop_path = drop_path
                )
            )

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)
