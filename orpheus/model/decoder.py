import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm

from .blocks_1d import MBConv, DepthwiseSeparableConvWN, EnhancedResBlock, Upsample

class Decoder(nn.Module):
    def __init__(
        self,
        h_dims,
        latent_dim,
        scales,
        blocks_per_stage,
        layers_per_blocks,
        se_ratio=None
    ):
        super().__init__()

        self.from_latent = nn.Sequential(
            weight_norm(nn.Conv1d(latent_dim, h_dims[0] * 2, kernel_size=3, padding="same")),
            nn.LeakyReLU(0.2),
            Upsample(scale_factor=2),
            nn.Conv1d(h_dims[0] * 2, h_dims[0], kernel_size=3, padding="same")
        )

        stages = []
        for i in range(len(h_dims)-1):
            in_channels, out_channels = h_dims[i], h_dims[i+1]

            decoder_stage = DecoderStage(in_channels, out_channels, scales[i], blocks_per_stage[i], layers_per_blocks[i], se_ratio, i+1 == len(h_dims) - 1)
            stages.append(decoder_stage)

        self.conv = nn.Sequential(*stages)

    def forward(self, z):
        out = self.from_latent(z)

        return torch.tanh(self.conv(out))


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale,
        num_blocks,
        layers_per_block,
        se_ratio,
        last_stage=False
    ):
        super().__init__()

        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                DecoderBlock(
                    in_channels,
                    3,
                    layers_per_block,
                    se_ratio=se_ratio
                )
            )
        
        blocks.append(
            nn.Sequential(
                Upsample(scale_factor=scale) if scale is not None else nn.Identity(),
                DepthwiseSeparableConvWN(in_channels, out_channels, (in_channels // out_channels) + 1 if last_stage else 3, padding="same")
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
        num_layers=4,
        dilation_factor=3,
        se_ratio=None
    ):
        super().__init__()

        conv = []

        for i in range(num_layers):
            dilation = dilation_factor ** i
            conv.append(
                EnhancedResBlock(
                    channels,
                    kernel,
                    padding = "same",
                    dilation = dilation,
                    bias = False,
                    se_ratio = se_ratio,
                    activation = nn.LeakyReLU(0.2)
                )
            )

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)
