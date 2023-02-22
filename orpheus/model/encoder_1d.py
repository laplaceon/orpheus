import torch
import torch.nn as nn

from torch.nn.utils import weight_norm

from .blocks_1d import MBConv, EnhancedResBlock

class Encoder(nn.Module):
    def __init__(
        self,
        h_dims,
        latent_dim,
        scales,
        blocks_per_stages,
        layers_per_blocks,
        se_ratio=None
    ):
        super().__init__()

        stages = []
        for i in range(len(h_dims)-1):
            in_channels, out_channels = h_dims[i], h_dims[i+1]

            encoder_stage = EncoderStage(in_channels, out_channels, scales[i], blocks_per_stages[i], layers_per_blocks[i], se_ratio, last_stage=(i+1 == len(h_dims) - 1))
            stages.append(encoder_stage)

        to_latent = nn.Sequential(
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(h_dims[-1], h_dims[-1] * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(h_dims[-1] * 2, latent_dim * 2, kernel_size=3, padding="same")),
            nn.Tanh()
        )

        stages.append(to_latent)

        self.conv = nn.Sequential(*stages)

    def forward(self, x):
        return self.conv(x)

class EncoderStage(nn.Module):
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

        if scale is None:
            kernel_size = (out_channels // in_channels) + 1
            expand = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"))
            blocks.append(expand)
        else:
            downscale = nn.Sequential(
                nn.LeakyReLU(0.2),
                weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size=scale*2, stride=scale, padding=scale//2))
            )
            blocks.append(downscale)

        for _ in range(num_blocks):
            blocks.append(
                EncoderBlock(
                    out_channels,
                    3,
                    layers_per_block,
                    se_ratio = se_ratio
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class EncoderBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel,
        num_layers,
        dilation_factor = 3,
        se_ratio = None
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