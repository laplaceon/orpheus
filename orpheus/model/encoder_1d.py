import torch
import torch.nn as nn

from .blocks_1d import MBConv, ResBlock

class Encoder(nn.Module):
    def __init__(
        self,
        sequence_length,
        latent_dim,
        h_dims,
        scales,
        blocks_per_stages,
        layers_per_blocks,
        se_ratio=None
    ):
        super().__init__()

        stages = []
        for i in range(len(h_dims)-1):
            in_channels, out_channels = h_dims[i], h_dims[i+1]

            encoder_stage = EncoderStage(in_channels, out_channels, scales[i], blocks_per_stages[i], layers_per_blocks[i], se_ratio, first_stage=(i == 0), last_stage=(i+1 == len(h_dims) - 1))
            stages.append(encoder_stage)

        to_latent = nn.Sequential(
            nn.Conv1d(h_dims[-1], h_dims[-1] * 2, 4, 2, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(h_dims[-1] * 2, latent_dim * 2, 3, padding="same"),
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
        first_stage=False,
        last_stage=False
    ):
        super().__init__()

        blocks = []

        if first_stage:
            kernel = out_channels // in_channels
            expand = nn.Conv1d(in_channels, out_channels, kernel_size=kernel+1, padding=3)
            blocks.append(expand)
        else:
            downscale = nn.Conv1d(in_channels, out_channels, kernel_size=scale*2, stride=scale, padding=scale//2)
            blocks.append(downscale)

        for _ in range(num_blocks):
            blocks.append(
                EncoderBlock(
                    out_channels,
                    3,
                    layers_per_block,
                    se_ratio=se_ratio
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.blocks(x)
        return out

class EncoderBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel,
        num_layers,
        dilation_factor=3,
        se_ratio=None
    ):
        super().__init__()

        conv = []

        for i in range(num_layers):
            dilation = dilation_factor ** i
            conv.append(ResBlock(channels, channels, kernel, dilation=dilation, activation=nn.LeakyReLU(negative_slope=0.2)))

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)