import torch
import torch.nn as nn

from .blocks_1d import MBConv, DepthwiseSeparableConv

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
            MBConv(h_dims[-1], h_dims[-1] * 2, kernel_size=4, downsample_factor=2, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)),
            MBConv(h_dims[-1] * 2, latent_dim * 2, kernel_size=3, padding="same", activation=nn.LeakyReLU(negative_slope=0.2)),
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
            expand = MBConv(in_channels, out_channels, kernel_size, padding="same", activation=nn.LeakyReLU(negative_slope=0.2))
            
            blocks.append(expand)
        else:
            downscale = MBConv(in_channels, out_channels, kernel_size=scale*2, downsample_factor=scale, padding=scale//2, activation=nn.LeakyReLU(negative_slope=0.2))
            blocks.append(downscale)

        blocks.append(
            EncoderBlock(
                out_channels,
                3,
                layers_per_block - 1,
                se_ratio = se_ratio,
                first_block = True
            )
        )

        for _ in range(num_blocks - 1):
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
        se_ratio = None,
        first_block = False
    ):
        super().__init__()

        conv = []

        offset = 1 if first_block else 0

        for i in range(num_layers):
            dilation = dilation_factor ** (i + offset)
            conv.append(
                MBConv(
                    channels,
                    channels,
                    kernel,
                    padding = "same",
                    dilation = dilation,
                    expansion_rate = 2,
                    se_ratio = se_ratio,
                    activation = nn.LeakyReLU(negative_slope=0.2)
                )
            )

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)