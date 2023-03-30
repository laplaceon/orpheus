import torch
import torch.nn as nn

from torch.nn.utils import weight_norm

from .blocks.odconv import ODConv1d as DyConv1d
from .blocks.enc import EBlock_R, EBlock_DS, EBlock_DOWN
from .blocks.attn import AttnBlock
from einops.layers.torch import Rearrange

class Encoder(nn.Module):
    def __init__(
        self,
        h_dims,
        latent_dim,
        scales,
        attns,
        blocks_per_stages,
        layers_per_blocks
    ):
        super().__init__()

        stages = []
        for i in range(len(h_dims)-1):
            in_channels, out_channels = h_dims[i], h_dims[i+1]

            encoder_stage = EncoderStage(in_channels, out_channels, scales[i], attns[i], blocks_per_stages[i], layers_per_blocks[i], last_stage=(i+1 == len(h_dims) - 1))
            stages.append(encoder_stage)

        to_latent = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.GroupNorm(16, h_dims[-1]),
            nn.Conv1d(h_dims[-1], h_dims[-1] * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # nn.GroupNorm(16, h_dims[-1] * 2),
            weight_norm(nn.Conv1d(h_dims[-1] * 2, latent_dim, kernel_size=3, padding="same")),
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
        attns,
        num_blocks,
        layers_per_block,
        last_stage=False
    ):
        super().__init__()

        blocks = []

        if scale is None:
            expand = nn.Conv1d(in_channels, out_channels, 7, padding="same", bias=False)
            blocks.append(expand)

            for _ in range(num_blocks):
                blocks.append(
                    EncoderBlock(
                        out_channels,
                        3,
                        layers_per_block
                    )
                )
        else:
            downscale = EBlock_DOWN(in_channels, out_channels, scale, expansion_factor=1.4, activation=nn.LeakyReLU(0.2))

            blocks.append(downscale)

            for i in range(num_blocks):
                blocks.append(
                    EncoderBlock(
                        out_channels,
                        3,
                        layers_per_block,
                        ds = True,
                        attn = attns[i]
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
        dilation_factor = 2,
        ds = False,
        attn = False
    ):
        super().__init__()

        conv = []

        for i in range(num_layers):
            dilation = dilation_factor ** i

            conv.append(
                EBlock_DS(
                    channels,
                    kernel,
                    padding = "same",
                    dilation = dilation,
                    bias = False,
                    expansion_factor = 1.4,
                    activation = nn.LeakyReLU(0.2),
                    dynamic = True
                ) if ds else

                EBlock_R(
                    channels,
                    kernel,
                    padding = "same",
                    dilation = dilation,
                    bias = False,
                    activation = nn.LeakyReLU(0.2),
                    dynamic = False
                )
            )

            if attn:
                conv.append(AttnBlock(channels, bucket=2048, expansion_factor_feedforward=1., dropout=0.1))

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)