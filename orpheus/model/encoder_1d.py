import torch
import torch.nn as nn

from torch.nn.utils import weight_norm

from .blocks_1d import MBConv, EnhancedResBlock, EBlockV2_R, EBlockV2_DS, EBlockV2_DOWN, AttnBlock
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
        attn = False
    ):
        super().__init__()

        stages = []
        for i in range(len(h_dims)-1):
            in_channels, out_channels = h_dims[i], h_dims[i+1]

            encoder_stage = EncoderStage(in_channels, out_channels, scales[i], attns[i], blocks_per_stages[i], layers_per_blocks[i], last_stage=(i+1 == len(h_dims) - 1))
            stages.append(encoder_stage)

        to_latent = nn.Sequential(
            nn.LeakyReLU(0.2),
            Rearrange('b c l -> b l c'),
            nn.LayerNorm(h_dims[-1]),
            Rearrange('b l c -> b c l'),
            nn.Conv1d(h_dims[-1], h_dims[-1] * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
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
        last_stage=False,
        attn=False
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
            downscale = EBlockV2_DOWN(in_channels, out_channels, scale, expansion_factor=1.5, activation=nn.LeakyReLU(0.2))

            blocks.append(downscale)

            for _ in range(num_blocks):
                blocks.append(
                    EncoderBlock(
                        out_channels,
                        3,
                        layers_per_block,
                        ds = True,
                        attn = attn
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
        ds = False,
        attn = False
    ):
        super().__init__()

        conv = []

        for i in range(num_layers):
            dilation = dilation_factor ** i

            if attn:
                conv.append(AttnBlock(channels, 2048))

            conv.append(
                EBlockV2_DS(
                    channels,
                    kernel,
                    padding = "same",
                    dilation = dilation,
                    bias = False,
                    expansion_factor = 2,
                    activation = nn.LeakyReLU(0.2)
                ) if ds else

                EBlockV2_R(
                    channels,
                    kernel,
                    padding = "same",
                    dilation = dilation,
                    bias = False,
                    activation = nn.LeakyReLU(0.2)
                )
            )

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)