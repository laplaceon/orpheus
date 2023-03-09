import torch
import torch.nn as nn

from torch.nn.utils import weight_norm

from .blocks_1d import MBConv, EnhancedResBlock, EBlockV2, EBlockV2_dw
from cape import CAPE1d
from memory_efficient_attention_pytorch import Attention
from einops.layers.torch import Rearrange

class Encoder(nn.Module):
    def __init__(
        self,
        h_dims,
        latent_dim,
        scales,
        blocks_per_stages,
        layers_per_blocks
    ):
        super().__init__()

        stages = []
        for i in range(len(h_dims)-1):
            in_channels, out_channels = h_dims[i], h_dims[i+1]

            encoder_stage = EncoderStage(in_channels, out_channels, scales[i], blocks_per_stages[i], layers_per_blocks[i], last_stage=(i+1 == len(h_dims) - 1))
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
        last_stage=False
    ):
        super().__init__()

        blocks = []

        if scale is None:
            expand = nn.Conv1d(in_channels, out_channels, 7, padding="same", bias=False)
            blocks.append(expand)

            self.scale = None
        else:
            downscale = nn.Sequential(
                nn.LeakyReLU(0.2),
                Rearrange('b c l -> b l c'),
                nn.LayerNorm(in_channels),
                Rearrange('b l c -> b c l'),
                nn.Conv1d(in_channels, out_channels, kernel_size=scale*2, stride=scale, padding=scale//2, bias=False)
            )
            
            self.scale = downscale

            attn = Attention(
                dim = out_channels,
                dim_head = out_channels // 8,
                heads = 8,
                causal = False,
                memory_efficient = True,
                q_bucket_size = 1024,
                k_bucket_size = 2048,
                dropout = 0.1
            )

            self.attn = nn.Sequential(
                Rearrange('b c l -> l b c'),
                CAPE1d(d_model=out_channels),
                Rearrange('l b c -> b l c'),
                nn.LayerNorm(out_channels),
                attn
            )

            # self.ff = nn.Sequential(
            #     nn.LayerNorm(out_channels),
            #     nn.Linear(out_channels, out_channels)
            # )

        for _ in range(num_blocks):
            blocks.append(
                EncoderBlock(
                    out_channels,
                    3,
                    layers_per_block
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.scale is not None:
            x = self.scale(x)
            x = x.transpose(1, 2) + self.attn(x)
            x = x.transpose(1, 2)

        return self.blocks(x)

class EncoderBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel,
        num_layers,
        dilation_factor = 3
    ):
        super().__init__()

        conv = []

        for i in range(num_layers):
            dilation = dilation_factor ** i
            dilation = 1
            conv.append(
                EBlockV2_dw(
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