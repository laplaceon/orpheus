import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from torch.nn.utils import weight_norm

from .blocks.dec import  DBlock_DS, DBlock_R

class MultiBranchProbabilisticDecoder(nn.Module):
    def __init__(
        self,
        penulti_dim,
        probabilistic_outputs
    ):
        super().__init__()

        self.prob_conv = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.GroupNorm(8, penulti_dim),
            nn.Conv1d(penulti_dim, probabilistic_outputs, 7, padding=3),
            # nn.GELU(),
            # nn.Conv1d(probabilistic_outputs, probabilistic_outputs, 1, bias=False)
        )
    
    def forward(self, x):
        return self.prob_conv(x)

class Decoder(nn.Module):
    def __init__(
        self,
        h_dims,
        latent_dim,
        scales,
        ds_expansion_factor,
        blocks_per_stage,
        layers_per_blocks,
        drop_path=0.
    ):
        super().__init__()

        self.from_latent = weight_norm(nn.Conv1d(latent_dim, h_dims[0] * 2, kernel_size=3, padding=1))

        stages = []
        h_dims_new = [h_dims[0] * 2] + h_dims[:-1]
        for i in range(len(h_dims)-1):
            in_channels, out_channels = h_dims_new[i], h_dims_new[i+1]

            decoder_stage = DecoderStage(in_channels, out_channels, scales[i], ds_expansion_factor, blocks_per_stage[i], layers_per_blocks[i], i+1 == len(h_dims) - 1, drop_path=drop_path)
            stages.append(decoder_stage)

        self.final_conv = nn.Sequential(
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(h_dims[-2], h_dims[-1], 7, padding=3)),
            nn.Tanh()
        )

        self.conv = nn.Sequential(*stages)

    def forward(self, z):
        out = self.from_latent(z)
        out = self.conv(out)
        return self.final_conv(out), out

class DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale,
        ds_expansion_factor,
        num_blocks,
        layers_per_block,
        last_stage=False,
        drop_path=0.
    ):
        super().__init__()

        blocks = []

        blocks.append(
            nn.Sequential(
                nn.LeakyReLU(0.2),
                weight_norm(nn.ConvTranspose1d(in_channels, out_channels, scale * 2, stride=scale, padding=scale//2, bias=False))
            )
        )

        for _ in range(num_blocks):
            blocks.append(
                DecoderBlock(
                    out_channels,
                    3,
                    layers_per_block,
                    ds_expansion_factor = ds_expansion_factor,
                    ds = True,
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
        num_layers = 2,
        dilation_factor = 2,
        ds_expansion_factor = 2.,
        ds = False,
        drop_path = 0.
    ):
        super().__init__()

        conv = []

        for i in range(num_layers):
            dilation = dilation_factor ** i
            conv.append(
                DBlock_DS(
                    channels,
                    kernel,
                    dilation = dilation,
                    bias = False,
                    expansion_factor = ds_expansion_factor,
                    activation = nn.LeakyReLU(0.2),
                    dynamic = False,
                    drop_path = drop_path
                ) if ds else
                
                DBlock_R(
                    channels,
                    kernel,
                    dilation = dilation,
                    bias = False,
                    activation = nn.LeakyReLU(0.2),
                    drop_path = drop_path
                )
            )

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)
