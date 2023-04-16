import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from torch.nn.utils import weight_norm

from .blocks.dec import CausalDBlock, CausalConv1d
from .blocks.conv import CausalConvTranspose1d

from .decoder import DecoderStage

class ProbabilisticDecoder(nn.Module):
    def __init__(
        self,
        h_dims,
        scales,
        blocks_per_stage,
        layers_per_blocks,
        drop_path = 0.
    ):
        super().__init__()

        stages = []
        for i in range(len(h_dims)-1):
            in_channels, out_channels = h_dims[i], h_dims[i+1]

            decoder_stage = DecoderStage(in_channels, out_channels, scales[i], blocks_per_stage[i], layers_per_blocks[i], i+1 == len(h_dims) - 1, drop_path=drop_path)
            stages.append(decoder_stage)

        final_conv = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.GroupNorm(8, h_dims[-2]),
            CausalConv1d(h_dims[-2], h_dims[-1], 7)
        )

        stages.append(final_conv)

        self.conv = nn.Sequential(*stages)

    def forward(self, x):
        return self.conv(x)