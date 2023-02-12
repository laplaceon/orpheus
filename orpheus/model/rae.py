import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from .encoder_2d import Encoder
from .decoder import Decoder, SynthDecoder

class Orpheus(nn.Module):
    def __init__(
        self,
        sequence_length,
        h_dims=(1, 8, 16, 32, 64, 128),
        scales=(2, 2, 2, 2, 2),
        blocks_per_stages=(2, 2, 2, 2, 2),
        layers_per_blocks=(2, 2, 2, 2, 2),
        se_ratio=0.25,
        codebook_width=256
    ):
        super().__init__()

        z_length = sequence_length
        for x in scales:
            z_length /= x

        self.z_shape = (int(z_length), h_dims[-1])

        self.encoder = Encoder(sequence_length, codebook_width, h_dims, scales, blocks_per_stages, layers_per_blocks, se_ratio)

        # h_dims_dec = [128, 64, 48, 32, 32, 16, 16, 8, 4, 2, 1]
        # scales_dec = [2, 2, 4, 4, 4, 2, 2, 2, 2, 2]
        # blocks_per_stages_dec = [1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1]
        # layers_per_blocks_dec = [3, 4, 4, 4, 3, 2, 2, 2, 2, 2, 2]

        # self.decoder = Decoder(sequence_length, h_dims_dec, scales_dec, blocks_per_stages_dec, layers_per_blocks_dec, se_ratio)
        # self.decoder = Decoder(sequence_length, h_dims[::-1], scales[::-1], blocks_per_stages[::-1], layers_per_blocks[::-1], se_ratio)
        self.decoder = SynthDecoder(sequence_length, h_dims[-1])

    def encode(self, x):
        return self.encoder(x)

    def reparametrize(self, z):
        mean, scale = z.chunk(2, dim=1)

        std = F.softplus(scale) + 1e-4
        var = std ** 2
        logvar = torch.log(var)

        z = torch.randn_like(mean) * std + mean
        kl = (mean ** 2 + var - logvar - 1).sum(1).mean()

        return z, kl

    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, x):
        encoded = self.encode(x)
        z, kl = self.reparametrize(encoded)
        out = self.decode(z)

        return out, kl