import torch
import torch.nn as nn
import torch.nn.functional as F
from .pqmf_old import PQMF

import math

from .encoder_1d import Encoder
from .decoder import Decoder, SynthDecoder

class Orpheus(nn.Module):
    def __init__(
        self,
        sequence_length,
        latent_dim=128,
        h_dims=(16, 96, 192, 384, 768),
        scales=(4, 4, 4, 4),
        blocks_per_stages=(1, 1, 1, 1),
        layers_per_blocks=(3, 3, 3, 3),
        se_ratio=0.25
    ):
        super().__init__()

        self.pqmf = PQMF(h_dims[0])

        self.encoder = Encoder(sequence_length, latent_dim, h_dims, scales, blocks_per_stages, layers_per_blocks, se_ratio)
        self.decoder = Decoder(sequence_length, latent_dim, h_dims[::-1], scales[::-1], blocks_per_stages[::-1], layers_per_blocks[::-1], se_ratio)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        out = self.decoder(z)
        return out

    def reparametrize(self, z):
        mean, scale = z.chunk(2, dim=1)

        std = F.softplus(scale) + 1e-4
        var = std ** 2
        logvar = torch.log(var)

        z = torch.randn_like(mean) * std + mean
        kl = (mean ** 2 + var - logvar - 1).sum(1).mean()

        return z, kl


    def forward(self, x):
        x_s = self.pqmf.analysis(x)
        encoded = self.encode(x_s)
        z, kl = self.reparametrize(encoded)
        out_s = self.decode(z)
        out = self.pqmf.synthesis(out_s)

        # print("z_out", z.shape, out_s.shape, out.shape)

        return out, kl