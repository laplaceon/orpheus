import torch
import torch.nn as nn
import torch.nn.functional as F
from .pqmf import PQMF

from .encoder_1d import Encoder
from .decoder import Decoder

class Orpheus(nn.Module):
    def __init__(
        self,
        sequence_length,
        h_dims=(16, 64, 128, 256, 512),
        latent_dim=128,
        scales=(None, 4, 4, 4),
        blocks_per_stages=(1, 1, 1, 1),
        layers_per_blocks=(3, 3, 3, 3),
        se_ratio=0.25
    ):
        super().__init__()

        self.pqmf = PQMF(h_dims[0], 100)

        self.encoder = Encoder(h_dims, latent_dim, scales, blocks_per_stages, layers_per_blocks, se_ratio)
        self.decoder = Decoder(h_dims[::-1], latent_dim, scales[::-1], blocks_per_stages[::-1], layers_per_blocks[::-1], se_ratio)

    def decompose(self, x):
        return self.pqmf(x)

    def encode(self, x):
        return self.encoder(x)

    def recompose(self, x):
        return self.pqmf.inverse(x)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, z, return_vars=False):
        mean, scale = z.chunk(2, dim=1)

        std = F.softplus(scale) + 1e-4
        var = std ** 2
        logvar = torch.log(var)

        z = torch.randn_like(mean) * std + mean
        kl = (mean ** 2 + var - logvar - 1).sum(1).mean()

        if return_vars:
            return z, kl, mean, std

        return z, kl


    def forward(self, x):
        encoded = self.encode(x)
        z, kl = self.reparameterize(encoded)
        out = self.decode(z)

        return out, kl