import torch
import torch.nn as nn
import torch.nn.functional as F
from .pqmf import PQMF

from .encoder_1d import Encoder
from .decoder import Decoder, FutureDecoder

class Orpheus(nn.Module):
    def __init__(
        self,
        sequence_length,
        h_dims=[16, 80, 160, 320, 640],
        latent_dim=128,
        scales=[4, 4, 4, 2],
        blocks_per_stages=[1, 1, 1, 1],
        layers_per_blocks=[3, 3, 3, 3],
        fast_recompose=True
    ):
        super().__init__()

        self.pqmf = PQMF(h_dims[0], 100, fast_recompose)

        self.encoder = Encoder(h_dims, latent_dim, [None] + scales, blocks_per_stages, layers_per_blocks)
        self.decoder = Decoder(h_dims[::-1], latent_dim, scales[::-1], blocks_per_stages[::-1], layers_per_blocks[::-1])

        # self.future_decoder = FutureDecoder(h_dims[::-1], latent_dim, scales[::-1], blocks_per_stages[::-1], layers_per_blocks[::-1])

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

    # def predict_future(self, z):
    #     return self.future_decoder(z)

    def forward(self, x):
        encoded = self.encode(x)
        z, kl = self.reparameterize(encoded)
        out = self.decode(z)

        return out, kl