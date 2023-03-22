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
        # enc_h_dims=[16, 64, 128, 256, 512],
        enc_h_dims=[16, 80, 160, 320, 640],
        dec_h_dims=[640, 320, 160, 80, 16],
        latent_dim=128,
        enc_scales=[4, 4, 4, 2],
        enc_attns=[[None, None, None], [None, None, None], [True, True, True]],
        dec_scales=[2, 4, 4, 4],
        enc_blocks_per_stages=[1, 1, 1, 1],
        enc_layers_per_blocks=[3, 3, 3, 2],
        dec_blocks_per_stages=[1, 1, 1, 1],
        dec_layers_per_blocks=[2, 3, 3, 3],
        attn=False,
        fast_recompose=True
    ):
        super().__init__()

        self.pqmf = PQMF(enc_h_dims[0], 100, fast_recompose)

        self.encoder = Encoder(enc_h_dims, latent_dim, [None] + enc_scales, [None] + enc_attns, enc_blocks_per_stages, enc_layers_per_blocks, attn=attn)
        self.decoder = Decoder(dec_h_dims, latent_dim, dec_scales, dec_blocks_per_stages, dec_layers_per_blocks)

        # self.future_decoder = FutureDecoder(h_dims[::-1], latent_dim, scales[::-1], blocks_per_stages[::-1], layers_per_blocks[::-1])

    def decompose(self, x):
        return self.pqmf(x)

    def encode(self, x):
        return self.encoder(x)

    def recompose(self, x):
        return self.pqmf.inverse(x)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize_vae(self, z, return_vars=False):
        mean, scale = z.chunk(2, dim=1)

        std = F.softplus(scale) + 1e-4
        var = std ** 2
        logvar = torch.log(var)

        z = torch.randn_like(mean) * std + mean
        kl = (mean ** 2 + var - logvar - 1).sum(1).mean()

        if return_vars:
            return z, kl, mean, std

        return z, kl

    def compute_mean_kernel(self, x, y):
        kernel_input = (x[:, None] - y[None]).pow(2).mean(2) / x.shape[-1]
        return torch.exp(-kernel_input).mean()

    def compute_mmd(self, x, y):
        x_kernel = self.compute_mean_kernel(x, x)
        y_kernel = self.compute_mean_kernel(y, y)
        xy_kernel = self.compute_mean_kernel(x, y)
        mmd = x_kernel + y_kernel - 2 * xy_kernel
        return mmd

    def reparameterize_wae(self, z):
        z_reshaped = z.permute(0, 2, 1).reshape(-1, z.shape[1])
        reg = self.compute_mmd(z_reshaped, torch.randn_like(z_reshaped))

        # if self.noise_augmentation:
        if False:
            noise = torch.randn(z.shape[0], self.noise_augmentation,
                                z.shape[-1]).type_as(z)
            z = torch.cat([z, noise], 1)

        return z, reg.mean()

    # def predict_future(self, z):
    #     return self.future_decoder(z)

    def forward(self, x):
        encoded = self.encode(x)
        z, kl = self.reparameterize_wae(encoded)
        out = self.decode(z)

        return out, kl