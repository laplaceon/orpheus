import torch
import torch.nn as nn
import torch.nn.functional as F
from .pqmf import PQMF

from .encoder_1d import Encoder
from .decoder import Decoder, MiddleDecoder

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

        self.middle_decoder = MiddleDecoder(dec_h_dims[1:], latent_dim, dec_scales[1:], dec_blocks_per_stages[1:], dec_layers_per_blocks[1:])

    def decompose(self, x):
        return self.pqmf(x)

    def encode(self, x):
        return self.encoder(x)

    def recompose(self, x):
        return self.pqmf.inverse(x)

    def decode(self, z):
        return self.decoder(z)

    def predict_middle(self, z):
        return self.middle_decoder(z)

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)

        return out