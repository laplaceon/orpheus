import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from .encoder_2d import Encoder
from .quantizer_2d import VQEmbedding
from .decoder import Decoder, SynthDecoder

class Orpheus(nn.Module):
    def __init__(
        self,
        sequence_length,
        h_dims=(1, 8, 16, 32, 64, 128),
        scales=(2, 2, 2, 2, 2),
        blocks_per_stages=(3, 3, 2, 2, 1),
        layers_per_blocks=(2, 2, 2, 2, 2),
        se_ratio=0.5,
        codebook_width=256
    ):
        super().__init__()

        z_length = sequence_length
        for x in scales:
            z_length /= x

        self.z_shape = (int(z_length), h_dims[-1])

        self.encoder = Encoder(sequence_length, codebook_width, h_dims, scales, blocks_per_stages, layers_per_blocks, se_ratio)
        # self.decoder = Decoder(sequence_length, h_dims[::-1], scales[::-1], blocks_per_stages[::-1], layers_per_blocks[::-1], se_ratio)
        self.decoder = SynthDecoder(sequence_length, h_dims[-1], se_ratio)

        self.codebook = VQEmbedding(codebook_width, h_dims[-1])

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x