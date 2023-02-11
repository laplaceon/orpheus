import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from .encoder_2d import Encoder
from .quantizer_2d import SQEmbedding
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

        h_dims_dec = [128, 64, 48, 32, 32, 16, 16, 8, 4, 2, 1]
        scales_dec = [2, 2, 4, 4, 4, 2, 2, 2, 2, 2]
        blocks_per_stages_dec = [1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1]
        layers_per_blocks_dec = [3, 4, 4, 4, 3, 2, 2, 2, 2, 2, 2]

        # self.decoder = Decoder(sequence_length, h_dims_dec, scales_dec, blocks_per_stages_dec, layers_per_blocks_dec, se_ratio)
        # self.decoder = Decoder(sequence_length, h_dims[::-1], scales[::-1], blocks_per_stages[::-1], layers_per_blocks[::-1], se_ratio)
        self.decoder = SynthDecoder(sequence_length, h_dims[-1], se_ratio)

        self.codebook = SQEmbedding(codebook_width, h_dims[-1])

    def encode(self, x):
        z_e_x = self.encoder(x)
        # latents = self.codebook(z_e_x)
        return z_e_x

    def decode(self, latents):
        # z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        # x_tilde = self.decoder(z_q_x)
        x_tilde = self.decoder(latents)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        # z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        # x_tilde = self.decoder(z_q_x_st)
        x_tilde = self.decoder(z_e_x)
        # print(x_tilde.shape)
        # print(z_e_x.shape, x_tilde.shape)
        return x_tilde, z_e_x