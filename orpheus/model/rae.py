import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from model.encoder import Encoder
from model.decoder import Decoder, SynthDecoder

class RawAudioEncoder(nn.Module):
    def __init__(
        self,
        sequence_length,
        h_dims=(1, 8, 32, 32, 32, 24),
        scales=(3, 3, 3, 2, 2),
        blocks_per_stages=(2, 2, 4, 4, 3),
        layers_per_blocks=(4, 4, 4, 4, 4),
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

    def encode(self, x, deterministic=False, temperature=0.5):
        return self.encoder(x, deterministic, temperature)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x, deterministic=True)
        return self.decode(z)
