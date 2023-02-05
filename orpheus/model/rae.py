import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from .encoder import Encoder
from .decoder import Decoder, SynthDecoder

class Orpheus(nn.Module):
    def __init__(
        self,
        sequence_length,
        h_dims=(64, 96, 144, 216, 256),
        scales=(2, 2, 2, 2),
        blocks_per_stages=(1, 1, 1, 1),
        layers_per_blocks=(2, 2, 2, 2),
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

        # self.quantizer = VectorQuantizer(codebook_width, h_dims[-1])

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # quantized_inputs, vq_loss = self.quantizer(self.encode(x))

        # return [self.decode(quantized_inputs), vq_loss]

        z = self.encode(x)

        # print(z.shape)

        return self.decode(z)