import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm

from .blocks_1d import MBConv, DepthwiseSeparableConvWN, DepthwiseSeparableConv, EnhancedResBlock, Upsample, DBlockV2

class Decoder(nn.Module):
    def __init__(
        self,
        h_dims,
        latent_dim,
        scales,
        blocks_per_stage,
        layers_per_blocks
    ):
        super().__init__()

        self.from_latent = weight_norm(nn.Conv1d(latent_dim, h_dims[0] * 2, kernel_size=3, padding="same"))

        stages = []
        h_dims_new = [h_dims[0] * 2] + h_dims[:-1]
        for i in range(len(h_dims)-1):
            in_channels, out_channels = h_dims_new[i], h_dims_new[i+1]

            decoder_stage = DecoderStage(in_channels, out_channels, scales[i], blocks_per_stage[i], layers_per_blocks[i], i+1 == len(h_dims) - 1)
            stages.append(decoder_stage)

        final_conv = nn.Sequential(
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(h_dims[-2], h_dims[-1], 7, padding="same")),
            nn.Tanh()
        )

        stages.append(final_conv)

        self.conv = nn.Sequential(*stages)

    def forward(self, z):
        out = self.from_latent(z)
        return self.conv(out)

class DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale,
        num_blocks,
        layers_per_block,
        last_stage=False
    ):
        super().__init__()

        blocks = []

        blocks.append(
            nn.Sequential(
                nn.LeakyReLU(0.2),
                # Upsample(scale_factor=scale),
                # DepthwiseSeparableConvWN(in_channels, out_channels, scale * 2, padding="same")
                nn.ConvTranspose1d(in_channels, out_channels, scale * 2, stride=scale, padding=scale//2, bias=False)
                # DepthwiseSeparableConvWN(in_channels, out_channels, (in_channels // out_channels) + 1 if last_stage else 3, padding="same")
            )
        )

        for _ in range(num_blocks):
            blocks.append(
                DecoderBlock(
                    out_channels,
                    3,
                    layers_per_block
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class DecoderBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel,
        num_layers=4,
        dilation_factor=3
    ):
        super().__init__()

        conv = []

        for i in range(num_layers):
            dilation = dilation_factor ** i
            conv.append(
                # EnhancedResBlock(
                #     channels,
                #     kernel,
                #     padding = "same",
                #     dilation = dilation,
                #     bias = False,
                #     se_ratio = se_ratio,
                #     activation = nn.LeakyReLU(0.2)
                # )

                DBlockV2(
                    channels,
                    kernel,
                    padding = "same",
                    dilation = dilation,
                    bias = False,
                    activation = nn.LeakyReLU(0.2)
                )
            )

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class FutureDecoder(nn.Module):
    def __init__(
        self,
        h_dims,
        latent_dim,
        scales,
        blocks_per_stage,
        layers_per_blocks
    ):
        super().__init__()

        conv = [Decoder([x//2 for x in h_dims[:-1]] + [h_dims[-1]], latent_dim, scales[:-1] + [scales[-1]//2], blocks_per_stage, layers_per_blocks)]

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)