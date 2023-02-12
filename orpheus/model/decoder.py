import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks_1d import MBConv, DepthwiseSeparableConv, ResBlock
from .synth import SinusoidalSynthesizer, FIRNoiseSynth, Reverb
from .pqmf_old import PQMF
from .newt import NEWT

class SynthDecoder(nn.Module):
    def __init__(
        self,
        sequence_length,
        dim,
        num_bands = 4,
        num_waveshapers = 32
    ):
        super().__init__()

        self.pqmf = PQMF(num_bands)
        # self.filter = FIRNoiseSynth(510, dim*2)

        self.net1 = nn.Sequential(
            nn.Conv1d(dim, dim*2, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(dim*2),
            MBConv(dim*2, dim*2, dim*2),
            nn.LeakyReLU()
        )

        paths = [NewtPath(num_waveshapers, dim*2)] * num_bands
        self.paths = nn.ModuleList(paths)

    def forward(self, z):
        z = z.view(z.shape[0], z.shape[1], -1)

        z_n = self.net1(z)

        bands = []
        
        for path in self.paths:
            bands.append(path(z_n))

        fused = torch.cat(bands, dim=1)

        # print(fused.shape)

        signal = self.pqmf.synthesis(fused)

        return signal

class NewtPath(nn.Module):
    def __init__(
        self,
        num_waveshapers,
        dim
    ):
        super().__init__()

        # self.newt = NEWT(num_waveshapers, dim)

        self.upscales = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, 3, padding=1),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.Conv1d(dim, dim // 4, 3, padding=1),
                nn.BatchNorm1d(dim // 4),
                nn.LeakyReLU(),
                ResBlock(dim // 4, dim // 4, activation=nn.LeakyReLU())
            ),
            nn.Sequential(
                nn.Conv1d(dim // 4, dim // 16, 3, padding=1),
                nn.BatchNorm1d(dim // 16),
                nn.LeakyReLU(),
                ResBlock(dim // 16, dim // 16, activation=nn.LeakyReLU())
            ),
            nn.Sequential(
                nn.Conv1d(dim // 16, dim // 64, 3, padding=1),
                nn.BatchNorm1d(dim // 64),
                nn.LeakyReLU(),
                ResBlock(dim // 64, dim // 64, activation=nn.LeakyReLU())
            ),
            nn.Sequential(
                nn.Conv1d(dim // 64, dim // 128, 3, padding=1),
                nn.BatchNorm1d(dim // 128),
                nn.LeakyReLU(),
                ResBlock(dim // 128, dim // 128, activation=nn.LeakyReLU())
            ),
            nn.Sequential(
                nn.Conv1d(dim // 128, dim // 256, 3, padding=1),
                nn.BatchNorm1d(dim // 256),
                nn.LeakyReLU(),
                ResBlock(dim // 256, dim // 256, activation=nn.Tanh())
            )
        ])

    def forward(self, x):
        # print("x", x.shape)
        out = F.interpolate(x, scale_factor=4, mode="linear")
        out = self.upscales[0](out)
        out = F.interpolate(out, scale_factor=4, mode="linear")
        out = self.upscales[1](out)
        out = F.interpolate(out, scale_factor=4, mode="linear")
        out = self.upscales[2](out)
        out = F.interpolate(out, scale_factor=4, mode="linear")
        out = self.upscales[3](out)
        out = F.interpolate(out, scale_factor=4, mode="linear")
        out = self.upscales[4](out)
        out = F.interpolate(out, scale_factor=2, mode="linear")
        out = self.upscales[5](out)

        return out

class Decoder(nn.Module):
    def __init__(
        self,
        sequence_length,
        h_dims,
        scales,
        blocks_per_stage,
        layers_per_blocks,
        se_ratio=None
    ):
        super().__init__()

        stages = []
        for i in range(len(h_dims)-1):
            in_channels, out_channels = h_dims[i], h_dims[i+1]

            decoder_stage = DecoderStage(in_channels, out_channels, 3, scales[i], blocks_per_stage[i], layers_per_blocks[i], se_ratio, i+1 == len(h_dims) - 1)
            stages.append(decoder_stage)

        self.conv = nn.Sequential(*stages)

    def forward(self, z):
        # z = z.view(z.shape[0], z.shape[1], -1)
        return self.conv(z)


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        scale,
        num_blocks,
        layers_per_block,
        se_ratio,
        last_stage=False
    ):
        super().__init__()

        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                DecoderBlock(
                    in_channels,
                    kernel,
                    layers_per_block,
                    se_ratio=se_ratio
                )
            )

        self.blocks = nn.Sequential(*blocks)

        self.post_upscale = DepthwiseSeparableConv(in_channels, out_channels, 3, padding="same", bias=True)
        self.activation = nn.Tanh() if last_stage else nn.SiLU()

        self.scale = scale

        self.last_stage = last_stage

    def forward(self, x):
        out = self.blocks(x)
        out = self.post_upscale(F.interpolate(out, scale_factor=self.scale))
        if self.last_stage:
            out = F.pad(out, (0, 1))
        return self.activation(out)

class DecoderBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel,
        num_layers,
        default_num_layers=4,
        dilation_factor=3,
        se_ratio=None
    ):
        super().__init__()

        if num_layers is None:
            num_layers = default_num_layers

        dilate = 1
        conv = [ResBlock(channels, channels, kernel)]

        for i in range(num_layers-1):
            dilation = dilation_factor ** (i+1)
            conv.append(ResBlock(channels, channels, kernel, dilation=1))

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)
