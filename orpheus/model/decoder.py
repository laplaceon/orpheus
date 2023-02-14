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
        num_bands = 8,
        waveshapers_per_band = 6,
        sample_rate=44100
    ):
        super().__init__()

        # self.filter = FIRNoiseSynth(510, dim*2)

        self.net1 = nn.Sequential(
            nn.Conv1d(dim, dim*2, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(dim*2),
            MBConv(dim*2, dim*2, dim*2),
            nn.LeakyReLU()
        )

        paths = [NewtPath(sequence_length // num_bands, sample_rate / num_bands, waveshapers_per_band, dim*2)] * num_bands
        self.paths = nn.ModuleList(paths)

    def forward(self, z):
        z = z.view(z.shape[0], z.shape[1], -1)

        z_n = self.net1(z)

        bands = []
        
        for path in self.paths:
            bands.append(path(z_n, z))

        fused = torch.cat(bands, dim=1)

        # print(fused.shape)

        signal = self.pqmf.synthesis(fused)

        return signal

class NewtPath(nn.Module):
    def __init__(
        self,
        sequence_length,
        sample_rate,
        num_waveshapers,
        dim,
    ):
        super().__init__()

        self.linear_synth = SinusoidalSynthesizer(sequence_length, sample_rate)
        self.newt = NEWT(num_waveshapers, dim // 2)

    def forward(self, x, z):
        amplitudes = x
        frequencies = x

        controls = self.linear_synth.get_controls(amplitudes, frequencies)
        synth_out = self.linear_synth.get_signal(controls["amplitudes"], controls["frequencies"])
        newt_out = self.newt(synth_out.unsqueeze(1), z)

        return torch.tanh(newt_out)

class Decoder(nn.Module):
    def __init__(
        self,
        sequence_length,
        latent_dim,
        h_dims,
        scales,
        blocks_per_stage,
        layers_per_blocks,
        se_ratio=None
    ):
        super().__init__()

        self.from_latent = nn.Sequential(
            nn.Conv1d(latent_dim, h_dims[0] * 2, kernel_size=3, padding="same", bias=False),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.to_h_dims = nn.Sequential(
            nn.Conv1d(h_dims[0] * 2, h_dims[0], 3, padding="same", bias=False),
            nn.LeakyReLU(negative_slope=0.2)
        )

        stages = []
        for i in range(len(h_dims)-1):
            in_channels, out_channels = h_dims[i], h_dims[i+1]

            decoder_stage = DecoderStage(in_channels, out_channels, scales[i], blocks_per_stage[i], layers_per_blocks[i], se_ratio, i+1 == len(h_dims) - 1)
            stages.append(decoder_stage)

        self.conv = nn.Sequential(*stages)

    def forward(self, z):
        out = self.from_latent(z)
        out = self.to_h_dims(F.interpolate(out, scale_factor=2))

        return self.conv(out)


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
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
                    3,
                    layers_per_block,
                    se_ratio=se_ratio
                )
            )

        self.blocks = nn.Sequential(*blocks)

        self.post_upscale = DepthwiseSeparableConv(in_channels, out_channels, 7 if last_stage else 3, padding="same")
        self.activation = nn.Tanh() if last_stage else nn.LeakyReLU()

        self.scale = scale

        self.last_stage = last_stage

    def forward(self, x):
        out = self.blocks(x)
        if not self.last_stage:
            out = F.interpolate(out, scale_factor=self.scale)
        out = self.post_upscale(out)
        return self.activation(out)

class DecoderBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel,
        num_layers=4,
        dilation_factor=3,
        se_ratio=None
    ):
        super().__init__()

        conv = []

        for i in range(num_layers):
            dilation = dilation_factor ** i
            conv.append(ResBlock(channels, channels, kernel, dilation=dilation, activation=nn.LeakyReLU(negative_slope=0.2)))

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)
