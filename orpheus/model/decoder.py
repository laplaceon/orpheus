import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks_1d import MBConv
from .synth import SinusoidalSynthesizer, FIRNoiseSynth, Reverb

class SynthDecoder(nn.Module):
    def __init__(
        self,
        sequence_length,
        dim,
        se_ratio
    ):
        super().__init__()

        self.net1 = nn.Sequential(
            nn.Conv1d(dim, dim*2, 3, padding=1),
            nn.SiLU(),
            nn.BatchNorm1d(dim*2),
            MBConv(dim*2, dim*2, dim*2),
            nn.Tanh()
        )

        self.net2 = nn.Sequential(
            nn.Conv1d(dim, dim*2, 3, padding=1),
            nn.SiLU(),
            nn.BatchNorm1d(dim*2),
            MBConv(dim*2, dim*2, dim*2, se_ratio=se_ratio),
            nn.ReLU()
        )

        self.net3 = nn.Sequential(
            nn.Conv1d(dim, dim*2, 3, padding=1),
            nn.SiLU(),
            nn.BatchNorm1d(dim*2),
            MBConv(dim*2, dim*2, dim*2),
            nn.Tanh()
        )

        self.synth = SinusoidalSynthesizer(sequence_length, 44100)
        self.filter = FIRNoiseSynth(510, 256)
        self.reverb = Reverb(sequence_length, 44100)

    def forward(self, z):
        z = z.view(-1, z.shape[1], z.shape[2]*z.shape[3])

        amplitudes = self.net1(z)
        frequencies = self.net2(z)
        filter_in = self.net3(z)
        noise = self.filter(filter_in)
        noise = F.interpolate(noise, scale_factor=32, mode="linear").squeeze(1)

        controls = self.synth.get_controls(amplitudes, frequencies)
        synth_out = self.synth.get_signal(controls["amplitudes"], controls["frequencies"])

        audio = self.reverb(synth_out + noise)

        return audio

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

    def forward(self, x):
        return self.conv(x)


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
        for i in range(num_blocks):
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

    def forward(self, x):
        out = self.blocks(x)
        out = self.post_upscale(F.interpolate(out, scale_factor=self.scale))
        return self.activation(out)

class DecoderBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel,
        num_layers,
        default_num_layers=4,
        dilation_factor=2,
        se_ratio=None
    ):
        super().__init__()

        if num_layers is None:
            num_layers = default_num_layers

        dilate = 1
        conv = [MBConv(channels, channels*2, channels, kernel, se_ratio=se_ratio)]

        for i in range(num_layers-1):
            dilation = dilation_factor ** (i+1)
            conv.append(MBConv(channels, channels*2, channels, kernel, dilation=dilation, se_ratio=se_ratio))

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)
