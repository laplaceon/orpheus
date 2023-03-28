import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks_1d import Upsample
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