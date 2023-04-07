import torch
import torch.nn as nn
import numpy as np
import librosa as li
import torchaudio
from einops import rearrange
from typing import Callable, Optional, Sequence, Union

def relative_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    norm: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    return norm(x - y) / norm(x)

def mean_difference(target: torch.Tensor,
                    value: torch.Tensor,
                    norm: str = 'L1',
                    relative: bool = False):
    diff = target - value
    if norm == 'L1':
        diff = diff.abs().mean()
        if relative:
            diff = diff / target.abs().mean()
        return diff
    elif norm == 'L2':
        diff = (diff * diff).mean()
        if relative:
            diff = diff / (target * target).mean()
        return diff
    else:
        raise Exception(f'Norm must be either L1 or L2, got {norm}')

def masked_mean_difference(target: torch.Tensor,
                    value: torch.Tensor,
                    mask: torch.Tensor,
                    norm: str = 'L1',
                    relative: bool = False):
    diff = target - value
    diff = diff.transpose(1, 2).view(mask.shape[0], mask.shape[1], -1)
    target = target.transpose(1, 2).view(mask.shape[0], mask.shape[1], -1)
    if norm == 'L1':
        diff = diff.abs().mean(dim=-1)
        if relative:
            diff = diff / target.abs().mean(dim=-1)
        return (diff * mask).sum() / mask.sum()
    elif norm == 'L2':
        diff = (diff * diff).mean(dim=-1)
        if relative:
            diff = diff / (target * target).mean(dim=-1)
        return (diff * mask).sum() / mask.sum()
    else:
        raise Exception(f'Norm must be either L1 or L2, got {norm}')

class MelScale(nn.Module):
    def __init__(self, sample_rate: int, n_fft: int, n_mels: int) -> None:
        super().__init__()
        mel = li.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
        mel = torch.from_numpy(mel).float()
        self.register_buffer('mel', mel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mel = self.mel.type_as(x)
        y = torch.einsum('bft,mf->bmt', x, mel)
        return y


class MultiScaleSTFT(nn.Module):

    def __init__(self,
                 scales: Sequence[int],
                 sample_rate: int,
                 magnitude: bool = True,
                 normalized: bool = False,
                 num_mels: Optional[int] = None) -> None:
        super().__init__()
        self.scales = scales
        self.magnitude = magnitude
        self.num_mels = num_mels

        self.stfts = []
        self.mel_scales = []
        for scale in scales:
            self.stfts.append(
                torchaudio.transforms.Spectrogram(
                    n_fft=scale,
                    win_length=scale,
                    hop_length=scale // 4,
                    normalized=normalized,
                    power=None,
                ))
            if num_mels is not None:
                self.mel_scales.append(
                    MelScale(
                        sample_rate=sample_rate,
                        n_fft=scale,
                        n_mels=num_mels,
                    ))
            else:
                self.mel_scales.append(None)

        self.stfts = nn.ModuleList(self.stfts)
        self.mel_scales = nn.ModuleList(self.mel_scales)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        x = rearrange(x, "b c t -> (b c) t")
        stfts = []
        for stft, mel in zip(self.stfts, self.mel_scales):
            y = stft(x)
            if mel is not None:
                y = mel(y)
            if self.magnitude:
                y = y.abs()
            else:
                y = torch.stack([y.real, y.imag], -1)
            stfts.append(y)

        return stfts


class AudioDistanceV1(nn.Module):

    def __init__(self, multiscale_stft: nn.Module,
                 log_epsilon: float) -> None:
        super().__init__()
        self.multiscale_stft = multiscale_stft
        self.log_epsilon = log_epsilon

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        stfts_x = self.multiscale_stft(x)
        stfts_y = self.multiscale_stft(y)
        distance = 0.

        for x, y in zip(stfts_x, stfts_y):
            logx = torch.log(x + self.log_epsilon)
            logy = torch.log(y + self.log_epsilon)

            lin_distance = mean_difference(x, y, norm='L2', relative=True)
            log_distance = mean_difference(logx, logy, norm='L1')

            distance = distance + lin_distance + log_distance

        return {'spectral_distance': distance}

class AudioDistanceMasked(nn.Module):

    def __init__(self, multiscale_stft: nn.Module,
                 log_epsilon: float) -> None:
        super().__init__()
        self.multiscale_stft = multiscale_stft
        self.log_epsilon = log_epsilon

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        stfts_x = self.multiscale_stft(x)
        stfts_y = self.multiscale_stft(y)
        distance = 0.

        for x, y in zip(stfts_x, stfts_y):
            logx = torch.log(x + self.log_epsilon)
            logy = torch.log(y + self.log_epsilon)

            lin_distance = masked_mean_difference(x, y, mask, norm='L2', relative=True)
            log_distance = masked_mean_difference(logx, logy, mask, norm='L1')

            distance = distance + lin_distance + log_distance

        return {'spectral_distance': distance}

class WeightedInstantaneousSpectralDistance(nn.Module):

    def __init__(self,
                 multiscale_stft: Callable[[], MultiScaleSTFT],
                 weighted: bool = False) -> None:
        super().__init__()
        self.multiscale_stft = multiscale_stft()
        self.weighted = weighted

    def phase_to_instantaneous_frequency(self,
                                         x: torch.Tensor) -> torch.Tensor:
        x = self.unwrap(x)
        x = self.derivative(x)
        return x

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., 1:] - x[..., :-1]

    def unwrap(self, x: torch.Tensor) -> torch.Tensor:
        x = self.derivative(x)
        x = (x + np.pi) % (2 * np.pi)
        return (x - np.pi).cumsum(-1)

    def forward(self, target: torch.Tensor, pred: torch.Tensor):
        stfts_x = self.multiscale_stft(target)
        stfts_y = self.multiscale_stft(pred)
        spectral_distance = 0.
        phase_distance = 0.

        for x, y in zip(stfts_x, stfts_y):
            assert x.shape[-1] == 2

            x = torch.view_as_complex(x)
            y = torch.view_as_complex(y)

            # AMPLITUDE DISTANCE
            x_abs = x.abs()
            y_abs = y.abs()

            logx = torch.log1p(x_abs)
            logy = torch.log1p(y_abs)

            lin_distance = mean_difference(x_abs,
                                           y_abs,
                                           norm='L2',
                                           relative=True)
            log_distance = mean_difference(logx, logy, norm='L1')

            spectral_distance = spectral_distance + lin_distance + log_distance

            # PHASE DISTANCE
            x_if = self.phase_to_instantaneous_frequency(x.angle())
            y_if = self.phase_to_instantaneous_frequency(y.angle())

            if self.weighted:
                mask = torch.clip(torch.log1p(x_abs[..., 2:]), 0, 1)
                x_if = x_if * mask
                y_if = y_if * mask

            phase_distance = phase_distance + mean_difference(
                x_if, y_if, norm='L2')

        return {
            'spectral_distance': spectral_distance,
            'phase_distance': phase_distance
        }


class EncodecAudioDistance(nn.Module):

    def __init__(self, scales: int,
                 spectral_distance: Callable[[int], nn.Module]) -> None:
        super().__init__()
        self.waveform_distance = WaveformDistance(norm='L1')
        self.spectral_distances = nn.ModuleList(
            [spectral_distance(scale) for scale in scales])

    def forward(self, x, y):
        waveform_distance = self.waveform_distance(x, y)
        spectral_distance = 0
        for dist in self.spectral_distances:
            spectral_distance = spectral_distance + dist(x, y)

        return {
            'waveform_distance': waveform_distance,
            'spectral_distance': spectral_distance
        }


class WaveformDistance(nn.Module):

    def __init__(self, norm: str) -> None:
        super().__init__()
        self.norm = norm

    def forward(self, x, y):
        return mean_difference(y, x, self.norm)


class SpectralDistance(nn.Module):

    def __init__(
        self,
        n_fft: int,
        sampling_rate: int,
        norm: Union[str, Sequence[str]],
        power: Union[int, None],
        normalized: bool,
        mel: Optional[int] = None,
    ) -> None:
        super().__init__()
        if mel:
            self.spec = torchaudio.transforms.MelSpectrogram(
                sampling_rate,
                n_fft,
                hop_length=n_fft // 4,
                n_mels=mel,
                power=power,
                normalized=normalized,
                center=False,
                pad_mode=None,
            )
        else:
            self.spec = torchaudio.transforms.Spectrogram(
                n_fft,
                hop_length=n_fft // 4,
                power=power,
                normalized=normalized,
                center=False,
                pad_mode=None,
            )

        if isinstance(norm, str):
            norm = (norm, )
        self.norm = norm

    def forward(self, x, y):
        x = self.spec(x)
        y = self.spec(y)

        distance = 0
        for norm in self.norm:
            distance = distance + mean_difference(y, x, norm)
        return distance

