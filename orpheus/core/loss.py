import torch
import torch.nn as nn
import numpy as np
import librosa as li
import torchaudio
from einops import rearrange
from typing import Callable, Optional, Sequence, Union, Tuple

import torch.nn.functional as F

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
    
    value = value.transpose(1, 2).view(mask.shape[0], mask.shape[1], -1)
    target = target.transpose(1, 2).view(mask.shape[0], mask.shape[1], -1)

    # mean = target.mean(dim=-1, keepdim=True)
    # var = target.var(dim=-1, keepdim=True)
    # ntarget = (target - mean) / (var + 1.e-6)**.5

    diff = target - value
    # ndiff = ntarget - value

    unmask = 1. - mask

    if norm == 'L1':
        diff = diff.abs().mean(dim=-1)
        # ndiff = ndiff.abs().mean(dim=-1)
        if relative:
            diff = diff / target.abs().mean(dim=-1)
            # ndiff = ndiff / ntarget.abs().mean(dim=-1)
    elif norm == 'L2':
        diff = (diff * diff).mean(dim=-1)
        # ndiff = (ndiff * ndiff).mean(dim=-1)
        if relative:
            diff = diff / (target * target).mean(dim=-1)
            # ndiff = ndiff / (ntarget * ntarget).mean(dim=-1)
    else:
        raise Exception(f'Norm must be either L1 or L2, got {norm}')
    
    masked_diff = (diff * mask).sum() / mask.sum()
    unmasked_diff = (diff * unmask).sum() / unmask.sum()

    return masked_diff, unmasked_diff

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
                 log_epsilon: float,
                 masked_weights: Tuple[float] = (1., 0.)) -> None:
        super().__init__()
        self.multiscale_stft = multiscale_stft
        self.log_epsilon = log_epsilon
        self.masked_weights = masked_weights

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        stfts_x = self.multiscale_stft(x)
        stfts_y = self.multiscale_stft(y)
        distance = 0.
        masked_distance = 0.
        unmasked_distance = 0.
        
        masked_weight = self.masked_weights[0]
        unmasked_weight = self.masked_weights[1]

        for x, y in zip(stfts_x, stfts_y):
            logx = torch.log(x + self.log_epsilon)
            logy = torch.log(y + self.log_epsilon)

            lin_distance_masked, lin_distance_unmasked = masked_mean_difference(x, y, mask, norm='L2', relative=True)
            log_distance_masked, log_distance_unmasked = masked_mean_difference(logx, logy, mask, norm='L1')

            lin_distance = masked_weight * lin_distance_masked + unmasked_weight * lin_distance_unmasked
            log_distance = masked_weight * log_distance_masked + unmasked_weight * log_distance_unmasked

            distance = distance + lin_distance + log_distance
            masked_distance = masked_distance + lin_distance_masked + log_distance_masked
            unmasked_distance = unmasked_distance + lin_distance_unmasked + log_distance_unmasked

        return {'spectral_distance': distance, 'spectral_distance_masked': masked_distance, 'spectral_distance_unmasked': unmasked_distance}

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

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

def _compute_inv_stdv(logits, distribution_base='std', min_mol_logscale=-250., gradient_smoothing_beta=0.6931472):
    softplus = nn.Softplus(beta=gradient_smoothing_beta)
    if distribution_base == 'std':
        scales = torch.maximum(softplus(logits),
                               torch.as_tensor(np.exp(min_mol_logscale)))
        inv_stdv = 1. / scales  # Not stable for sharp distributions
        log_scales = torch.log(scales)

    elif distribution_base == 'logstd':
        log_scales = torch.maximum(logits, torch.as_tensor(np.array(min_mol_logscale)))
        inv_stdv = torch.exp(-gradient_smoothing_beta * log_scales)
    else:
        raise ValueError(f'distribution base {distribution_base} not known!!')

    return inv_stdv, log_scales

def discretized_mix_logistic_loss(input, target, num_classes=256,
                                  log_scale_min=-7.0, reduce=True):
    """Discretized mixture of logistic distributions loss
    Note that it is assumed that input is scaled to [-1, 1].
    Args:
        y_hat (Tensor): Predicted output (B x C x T)
        y (Tensor): Target (B x T x 1).
        num_classes (int): Number of classes
        log_scale_min (float): Log scale minimum value
        reduce (bool): If True, the losses are averaged or summed for each
          minibatch.
    Returns
        Tensor: loss
    """
    assert input.dim() == 3
    assert input.size(1) % 3 == 0
    nr_chan = target.size(2)
    nr_mix = input.size(1) // (3 * nr_chan)

    # (B x T x C)
    input = input.transpose(1, 2)

    logit_probs = input[:, :, :nr_mix*nr_chan]
    means = input[:, :, nr_mix*nr_chan:2 * nr_mix*nr_chan]
    log_scales = torch.clamp(input[:, :, 2 * nr_mix * nr_chan:3 * nr_mix * nr_chan], min=log_scale_min)

    # B x T x 1 -> B x T x num_mixtures
    # print(y.shape, means.shape)
    # y = y.expand_as(means)
    target = target.repeat(1, 1, nr_mix)

    centered_y = target - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    # equivalent: torch.log(torch.sigmoid(plus_in))
    log_cdf_plus = plus_in - F.softplus(plus_in)

    # log probability for edge case of 255 (before scaling)
    # equivalent: (1 - torch.sigmoid(min_in)).log()
    log_one_minus_cdf_min = -F.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # tf equivalent
    """
    log_probs = tf.where(x < -0.999, log_cdf_plus,
                         tf.where(x > 0.999, log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                           tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log(127.5))))
    """
    # TODO: cdf_delta <= 1e-5 actually can happen. How can we choose the value
    # for num_classes=65536 case? 1e-7? not sure..
    inner_inner_cond = (cdf_delta > 1e-5).float()

    inner_inner_out = inner_inner_cond * \
        torch.log(torch.clamp(cdf_delta, min=1e-12)) + \
        (1. - inner_inner_cond) * (log_pdf_mid - np.log((num_classes - 1) / 2))
    inner_cond = (target > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (target < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

    log_probs = log_probs + F.log_softmax(logit_probs, -1)

    if reduce:
        return -torch.sum(log_sum_exp(log_probs))
    else:
        return -log_sum_exp(log_probs).unsqueeze(-1)

def discretized_mix_logistic_loss_2(logits, targets, num_classes=256, num_mixtures=4, log_scale_min=-7.0):
    # Shapes:
    #    targets: B, C, L
    #    logits: B, M * (3 * C + 1), L

    assert len(targets.shape) == 3
    B, C, L = targets.size()

    min_pix_value, max_pix_value = -1, 1

    targets = targets.unsqueeze(2)  # B, C, 1, L

    logit_probs = logits[:, :num_mixtures, :]  # B, M, L
    l = logits[:, num_mixtures:, :]  # B, M*C*3 , L
    l = l.reshape(B, C, 3 * num_mixtures, L)  # B, C, 3 * M, L

    model_means = l[:, :, :num_mixtures, :]  # B, C, M, L
    means = model_means

    inv_stdv, log_scales = _compute_inv_stdv(
        l[:, :, num_mixtures: 2 * num_mixtures, :], distribution_base='logstd')

    # model_coeffs = torch.tanh(
    #     l[:, :, 2 * num_output_mixtures: 3 * num_output_mixtures, :])  # B, C, M, H, W

    centered = targets - means  # B, C, M, L

    plus_in = inv_stdv * (centered + 1. / num_classes)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered - 1. / num_classes)
    cdf_min = torch.sigmoid(min_in)

    log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255 (before scaling)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min  # B, C, M, L

    mid_in = inv_stdv * centered
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in this code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
    # which is mapped to 0.9922
    broadcast_targets = torch.broadcast_to(targets, size=[B, C, num_mixtures, L])
    log_probs = torch.where(broadcast_targets == min_pix_value, log_cdf_plus,
                            torch.where(broadcast_targets == max_pix_value, log_one_minus_cdf_min,
                                        torch.where(cdf_delta > 1e-5,
                                                    torch.log(torch.clamp(cdf_delta, min=1e-12)),
                                                    log_pdf_mid - np.log(num_classes / 2))))  # B, C, M, L

    log_probs = torch.sum(log_probs, dim=1) + F.log_softmax(logit_probs, dim=1)  # B, M, L
    negative_log_probs = -torch.logsumexp(log_probs, dim=1)  # B, L

    expected = torch.sum(means * logit_probs.unsqueeze(1), dim=2)

    return negative_log_probs, expected

class DiscretizedMixtureLoss1(nn.Module):
    def __init__(
        self,
        num_classes = 256
    ):
        super().__init__()

        self.num_classes = num_classes

    def forward(self, input, target, mask=None):
        mask = mask.expand_as(target)

        losses = discretized_mix_logistic_loss(
            input, target, num_classes=self.num_classes, reduce=False)
        assert losses.size() == target.size()

        losses = losses.squeeze(2)

        if mask is not None:
           return ((losses * mask).sum()) / mask.sum()
        
        return torch.sum(losses)
    
class DiscretizedMixtureLoss2(nn.Module):
    def __init__(
        self,
        num_classes = 256
    ):
        super().__init__()

        self.num_classes = num_classes

    def forward(self, input, target, mask=None):
        # mask = mask.expand_as(target)

        losses, expected = discretized_mix_logistic_loss_2(
            input, target, num_classes=self.num_classes)
        # assert losses.size() == target.size()

        if mask is not None:
           return ((losses * mask).sum()) / mask.sum(), expected
        
        return torch.sum(losses), expected