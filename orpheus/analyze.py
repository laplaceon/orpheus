import torch
import torch.nn.functional as F
import torchaudio
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

from torch_audiomentations import SomeOf, OneOf, PolarityInversion, AddColoredNoise, Gain, HighPassFilter, LowPassFilter

import random

from phase import unwrap
from model.pqmf import PQMF

from torch.distributions import Normal

from core.utils import min_max_scale
from core.loss import MelScale
from model.mask import gen_random_mask_1d, upsample_mask

from einops import rearrange

apply_augmentations = SomeOf(
    num_transforms = (1, 3),
    transforms = [
        PolarityInversion(),
        AddColoredNoise(),
        Gain(),
        OneOf(
            transforms = [
                HighPassFilter(),
                LowPassFilter()
            ]
        )
    ]
)

to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

pqmf = PQMF(16, 100)

def torch_unwrap(x):
    # Port from np.unwrap
    dx = torch_diff(x)
    dx_m = ((dx + np.pi) % (2 * np.pi)) - np.pi
    dx_m[(dx_m == -np.pi) & (dx > 0)] = np.pi
    x_adj = dx_m - dx
    x_adj[dx.abs() < np.pi] = 0

    return x + x_adj.cumsum(-1)

def torch_diff(x):
    return F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (1, 0))

def get_complex_spectrogram(t, to, batches=(5, 15)):
    for k in t:
        data = t[k]
        consumable = data.shape[0] - (data.shape[0] % sequence_length)
        data = torch.stack(torch.split(data[:consumable], sequence_length)).cuda()

        data = apply_augmentations(data.unsqueeze(1))

        t[k] = to(data[batches[0]:batches[1]]).cuda()

    return t

def visualize_magnitude_and_phase(c, to_mel, resize):
    for k in c:
        spec = c[k]

        m, p = spec.abs().pow(2), spec.angle()
        unwrapped_p = torch_unwrap(p)
        inf = torch_diff(unwrapped_p)
        m_mel = to_db(resize(to_mel(m))).cpu()
        p_mel = resize(to_mel(p)).cpu()
        up_mel = resize(to_mel(unwrapped_p)).cpu()
        if_mel = resize(to_mel(inf)).cpu()

        # combo = torch.cat([m_mel, if_mel, torch.zeros_like(m_mel)], dim=1)

        for i in range(spec.shape[0]):
            plt.figure()

            f, axarr = plt.subplots(1,5) 

            axarr[0].imshow(m[i].permute(1, 2, 0).cpu())
            axarr[1].imshow(m_mel[i].permute(1, 2, 0))
            axarr[2].imshow(p_mel[i].permute(1, 2, 0))
            axarr[3].imshow(up_mel[i].permute(1, 2, 0).cpu())
            axarr[4].imshow(if_mel[i].permute(1, 2, 0))

            # axarr[0].imshow(m[i].permute(1, 2, 0).cpu())
            # axarr[1].imshow(p[i].permute(1, 2, 0).cpu())
            # axarr[2].imshow(unwrapped_p[i].permute(1, 2, 0).cpu())
            # axarr[3].imshow(inf[i].permute(1, 2, 0).cpu())
            # axarr[4].imshow(combo[i].permute(1, 2, 0).cpu())
            
            plt.show()

def count_pos_neg(x):
    pos = (x >= 0).to(torch.int16)
    num_pos = (pos == 1).sum()
    num_neg = (pos == 0).sum()
    print(x.shape, num_pos.item(), num_neg.item())

def load_audio_clips(l):
    data_map = {}

    for file in l:
        data, _ = torchaudio.load(file)
        bal = random.uniform(0.25, 0.75)

        if data.shape[0] == 2:
            data = bal * data[0, :] + (1 - bal) * data[1, :]
        else:
            data = data[0, :]
    
        data_map[file] = data

        # count_pos_neg(data)
    
    return data_map

def mel_spec(x, scale, normalized=False, num_mels=None, sample_rate=44100):
    B = x.size(0)
    x = rearrange(x, "b c t -> (b c) t")

    to_spec = torchaudio.transforms.Spectrogram(
        n_fft=scale,
        win_length=scale,
        hop_length=scale // 4,
        normalized=normalized,
        power=None,
    )

    spec = to_spec(x)

    if num_mels is not None:
        to_mel = MelScale(
            sample_rate=sample_rate,
            n_fft=scale,
            n_mels=num_mels,
        )

        spec = to_mel(spec).abs()

    _, C, L = spec.size()
    spec = spec.view(B, -1, C, L)
    
    return spec

def get_trunc_norm(min, max, mu, std):
    return stats.truncnorm((min - mu) / std, (max - mu) / std, loc=mu, scale=std)

bitrate = 44100
# sequence_length = 262144
sequence_length = 131072
n_fft = 2048
n_stft = n_fft // 2 + 1
n_mels = 64

# to_complex = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=None).cuda()
# to_mel = torchaudio.transforms.MelScale(sample_rate=bitrate, n_stft=n_fft // 2 + 1, n_mels=n_mels).cuda()
# resize = torchvision.transforms.Resize((n_stft, n_stft)).cuda()

clips = load_audio_clips(["../input/Synthwave Coolin'.wav", "../input/Waiting For The End [Official Music Video] - Linkin Park-HQ.wav"])
clips = list(clips.values())[1][sequence_length*5:sequence_length*6].unsqueeze(0).unsqueeze(1)

def test_wave_spectral_prob_relationship():
    # clips = pqmf(clips)

    # clips = torch.tensor(get_trunc_norm(-1, 1, 0, 1).rvs(sequence_length)).float().unsqueeze(0).unsqueeze(1)

    means = clips
    variances = torch.rand(1, sequence_length) * 0.1

    dist = Normal(clips, variances)
    sampled = dist.sample_n(32)

    print(sampled.shape, sampled.min(), sampled.max())

    print(means.shape, means.min(), means.max())
    print(variances.shape, variances.min(), variances.max())
    spec = mel_spec(sampled, 2048, normalized=False)
    print(spec.shape, spec.min(), spec.max())
    # spectrograms = get_complex_spectrogram(clips, to_complex)
    # visualize_magnitude_and_phase(spectrograms, to_mel, resize)

def visualize_spec(specs):
    plt.figure()

    s = to_db(specs[0].permute(1, 2, 0))
    mask = to_db(specs[1].permute(1, 2, 0))
    
    plt.subplot(1, 3, 1)
    plt.imshow(s)
    plt.subplot(1, 3, 3)
    plt.imshow(mask)
    plt.subplot(1, 3, 2)
    plt.imshow(s * (1. - mask))
    
    plt.colorbar()
    plt.show()

# mask = gen_random_mask_1d(clips, 0.1, 2048)
# wav_mask = upsample_mask(mask, 2048).unsqueeze(1)

# print(wav_mask.shape, clips.shape)

# clips_mel = mel_spec(clips, 2048, num_mels=128)
# mask_mel = min_max_scale(mel_spec(wav_mask, 2048, num_mels=128))

# print(mask, mask_mel.min().item(), mask_mel.max().item())

# visualize_spec([clips_mel[0], mask_mel[0]])

# print(clips.shape, clips.min(), clips.max())
# spec = to_mel(clips, 2048, normalized=False)
# print(spec)
# print(spec.shape, spec.min(), spec.max())
# spec = min_max_scale(spec)
# print(spec)
# print(spec.shape, spec.min(), spec.max())

nums = torch.rand(5,)
logprobs = torch.log_softmax(nums, dim=0)
probs = torch.softmax(nums, dim=0)
exp = torch.exp(logprobs)
log = torch.log(probs)

print(nums, logprobs, exp, log)