import torch
import torch.nn.functional as F
import torchaudio
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torch_audiomentations import SomeOf, OneOf, PolarityInversion, AddColoredNoise, Gain, HighPassFilter, LowPassFilter

import random

from phase import unwrap

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
    
    return data_map

bitrate = 44100
# sequence_length = 262144
sequence_length = 131072
n_fft = 2048
n_stft = n_fft // 2 + 1
n_mels = 64

to_complex = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=None).cuda()
to_mel = torchaudio.transforms.MelScale(sample_rate=bitrate, n_stft=n_fft // 2 + 1, n_mels=n_mels).cuda()
resize = torchvision.transforms.Resize((n_stft, n_stft)).cuda()

clips = load_audio_clips(["../input/Synthwave Coolin'.wav", "../input/Waiting For The End [Official Music Video] - Linkin Park-HQ.wav"])
spectrograms = get_complex_spectrogram(clips, to_complex)
visualize_magnitude_and_phase(spectrograms, to_mel, resize)