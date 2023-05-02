import torch
from torch import nn
import torch.nn.functional as F
import io

from core.process import convert_pcm, quantize_waveform

import torchaudio
from torchaudio.transforms import Resample
from torchaudio.functional import mu_law_encoding, mu_law_decoding, apply_codec

from torch_audiomentations import PolarityInversion

from slugify import slugify

import pydub

from model.mask import gen_random_mask_1d

pool = 16
q_channels = 256
block = 2048

sample_rate = 44100

polarity_inversion = PolarityInversion(p=1.0, sample_rate=sample_rate)

resample = Resample(sample_rate, sample_rate // pool)

def get_song_features(file):
    data, rate = torchaudio.load(file)
    bal = 0.5

    if data.shape[0] == 2:
        data = bal * data[0, :] + (1 - bal) * data[1, :]
    else:
        data = data[0, :]

    # consumable = data.shape[0] - (data.shape[0] % pool)
    consumable = data.shape[0] - (data.shape[0] % block)
    data_q = data[:consumable].unsqueeze(0).unsqueeze(1)

    # data_q_inv = polarity_inversion(data_q)
    # print(data_q.shape)

    # print("dq", torch.min(data_q).item(), torch.max(data_q).item(), data_q)
    # print("dqi", torch.min(data_q_inv).item(), torch.max(data_q_inv).item(), data_q_inv)
    # dm1 = convert_pcm(data_q.squeeze(0))
    # dm2 = convert_pcm(data_q_inv.squeeze(0))
    # print("aac", torch.min(dm1).item(), torch.max(dm1).item(), dm1)
    # print("aaci", torch.min(dm2).item(), torch.max(dm2).item(), dm2)
    # print(apply_codec(data_q.squeeze(0), sample_rate, "adts"))
    # data_q = quantize_waveform(dm1, 256, pool=1)
    # print(data_q, data_q.shape, data_q.min().item(), data_q.max().item())
    # data_q = quantize_waveform(dm1, 256, pool=pool)
    # print(data_q, data_q.shape, data_q.min().item(), data_q.max().item())
    # print()

    # torchaudio.save(f"../output/test_m4.mp3", dm2, sample_rate, format="mp3")

    # song = pydub.AudioSegment(dm1.numpy().tobytes(), frame_rate=sample_rate, sample_width=4, channels=1)
    # song.export(f"../output/test_m4.mp3", format="mp3")
    # data_inv_mp3 = convert_pcm(data_q_inv.squeeze(0))
    # print(data_mp3)
    # print(data_inv_mp3)

    # data_q = F.avg_pool1d(data_q, pool)
    data_q = resample(data_q)
    # data_mp3_quantized = quantize_waveform(data_mp3, 256)
    # print(data_mp3_quantized)
    # print(data_q_inv)
    data_q = mu_law_encoding(data_q, q_channels)
    # data_q_inv = mu_law_encoding(data_q_inv, q_channels)
    # data_q = apply_codec(data_q.squeeze(0), rate, **{"format": "mp3", "compression": -4.2})
    # print(data_q)
    # print(data_q_inv)

    data_q = mu_law_decoding(data_q, q_channels)
    rate = rate // pool

    # mask = gen_random_mask_1d(data, 0.95, block)

    # data_q = data * (1. - mask.repeat(1, 1, data.shape[2] // mask.shape[1]))

    return data_q, rate

test_files = [
    # "lost.mp3",
    "Synthwave Coolin'.wav",
    "Waiting For The End [Official Music Video] - Linkin Park-HQ.wav",
    "q1.wav"
]

for test_file in test_files:
    out, rate = get_song_features(f"../input/{test_file}")
    print(out)
    torchaudio.save(f"../output/{slugify(test_file)}_rq.wav", out.squeeze(0), rate)