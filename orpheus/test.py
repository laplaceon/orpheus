import torch
from torch import nn
import torch.nn.functional as F

import torchaudio
from torchaudio.functional import mu_law_encoding, mu_law_decoding

from slugify import slugify

from model.mask import gen_random_mask_1d

pool = 16
q_channels = 256
block = 2048


def get_song_features(file):
    data, rate = torchaudio.load(file)
    bal = 0.5

    if data.shape[0] == 2:
        data = bal * data[0, :] + (1 - bal) * data[1, :]
    else:
        data = data[0, :]

    # consumable = data.shape[0] - (data.shape[0] % pool)
    consumable = data.shape[0] - (data.shape[0] % block)
    data = data[:consumable].unsqueeze(0).unsqueeze(1)

    # data_q = F.avg_pool1d(data, pool)
    # data_q = mu_law_encoding(data_q, q_channels)
    # data_q = mu_law_decoding(data_q, q_channels)
    # rate = rate // pool

    mask = gen_random_mask_1d(data, 0.5, block)

    data_q = data * (1. - mask.repeat(1, 1, data.shape[2] // mask.shape[1]))

    return data_q, rate

test_files = [
    "Synthwave Coolin'.wav",
    "Waiting For The End [Official Music Video] - Linkin Park-HQ.wav",
    "q1.wav"
]

for test_file in test_files:
    out, rate = get_song_features(f"../input/{test_file}")
    print(out)
    torchaudio.save(f"../output/{slugify(test_file)}_m2.wav", out.squeeze(0), rate)

# x = torch.rand(2, 1, 8192) * 2 - 1
# x_mu = mu_law_encoding(x.squeeze(1), q_channels)

# y = torch.randn(2, q_channels, 8192)

# ce = F.cross_entropy(y, x_mu, reduction="none")
# mask = torch.ones(2, 64).repeat(1, 2048 // 16)

# ce_masked = (ce * mask).sum() / mask.sum()

# print(torch.mean(ce))
# print(ce_masked)