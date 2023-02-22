from model.pqmf_old import PQMF as PQMF1
from model.pqmf_rave import PQMF as PQMF2

import torch
import torchaudio

torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
torchaudio.set_audio_backend("soundfile")

from slugify import slugify

bitrate = 44100
sequence_length = 131072

bands = 16

pqmf1 = PQMF1(bands).cuda()
pqmf2 = PQMF2(bands, 100, False).cuda()

files = [
    "../input/Waiting For The End [Official Music Video] - Linkin Park-HQ.wav",
    "../input/Synthwave Coolin'.wav"
]

for file in files:
    data, rate = torchaudio.load(file)
    bal = 0.5

    if data.shape[0] == 2:
        data = bal * data[0, :] + (1 - bal) * data[1, :]
    else:
        data = data[0, :]

    consumable = data.shape[0] - (data.shape[0] % sequence_length)

    data = torch.stack(torch.split(data[:consumable], sequence_length)).cuda()
    data_spec = data[:15].unsqueeze(1)

    pqmf1_mb = pqmf1.analysis(data_spec)
    pqmf1_rec = pqmf1.synthesis(pqmf1_mb).flatten()

    pqmf2_mb = pqmf2(data_spec)
    pqmf2_rec = pqmf2.inverse(pqmf2_mb).flatten()

    print(pqmf1_rec.shape, pqmf2_rec.shape)

    torchaudio.save(f"../output/{slugify(file)}_pqmf1.wav", pqmf1_rec.cpu().unsqueeze(0), bitrate)
    torchaudio.save(f"../output/{slugify(file)}_pqmf2.wav", pqmf2_rec.cpu().unsqueeze(0), bitrate)