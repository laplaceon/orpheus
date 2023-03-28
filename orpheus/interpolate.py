import torch
import torchaudio

import scipy
from model.rae import Orpheus

torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
torchaudio.set_audio_backend("soundfile")

bitrate = 44100
sequence_length = 131072

def load_audio_clips(l):
    data_map = {}

    for file in l:
        data, _ = torchaudio.load(file)

        if data.shape[0] == 2:
            data = 0.5 * data[0, :] + 0.5 * data[1, :]
        else:
            data = data[0, :]
    
        data_map[file] = data
    
    return data_map

def interpolate(model, clip1, clip2, weights=[0.5, 0.5]):
    clips = load_audio_clips([clip1, clip2])

    n = 6

    segments = torch.stack([clips[clip1][:sequence_length*n], clips[clip2][:sequence_length*n]]).unsqueeze(1)
    z = model.encode(model.decompose(segments))

    weights = torch.tensor(weights)
    weighted_z = z.transpose(0, 2) * weights
    z_n = torch.sum(weighted_z.transpose(0, 2), dim=0)

    out = model.recompose(model.decode(z_n.unsqueeze(0))).squeeze(0)

    z_parts = model.encode(model.decompose(segments))
    segment_recons = model.recompose(model.decode(z_parts))

    return out, segment_recons[0], segment_recons[1]

model = Orpheus(sequence_length, fast_recompose=True)
model.load_state_dict(torch.load("../models/rae_40.pt"))
model.eval()

with torch.no_grad():
    fused, segment_1, segment_2 = interpolate(model, "../input/KORDHELL - MURDER IN MY MIND.wav", "../input/Prodigy  - Out of Space [Breakbeat Remix].wav")
    torchaudio.save(f"../output/fused7.wav", fused, bitrate)
    torchaudio.save(f"../output/segment1.wav", segment_1, bitrate)
    torchaudio.save(f"../output/segment2.wav", segment_2, bitrate)