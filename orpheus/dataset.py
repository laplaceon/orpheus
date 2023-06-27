import torchaudio
from torch.utils.data import Dataset
import random
from math import floor

import os

def aggregate_wavs(dirs, random_state=4):
    files = []

    for i, dir in enumerate(dirs):
        files += [
            # f"{dir}/{x}" for x in os.listdir(dir) if x.endswith(".wav")
            {"file": f"{dir}/{x}", "genre": i} for x in os.listdir(dir) if x.endswith(".wav")
        ]
    
    random.Random(random_state).shuffle(files)

    return files

class AudioFileDataset(Dataset):
    def __init__(self, inputs, extract_length, bitrate=44100, multiplier=1):
        self.multiplier = multiplier

        self.bitrate = bitrate
        self.extract_length = extract_length

        self.files = [None] * len(inputs)
        self.genres = [None] * len(inputs)

        for i in range(len(inputs)):
            self.files[i] = torchaudio.load(inputs[i]["file"])
            self.genres[i] = inputs[i]["genre"]

    def __getitem__(self, i):
        idx = floor(i / self.multiplier)
        data, rate = self.files[idx]

        bal = random.uniform(0.25, 0.75)

        if data.shape[0] == 2:
            data = bal * data[0, :] + (1 - bal) * data[1, :]
        else:
            data = data[0, :]

        extract_length = self.extract_length
        sample = None

        # Convert rate to desired one
        if rate < self.bitrate:
            multiplier = int(self.bitrate / rate)
            extract_length /= multiplier
            sample_idx = random.randrange(0, data.shape[0] - extract_length)
            sample = data[sample_idx:sample_idx+extract_length].repeat_interleave(multiplier)
        else:
            sample_idx = random.randrange(0, data.shape[0] - extract_length)
            sample = data[sample_idx:sample_idx+extract_length]

        item = {
            "input": sample,
            "genre": self.genres[idx]
        }

        return item

    def __len__(self):
        return len(self.files) * self.multiplier
