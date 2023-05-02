import torch
import numpy as np
import io
import torchaudio
import torch.nn.functional as F

from pydub import AudioSegment

def quantize_waveform(tensor, n_bins, bits=16, pool=1):
    # Determine the min and max values of the tensor
    t_min = -(2**(bits-1))
    t_max = (2**(bits-1))-1
    
    # Compute the range and width of each bin
    bin_range = t_max - t_min
    bin_width = bin_range / n_bins
    
    # Shift the tensor so that the minimum value is at 0
    tensor = tensor - t_min
    
    # Compute the bin indices for each element in the tensor
    bin_indices = (tensor / bin_width).floor() if pool == 1 else F.avg_pool1d(tensor / bin_width, pool).floor()
    
    return bin_indices.clamp(0, n_bins - 1)

def write_audio_tensor(input, sample_rate=44100, params={"format": "mp3", "bitrate": "128k"}):
    assert len(input.shape) == 2 and input.shape[0] == 1
    with io.BytesIO() as bytes:
        torchaudio.save(bytes, input, sample_rate, format="wav")
        bytes.seek(0)
        
        sound = AudioSegment.from_file(bytes, frame_rate=sample_rate, format="wav")
        bytes.seek(0)
        bytes.truncate(0)

        sound.export(bytes, **params)
        bytes.seek(0)
        
        sound = AudioSegment.from_file(bytes, frame_rate=sample_rate, format="mp3")

        sound_np = np.array([sound.get_array_of_samples()])

        return torch.from_numpy(sound_np).int()

def convert_pcm(input, sample_rate=44100, params={"format": "mp3", "bitrate": "128k"}):
    assert len(input.shape) == 2 and input.shape[0] == 1
    with io.BytesIO() as bytes:
        torchaudio.save(bytes, input, sample_rate, format="wav")
        bytes.seek(0)
        
        sound = AudioSegment.from_file(bytes, frame_rate=sample_rate, format="wav")
        bytes.seek(0)
        bytes.truncate(0)

        sound.export(bytes, **params)
        bytes.seek(0)
        
        sound = AudioSegment.from_file(bytes, frame_rate=sample_rate, format="mp3")

        sound_np = np.array([sound.get_array_of_samples()])

        return torch.from_numpy(sound_np).int()