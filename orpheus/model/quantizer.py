import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def quantize(num, quantization_channels=256):
    # Calculate the size of each bin
    bin_size = 2 / quantization_channels
    
    # Find the bin that each number falls into
    bin_num = torch.floor((num + 1) / bin_size)
    
    # Map the bin numbers to their corresponding quantized values
    quantized_num = -1 + bin_num * bin_size + bin_size / 2
    
    # If a number is exactly at the upper bound, put it in the last bin
    quantized_num[bin_num == quantization_channels] = -1 + (quantization_channels - 1) * bin_size + bin_size / 2
    
    return quantized_num

def mu_law_encode(audio, quantization_channels=256):
    """
    Quantize waveform amplitudes.
    Reference: https://github.com/vincentherrmann/pytorch-wavenet/blob/master/audio_data.py
    """
    mu = float(quantization_channels - 1)
    quantize_space = np.linspace(-1, 1, quantization_channels)

    quantized = np.sign(audio) * np.log(1 + mu * np.abs(audio)) / np.log(mu + 1)
    quantized = np.digitize(quantized, quantize_space) - 1

    return quantized

import torch

def mu_law_encoding(signal, mu=255):
    # Apply mu-law compression to signal
    signal = torch.sign(signal) * torch.log1p(mu * torch.abs(signal)) / torch.log1p(mu)
    
    # Quantize the signal to integer values between 0 and mu
    signal = (signal + 1) / 2 * mu
    signal = torch.floor(signal)
    
    # Convert the signal to an integer data type
    signal = signal.to(torch.int32)
    
    return signal


def mu_law_decode(output, quantization_channels=256):
    """
    Recovers waveform from quantized values.
    Reference: https://github.com/vincentherrmann/pytorch-wavenet/blob/master/audio_data.py
    """
    mu = float(quantization_channels - 1)

    expanded = (output / quantization_channels) * 2. - 1
    waveform = np.sign(expanded) * (
                   np.exp(np.abs(expanded) * np.log(mu + 1)) - 1
               ) / mu

    return waveform
