import torch

from synth_tf import Sinusoidal as SinusoidalTf
from synth import SinusoidalSynthesizer as SinusoidalPt
import numpy as np
import tensorflow as tf

a = np.random.randn(16, 24, 1648)
f = np.random.randn(16, 24, 1648)

a = np.transpose(a, (0, 2, 1))
f = np.transpose(f, (0, 2, 1))

n_samples=177984
sample_rate=44100

a_pt = torch.Tensor(a)
f_pt = torch.Tensor(f)

a_tf = tf.convert_to_tensor(a)
f_tf = tf.convert_to_tensor(f)

def cd1(tensor_pt, tensor_tf):
    a = tensor_pt.numpy()
    b = tensor_tf.numpy()

    mse = (np.square(a - b)).mean(axis=None)

    return mse

def cd2(tensor_pt, tensor_tf):
    a = tensor_pt.numpy()
    b = tensor_tf.numpy()

    return a - b

sinusoidal_tf = SinusoidalTf(n_samples, sample_rate)
sinusoidal_pt = SinusoidalPt(n_samples, sample_rate)

scaled_tf = sinusoidal_tf.get_controls(a_tf, f_tf)
scaled_pt = sinusoidal_pt.get_controls(a_pt, f_pt)

amp_diff = cd2(scaled_tf['amplitudes'], scaled_pt['amplitudes'])
freq_diff = cd2(scaled_tf['frequencies'], scaled_pt['frequencies'])

signal_tf = sinusoidal_tf.get_signal(scaled_tf['amplitudes'], scaled_tf['frequencies'])
signal_pt = sinusoidal_pt.get_signal(scaled_pt['amplitudes'], scaled_pt['frequencies'])

# (ae, fe, signal_tf) = sinusoidal_tf.get_signal(scaled_tf['amplitudes'], scaled_tf['frequencies'])
# signal_pt = sinusoidal_pt.get_signal(torch.tensor(ae), torch.tensor(fe))

amp_diff = cd1(signal_tf[0], signal_pt[0])
freq_diff = cd1(signal_tf[1], signal_pt[1])

# signal_diff = cd2(signal_tf, signal_pt)
# print(signal_diff)
print(amp_diff, freq_diff)
