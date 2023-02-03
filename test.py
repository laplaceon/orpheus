import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def spectrogram(xs):
  return signal.spectrogram(xs, 256, window=('hann'), nperseg=256, nfft=256, noverlap=200, mode='complex')

def build_training_data(batch_size, sample_size):
  xs = np.random.randn(batch_size, sample_size)
  f, t, ys = spectrogram(xs)

  (num_rows, num_cols) = (ys.shape[1], ys.shape[2])
  print(ys.shape)

  ys = ys.reshape(batch_size, num_rows * num_cols)
  Ys = np.hstack([ys.real, ys.imag])
  return (xs, Ys, num_rows, num_cols)

size = 2048
N = 15000

(xs, ys, rows, cols) = build_training_data(N, size)