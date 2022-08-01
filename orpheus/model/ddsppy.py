import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import numpy as np
import librosa as li
# import crepe
import math

def safe_log(x):
    return torch.log(x + 1e-7)


@torch.no_grad()
def mean_std_loudness(dataset):
    mean = 0
    std = 0
    n = 0
    for _, _, l in dataset:
        n += 1
        mean += (l.mean().item() - mean) / n
        std += (l.std().item() - std) / n
    return mean, std


def multiscale_fft(signal, scales, overlap):
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts


def resample(x, factor: int):
    batch, frame, channel = x.shape
    x = x.permute(0, 2, 1).reshape(batch * channel, 1, frame)

    window = torch.hann_window(
        factor * 2,
        dtype=x.dtype,
        device=x.device,
    ).reshape(1, 1, -1)
    y = torch.zeros(x.shape[0], x.shape[1], factor * x.shape[2]).to(x)
    y[..., ::factor] = x
    y[..., -1:] = x[..., -1:]
    y = torch.nn.functional.pad(y, [factor, factor])
    y = torch.nn.functional.conv1d(y, window)[..., :-1]

    y = y.reshape(batch, channel, factor * frame).permute(0, 2, 1)

    return y


def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
    return signal.permute(0, 2, 1)


def remove_above_nyquist(amplitudes, pitch, sampling_rate):
    n_harm = amplitudes.shape[-1]
    pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)
    aa = (pitches < sampling_rate / 2).float() + 1e-4
    return amplitudes * aa


def scale_function(x):
    return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-7


def extract_loudness(signal, sampling_rate, block_size, n_fft=2048):
    S = li.stft(
        signal,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
    )
    S = np.log(abs(S) + 1e-7)
    f = li.fft_frequencies(sampling_rate, n_fft)
    a_weight = li.A_weighting(f)

    S = S + a_weight.reshape(-1, 1)

    S = np.mean(S, 0)[..., :-1]

    return S


# def extract_pitch(signal, sampling_rate, block_size):
#     length = signal.shape[-1] // block_size
#     f0 = crepe.predict(
#         signal,
#         sampling_rate,
#         step_size=int(1000 * block_size / sampling_rate),
#         verbose=1,
#         center=True,
#         viterbi=True,
#     )
#     f0 = f0[1].reshape(-1)[:-1]
#
#     if f0.shape[-1] != length:
#         f0 = np.interp(
#             np.linspace(0, 1, length, endpoint=False),
#             np.linspace(0, 1, f0.shape[-1], endpoint=False),
#             f0,
#         )
#
#     return f0


def mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)


def gru(n_input, hidden_size):
    return nn.GRU(n_input * hidden_size, hidden_size, batch_first=True)


def harmonic_synth(pitch, amplitudes, sampling_rate):
    n_harmonic = amplitudes.shape[-1]
    omega = torch.cumsum(2 * math.pi * pitch / sampling_rate, 1)
    omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
    signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
    return signal


def amp_to_impulse_response(amp, target_size):
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp


def fft_convolve(signal, kernel):
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output

def _add_depth_axis(freqs: torch.Tensor, depth: int = 1) -> torch.Tensor:
  """Turns [batch, time, sinusoids*depth] to [batch, time, sinusoids, depth]."""
  freqs = freqs.unsqueeze(-1)
  # Unpack sinusoids dimension.
  n_batch, n_time, n_combined, _ = freqs.shape
  n_sinusoids = int(n_combined) // depth
  return torch.reshape(freqs, (n_batch, n_time, n_sinusoids, depth))


def frequencies_sigmoid(freqs: torch.Tensor,
                        depth: int = 1,
                        hz_min: float = 0.0,
                        hz_max: float = 8000.0) -> torch.Tensor:
  """Sum of sigmoids to logarithmically scale network outputs to frequencies.
  Args:
    freqs: Neural network outputs, [batch, time, n_sinusoids * depth] or
      [batch, time, n_sinusoids, depth].
    depth: If freqs is 3-D, the number of sigmoid components per a sinusoid to
      unroll from the last dimension.
    hz_min: Lowest frequency to consider.
    hz_max: Highest frequency to consider.
  Returns:
    A tensor of frequencies in hertz [batch, time, n_sinusoids].
  """
  if len(freqs.shape) == 3:
    # Add depth: [B, T, N*D] -> [B, T, N, D]
    freqs = _add_depth_axis(freqs, depth)
  else:
    depth = int(freqs.shape[-1])

  # Probs: [B, T, N, D]
  f_probs = F.sigmoid(freqs)

  # [B, T N]
  # Partition frequency space in factors of 2, limit to range [hz_max, hz_min].
  hz_scales = []
  hz_min_copy = hz_min
  remainder = hz_max - hz_min
  scale_factor = remainder**(1.0 / depth)
  for i in range(depth):
    if i == (depth - 1):
      # Last depth element goes between minimum and remainder.
      hz_max = remainder
      hz_min = hz_min_copy
    else:
      # Reduce max by a constant factor for each depth element.
      hz_max = remainder * (1.0 - 1.0 / scale_factor)
      hz_min = 0
      remainder -= hz_max

    hz_scales.append(unit_to_hz(f_probs[..., i],
                                hz_min=hz_min,
                                hz_max=hz_max))

  return torch.sum(torch.stack(hz_scales, axis=-1), axis=-1)


def resample_m(
                inputs,
                n_timesteps,
                method = 'linear',
                add_endpoint = True):
  """Interpolates a tensor from n_frames to n_timesteps.
  Args:
    inputs: Framewise 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_frames],
      [batch_size, n_frames], [batch_size, n_frames, channels], or
      [batch_size, n_frames, n_freq, channels].
    n_timesteps: Time resolution of the output signal.
    method: Type of resampling, must be in ['nearest', 'linear', 'cubic',
      'window']. Linear and cubic ar typical bilinear, bicubic interpolation.
      'window' uses overlapping windows (only for upsampling) which is smoother
      for amplitude envelopes with large frame sizes.
    add_endpoint: Hold the last timestep for an additional step as the endpoint.
      Then, n_timesteps is divided evenly into n_frames segments. If false, use
      the last timestep as the endpoint, producing (n_frames - 1) segments with
      each having a length of n_timesteps / (n_frames - 1).
  Returns:
    Interpolated 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_timesteps],
      [batch_size, n_timesteps], [batch_size, n_timesteps, channels], or
      [batch_size, n_timesteps, n_freqs, channels].
  Raises:
    ValueError: If method is 'window' and input is 4-D.
    ValueError: If method is not one of 'nearest', 'linear', 'cubic', or
      'window'.
  """
  is_1d = len(inputs.shape) == 1
  is_2d = len(inputs.shape) == 2
  is_4d = len(inputs.shape) == 4

  # Ensure inputs are at least 3d.
  if is_1d:
    # inputs = inputs[tf.newaxis, :, tf.newaxis]
    inputs = inputs[None, :, None]
  elif is_2d:
    # inputs = inputs[:, :, tf.newaxis]
    inputs = inputs[:, :, None]

  def _image_resize(method):
    """Closure around tf.image.resize."""
    # Image resize needs 4-D input. Add/remove extra axis if not 4-D.
    # outputs = inputs[:, :, tf.newaxis, :] if not is_4d else inputs
    outputs = inputs[:, :, None, :] if not is_4d else inputs
    # outputs = tf.compat.v1.image.resize(outputs,
    #                                     [n_timesteps, outputs.shape[2]],
    #                                     method=method,
    #                                     align_corners=not add_endpoint)
    outpouts = F.interpolate(outputs, (n_timesteps, outputs.shape[2]), mode=method, align_corners=not add_endpoint)
    return outputs[:, :, 0, :] if not is_4d else outputs

  # Perform resampling.
  if method == 'nearest':
    outputs = _image_resize('nearest')
  elif method == 'linear':
    outputs = _image_resize('bilinear')
  elif method == 'cubic':
    outputs = _image_resize('bicubic')
  elif method == 'window':
    outputs = upsample_with_windows(inputs, n_timesteps, add_endpoint)
  else:
    raise ValueError('Method ({}) is invalid. Must be one of {}.'.format(
        method, "['nearest', 'linear', 'cubic', 'window']"))

  # Return outputs to the same dimensionality of the inputs.
  if is_1d:
    outputs = outputs[0, :, 0]
  elif is_2d:
    outputs = outputs[:, :, 0]

  return outputs


def upsample_with_windows(inputs: torch.Tensor,
                          n_timesteps: int,
                          add_endpoint: bool = True) -> torch.Tensor:
  """Upsample a series of frames using using overlapping hann windows.
  Good for amplitude envelopes.
  Args:
    inputs: Framewise 3-D tensor. Shape [batch_size, n_frames, n_channels].
    n_timesteps: The time resolution of the output signal.
    add_endpoint: Hold the last timestep for an additional step as the endpoint.
      Then, n_timesteps is divided evenly into n_frames segments. If false, use
      the last timestep as the endpoint, producing (n_frames - 1) segments with
      each having a length of n_timesteps / (n_frames - 1).
  Returns:
    Upsampled 3-D tensor. Shape [batch_size, n_timesteps, n_channels].
  Raises:
    ValueError: If input does not have 3 dimensions.
    ValueError: If attempting to use function for downsampling.
    ValueError: If n_timesteps is not divisible by n_frames (if add_endpoint is
      true) or n_frames - 1 (if add_endpoint is false).
  """
  # inputs = tf_float32(inputs)

  if len(inputs.shape) != 3:
    raise ValueError('Upsample_with_windows() only supports 3 dimensions, '
                     'not {}.'.format(inputs.shape))

  # Mimic behavior of tf.image.resize.
  # For forward (not endpointed), hold value for last interval.
  if add_endpoint:
    inputs = torch.concat([inputs, inputs[:, -1:, :]], axis=1)

  n_frames = int(inputs.shape[1])
  n_intervals = (n_frames - 1)

  if n_frames >= n_timesteps:
    raise ValueError('Upsample with windows cannot be used for downsampling'
                     'More input frames ({}) than output timesteps ({})'.format(
                         n_frames, n_timesteps))

  if n_timesteps % n_intervals != 0.0:
    minus_one = '' if add_endpoint else ' - 1'
    raise ValueError(
        'For upsampling, the target the number of timesteps must be divisible '
        'by the number of input frames{}. (timesteps:{}, frames:{}, '
        'add_endpoint={}).'.format(minus_one, n_timesteps, n_frames,
                                   add_endpoint))

  # Constant overlap-add, half overlapping windows.
  hop_size = n_timesteps // n_intervals
  window_length = 2 * hop_size
  window = torch.hann_window(window_length)  # [window]

  # Transpose for overlap_and_add.
  x = inputs.permute(0, 2, 1)  # [batch_size, n_channels, n_frames]

  # Broadcast multiply.
  # Add dimension for windows [batch_size, n_channels, n_frames, window].
  x = x[:, :, :, None]
  window = window[None, None, None, :]
  x_windowed = (x * window)
  x = overlap_and_add(x_windowed, hop_size)

  # Transpose back.
  x = x.permute(0, 2, 1)  # [batch_size, n_timesteps, n_channels]

  # Trim the rise and fall of the first and last window.
  return x[:, hop_size:-hop_size, :]

def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.
    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where
        output_size = (frames - 1) * frame_step + frame_length
    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length
    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class SinusoidalSynthesizer(nn.Module):
    def __init__(
        self,
        n_samples=108,
        sample_rate=44100,
        amp_resample_method='window',
        amp_scale_fn=scale_function,
        freq_scale_fn=frequencies_sigmoid,
        name='sinusoidal'
    ):
        super().__init__()

        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.amp_resample_method = amp_resample_method
        self.amp_scale_fn = amp_scale_fn
        self.freq_scale_fn = freq_scale_fn

    def get_controls(self, amplitudes, frequencies):
        if self.amp_scale_fn is not None:
            amplitudes = self.amp_scale_fn(amplitudes)

        if self.freq_scale_fn is not None:
            frequencies = self.freq_scale_fn(frequencies)
            amplitudes = remove_above_nyquist(amplitudes, frequencies, self.sample_rate)

        return amplitudes, frequencies

    def get_signal(self, amplitudes, frequencies):

        amplitude_envelopes = resample_m(amplitudes, self.n_samples, method=self.amp_resample_method)
        frequency_envelopes = resample_m(frequencies, self.n_samples)


        return amplitude_envelopes, frequency_envelopes
