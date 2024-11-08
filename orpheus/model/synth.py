import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import torch.fft as fft

from typing import Callable

import torchaudio

def safe_divide(numerator, denominator, eps=1e-7):
  """Avoid dividing by zero by adding a small epsilon."""
  safe_denominator = torch.where(denominator == 0.0, eps, denominator)
  return numerator / safe_denominator

def safe_log(x, eps=1e-5):
  """Avoid taking the log of a non-positive number."""
  if not isinstance(x, torch.Tensor):
      x = torch.tensor(x)
  safe_x = torch.where(x <= 0.0, eps, x)
  return torch.log(safe_x)

def logb(x, base=2.0, eps=1e-5):
  """Logarithm with base as an argument."""
  return safe_divide(safe_log(x, eps), safe_log(base, eps), eps)

def remove_above_nyquist(frequency_envelopes: torch.Tensor,
                         amplitude_envelopes: torch.Tensor,
                         sample_rate: int = 16000) -> torch.Tensor:
  """Set amplitudes for oscillators above nyquist to 0.
  Args:
    frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape
      [batch_size, n_samples, n_sinusoids].
    amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size,
      n_samples, n_sinusoids].
    sample_rate: Sample rate in samples per a second.
  Returns:
    amplitude_envelopes: Sample-wise filtered oscillator amplitude.
      Shape [batch_size, n_samples, n_sinusoids].
  """
  # frequency_envelopes = tf_float32(frequency_envelopes)
  # amplitude_envelopes = tf_float32(amplitude_envelopes)

  amplitude_envelopes = torch.where(
      torch.ge(frequency_envelopes, sample_rate / 2.0),
      torch.zeros_like(amplitude_envelopes), amplitude_envelopes)
  return amplitude_envelopes

def unit_to_midi(unit,
                 midi_min = 20.0,
                 midi_max = 90.0,
                 clip: bool = False):
  """Map the unit interval [0, 1] to MIDI notes."""
  unit = torch.clamp(unit, 0.0, 1.0) if clip else unit
  return midi_min + (midi_max - midi_min) * unit

def midi_to_hz(notes, midi_zero_silence: bool = False):
  """TF-compatible midi_to_hz function.
  Args:
    notes: Tensor containing encoded pitch in MIDI scale.
    midi_zero_silence: Whether to output 0 hz for midi 0, which would be
      convenient when midi 0 represents silence. By defualt (False), midi 0.0
      corresponds to 8.18 Hz.
  Returns:
    hz: Frequency of MIDI in hz, same shape as input.
  """
  # notes = torch.tensor(notes)
  hz = 440.0 * (2.0 ** ((notes - 69.0) / 12.0))
  # Map MIDI 0 as 0 hz when MIDI 0 is silence.
  if midi_zero_silence:
    hz = torch.where(torch.equal(notes, 0.0), 0.0, hz)
  return hz

def hz_to_midi(frequencies):
  """TF-compatible hz_to_midi function."""
  frequencies = torch.tensor(frequencies)
  notes = 12.0 * (logb(frequencies, 2.0) - logb(440.0, 2.0)) + 69.0
  # Map 0 Hz to MIDI 0 (Replace -inf MIDI with 0.)
  notes = torch.where(torch.le(frequencies, 0.0), 0.0, notes)
  return notes

def unit_to_hz(unit,
               hz_min,
               hz_max,
               clip: bool = False):
  """Map unit interval [0, 1] to [hz_min, hz_max], scaling logarithmically."""
  midi = unit_to_midi(unit,
                      midi_min=hz_to_midi(hz_min),
                      midi_max=hz_to_midi(hz_max),
                      clip=clip)
  return midi_to_hz(midi)

def remove_above_nyquist(frequency_envelopes: torch.Tensor,
                         amplitude_envelopes: torch.Tensor,
                         sample_rate: int = 16000) -> torch.Tensor:
  """Set amplitudes for oscillators above nyquist to 0.
  Args:
    frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape
      [batch_size, n_samples, n_sinusoids].
    amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size,
      n_samples, n_sinusoids].
    sample_rate: Sample rate in samples per a second.
  Returns:
    amplitude_envelopes: Sample-wise filtered oscillator amplitude.
      Shape [batch_size, n_samples, n_sinusoids].
  """
  # frequency_envelopes = torch.tensor(frequency_envelopes)
  # amplitude_envelopes = torch.tensor(amplitude_envelopes)

  amplitude_envelopes = torch.where(
      torch.greater_equal(frequency_envelopes, sample_rate / 2.0),
      torch.zeros_like(amplitude_envelopes), amplitude_envelopes)
  return amplitude_envelopes

# def remove_above_nyquist(amplitudes, pitch, sampling_rate):
#     n_harm = amplitudes.shape[-1]
#     pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)
#     aa = (pitches < sampling_rate / 2).float() + 1e-4
#     return amplitudes * aa

def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
    # exp = torch.log(torch.Tensor([exponent]).type(torch.FloatTensor)).detach().item()
    exp = math.log(exponent)
    return max_value * torch.sigmoid(x)**exp + threshold

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
  f_probs = torch.sigmoid(freqs)

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

    # print(hz_scales)
    # return f_probs

  return torch.sum(torch.stack(hz_scales, axis=-1), axis=-1)

def resample(inputs: torch.Tensor,
             n_timesteps: int,
             method = 'linear',
             add_endpoint: bool = True) -> torch.Tensor:
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
  # inputs = torch.tensor(inputs)
  is_1d = len(inputs.shape) == 1
  is_2d = len(inputs.shape) == 2
  is_4d = len(inputs.shape) == 4

  # print(inputs.shape)

  # Ensure inputs are at least 3d.
  if is_1d:
    inputs = inputs[None, :, None]
  elif is_2d:
    inputs = inputs[:, :, None]

  # def _image_resize(method):
  #   # Image resize needs 4-D input. Add/remove extra axis if not 4-D.
  #   outputs = inputs[:, :, None, :] if not is_4d else inputs
  #   outputs = outputs.permute(0, 3, 1, 2)
  #   outputs = F.interpolate(outputs, size=(n_timesteps, outputs.shape[3]), mode=method, align_corners=not add_endpoint)
  #   outputs = outputs.permute(0, 2, 3, 1)
  #   return outputs[:, :, 0, :] if not is_4d else outputs

  def _image_resize(method):
    outputs = inputs.permute(0, 2, 1)
    outputs = F.interpolate(outputs, size=(n_timesteps,), mode=method, align_corners=not add_endpoint)
    outputs = outputs.permute(0, 2, 1)
    return outputs

  # Perform resampling.
  if method == 'nearest':
    outputs = _image_resize('nearest')
  elif method == 'area':
    outputs = _image_resize('area')
  elif method == 'exact':
    outputs = _image_resize('nearest-exact')
  elif method == 'linear':
    outputs = _image_resize('linear')
  elif method == 'cubic':
    outputs = _image_resize('bicubic')
  elif method == 'window':
    outputs = upsample_with_windows(inputs, n_timesteps, add_endpoint)
    # outputs = torchaudio.functional.resample(inputs, 512, n_timesteps)
  else:
    raise ValueError('Method ({}) is invalid. Must be one of {}.'.format(
        method, "['nearest', 'linear', 'cubic', 'window']"))

  # Return outputs to the same dimensionality of the inputs.
  if is_1d:
    outputs = outputs[0, :, 0]
  elif is_2d:
    outputs = outputs[:, :, 0]

  # print(outputs.shape)

  return outputs

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

    subframe_signal = signal.reshape(*outer_dimensions, -1, subframe_length)
    # subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step).to(subframe_signal)
    # frame = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.clone().long().to(subframe_signal.device)
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result

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

  # print(inputs.shape)

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
  window = torch.hann_window(window_length).to(inputs.device)  # [window]

  # Transpose for overlap_and_add.
  x = inputs.transpose(1, 2)  # [batch_size, n_channels, n_frames]

  # Broadcast multiply.
  # Add dimension for windows [batch_size, n_channels, n_frames, window].
  x = x[:, :, :, None]
  window = window[None, None, None, :]
  x_windowed = (x * window)
  # print(x_windowed.shape, hop_size)
  x = overlap_and_add(x_windowed, hop_size)

  # Transpose back.
  x = x.transpose(1, 2)  # [batch_size, n_timesteps, n_channels]

  # Trim the rise and fall of the first and last window.
  return x[:, hop_size:-hop_size, :]


def pad_axis(x, padding=(0, 0), axis=0, **pad_kwargs):
  """Pads only one axis of a tensor.
  Args:
    x: Input tensor.
    padding: Tuple of number of samples to pad (before, after).
    axis: Which axis to pad.
    **pad_kwargs: Other kwargs to pass to tf.pad.
  Returns:
    A tensor padded with padding along axis.
  """
  n_end_dims = len(x.shape) - axis - 1
  n_end_dims *= n_end_dims > 0
  paddings = [[0, 0]] * axis + [list(padding)] + [[0, 0]] * n_end_dims
  return F.pad(x, tuple(paddings), **pad_kwargs)

def angular_cumsum(angular_frequency, chunk_size=1000):
  """Get phase by cumulative sumation of angular frequency.
  Custom cumsum splits first axis into chunks to avoid accumulation error.
  Just taking tf.sin(tf.cumsum(angular_frequency)) leads to accumulation of
  phase errors that are audible for long segments or at high sample rates. Also,
  in reduced precision settings, cumsum can overflow the threshold.
  During generation, if syntheiszed examples are longer than ~100k samples,
  consider using angular_sum to avoid noticible phase errors. This version is
  currently activated by global gin injection. Set the gin parameter
  `oscillator_bank.use_angular_cumsum=True` to activate.
  Given that we are going to take the sin of the accumulated phase anyways, we
  don't care about the phase modulo 2 pi. This code chops the incoming frequency
  into chunks, applies cumsum to each chunk, takes mod 2pi, and then stitches
  them back together by adding the cumulative values of the final step of each
  chunk to the next chunk.
  Seems to be ~30% faster on CPU, but at least 40% slower on TPU.
  Args:
    angular_frequency: Radians per a sample. Shape [batch, time, ...].
      If there is no batch dimension, one will be temporarily added.
    chunk_size: Number of samples per a chunk. to avoid overflow at low
       precision [chunk_size <= (accumulation_threshold / pi)].
  Returns:
    The accumulated phase in range [0, 2*pi], shape [batch, time, ...].
  """
  # Get tensor shapes.
  n_batch = angular_frequency.shape[0]
  n_time = angular_frequency.shape[1]
  n_dims = len(angular_frequency.shape)
  n_ch_dims = n_dims - 2

  # Pad if needed.
  remainder = n_time % chunk_size
  if remainder:
    pad_amount = chunk_size - remainder
    angular_frequency = pad_axis(angular_frequency, [0, pad_amount], axis=1)

  # Split input into chunks.
  length = angular_frequency.shape[1]
  n_chunks = int(length / chunk_size)
  chunks = torch.reshape(angular_frequency,
                      tuple([n_batch, n_chunks, chunk_size] + [-1] * n_ch_dims))
  phase = torch.cumsum(chunks, dim=2)

  # Add offsets.
  # Offset of the next row is the last entry of the previous row.
  offsets = phase[:, :, -1:, ...] % (2.0 * np.pi)
  offsets = pad_axis(offsets, [1, 0], axis=1)
  offsets = offsets[:, :-1, ...]

  # Offset is cumulative among the rows.
  offsets = torch.cumsum(offsets, dim=1) % (2.0 * np.pi)
  phase = phase + offsets

  # Put back in original shape.
  phase = phase % (2.0 * np.pi)
  phase = tf.reshape(phase, tuple([n_batch, length] + [-1] * n_ch_dims))

  # Remove padding if added it.
  if remainder:
    phase = phase[:, :n_time]
  return phase

def fft_convolve(signal, kernel):
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output

def oscillator_bank(frequency_envelopes: torch.Tensor,
                    amplitude_envelopes: torch.Tensor,
                    sample_rate: int = 16000,
                    sum_sinusoids: bool = True,
                    use_angular_cumsum: bool = False) -> torch.Tensor:
  """Generates audio from sample-wise frequencies for a bank of oscillators.
  Args:
    frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape
      [batch_size, n_samples, n_sinusoids].
    amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size,
      n_samples, n_sinusoids].
    sample_rate: Sample rate in samples per a second.
    sum_sinusoids: Add up audio from all the sinusoids.
    use_angular_cumsum: If synthesized examples are longer than ~100k audio
      samples, consider use_angular_cumsum to avoid accumulating noticible phase
      errors due to the limited precision of tf.cumsum. Unlike the rest of the
      library, this property can be set with global dependency injection with
      gin. Set the gin parameter `oscillator_bank.use_angular_cumsum=True`
      to activate. Avoids accumulation of errors for generation, but don't use
      usually for training because it is slower on accelerators.
  Returns:
    wav: Sample-wise audio. Shape [batch_size, n_samples, n_sinusoids] if
      sum_sinusoids=False, else shape is [batch_size, n_samples].
  """
  # frequency_envelopes = tf_float32(frequency_envelopes)
  # amplitude_envelopes = tf_float32(amplitude_envelopes)

  # Don't exceed Nyquist.
  amplitude_envelopes = remove_above_nyquist(frequency_envelopes,
                                             amplitude_envelopes,
                                             sample_rate)

  # Angular frequency, Hz -> radians per sample.
  omegas = frequency_envelopes * (2.0 * np.pi)  # rad / sec
  omegas = omegas / float(sample_rate)  # rad / sample

  # Accumulate phase and synthesize.
  if use_angular_cumsum:
    # Avoids accumulation errors.
    phases = angular_cumsum(omegas)
  else:
    phases = torch.cumsum(omegas, dim=1)

  # Convert to waveforms.
  wavs = torch.sin(phases)
  audio = amplitude_envelopes * wavs  # [mb, n_samples, n_sinusoids]
  if sum_sinusoids:
    audio = torch.sum(audio, dim=-1)  # [mb, n_samples]
  return audio

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

# def frequency_filter(audio: torch.Tensor,
#                      magnitudes: torch.Tensor,
#                      window_size: int = 0) -> torch.Tensor:
#   """Filter audio with a finite impulse response filter.
#   Args:
#     audio: Input audio. Tensor of shape [batch, audio_timesteps].
#     magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
#       n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
#       last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
#       f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
#       audio into equally sized frames to match frames in magnitudes.
#     window_size: Size of the window to apply in the time domain. If window_size
#       is less than 1, it is set as the default (n_frequencies).
#     padding: Either 'valid' or 'same'. For 'same' the final output to be the
#       same size as the input audio (audio_timesteps). For 'valid' the audio is
#       extended to include the tail of the impulse response (audio_timesteps +
#       window_size - 1).
#   Returns:
#     Filtered audio. Tensor of shape
#         [batch, audio_timesteps + window_size - 1] ('valid' padding) or shape
#         [batch, audio_timesteps] ('same' padding).
#   """
#   impulse_response = frequency_impulse_response(magnitudes,
#                                                 window_size=window_size)
#   return fft_convolve(audio, impulse_response)

class SinusoidalSynthesizer(nn.Module):
    def __init__(
        self,
        n_samples=64000,
        sample_rate=16000,
        amp_resample_method='window',
        amp_scale_fn=exp_sigmoid,
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
            amplitudes = remove_above_nyquist(frequencies, amplitudes, self.sample_rate)

        return {'amplitudes': amplitudes, 'frequencies': frequencies}

    def get_signal(self, amplitudes, frequencies):
        amplitude_envelopes = resample(amplitudes, self.n_samples, method=self.amp_resample_method)
        frequency_envelopes = resample(frequencies, self.n_samples)

        signal = oscillator_bank(frequency_envelopes=frequency_envelopes,
                                      amplitude_envelopes=amplitude_envelopes,
                                      sample_rate=self.sample_rate)

        return signal


class Reverb(nn.Module):
    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1))

        return x

# class FilteredNoise(nn.Module):
#   """Synthesize audio by filtering white noise."""

#   def __init__(self,
#                n_samples=64000,
#                window_size=257,
#                scale_fn=exp_sigmoid,
#                initial_bias=-5.0,
#                name='filtered_noise'):
#     super().__init__(name=name)
#     self.n_samples = n_samples
#     self.window_size = window_size
#     self.scale_fn = scale_fn
#     self.initial_bias = initial_bias

#   def get_controls(self, magnitudes):
#     """Convert network outputs into a dictionary of synthesizer controls.
#     Args:
#       magnitudes: 3-D Tensor of synthesizer parameters, of shape [batch, time,
#         n_filter_banks].
#     Returns:
#       controls: Dictionary of tensors of synthesizer controls.
#     """
#     # Scale the magnitudes.
#     if self.scale_fn is not None:
#       magnitudes = self.scale_fn(magnitudes + self.initial_bias)

#     return {'magnitudes': magnitudes}

#   def get_signal(self, magnitudes):
#     """Synthesize audio with filtered white noise.
#     Args:
#       magnitudes: Magnitudes tensor of shape [batch, n_frames, n_filter_banks].
#         Expects float32 that is strictly positive.
#     Returns:
#       signal: A tensor of harmonic waves of shape [batch, n_samples, 1].
#     """
#     batch_size = int(magnitudes.shape[0])
#     signal = torch.rand(batch_size, self.n_samples) * 2 - 1
#     return frequency_filter(signal, magnitudes, window_size=self.window_size)

class FilteredNoise(nn.Module):
    def __init__(self, frame_length = 64, attenuate_gain = 1e-2, device = 'cuda'):
        super(FilteredNoise, self).__init__()
        
        self.frame_length = frame_length
        self.device = device
        self.attenuate_gain = attenuate_gain
        
    def forward(self, z):
        """
        Compute linear-phase LTI-FVR (time-varient in terms of frame by frame) filter banks in batch from network output,
        and create time-varying filtered noise by overlap-add method.
        
        Argument:
            z['H'] : filter coefficient bank for each batch, which will be used for constructing linear-phase filter.
                - dimension : (batch_num, frame_num, filter_coeff_length)
        
        """
        
        batch_num, frame_num, filter_coeff_length = z.shape
        self.filter_window = nn.Parameter(torch.hann_window(filter_coeff_length * 2 - 1, dtype = torch.float32), requires_grad = False).to(self.device)
        
        INPUT_FILTER_COEFFICIENT = z
        
        # Desired linear-phase filter can be obtained by time-shifting a zero-phase form (especially to a causal form to be real-time),
        # which has zero imaginery part in the frequency response. 
        # Therefore, first we create a zero-phase filter in frequency domain.
        # Then, IDFT & make it causal form. length IDFT-ed signal size can be both even or odd, 
        # but we choose odd number such that a single sample can represent the center of impulse response.
        ZERO_PHASE_FR_BANK = INPUT_FILTER_COEFFICIENT.unsqueeze(-1).expand(batch_num, frame_num, filter_coeff_length, 2).contiguous()
        ZERO_PHASE_FR_BANK[..., 1] = 0
        ZERO_PHASE_FR_BANK = ZERO_PHASE_FR_BANK.view(-1, filter_coeff_length, 2)
        # zero_phase_ir_bank = fft.irfft(ZERO_PHASE_FR_BANK, 1, signal_sizes = (filter_coeff_length * 2 - 1,))
        zero_phase_ir_bank = fft.irfft(torch.view_as_complex(ZERO_PHASE_FR_BANK), n = (filter_coeff_length * 2 - 1))
           
        # Make linear phase causal impulse response & Hann-window it.
        # Then zero pad + DFT for linear convolution.
        linear_phase_ir_bank = zero_phase_ir_bank.roll(filter_coeff_length - 1, 1)
        windowed_linear_phase_ir_bank = linear_phase_ir_bank * self.filter_window.view(1, -1)
        zero_paded_windowed_linear_phase_ir_bank = nn.functional.pad(windowed_linear_phase_ir_bank, (0, self.frame_length - 1))
        ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK = fft.rfft(torch.view_as_complex(zero_paded_windowed_linear_phase_ir_bank))
        
        # Generate white noise & zero pad & DFT for linear convolution.
        noise = torch.rand(batch_num, frame_num, self.frame_length, dtype = torch.float32).view(-1, self.frame_length).to(self.device) * 2 - 1
        zero_paded_noise = nn.functional.pad(noise, (0, filter_coeff_length * 2 - 2))
        ZERO_PADED_NOISE = fft.rfft(torch.view_as_complex(zero_paded_noise))

        # Convolve & IDFT to make filtered noise frame, for each frame, noise band, and batch.
        FILTERED_NOISE = torch.zeros_like(ZERO_PADED_NOISE).to(self.device)
        FILTERED_NOISE[:, :, 0] = ZERO_PADED_NOISE[:, :, 0] * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :, 0] \
            - ZERO_PADED_NOISE[:, :, 1] * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :, 1]
        FILTERED_NOISE[:, :, 1] = ZERO_PADED_NOISE[:, :, 0] * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :, 1] \
            + ZERO_PADED_NOISE[:, :, 1] * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :, 0]
        filtered_noise = fft.irfft(torch.view_as_complex(FILTERED_NOISE)).view(batch_num, frame_num, -1) * self.attenuate_gain         
                
        # Overlap-add to build time-varying filtered noise.
        overlap_add_filter = torch.eye(filtered_noise.shape[-1], requires_grad = False).unsqueeze(1).to(self.device)
        output_signal = nn.functional.conv_transpose1d(filtered_noise.transpose(1, 2), 
                                                       overlap_add_filter, 
                                                       stride = self.frame_length, 
                                                       padding = 0).squeeze(1)
        
        return output_signal

class FIRNoiseSynth(nn.Module):
    def __init__(
        self, ir_length: int, hop_length: int, window_fn: Callable = torch.hann_window
    ):
        super().__init__()
        self.ir_length = ir_length
        self.hop_length = hop_length
        self.register_buffer("window", window_fn(ir_length))

    def forward(self, H_re):
        H_im = torch.zeros_like(H_re)
        H_z = torch.complex(H_re, H_im)

        h = torch.fft.irfft(H_z.transpose(1, 2))
        h = h.roll(self.ir_length // 2, -1)
        h = h * self.window.view(1, 1, -1)
        H = torch.fft.rfft(h)

        noise = torch.rand(self.hop_length * H_re.shape[-1] - 1, device=H_re.device)
        X = torch.stft(noise, self.ir_length, self.hop_length, return_complex=True)
        X = X.unsqueeze(0)
        Y = X * H.transpose(1, 2)
        y = torch.istft(Y, self.ir_length, self.hop_length, center=False)
        return y.unsqueeze(1)[:, :, : H_re.shape[-1] * self.hop_length]