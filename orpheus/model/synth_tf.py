import tensorflow as tf
import math
import numpy as np

tfkl = tf.keras.layers

def tf_float32(x):
  """Ensure array/tensor is a float32 tf.Tensor."""
  if isinstance(x, tf.Tensor):
    return tf.cast(x, dtype=tf.float32)  # This is a no-op if x is float32.
  else:
    return tf.convert_to_tensor(x, tf.float32)

def safe_divide(numerator, denominator, eps=1e-7):
  """Avoid dividing by zero by adding a small epsilon."""
  safe_denominator = tf.where(denominator == 0.0, eps, denominator)
  return numerator / safe_denominator

def safe_log(x, eps=1e-5):
  """Avoid taking the log of a non-positive number."""
  safe_x = tf.where(x <= 0.0, eps, x)
  return tf.math.log(safe_x)

def logb(x, base=2.0, eps=1e-5):
  """Logarithm with base as an argument."""
  return safe_divide(safe_log(x, eps), safe_log(base, eps), eps)

def remove_above_nyquist(frequency_envelopes: tf.Tensor,
                         amplitude_envelopes: tf.Tensor,
                         sample_rate: int = 16000) -> tf.Tensor:
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
  frequency_envelopes = tf_float32(frequency_envelopes)
  amplitude_envelopes = tf_float32(amplitude_envelopes)

  amplitude_envelopes = tf.where(
      tf.greater_equal(frequency_envelopes, sample_rate / 2.0),
      tf.zeros_like(amplitude_envelopes), amplitude_envelopes)
  return amplitude_envelopes

def unit_to_midi(unit,
                 midi_min = 20.0,
                 midi_max = 90.0,
                 clip: bool = False):
  """Map the unit interval [0, 1] to MIDI notes."""
  unit = tf.clip_by_value(unit, 0.0, 1.0) if clip else unit
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
  notes = tf_float32(notes)
  hz = 440.0 * (2.0 ** ((notes - 69.0) / 12.0))
  # Map MIDI 0 as 0 hz when MIDI 0 is silence.
  if midi_zero_silence:
    hz = tf.where(tf.equal(notes, 0.0), 0.0, hz)
  return hz

def hz_to_midi(frequencies):
  """TF-compatible hz_to_midi function."""
  frequencies = tf_float32(frequencies)
  notes = 12.0 * (logb(frequencies, 2.0) - logb(440.0, 2.0)) + 69.0
  # Map 0 Hz to MIDI 0 (Replace -inf MIDI with 0.)
  notes = tf.where(tf.less_equal(frequencies, 0.0), 0.0, notes)
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

def _add_depth_axis(freqs: tf.Tensor, depth: int = 1) -> tf.Tensor:
  """Turns [batch, time, sinusoids*depth] to [batch, time, sinusoids, depth]."""
  freqs = freqs[..., tf.newaxis]
  # Unpack sinusoids dimension.
  n_batch, n_time, n_combined, _ = freqs.shape
  n_sinusoids = int(n_combined) // depth
  return tf.reshape(freqs, [n_batch, n_time, n_sinusoids, depth])

def frequencies_sigmoid(freqs: tf.Tensor,
                        depth: int = 1,
                        hz_min: float = 0.0,
                        hz_max: float = 8000.0) -> tf.Tensor:
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
  f_probs = tf.nn.sigmoid(freqs)

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

  return tf.reduce_sum(tf.stack(hz_scales, axis=-1), axis=-1)

def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
  """Exponentiated Sigmoid pointwise nonlinearity.
  Bounds input to [threshold, max_value] with slope given by exponent.
  Args:
    x: Input tensor.
    exponent: In nonlinear regime (away from x=0), the output varies by this
      factor for every change of x by 1.0.
    max_value: Limiting value at x=inf.
    threshold: Limiting value at x=-inf. Stablizes training when outputs are
      pushed to 0.
  Returns:
    A tensor with pointwise nonlinearity applied.
  """
  x = tf_float32(x)
  return max_value * tf.nn.sigmoid(x)**tf.math.log(exponent) + threshold


def resample(inputs: tf.Tensor,
             n_timesteps: int,
             method = 'linear',
             add_endpoint: bool = True) -> tf.Tensor:
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

  inputs = tf_float32(inputs)
  is_1d = len(inputs.shape) == 1
  is_2d = len(inputs.shape) == 2
  is_4d = len(inputs.shape) == 4

  # print(inputs.shape)

  # Ensure inputs are at least 3d.
  if is_1d:
    inputs = inputs[tf.newaxis, :, tf.newaxis]
  elif is_2d:
    inputs = inputs[:, :, tf.newaxis]

  def _image_resize(method):
    """Closure around tf.image.resize."""
    # Image resize needs 4-D input. Add/remove extra axis if not 4-D.
    outputs = inputs[:, :, tf.newaxis, :] if not is_4d else inputs
    outputs = tf.compat.v1.image.resize(outputs,
                                        [n_timesteps, outputs.shape[2]],
                                        method=method,
                                        align_corners=not add_endpoint)
    return outputs[:, :, 0, :] if not is_4d else outputs

  # Perform resampling.
  if method == 'nearest':
    outputs = _image_resize(tf.compat.v1.image.ResizeMethod.NEAREST_NEIGHBOR)
  elif method == 'linear':
    outputs = _image_resize(tf.compat.v1.image.ResizeMethod.BILINEAR)
  elif method == 'cubic':
    outputs = _image_resize(tf.compat.v1.image.ResizeMethod.BICUBIC)
  elif method == 'window':
    outputs = upsample_with_windows(inputs, n_timesteps, add_endpoint)
  else:
    raise ValueError('Method ({}) is invalid. Must be one of {}.'.format(
        method, "['nearest', 'linear', 'cubic', 'window']"))

  # print(outputs.shape)

  return outputs

  # Return outputs to the same dimensionality of the inputs.
  if is_1d:
    outputs = outputs[0, :, 0]
  elif is_2d:
    outputs = outputs[:, :, 0]

  return outputs


def upsample_with_windows(inputs: tf.Tensor,
                          n_timesteps: int,
                          add_endpoint: bool = True) -> tf.Tensor:
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
    inputs = tf.concat([inputs, inputs[:, -1:, :]], axis=1)

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
  window = tf.signal.hann_window(window_length)  # [window]

  # Transpose for overlap_and_add.
  x = tf.transpose(inputs, perm=[0, 2, 1])  # [batch_size, n_channels, n_frames]

  # Broadcast multiply.
  # Add dimension for windows [batch_size, n_channels, n_frames, window].
  x = x[:, :, :, tf.newaxis]
  window = window[tf.newaxis, tf.newaxis, tf.newaxis, :]
  x_windowed = (x * window)
  # print(x_windowed.shape, hop_size)
  x = tf.signal.overlap_and_add(x_windowed, hop_size)

  # Transpose back.
  x = tf.transpose(x, perm=[0, 2, 1])  # [batch_size, n_timesteps, n_channels]

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
  return tf.pad(x, paddings, **pad_kwargs)

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
  chunks = tf.reshape(angular_frequency,
                      [n_batch, n_chunks, chunk_size] + [-1] * n_ch_dims)
  phase = tf.cumsum(chunks, axis=2)

  # Add offsets.
  # Offset of the next row is the last entry of the previous row.
  offsets = phase[:, :, -1:, ...] % (2.0 * np.pi)
  offsets = pad_axis(offsets, [1, 0], axis=1)
  offsets = offsets[:, :-1, ...]

  # Offset is cumulative among the rows.
  offsets = tf.cumsum(offsets, axis=1) % (2.0 * np.pi)
  phase = phase + offsets

  # Put back in original shape.
  phase = phase % (2.0 * np.pi)
  phase = tf.reshape(phase, [n_batch, length] + [-1] * n_ch_dims)

  # Remove padding if added it.
  if remainder:
    phase = phase[:, :n_time]
  return phase


def oscillator_bank(frequency_envelopes: tf.Tensor,
                    amplitude_envelopes: tf.Tensor,
                    sample_rate: int = 16000,
                    sum_sinusoids: bool = True,
                    use_angular_cumsum: bool = False) -> tf.Tensor:
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

  # print(amplitude_envelopes.shape, frequency_envelopes.shape)

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
    phases = tf.cumsum(omegas, axis=1)

  # Convert to waveforms.
  wavs = tf.sin(phases)
  audio = amplitude_envelopes * wavs  # [mb, n_samples, n_sinusoids]
  if sum_sinusoids:
    audio = tf.reduce_sum(audio, axis=-1)  # [mb, n_samples]
  return audio

class Processor(tfkl.Layer):
  """Abstract base class for signal processors.
  Since most effects / synths require specificly formatted control signals
  (such as amplitudes and frequenices), each processor implements a
  get_controls(inputs) method, where inputs are a variable number of tensor
  arguments that are typically neural network outputs. Check each child class
  for the class-specific arguments it expects. This gives a dictionary of
  controls that can then be passed to get_signal(controls). The
  get_outputs(inputs) method calls both in succession and returns a nested
  output dictionary with all controls and signals.
  """

  def __init__(self, name, trainable: bool = False):
    super().__init__(name=name, trainable=trainable, autocast=False)

  def call(self,
           *args: tf.Tensor,
           return_outputs_dict: bool = False,
           **kwargs) -> tf.Tensor:
    """Convert input tensors arguments into a signal tensor."""
    # Don't use `training` or `mask` arguments from keras.Layer.
    for k in ['training', 'mask']:
      if k in kwargs:
        _ = kwargs.pop(k)

    controls = self.get_controls(*args, **kwargs)
    signal = self.get_signal(**controls)
    if return_outputs_dict:
      return dict(signal=signal, controls=controls)
    else:
      return signal

  def get_controls(self, *args: tf.Tensor, **kwargs: tf.Tensor):
    """Convert input tensor arguments into a dict of processor controls."""
    raise NotImplementedError

  def get_signal(self, *args: tf.Tensor, **kwargs: tf.Tensor) -> tf.Tensor:
    """Convert control tensors into a signal tensor."""
    raise NotImplementedError

class Sinusoidal(Processor):
  """Synthesize audio with a bank of arbitrary sinusoidal oscillators."""

  def __init__(self,
               n_samples=64000,
               sample_rate=16000,
               amp_scale_fn=exp_sigmoid,
               amp_resample_method='window',
               freq_scale_fn=frequencies_sigmoid,
               name='sinusoidal'):
    super().__init__(name=name)
    self.n_samples = n_samples
    self.sample_rate = sample_rate
    self.amp_scale_fn = amp_scale_fn
    self.amp_resample_method = amp_resample_method
    self.freq_scale_fn = freq_scale_fn

  def get_controls(self, amplitudes, frequencies):
    """Convert network output tensors into a dictionary of synthesizer controls.
    Args:
      amplitudes: 3-D Tensor of synthesizer controls, of shape
        [batch, time, n_sinusoids].
      frequencies: 3-D Tensor of synthesizer controls, of shape
        [batch, time, n_sinusoids]. Expects strictly positive in Hertz.
    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    # Scale the inputs.
    if self.amp_scale_fn is not None:
      amplitudes = self.amp_scale_fn(amplitudes)

    if self.freq_scale_fn is not None:
      frequencies = self.freq_scale_fn(tf_float32(frequencies))
      amplitudes = remove_above_nyquist(frequencies, amplitudes, self.sample_rate)

    return {'amplitudes': amplitudes,
            'frequencies': frequencies}

  def get_signal(self, amplitudes, frequencies):
    """Synthesize audio with sinusoidal synthesizer from controls.
    Args:
      amplitudes: Amplitude tensor of shape [batch, n_frames, n_sinusoids].
        Expects float32 that is strictly positive.
      frequencies: Tensor of shape [batch, n_frames, n_sinusoids].
        Expects float32 in Hertz that is strictly positive.
    Returns:
      signal: A tensor of harmonic waves of shape [batch, n_samples].
    """
    # Create sample-wise envelopes.
    # amplitude_envelopes = resample(amplitudes, self.n_samples,
    #                                     method=self.amp_resample_method)
    # frequency_envelopes = resample(frequencies, self.n_samples)

    signal = oscillator_bank(frequency_envelopes=frequencies,
                                  amplitude_envelopes=amplitudes,
                                  sample_rate=self.sample_rate)

    return signal
