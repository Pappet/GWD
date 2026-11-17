"""The Keras layers of the chirp telescope: ChirpLens and BayesPooling.

gwd.training.telescope builds the model from them and explains it. They live in a
module of their own that needs only TensorFlow, numpy, and gwd.core.lens, because
gwd.registry.load_model imports this module to register the layers with Keras (package
"gwd"): without it a saved telescope could not be loaded, and with all of
gwd.training.telescope the registry would pull in gwd.data as well.

The layers compute in float32 and complex64 under every precision policy.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, utils

from gwd.core import lens as optics

# What training adds to a mother curve is a smooth curve of its own: this many knots,
# evenly spread over the nodes of the Grid and interpolated linearly between them.
CURVE_KNOTS = 64
# Nodes of the gain over the band, and of the evidence curve from rho 0 to its maximum.
GAIN_NODES = 64
EVIDENCE_NODES = 33
EVIDENCE_RHO_MAX = 32.0


def _interpolate(values, position, last, offset=0):
    """values between its nodes at position (in nodes), linear.

    Nodes 0 ... last; beyond node last the last segment is continued. offset (whole
    nodes) is added to the node index, to read one row of a flattened table.
    """
    index = tf.clip_by_value(tf.floor(position), 0.0, last - 1.0)
    fraction = position - index
    index = tf.cast(index, tf.int32) + offset
    low = tf.gather(values, index)
    return low + fraction * (tf.gather(values, index + 1) - low)


@utils.register_keras_serializable(package="gwd")
class ChirpLens(layers.Layer):
    """A sample to its focal plane: rho, (batch, channels, merger times).

    ladders: one list of total masses (solar masses) per mother curve; every mass is a
        channel, in this order.
    mass_ratios: the mass ratio m1 / m2 of each mother curve.
    grid, frame: the fields of gwd.core.lens.Grid and Frame.
    window: the merger times kept, (first, last) in seconds from the sample's start.
    train_curves, train_gain: whether the mother curves and the band gain are trained.

    Weights: amplitude_start and phase_start (mother curves x nodes, fixed) hold the
    start curves; amplitude_delta and phase_delta (mother curves x curve_knots) are
    what training adds to them, on knots spread evenly over the nodes (added());
    gain (gain_nodes values, a log factor interpolated over the band); noise_power and
    whitening (per bin, fixed): the mean power of noise-only samples and 1 / ASD of the
    noise the signals were injected into. set_start() and set_noise() fill them.
    """

    def __init__(
        self,
        ladders,
        mass_ratios,
        grid,
        frame,
        window,
        curve_knots=CURVE_KNOTS,
        gain_nodes=GAIN_NODES,
        train_curves=True,
        train_gain=True,
        **kwargs,
    ):
        kwargs.setdefault("dtype", "float32")
        super().__init__(**kwargs)
        self.ladders = [[float(mass) for mass in masses] for masses in ladders]
        self.mass_ratios = [float(ratio) for ratio in mass_ratios]
        self.grid = dict(grid)
        self.frame = {**frame, "band": [float(edge) for edge in frame["band"]]}
        self.window = [float(window[0]), float(window[1])]
        self.curve_knots = int(curve_knots)
        self.gain_nodes = int(gain_nodes)
        self.train_curves = bool(train_curves)
        self.train_gain = bool(train_gain)
        self.optics_grid = optics.Grid(**self.grid)
        self.optics_frame = optics.Frame(
            **{**self.frame, "band": tuple(self.frame["band"])}
        )
        self.channels = sum(len(masses) for masses in self.ladders)

    def build(self, input_shape):
        low, high = self.optics_frame.bins
        curves = (len(self.ladders), self.optics_grid.nodes)
        knots = (len(self.ladders), self.curve_knots)

        def weight(name, shape, value, trainable):
            return self.add_weight(
                name=name, shape=shape, initializer=value, trainable=trainable
            )

        self.amplitude_start = weight("amplitude_start", curves, "zeros", False)
        self.phase_start = weight("phase_start", curves, "zeros", False)
        self.amplitude_delta = weight(
            "amplitude_delta", knots, "zeros", self.train_curves
        )
        self.phase_delta = weight("phase_delta", knots, "zeros", self.train_curves)
        self.gain = weight("gain", (self.gain_nodes,), "zeros", self.train_gain)
        self.noise_power = weight("noise_power", (high - low,), "ones", False)
        self.whitening = weight("whitening", (high - low,), "ones", False)
        super().build(input_shape)

    def set_start(self, amplitude, phase):
        """Set the start curves (mother curves x nodes) and forget what was learned."""
        self.amplitude_start.assign(np.asarray(amplitude, dtype=np.float32))
        self.phase_start.assign(np.asarray(phase, dtype=np.float32))
        self.amplitude_delta.assign(tf.zeros_like(self.amplitude_delta))
        self.phase_delta.assign(tf.zeros_like(self.phase_delta))

    def set_noise(self, noise_power, whitening):
        """Set the noise power and 1 / ASD in the band's bins."""
        self.noise_power.assign(np.asarray(noise_power, dtype=np.float32))
        self.whitening.assign(np.asarray(whitening, dtype=np.float32))

    def added(self):
        """What training added to the curves, on the nodes: (amplitude, phase).

        Each is mother curves x nodes: the knots of amplitude_delta and phase_delta,
        interpolated linearly over the node index.
        """
        nodes, knots = self.optics_grid.nodes, self.curve_knots
        position = np.linspace(0.0, knots - 1.0, nodes).astype(np.float32)[None, :]
        row = (np.arange(len(self.ladders)) * knots).astype(np.int32)[:, None]

        def on_nodes(delta):
            return _interpolate(tf.reshape(delta, [-1]), position, knots - 1.0, row)

        return on_nodes(self.amplitude_delta), on_nodes(self.phase_delta)

    def kernels(self):
        """The spectrum of every channel's kernel in the band: complex64 (channels, bins).

        gwd.core.lens.kernel_spectra with the current curves, in TensorFlow.
        """
        grid, frame = self.optics_grid, self.optics_frame
        masses = np.concatenate(self.ladders).astype(np.float32)
        mother = np.repeat(
            np.arange(len(self.ladders)), [len(masses) for masses in self.ladders]
        )
        warp_low = np.arcsinh(grid.u_min / grid.scale)
        warp_high = np.arcsinh(grid.u_max / grid.scale)

        u = tf.constant(frame.times, tf.float32)[None, :] / masses[:, None]
        position = (tf.math.asinh(u / grid.scale) - warp_low) * (
            (grid.nodes - 1) / (warp_high - warp_low)
        )
        inside = (position >= 0.0) & (position <= grid.nodes - 1.0)
        position = tf.clip_by_value(position, 0.0, grid.nodes - 1.0)
        # Each mother curve is one row of the tables, which are read flattened.
        row = (mother * grid.nodes).astype(np.int32)[:, None]

        added_amplitude, added_phase = self.added()
        amplitude = tf.reshape(self.amplitude_start + added_amplitude, [-1])
        phase = tf.reshape(self.phase_start + added_phase, [-1])
        a = _interpolate(amplitude, position, grid.nodes - 1.0, row)
        p = _interpolate(phase, position, grid.nodes - 1.0, row)
        a = a * tf.cast(inside, tf.float32) * frame.fade.astype(np.float32)[None, :]
        kernel = tf.complex(a * tf.cos(p), a * tf.sin(p))
        kernel = tf.pad(kernel, [[0, 0], [0, frame.n_samples - frame.length]])
        low, high = frame.bins
        return tf.signal.fft(kernel)[:, low:high] * frame.shift.astype(np.complex64)

    def band_weight(self):
        """The real factor on every bin of the band: whitening * gain / sqrt(noise power).

        A signal went through the same filters as the noise, so the matched filter for
        it divides the whitened kernel by the root of the noise power (the notches and
        the edges of the band-pass). The trainable gain starts at 1.
        """
        low, high = self.optics_frame.bins
        position = np.linspace(0.0, self.gain_nodes - 1.0, high - low)
        gain = _interpolate(
            self.gain, position.astype(np.float32), self.gain_nodes - 1.0
        )
        return self.whitening * tf.exp(gain) / tf.sqrt(self.noise_power)

    def call(self, inputs):
        frame = self.optics_frame
        low, high = frame.bins
        first, stop = frame.window(*self.window)
        samples = tf.cast(tf.reshape(inputs, [-1, frame.n_samples]), tf.float32)
        data = tf.signal.rfft(samples)[:, low:high]
        kernels = self.kernels()
        weight = self.band_weight()
        energy = tf.math.real(kernels) ** 2 + tf.math.imag(kernels) ** 2
        # The variance of a pixel in noise; rho^2 = 2 |pixel|^2 / variance.
        variance = tf.reduce_sum(self.noise_power * weight**2 * energy, axis=1)
        # tf.complex and not tf.cast: the gradient of a cast to complex64 warns that it
        # drops an imaginary part.
        filters = tf.math.conj(kernels) * tf.complex(weight, tf.zeros_like(weight))
        spectrum = data[:, None, :] * filters[None, :, :]
        spectrum = tf.pad(spectrum, [[0, 0], [0, 0], [low, frame.focal_samples - high]])
        focal = tf.signal.ifft(spectrum)[:, :, first:stop] * frame.focal_samples
        power = tf.math.real(focal) ** 2 + tf.math.imag(focal) ** 2
        return tf.sqrt(2.0 * power / variance[None, :, None] + 1e-12)

    def compute_output_shape(self, input_shape):
        first, stop = self.optics_frame.window(*self.window)
        return (input_shape[0], self.channels, stop - first)

    def get_config(self):
        config = super().get_config()
        config.update(
            ladders=self.ladders,
            mass_ratios=self.mass_ratios,
            grid=self.grid,
            frame=self.frame,
            window=self.window,
            curve_knots=self.curve_knots,
            gain_nodes=self.gain_nodes,
            train_curves=self.train_curves,
            train_gain=self.train_gain,
        )
        return config


@utils.register_keras_serializable(package="gwd")
class BayesPooling(layers.Layer):
    """The focal plane to one logit: a log Bayes factor of signal against noise.

    logit = log sum over pixels of exp(evidence(rho) + bias[channel]) - log(times).
    evidence is a curve over rho, linear between nodes values from 0 to rho_max (and on
    beyond it); bias is one value per channel. With evidence = the log likelihood ratio
    of one pixel (gwd.core.lens.pixel_evidence) and bias = the log prior of the channel
    (channel_log_prior), the logit is the log Bayes factor for a signal at one unknown
    pixel, uniform over the merger times. set_start() fills both.
    """

    def __init__(self, nodes=EVIDENCE_NODES, rho_max=EVIDENCE_RHO_MAX, **kwargs):
        kwargs.setdefault("dtype", "float32")
        super().__init__(**kwargs)
        self.nodes = int(nodes)
        self.rho_max = float(rho_max)

    def build(self, input_shape):
        self.evidence = self.add_weight(
            name="evidence", shape=(self.nodes,), initializer="zeros"
        )
        self.bias = self.add_weight(
            name="bias", shape=(input_shape[1],), initializer="zeros"
        )
        super().build(input_shape)

    def rho_nodes(self):
        """The rho values of the evidence nodes."""
        return np.linspace(0.0, self.rho_max, self.nodes)

    def set_start(self, evidence, bias):
        self.evidence.assign(np.asarray(evidence, dtype=np.float32))
        self.bias.assign(np.asarray(bias, dtype=np.float32))

    def call(self, rho):
        position = rho * ((self.nodes - 1) / self.rho_max)
        evidence = _interpolate(self.evidence, position, self.nodes - 1.0)
        logit = tf.reduce_logsumexp(evidence + self.bias[None, :, None], axis=[1, 2])
        times = tf.cast(tf.shape(rho)[2], tf.float32)
        return (logit - tf.math.log(times))[:, None]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    def get_config(self):
        config = super().get_config()
        config.update(nodes=self.nodes, rho_max=self.rho_max)
        return config
