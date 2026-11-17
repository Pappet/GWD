"""The chirp telescope (experimental): a model that focuses a chirp before it looks.

The CNN in gwd.training.model adds a chirp up as energy: before its Dense layer one
position sees 65 ms of the sample, and three MaxPooling layers drop the phase. A chirp
of two light black holes runs through seconds of the window, and only a sum that keeps
the phase over all of it reaches the optimal SNR. The telescope keeps the phase until
the signal is focused:

1. Spectrum: the rfft of the sample, the bins of TELESCOPE_BAND only.
2. Lens (ChirpLens): the spectrum times one kernel per channel, then one inverse FFT.
   A channel is a mother curve of one mass ratio (amplitude and phase over t / M,
   gwd.core.lens) stretched to one total mass. The result is the focal plane, complex,
   channels x merger times: a chirp becomes one bright pixel at its mass and time.
3. The modulus of every pixel, divided by its noise level: rho. It is the only
   nonlinearity up to here. Noise alone gives rho^2 a chi-squared distribution with 2
   degrees of freedom.
4. Pooling (BayesPooling): the log-sum-exp of evidence(rho) + bias(channel) over all
   pixels, a log Bayes factor. Its sigmoid is the score.

There is no ReLU, no pooling in time, and no convolution in the time domain. The
trainable weights are what the mother curves differ by from their start values (a
smooth curve on CURVE_KNOTS knots each), a smooth gain over the band, the evidence
curve, and the channel biases: about a thousand numbers. With start "imr" the lens
starts as a bank of PyCBC waveforms, a matched filter; with start "newton" it starts
as the leading-order inspiral formula, which training can then correct. gwd eval lens
shows what training changed.

The layers compute in float32 and complex64, so gwd train runs the telescope in float32
also with a GPU. They are registered with Keras under the package "gwd", and
gwd.registry.load_model imports this module, so a saved telescope loads everywhere.
"""

from dataclasses import asdict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, utils

from gwd.config import (
    DURATION,
    LEARNING_RATE,
    N_SAMPLES,
    SAMPLE_RATE,
    TELESCOPE_BAND,
    TELESCOPE_MASS_RATIOS,
    TELESCOPE_NOISE_SAMPLES,
    TELESCOPE_OVERLAP,
    WAVEFORM_BACKEND,
)
from gwd.core import lens as optics
from gwd.data.injection import make_sample, simulated_psd
from gwd.data.preprocessing import normalize
from gwd.data.scenarios import telescope_problem

# Stored in each model's .json.
ARCHITECTURE = "ChirpTelescope_V1"

# Where the mother curves start: PyCBC's IMRPhenomD, or the leading-order inspiral.
STARTS = ("imr", "newton")
# What gwd train changes: everything, the pooling and the band gain (the lens stays as
# it started), or the mother curves only.
TRAINABLE = ("all", "pooling", "lens")

# The first word of the spawn keys of the noise samples: 'gwdl' in ASCII. Disjoint from
# the streams of online training ('gwdo') and of the test sets ('gwdt').
LENS_STREAM = 0x6777646C
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


def require_scenario(scenario):
    """Raise ValueError unless the telescope can be built for scenario.

    gwd.data.scenarios.telescope_problem says why not: it needs a V2 or V3 scenario
    with a mass range. None stands for a dataset without a known scenario.
    """
    problem = telescope_problem(scenario)
    if problem:
        raise ValueError(problem)


def frame_of(scenario):
    """The Frame of the kernels for scenario: they fit its merger times into the sample."""
    return optics.Frame(
        sample_rate=SAMPLE_RATE,
        n_samples=N_SAMPLES,
        pre_s=float(scenario.t_merger_range[0]),
        post_s=float(DURATION - scenario.t_merger_range[1]),
        band=TELESCOPE_BAND,
    )


def total_mass_range(scenario, mass_ratio):
    """(lowest, highest) total mass of scenario at mass_ratio = m1 / m2 >= 1."""
    low, high = scenario.masses
    if mass_ratio < 1 or mass_ratio > high / low:
        raise ValueError(
            f"{scenario.name} draws masses from {low} to {high}, so its mass ratios "
            f"lie between 1 and {high / low:g}; got {mass_ratio}"
        )
    return low * (1.0 + mass_ratio), high * (1.0 + mass_ratio) / mass_ratio


def noise_power(scenario, frame, seed=0, n=TELESCOPE_NOISE_SAMPLES):
    """The mean power of n noise-only samples of scenario in the bins of the band.

    The samples go through make_sample and normalize like every model input. Sample i
    draws from RandomState(MT19937(SeedSequence(seed, spawn_key=(LENS_STREAM, i)))), a
    stream of its own.
    """
    low, high = frame.bins
    total = np.zeros(high - low)
    for index in range(n):
        sequence = np.random.SeedSequence(seed, spawn_key=(LENS_STREAM, index))
        rng = np.random.RandomState(np.random.MT19937(sequence))
        sample = make_sample(scenario, rng, has_signal=False)
        total += np.abs(np.fft.rfft(normalize(sample.data))[low:high]) ** 2
    return total / n


def whitening_of(scenario, frame):
    """1 / ASD of the simulated noise of scenario in the bins of the band, at most 1."""
    low, high = frame.bins
    psd = np.asarray(simulated_psd(scenario.simulated_noise, frame.n_samples))[low:high]
    if np.any(psd <= 0):
        raise ValueError(
            f"The noise of {scenario.name} has no power in parts of the band {frame.band}"
        )
    return np.sqrt(psd.min() / psd)


def start_curves(start, mass_ratios, grid, backend=WAVEFORM_BACKEND):
    """(amplitude, phase), each mother curves x nodes, for start in STARTS."""
    if start not in STARTS:
        raise ValueError(f"start must be one of {STARTS}, got {start!r}")
    if start == "newton":
        curves = [optics.newtonian_curve(ratio, grid) for ratio in mass_ratios]
    else:
        curves = [
            optics.mother_curve(ratio, grid, backend=backend) for ratio in mass_ratios
        ]
    return np.array([c[0] for c in curves]), np.array([c[1] for c in curves])


def build_telescope(
    scenario,
    learning_rate=LEARNING_RATE,
    *,
    seed=0,
    start="imr",
    trainable="all",
    mass_ratios=TELESCOPE_MASS_RATIOS,
    overlap=TELESCOPE_OVERLAP,
    noise_samples=TELESCOPE_NOISE_SAMPLES,
    backend=WAVEFORM_BACKEND,
):
    """Build and compile the chirp telescope for the samples of scenario.

    Args:
        scenario: The Scenario of the training data (V2 or V3 with a mass range). It
            sets the masses and merger times the lens covers, the noise, and the SNR
            range of the evidence curve.
        learning_rate: Adam learning rate. Default: LEARNING_RATE in gwd.config.
        seed: Seeds the noise-only samples the noise power is measured from.
        start: "imr" starts the mother curves from PyCBC's IMRPhenomD, "newton" from
            the leading-order inspiral formula (no PyCBC waveform).
        trainable: "all", "pooling" (evidence, biases, and band gain; the mother curves
            stay as they start), or "lens" (the mother curves only).
        mass_ratios: One mother curve per mass ratio m1 / m2.
        overlap: Overlap of neighbouring channels of one mother curve.
        noise_samples: Noise-only samples drawn for the noise power.

    Returns:
        The compiled Keras model: input (N_SAMPLES, 1), output one score in [0, 1].
    """
    require_scenario(scenario)
    if trainable not in TRAINABLE:
        raise ValueError(f"trainable must be one of {TRAINABLE}, got {trainable!r}")
    frame = frame_of(scenario)
    ranges = [total_mass_range(scenario, ratio) for ratio in mass_ratios]
    lightest = min(low for low, _ in ranges)
    grid = optics.Grid(u_min=-frame.pre_s / lightest, u_max=frame.post_s / lightest)
    whitening = whitening_of(scenario, frame)
    amplitude, phase = start_curves(start, mass_ratios, grid, backend)
    ladders = [
        optics.mass_ladder(a, p, grid, low, high, overlap, frame, weight=whitening)
        for a, p, (low, high) in zip(amplitude, phase, ranges)
    ]

    lens = ChirpLens(
        ladders=[masses.tolist() for masses in ladders],
        mass_ratios=list(mass_ratios),
        grid=asdict(grid),
        frame=asdict(frame),
        window=scenario.t_merger_range,
        train_curves=trainable in ("all", "lens"),
        train_gain=trainable in ("all", "pooling"),
        name="lens",
    )
    pooling = BayesPooling(name="pooling", trainable=trainable in ("all", "pooling"))
    inputs = layers.Input(shape=(N_SAMPLES, 1))
    logit = pooling(lens(inputs))
    # The output is always float32, also under mixed_float16.
    score = layers.Activation("sigmoid", dtype="float32", name="score")(logit)
    model = models.Model(inputs, score, name="chirp_telescope")

    lens.set_start(amplitude, phase)
    lens.set_noise(noise_power(scenario, frame, seed, noise_samples), whitening)
    pooling.set_start(
        optics.pixel_evidence(pooling.rho_nodes(), scenario.snr_range),
        optics.channel_log_prior(ladders),
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
        # Keras would decide by itself whether to compile with XLA; the telescope was
        # only run without it.
        jit_compile=False,
    )
    return model
