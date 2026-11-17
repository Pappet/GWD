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

The layers (gwd.training.telescope_layers) compute in float32 and complex64, so gwd
train runs the telescope in float32 also with a GPU. They are registered with Keras
under the package "gwd", and gwd.registry.load_model imports their module, so a saved
telescope loads everywhere.
"""

from dataclasses import asdict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

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
from gwd.training.telescope_layers import (  # noqa: F401 (the constants are the API)
    CURVE_KNOTS,
    EVIDENCE_NODES,
    EVIDENCE_RHO_MAX,
    GAIN_NODES,
    BayesPooling,
    ChirpLens,
)

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
