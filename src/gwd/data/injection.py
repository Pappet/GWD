"""The one signal injection for training, evaluation, and the apps (P3.5).

make_sample draws a sample from a Scenario: noise (real or white), and for a signal
sample a chirp scaled to a peak SNR. The draw order is fixed, so a seed gives the same
samples on every machine:

1. the label, unless the caller passes has_signal;
2. one number that picks real or white noise, if the scenario uses real noise;
3. the noise: a real chunk (index, then crop offset if needed) or white noise;
4. the parameters (draw_params): masses and distance (random masses only, see
   gwd.core.population.random_bbh_params), then the merger time, then the SNR unless
   the caller fixes it.

For TRAIN_V1 this was the order gwd.data.generate used before P3.5, except that until
P5.3 two spins and an inclination were drawn after the distance and dropped. The other
callers drew the merger time and SNR, or label and noise, in another order.

rng is anything with the numpy RandomState methods (rand, uniform, randint, normal).
The default is np.random, the global RNG, as before. It is passed on to gwd.core
(white noise, random masses), which has no RNG of its own since P5.3.

The chirps come from the backend gwd.config.WAVEFORM_BACKEND ('pycbc'). Without PyCBC,
make_sample raises gwd.core.waveforms.WaveformBackendError (P5.2); there is no fallback
to the Newtonian approximation. The backend argument exists so the golden values and
tests can draw numpy-only Newton chirps on purpose.

Every signal sample also gets its optimal (matched filter) SNR, measured after the
scaling and without drawing from rng (P5.5, measure_optimal_snr). The peak SNR still
drives V1; the optimal SNR is recorded so the difference can be seen.
"""

import glob
import os
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from gwd.config import (
    BANDPASS,
    DATA_DIR_NOISE,
    DURATION,
    N_SAMPLES,
    SAMPLE_RATE,
    WAVEFORM_BACKEND,
)
from gwd.core import noise as core_noise
from gwd.core.population import random_bbh_params
from gwd.core.snr import optimal_snr_welch, optimal_snr_white, scale_signal_to_snr
from gwd.core.timebase import time_axis
from gwd.core.waveforms import generate_chirp
from gwd.data.preprocessing import normalize

# Distance of the fixed-mass scenarios: the default of generate_chirp. Random masses
# draw it uniform in volume (gwd.core.population). Either way it does not change the
# sample, because the chirp is normalized to peak 1 before scaling (metadata only).
FIXED_DISTANCE_MPC = 400.0
# Standard deviation of the white noise of every scenario.
WHITE_NOISE_SIGMA = 1.0


@dataclass(frozen=True)
class InjectionParams:
    mass1: float
    mass2: float
    distance: float
    t_merger: float
    snr: float


@dataclass(frozen=True)
class Sample:
    """data = noise + signal. signal and params are None for a noise-only sample.

    snr is the peak SNR the signal was scaled to (also as peak_snr), optimal_snr the
    optimal SNR measured by measure_optimal_snr; both are 0.0 without signal.
    """

    data: np.ndarray
    noise: np.ndarray
    signal: Optional[np.ndarray]
    has_signal: bool
    params: Optional[InjectionParams]
    snr: float  # the peak SNR the signal was scaled to; 0.0 without signal
    noise_kind: str  # "real" or "white"
    noise_file: Optional[str] = None  # file name of the real chunk, None for white
    waveform_backend: Optional[str] = None  # "pycbc" or "newton"; None without signal
    optimal_snr: float = 0.0  # measured, see measure_optimal_snr; 0.0 without signal

    @property
    def peak_snr(self):
        """The peak SNR (max|signal| over the noise std): the same value as snr."""
        return self.snr


def _rng(rng):
    return np.random if rng is None else rng


def white_noise(length, rng=None):
    """White noise of std WHITE_NOISE_SIGMA (1) from gwd.core.noise.white_noise."""
    return core_noise.white_noise(length, WHITE_NOISE_SIGMA, rng=_rng(rng))


def measure_optimal_snr(signal, noise, noise_kind):
    """The optimal SNR of signal in the noise of a V1 sample (P5.5).

    White noise: analytically, optimal_snr_white(signal, WHITE_NOISE_SIGMA) over the
    full band; the std is known, so it is not estimated from the noise.
    Real noise: optimal_snr_welch against the Welch PSD of the chunk the sample uses,
    restricted to BANDPASS (20-300 Hz). The chunk is band-passed but the V1 signal is
    not, so above the band its PSD is near 0 and the SNR would explode
    (evaluation:MISS-2). Both are measured on the scaled signal before normalization.
    """
    if noise_kind == "white":
        return optimal_snr_white(signal, WHITE_NOISE_SIGMA)
    return optimal_snr_welch(signal, noise, SAMPLE_RATE, *BANDPASS)


def real_noise_files(folder=DATA_DIR_NOISE):
    """The noise chunks in folder, sorted so a seed picks the same chunk everywhere."""
    return sorted(glob.glob(os.path.join(folder, "*.npy")))


def load_random_real_noise(length, files, rng=None):
    """Load a random chunk from files, crop it to length, and normalize it.

    The array of draw_real_noise(), without the file name.
    """
    return draw_real_noise(length, files, rng)[0]


def draw_real_noise(length, files, rng=None):
    """Return (noise, file name) for a random chunk from files, cropped and normalized.

    Falls back to white noise if files is empty, the chunk is too short, or it cannot
    be read; the file name is then None, and a chunk that is too short or unreadable
    warns. normalize keeps its default eps: on raw strain (std near 1e-21) the eps
    dominates, so the chunk ends up with a std near 1e-11, not 1. This is the V1
    behaviour, kept as is (see the P3.2 PR and docs/architecture.md).
    """
    rng = _rng(rng)
    if not files:
        return white_noise(length, rng), None

    path = files[rng.randint(len(files))]
    name = os.path.basename(path)
    try:
        data = np.load(path)
        if len(data) != length:
            if len(data) > length:
                start = rng.randint(0, len(data) - length)
                data = data[start : start + length]
            else:
                warnings.warn(
                    f"Noise chunk {name} has {len(data)} samples, fewer than "
                    f"{length}; using white noise instead.",
                    stacklevel=2,
                )
                return white_noise(length, rng), None
        return normalize(data), name
    except Exception as error:
        warnings.warn(
            f"Cannot read noise chunk {name} ({error}); using white noise instead.",
            stacklevel=2,
        )
        return white_noise(length, rng), None


def real_noise_provider(folder=DATA_DIR_NOISE):
    """A noise_provider for make_sample, or None if folder has no chunks.

    The provider returns (noise, file name), see draw_real_noise(). Its attribute
    files lists the chunks it draws from.
    """
    files = real_noise_files(folder)
    if not files:
        return None

    def provider(length, rng=None):
        return draw_real_noise(length, files, rng)

    provider.files = files
    return provider


def draw_params(scenario, rng=None, snr=None):
    """Draw the signal parameters of one sample from scenario.

    The SNR is drawn last; pass snr to fix it (for example for a sensitivity curve).
    """
    rng = _rng(rng)
    if scenario.mass_mode == "random":
        binary = random_bbh_params(rng, scenario.masses)
        m1, m2, distance = binary["mass1"], binary["mass2"], binary["distance"]
    else:
        m1, m2 = scenario.masses
        distance = FIXED_DISTANCE_MPC

    t_merger = rng.uniform(*scenario.t_merger_range)
    if snr is None:
        snr = rng.uniform(*scenario.snr_range)
    return InjectionParams(
        mass1=m1, mass2=m2, distance=distance, t_merger=t_merger, snr=snr
    )


def make_sample(
    scenario,
    rng=None,
    has_signal=None,
    snr=None,
    noise_provider=None,
    backend=WAVEFORM_BACKEND,
):
    """Draw one sample of N_SAMPLES points from scenario (draw order in the module doc).

    has_signal: None draws the label (50/50); True or False sets it.
    snr: fixes the SNR of a signal sample instead of drawing it.
    noise_provider: callable (length, rng) -> (real noise, file name), see
        real_noise_provider. A file name of None means the provider fell back to white
        noise. None means no real noise, so every sample gets white noise.
    backend: the waveform backend of the chirp; the pipeline keeps the default.
    """
    if scenario.condition_signal:
        raise NotImplementedError("Conditioning the signal comes with V2 (P6.2)")
    rng = _rng(rng)
    time = time_axis(SAMPLE_RATE, DURATION)

    if has_signal is None:
        has_signal = bool(rng.rand() > 0.5)

    use_real = False
    if scenario.real_noise_fraction > 0:
        # Drawn even without chunks, as in V1, so the random stream does not depend
        # on whether real noise exists.
        dice = rng.rand()
        use_real = noise_provider is not None and dice < scenario.real_noise_fraction
    noise_file = None
    if use_real:
        noise, noise_file = noise_provider(len(time), rng)
    else:
        noise = white_noise(len(time), rng)
    # A provider that fell back to white noise gives no file name.
    noise_kind = "white" if noise_file is None else "real"

    if not has_signal:
        return Sample(noise, noise, None, False, None, 0.0, noise_kind, noise_file)

    params = draw_params(scenario, rng, snr)
    chirp, _ = generate_chirp(
        time,
        backend=backend,
        t_merger=params.t_merger,
        mass1=params.mass1,
        mass2=params.mass2,
        distance_mpc=params.distance,
        normalize=True,
    )
    signal, actual_snr = scale_signal_to_snr(
        chirp, noise, params.snr, scenario.peak_reference
    )
    return Sample(
        noise + signal,
        noise,
        signal,
        True,
        params,
        actual_snr,
        noise_kind,
        noise_file,
        backend,
        measure_optimal_snr(signal, noise, noise_kind),
    )


def make_test_set(
    scenario, n_samples, rng=None, noise_provider=None, backend=WAVEFORM_BACKEND
):
    """n_samples samples with drawn labels, normalized, as (X, y).

    X has the shape (n_samples, N_SAMPLES, 1) the CNN takes, y holds 0 and 1.
    """
    X, y = [], []
    for _ in range(n_samples):
        sample = make_sample(
            scenario, rng, noise_provider=noise_provider, backend=backend
        )
        X.append(normalize(sample.data))
        y.append(1 if sample.has_signal else 0)
    return np.array(X).reshape(n_samples, N_SAMPLES, 1), np.array(y)
