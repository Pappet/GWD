"""The one signal injection for training, evaluation, and the apps (P3.5).

make_sample draws a sample from a Scenario: noise (real or white), and for a signal
sample a chirp scaled to a peak SNR. The draw order is fixed, so a seed gives the same
samples on every machine:

1. the label, unless the caller passes has_signal;
2. one number that picks real or white noise, if the scenario uses real noise;
3. the noise: a real chunk (index, then crop offset if needed) or white noise;
4. the parameters (draw_params): masses, distance, two spins and an inclination
   (random masses only), then the merger time, then the SNR unless the caller fixes it.

For TRAIN_V1 this is the order gwd.data.generate used before P3.5. The other callers
drew the merger time and SNR, or label and noise, in another order.

rng is anything with the numpy RandomState methods (rand, uniform, randint, normal).
The default is np.random, the global RNG, as before. gwd.core has no rng parameter
until P5.3, so this module draws white noise and random masses itself, with the same
calls as gwd.core.noise.generate_gaussian_noise and
gwd.core.waveforms.generate_random_bbh_parameters.

The chirps come from the backend gwd.config.WAVEFORM_BACKEND ('pycbc'). Without PyCBC,
make_sample raises gwd.core.waveforms.WaveformBackendError (P5.2); there is no fallback
to the Newtonian approximation. The backend argument exists so the golden values and
tests can draw numpy-only Newton chirps on purpose.
"""

import glob
import os
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from gwd.config import (
    DATA_DIR_NOISE,
    DURATION,
    N_SAMPLES,
    SAMPLE_RATE,
    WAVEFORM_BACKEND,
)
from gwd.core.timebase import time_axis
from gwd.core.waveforms import generate_chirp
from gwd.data.preprocessing import normalize

# Distance of the fixed-mass scenarios: the default of generate_chirp.
FIXED_DISTANCE_MPC = 400.0
# Random masses: distance uniform in volume up to this value. It does not change the
# sample, because the chirp is normalized to peak 1 before scaling (metadata only).
MAX_DISTANCE_MPC = 2000


@dataclass(frozen=True)
class InjectionParams:
    mass1: float
    mass2: float
    distance: float
    t_merger: float
    snr: float


@dataclass(frozen=True)
class Sample:
    """data = noise + signal. signal and params are None for a noise-only sample."""

    data: np.ndarray
    noise: np.ndarray
    signal: Optional[np.ndarray]
    has_signal: bool
    params: Optional[InjectionParams]
    snr: float  # the SNR the signal was scaled to; 0.0 without signal
    noise_kind: str  # "real" or "white"
    noise_file: Optional[str] = None  # file name of the real chunk, None for white
    waveform_backend: Optional[str] = None  # "pycbc" or "newton"; None without signal


def _rng(rng):
    return np.random if rng is None else rng


def white_noise(length, rng=None):
    """Unit white noise, the same call as gwd.core.noise.generate_gaussian_noise."""
    return _rng(rng).normal(loc=0.0, scale=1.0, size=length)


def scale_signal_to_snr(signal, noise, target_snr, reference="noise_std"):
    """Scale signal to a peak SNR and return (scaled signal, SNR).

    "noise_std" (TRAIN_V1): the peak becomes target_snr times the std of noise. A
    signal without peak returns zeros and SNR 0.0.
    "unit" (EVAL_V1, APP_V1): signal * target_snr. This is a peak SNR only because the
    chirp has peak 1 and the white noise a nominal std of 1; the measured std is not
    used.
    """
    if reference == "unit":
        return signal * target_snr, target_snr

    noise_level = np.std(noise)
    if noise_level == 0:
        noise_level = 1.0

    sig_max = np.max(np.abs(signal))
    if sig_max == 0:
        return signal, 0.0

    signal_normalized = signal / sig_max
    scaled_signal = signal_normalized * target_snr * noise_level
    return scaled_signal, target_snr


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
        # The draws of gwd.core.waveforms.generate_random_bbh_parameters, in its order.
        low, high = scenario.masses
        m1 = 10 ** rng.uniform(np.log10(low), np.log10(high))
        m2 = 10 ** rng.uniform(np.log10(low), np.log10(m1))  # m2 <= m1
        distance = (rng.uniform(0, 1) ** (1 / 3)) * MAX_DISTANCE_MPC
        # Two spins and the inclination are drawn and dropped, which keeps the V1
        # random stream; the chirp is spinless and face-on (CORE-16, P5.3).
        rng.uniform(-0.9, 0.9)
        rng.uniform(-0.9, 0.9)
        rng.uniform(-1, 1)
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
