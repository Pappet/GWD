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
make_sample raises gwd.core.waveforms.WaveformBackendError (P5.2); there is no
fallback. Since P6.1 'pycbc' is the only backend; the argument stays, so the backend
is still named and recorded.

Every signal sample also gets its optimal (matched filter) SNR, measured after the
scaling and without drawing from rng (P5.5, measure_optimal_snr). The peak SNR still
drives V1; the optimal SNR is recorded so the difference can be seen.

Scenario V2 (condition_signal, P6.2) injects first and conditions afterwards
(make_conditioned_sample):

1. the label, unless the caller passes has_signal;
2. the noise, pad_s longer on each side: Gaussian noise with the aLIGO design PSD
   (colored_noise_from_psd with reference_psd, 0 below SIMULATED_PSD_F_LOW) or white
   noise, raw, not yet filtered;
3. the parameters (draw_params), as in V1;
4. the chirp on the padded time axis, scaled so that its part inside the window has
   the drawn optimal SNR against the same PSD in BANDPASS (scale_factor_v2);
5. noise + signal conditioned together (condition_array: notches and band-pass, then
   the pads cut off), and normalized with eps 0 by the std of the sum.

Signal and noise thus pass the same filters, so no unfiltered chirp above 300 Hz is
left as a shortcut (DATA-12, APPX-1), and the normalization no longer depends on the
noise type (the V1 scale skew of real chunks, P3.2). V2 draws no real noise (D3 = B).

Scenario V3 (whiten) draws exactly as V2, step for step, so the same rng gives the same
label, noise, and parameters. Only step 5 differs: the padded sum is first whitened
with the known PSD of the simulated noise (simulated_psd, the PSD the noise was drawn
from), then conditioned and normalized as in V2. The chirp alone passes the same chain
to give the signal part. Whitening is linear and invertible in the band, so the drawn
optimal SNR keeps its meaning (D2 = B): it is the same matched filter SNR in the
whitened sample.
"""

import glob
import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
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
from gwd.core.interferometer import reference_psd
from gwd.core.population import random_bbh_params
from gwd.core.snr import (
    optimal_snr_psd,
    optimal_snr_welch,
    optimal_snr_white,
    peak_snr,
    scale_signal_to_snr,
)
from gwd.core.timebase import time_axis
from gwd.core.waveforms import generate_chirp
from gwd.data.preprocessing import condition_array, normalize

# Distance of the fixed-mass scenarios: the default of generate_chirp. Random masses
# draw it uniform in volume (gwd.core.population). Either way it does not change the
# sample, because the chirp is normalized to peak 1 before scaling (metadata only).
FIXED_DISTANCE_MPC = 400.0
# Standard deviation of the white noise of every scenario.
WHITE_NOISE_SIGMA = 1.0
# The simulated aLIGO noise of V2 has no power below this frequency (Hz). Below it the
# design PSD rises steeply (at 1 Hz it is 1e7 times its value at 100 Hz); the band-pass
# from 20 Hz removes that part anyway, and without it the filters need not suppress it.
SIMULATED_PSD_F_LOW = 10.0


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

    V1: snr is the peak SNR the signal was scaled to (also as peak_snr), optimal_snr
    the optimal SNR measured by measure_optimal_snr. V2: optimal_snr is the optimal
    SNR the signal was scaled to (params.snr), snr the peak SNR of the conditioned
    signal over the std of the conditioned noise, measured. Both are 0.0 without
    signal. In V2, data, noise, and signal are conditioned and normalized.
    """

    data: np.ndarray
    noise: np.ndarray
    signal: Optional[np.ndarray]
    has_signal: bool
    params: Optional[InjectionParams]
    snr: float  # the peak SNR (V1: scaled to, V2: measured); 0.0 without signal
    noise_kind: str  # "real", "white", or "aligo" (V2)
    noise_file: Optional[str] = None  # file name of the real chunk, None for white
    waveform_backend: Optional[str] = None  # "pycbc"; None without signal
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
        noise. None means no real noise, so every sample gets white noise. V2 scenarios
        draw no real noise and ignore it.
    backend: the waveform backend of the chirp; the pipeline keeps the default.
    A V2 scenario (condition_signal) goes to make_conditioned_sample.
    """
    rng = _rng(rng)
    if scenario.condition_signal:
        return make_conditioned_sample(scenario, rng, has_signal, snr, backend)
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


@lru_cache(maxsize=8)
def _psd_on_grid(kind, n, fs):
    """The one-sided PSD of the simulated noise on np.fft.rfftfreq(n, 1 / fs), read-only.

    'aligo': reference_psd, 0 below SIMULATED_PSD_F_LOW and at DC. 'white': the flat
    PSD 2 sigma^2 / fs of white noise of std WHITE_NOISE_SIGMA (0 at DC).
    """
    freqs = np.fft.rfftfreq(n, 1 / fs)
    if kind == "aligo":
        values = np.asarray(reference_psd(freqs, SIMULATED_PSD_F_LOW), dtype=float)
    elif kind == "white":
        values = np.full_like(freqs, 2 * WHITE_NOISE_SIGMA**2 / fs)
    else:
        raise ValueError(f"Unknown simulated noise {kind!r}")
    values[0] = 0.0
    values.flags.writeable = False
    return values


def simulated_psd(kind, n, fs=SAMPLE_RATE):
    """The PSD of simulated noise kind ('aligo' or 'white') of n samples at fs Hz.

    An array on np.fft.rfftfreq(n, 1 / fs), cached and read-only.
    """
    return _psd_on_grid(kind, n, fs)


def draw_simulated_noise(kind, n, rng, fs=SAMPLE_RATE):
    """n samples of raw simulated noise: 'aligo' (colored) or 'white' (std 1)."""
    if kind == "white":
        return white_noise(n, rng)
    return core_noise.colored_noise_from_psd(n, fs, simulated_psd(kind, n, fs), rng=rng)


def scale_factor_v2(window_signal, target_snr, kind, fs=SAMPLE_RATE):
    """The factor that gives window_signal the optimal SNR target_snr (V2, D2 = B).

    The optimal SNR is that of the signal inside the sample window, against the PSD of
    the simulated noise, in BANDPASS: the band that conditioning keeps. Raises
    ValueError for a signal without power in the band.
    """
    psd = simulated_psd(kind, len(window_signal), fs)
    current = optimal_snr_psd(window_signal, psd, fs, *BANDPASS)
    if current == 0:
        raise ValueError("cannot scale a signal without power in BANDPASS")
    return target_snr / current


def make_conditioned_sample(
    scenario, rng=None, has_signal=None, snr=None, backend=WAVEFORM_BACKEND
):
    """Draw one V2 or V3 sample: inject first, then condition and normalize together.

    V3 (scenario.whiten) whitens with the known PSD before conditioning (see the
    module doc); the draws are those of V2.

    The draw order and the steps are in the module doc. The returned data, noise, and
    signal are the conditioned window of N_SAMPLES points, divided by the std of the
    conditioned sum (eps 0; noise and data also minus its mean), so data has mean 0
    and std 1 and data = noise + signal up to rounding. Raises ImportError without
    gwpy and WaveformBackendError or ReferencePSDError without PyCBC.
    """
    if not scenario.condition_signal:
        raise ValueError(f"{scenario.name} is a V1 scenario; use make_sample")
    rng = _rng(rng)
    pad = round(scenario.pad_s * SAMPLE_RATE)
    pad_s = pad / SAMPLE_RATE
    total = N_SAMPLES + 2 * pad

    if has_signal is None:
        has_signal = bool(rng.rand() > 0.5)
    noise_kind = scenario.simulated_noise
    raw_noise = draw_simulated_noise(noise_kind, total, rng)
    # V3 whitens with the PSD the noise was drawn from; V2 does not whiten.
    psd = simulated_psd(noise_kind, total) if scenario.whiten else None

    if not has_signal:
        conditioned = condition_array(raw_noise, SAMPLE_RATE, pad_s, psd=psd)
        data = normalize(conditioned, eps=0.0)
        return Sample(data, data, None, False, None, 0.0, noise_kind)

    params = draw_params(scenario, rng, snr)
    time = time_axis(SAMPLE_RATE, total / SAMPLE_RATE, t0=-pad_s)
    chirp, _ = generate_chirp(
        time,
        backend=backend,
        t_merger=params.t_merger,
        mass1=params.mass1,
        mass2=params.mass2,
        distance_mpc=params.distance,
        normalize=True,
    )
    chirp = chirp * scale_factor_v2(
        chirp[pad : pad + N_SAMPLES], params.snr, noise_kind
    )

    conditioned = condition_array(raw_noise + chirp, SAMPLE_RATE, pad_s, psd=psd)
    conditioned_signal = condition_array(chirp, SAMPLE_RATE, pad_s, psd=psd)
    mean, std = np.mean(conditioned), np.std(conditioned)
    data = (conditioned - mean) / std
    signal = conditioned_signal / std
    noise = (conditioned - conditioned_signal - mean) / std
    return Sample(
        data,
        noise,
        signal,
        True,
        params,
        peak_snr(signal, float(np.std(noise))),
        noise_kind,
        None,
        backend,
        float(params.snr),
    )
