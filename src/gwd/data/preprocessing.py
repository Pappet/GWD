"""Conditioning and normalization of strain data (P3.2), and the padded crop (P4.3).

condition_strain works on a gwpy TimeSeries but never imports gwpy itself; it only
calls the methods of the series it gets, so this module imports without gwpy.
condition_array (P6.2) runs the same padded chain on a numpy array, for the simulated
samples of scenario V2; it imports gwpy when it is called.

Whitening (V3): whiten() divides a padded segment by its amplitude spectral density in
the frequency domain, so coloured noise becomes white with unit variance; it is the one
whitening of the package and uses numpy and scipy only. The preprocessing
"whiten_bandpass" (gwd.data.scenarios.PREPROCESSINGS) whitens before the notch and band-pass
chain: simulated V3 samples with the known PSD of their noise (condition_array with
psd), real data with the Welch PSD of the padded segment itself (whiten_strain, through
conditioned_crop with preprocessing="whiten_bandpass").
"""

import numpy as np
from scipy import signal as sp_signal

from gwd.config import BANDPASS, NOTCH_FREQS, WHITEN_PSD_SEGMENT_S, WHITEN_TAPER_S
from gwd.data.scenarios import DEFAULT_PREPROCESSING, PREPROCESSINGS

WHITEN = "whiten_bandpass"


def check_preprocessing(preprocessing):
    """Return preprocessing if it is one of PREPROCESSINGS, else raise ValueError."""
    if preprocessing not in PREPROCESSINGS:
        raise ValueError(
            f"Unknown preprocessing {preprocessing!r}; known: {', '.join(PREPROCESSINGS)}"
        )
    return preprocessing


def whiten(x, fs, psd, taper_s=0.0):
    """x divided by its amplitude spectral density, in the frequency domain.

    psd is the one-sided PSD of the noise in x (units of x squared per Hz) on the grid
    np.fft.rfftfreq(len(x), 1 / fs). Every bin is divided by sqrt(psd * fs / 2), so
    noise with that PSD comes out white with unit variance (white noise of std sigma has
    the PSD 2 sigma^2 / fs). Bins where psd is 0, and DC, are set to 0. The division is
    circular: the ends of x influence each other, so x should carry pads that are cut
    off afterwards. taper_s > 0 first multiplies x by a Tukey window that falls to 0
    over taper_s seconds at each end, for data whose two ends do not meet (real data).
    Linear in x, so whitening noise + signal equals whitening each and adding.
    """
    x = np.asarray(x, dtype=float)
    psd = np.asarray(psd, dtype=float)
    n = len(x)
    if psd.shape != (n // 2 + 1,):
        raise ValueError(
            f"The PSD has {psd.shape} values; {n} samples need {n // 2 + 1}"
        )
    if taper_s > 0:
        taper = round(taper_s * fs)
        if 2 * taper > n:
            raise ValueError(f"A taper of {taper_s} s is longer than half of x")
        x = x * sp_signal.windows.tukey(n, alpha=2 * taper / n)
    scale = np.zeros_like(psd)
    positive = psd > 0
    scale[positive] = 1.0 / np.sqrt(psd[positive] * fs / 2.0)
    scale[0] = 0.0
    return np.fft.irfft(np.fft.rfft(x) * scale, n)


def welch_psd_on_grid(x, fs, segment_s=WHITEN_PSD_SEGMENT_S):
    """The Welch PSD of x on np.fft.rfftfreq(len(x), 1 / fs), for whiten().

    Hann segments of segment_s seconds (at most half of x), half overlap, and the
    median of the segments instead of their mean, so a loud event in a few segments
    does not raise the estimate. Interpolated linearly onto the grid of x.
    """
    x = np.asarray(x, dtype=float)
    nperseg = min(round(segment_s * fs), len(x) // 2)
    if nperseg < 2:
        raise ValueError(f"{len(x)} samples are too short for a Welch PSD")
    freqs, psd = sp_signal.welch(x, fs=fs, nperseg=nperseg, average="median")
    return np.interp(np.fft.rfftfreq(len(x), 1 / fs), freqs, psd)


def whiten_strain(ts, fs, segment_s=WHITEN_PSD_SEGMENT_S, taper_s=WHITEN_TAPER_S):
    """Resample a padded gwpy TimeSeries to fs and whiten it with its own Welch PSD.

    The real data path of "whiten_bandpass": the PSD is estimated from the whole padded
    segment (welch_psd_on_grid), and the segment is tapered over taper_s at each end
    before it is whitened. Returns a series of the same type, with the same t0.
    """
    if ts.sample_rate.value != fs:
        ts = ts.resample(fs)
    values = np.asarray(ts.value, dtype=float)
    white = whiten(values, fs, welch_psd_on_grid(values, fs, segment_s), taper_s)
    return type(ts)(white, sample_rate=fs, t0=ts.t0)


def condition_strain(ts, fs, notches=NOTCH_FREQS, band=BANDPASS):
    """Resample a gwpy TimeSeries to fs, notch each frequency, then band-pass.

    This is a notch and band-pass chain, not whitening. Returns a numpy array.
    """
    if ts.sample_rate.value != fs:
        ts = ts.resample(fs)
    for freq in notches:
        ts = ts.notch(freq)
    conditioned = ts.bandpass(*band)
    return conditioned.value


def conditioned_crop(
    ts,
    fs,
    pad_s,
    notches=NOTCH_FREQS,
    band=BANDPASS,
    preprocessing=DEFAULT_PREPROCESSING,
):
    """Condition a padded gwpy TimeSeries as a whole, then cut pad_s off each end.

    preprocessing "whiten_bandpass" (V3) whitens the padded series with its own Welch
    PSD first (whiten_strain); "bandpass", the default, does not.

    The filters ring at the ends of the data they get. Conditioning a series that is
    pad_s longer on each side and cropping afterwards keeps that ringing out of the
    result, as for the training noise, which is conditioned in one piece and then cut
    (core:MISS-2). A series downloaded with fetch_strain(..., pad_s=pad_s) gives back
    exactly the requested span. Returns a numpy array.

    Raises ValueError if pad_s is negative or the series is not longer than the two pads.
    """
    if pad_s < 0:
        raise ValueError(f"pad_s must not be negative, got {pad_s}")
    if check_preprocessing(preprocessing) == WHITEN:
        ts = whiten_strain(ts, fs)
    conditioned = condition_strain(ts, fs, notches=notches, band=band)
    pad = round(pad_s * fs)
    if len(conditioned) <= 2 * pad:
        raise ValueError(
            f"{len(conditioned)} samples are too short for a pad of {pad} samples per side"
        )
    return conditioned[pad : len(conditioned) - pad]


GWPY_HINT = (
    "Install gwpy for scenarios V2, V3, and real data: pip install -e '.[gwdata]'"
)


def condition_array(x, fs, pad_s, notches=NOTCH_FREQS, band=BANDPASS, psd=None):
    """conditioned_crop of a numpy array x sampled at fs Hz (P6.2).

    The array is wrapped in a gwpy TimeSeries, so a simulated V2 sample goes through
    exactly the chain the field test and the Glitch Hunter apply to real data. The
    chain is linear: conditioning noise + signal equals conditioning each and adding.
    With psd (V3), x is first whitened with that known PSD of its noise (whiten(), on
    np.fft.rfftfreq(len(x), 1 / fs)), without a taper: simulated noise is periodic.
    Raises ImportError with the install command if gwpy is missing.
    """
    try:
        from gwpy.timeseries import TimeSeries
    except ImportError as error:
        raise ImportError(GWPY_HINT) from error
    x = np.asarray(x, dtype=float)
    if psd is not None:
        x = whiten(x, fs, psd)
    series = TimeSeries(x, sample_rate=fs)
    return conditioned_crop(series, fs, pad_s, notches=notches, band=band)


def normalize(x, eps=1e-10):
    """Z-score: (x - mean) / (std + eps).

    eps is absolute. On raw strain (std around 1e-21) it dominates the std and the
    result is not unit variance, so callers that normalize raw strain pass eps=0.
    """
    x = np.asarray(x)
    return (x - np.mean(x)) / (np.std(x) + eps)
