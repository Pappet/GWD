"""Conditioning and normalization of strain data (P3.2), and the padded crop (P4.3).

condition_strain works on a gwpy TimeSeries but never imports gwpy itself; it only
calls the methods of the series it gets, so this module imports without gwpy.
"""

import numpy as np

from gwd.config import BANDPASS, NOTCH_FREQS


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


def conditioned_crop(ts, fs, pad_s, notches=NOTCH_FREQS, band=BANDPASS):
    """Condition a padded gwpy TimeSeries as a whole, then cut pad_s off each end.

    The filters ring at the ends of the data they get. Conditioning a series that is
    pad_s longer on each side and cropping afterwards keeps that ringing out of the
    result, as for the training noise, which is conditioned in one piece and then cut
    (core:MISS-2). A series downloaded with fetch_strain(..., pad_s=pad_s) gives back
    exactly the requested span. Returns a numpy array.

    Raises ValueError if pad_s is negative or the series is not longer than the two pads.
    """
    if pad_s < 0:
        raise ValueError(f"pad_s must not be negative, got {pad_s}")
    conditioned = condition_strain(ts, fs, notches=notches, band=band)
    pad = round(pad_s * fs)
    if len(conditioned) <= 2 * pad:
        raise ValueError(
            f"{len(conditioned)} samples are too short for a pad of {pad} samples per side"
        )
    return conditioned[pad : len(conditioned) - pad]


def normalize(x, eps=1e-10):
    """Z-score: (x - mean) / (std + eps).

    eps is absolute. On raw strain (std around 1e-21) it dominates the std and the
    result is not unit variance, so callers that normalize raw strain pass eps=0.
    """
    x = np.asarray(x)
    return (x - np.mean(x)) / (np.std(x) + eps)
