"""Conditioning and normalization of strain data, one function each (P3.2).

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


def normalize(x, eps=1e-10):
    """Z-score: (x - mean) / (std + eps).

    eps is absolute. On raw strain (std around 1e-21) it dominates the std and the
    result is not unit variance, so callers that normalize raw strain pass eps=0.
    """
    x = np.asarray(x)
    return (x - np.mean(x)) / (np.std(x) + eps)
