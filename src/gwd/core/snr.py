"""Signal-to-noise ratios of an injected signal h: the peak SNR and the optimal SNR (P5.5).

peak_snr is the definition V1 scales to: the peak amplitude max|h| divided by the noise
standard deviation sigma. It is an amplitude ratio, not a matched filter SNR, so the
same value means a different detectability for other masses and other noise
(data-training:SNR-DEF, evaluation:MISS-2).

The optimal SNR is the matched filter SNR of h with its own template,

    rho_opt^2 = 4 * sum_k |H(f_k)|^2 / S(f_k) * df,

with H = rfft(h) / fs (the continuous Fourier transform, as in PyCBC), df = fs / n,
and S the one-sided noise PSD in units^2 / Hz. optimal_snr_psd sums the bins with
f_low <= f_k < f_high, the convention of pycbc.filter.sigma. For white noise of std
sigma the one-sided PSD is 2 sigma^2 / fs, and the full sum is sqrt(sum h^2) / sigma by
Parseval: optimal_snr_white. optimal_snr_welch estimates S from a noise series with
Welch's method first, for noise whose PSD is not known, such as a real chunk.

scale_to_snr scales h linearly to a target value of one of these definitions, and
measure measures it. scale_signal_to_snr is the V1 scaling of gwd.data.injection
(moved here in P5.5): a peak SNR against the measured noise std, or against a nominal
std of 1.

numpy and scipy only, no config: the caller passes fs and the band.
"""

import numpy as np
from scipy import signal as sp_signal

# The definitions measure and scale_to_snr know, with the keyword arguments each needs.
KINDS = {
    "peak": ("sigma",),
    "optimal_white": ("sigma",),
    "optimal_psd": ("psd", "fs", "f_low", "f_high"),
    "optimal_welch": ("noise", "fs", "f_low", "f_high"),
}
# Welch segment length in seconds: 0.25 s gives 4 Hz bins and 31 half-overlapping
# segments in a 4 s sample. Fewer segments bias 1 / S upward: against the true PSD,
# rho from 1 s segments came out about 7 % high, from 0.25 s about 1 % (P5.5).
WELCH_SEGMENT_S = 0.25


def peak_snr(h, sigma):
    """max|h| / sigma: the peak amplitude over the noise std (the V1 'snr')."""
    return float(np.max(np.abs(h))) / sigma


def optimal_snr_white(h, sigma):
    """sqrt(sum h^2) / sigma: the optimal SNR of h in white noise of std sigma.

    It does not depend on the sampling rate: sigma is the std of one noise sample.
    """
    return float(np.sqrt(np.sum(np.square(h)))) / sigma


def _psd_on_grid(psd, n, fs):
    freqs = np.fft.rfftfreq(n, 1 / fs)
    if callable(psd):
        values = np.zeros_like(freqs)
        values[1:] = psd(freqs[1:])
        return freqs, values
    values = np.asarray(psd, dtype=float)
    if values.ndim == 0:
        return freqs, np.full_like(freqs, float(values))
    if values.shape != freqs.shape:
        raise ValueError(
            f"psd has shape {values.shape}, but rfftfreq({n}, 1/{fs}) has {freqs.shape}"
        )
    return freqs, values


def optimal_snr_psd(h, psd, fs, f_low=None, f_high=None):
    """The optimal SNR of h (sampled at fs Hz) in noise with the one-sided PSD psd.

    psd is a function of the frequency in Hz, called with the positive frequencies of
    np.fft.rfftfreq(len(h), 1 / fs), an array on all of them, or a number for a flat
    PSD, in units^2 / Hz.
    The sum runs over the bins with f_low <= f < f_high, as in pycbc.filter.sigma;
    f_low None starts at the first bin above DC, f_high None ends at the Nyquist
    frequency, whose bin (even n) counts half, so a flat PSD 2 sigma^2 / fs gives
    optimal_snr_white for a signal without DC. Raises ValueError if the PSD is not
    finite and positive in the band.
    """
    h = np.asarray(h, dtype=float)
    n = len(h)
    freqs, values = _psd_on_grid(psd, n, fs)
    weights = np.ones_like(freqs)
    weights[0] = 0.0
    if n % 2 == 0 and f_high is None:
        weights[-1] = 0.5
    if f_low is not None:
        weights[freqs < f_low] = 0.0
    if f_high is not None:
        weights[freqs >= f_high] = 0.0
    band = weights > 0
    if not band.any():
        raise ValueError(f"no frequency bin in [{f_low}, {f_high}) Hz")
    if not np.all(np.isfinite(values[band])) or np.any(values[band] <= 0):
        raise ValueError(f"psd must be finite and > 0 in [{f_low}, {f_high}) Hz")

    spectrum = np.fft.rfft(h)[band] / fs
    df = fs / n
    rho_sq = 4.0 * df * np.sum(weights[band] * np.abs(spectrum) ** 2 / values[band])
    return float(np.sqrt(rho_sq))


def welch_psd(noise, fs, segment_s=WELCH_SEGMENT_S):
    """(frequencies, one-sided PSD) of noise by Welch's method (Hann, half overlap)."""
    nperseg = min(len(noise), int(round(segment_s * fs)))
    return sp_signal.welch(noise, fs=fs, nperseg=nperseg)


def optimal_snr_welch(h, noise, fs, f_low, f_high, segment_s=WELCH_SEGMENT_S):
    """optimal_snr_psd of h against the Welch PSD of noise, in [f_low, f_high) Hz.

    The Welch PSD is interpolated linearly onto the frequency grid of h. A band is
    needed: outside the pass band of filtered noise the estimate is near 0, and the
    SNR there would explode (evaluation:MISS-2).
    """
    freqs, psd = welch_psd(noise, fs, segment_s)
    grid = np.fft.rfftfreq(len(h), 1 / fs)
    return optimal_snr_psd(h, np.interp(grid, freqs, psd), fs, f_low, f_high)


def measure(h, kind, **kwargs):
    """The SNR of h by definition kind (a key of KINDS), with that kind's arguments.

    For example measure(h, "peak", sigma=1.0) or
    measure(h, "optimal_psd", psd=psd, fs=4096, f_low=20, f_high=300).
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {tuple(KINDS)}, got {kind!r}")
    unknown = set(kwargs) - set(KINDS[kind]) - {"segment_s"}
    if unknown:
        raise TypeError(f"{kind} takes {KINDS[kind]}, not {sorted(unknown)}")
    if kind == "peak":
        return peak_snr(h, **kwargs)
    if kind == "optimal_white":
        return optimal_snr_white(h, **kwargs)
    if kind == "optimal_psd":
        return optimal_snr_psd(h, **kwargs)
    return optimal_snr_welch(h, **kwargs)


def scale_to_snr(h, target, kind, **kwargs):
    """h scaled so that measure(result, kind, **kwargs) == target.

    Every definition is linear in h, so this is h * target / measure(h, ...).
    Raises ValueError for a signal whose SNR is 0.
    """
    current = measure(h, kind, **kwargs)
    if current == 0:
        raise ValueError(f"cannot scale a signal whose {kind} SNR is 0")
    return np.asarray(h) * (target / current)


def scale_signal_to_snr(signal, noise, target_snr, reference="noise_std"):
    """Scale signal to a peak SNR and return (scaled signal, SNR): the V1 scaling.

    "noise_std" (TRAIN_V1): the peak becomes target_snr times the std of noise. A
    signal without peak returns zeros and SNR 0.0.
    "unit" (EVAL_V1, APP_V1): signal * target_snr. This is a peak SNR only because the
    chirp has peak 1 and the white noise a nominal std of 1; the measured std is not
    used.
    The arithmetic is that of V1, so the golden values stay bit for bit.
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
