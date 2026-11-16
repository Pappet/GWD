"""
Noise for gravitational wave detector simulations.

white_noise draws Gaussian white noise with a given standard deviation.
colored_noise_from_psd draws Gaussian noise with a given one-sided power spectral
density (PSD): its Welch PSD matches the PSD, and its variance is the integral of the
PSD up to the Nyquist frequency, whatever the length and the sampling rate (P5.4,
CORE-9). pink_psd is the 1/f spectrum that generate_colored_noise applied before P5.4
and called 'seismic'; it is a shape, not a detector model. The aLIGO design PSD is
gwd.core.interferometer.reference_psd.

Every function draws from the rng it is given, a numpy Generator or RandomState, never
from the global numpy RNG (P5.3, core:MISS-3); the same seed gives the same noise.
"""

import numpy as np


def white_noise(n, sigma=1.0, *, rng):
    """Return n samples of Gaussian white noise with mean 0 and standard deviation sigma.

    Its one-sided PSD is flat at 2 sigma^2 / fs for a sampling rate fs.
    """
    return rng.normal(loc=0.0, scale=sigma, size=n)


def pink_psd(f, psd_at_f_ref=1.0, f_ref=10.0):
    """One-sided 1/f ('pink') PSD: psd_at_f_ref * f_ref / f, in units^2 / Hz.

    Infinite at f = 0; colored_noise_from_psd never evaluates a PSD there.
    """
    return psd_at_f_ref * f_ref / np.asarray(f, dtype=float)


def colored_noise_from_psd(n, fs, psd, *, rng):
    """Return n samples of Gaussian noise at fs Hz with the one-sided PSD psd.

    psd is a function of the frequency in Hz, called once with the positive frequencies
    np.fft.rfftfreq(n, 1 / fs)[1:], or an array of the PSD on all of
    np.fft.rfftfreq(n, 1 / fs). Its unit is units^2 / Hz, for example strain^2 / Hz.

    Each positive frequency bin f_k gets a complex Gaussian coefficient with
    E|X_k|^2 = n fs S(f_k) / 2, the Nyquist bin (even n) a real one, and the DC bin 0,
    so the noise has mean 0. With the irfft convention x = ifft(X), the expected
    variance is the sum of S(f_k) df over the positive bins (the Nyquist bin counted
    half), df = fs / n: the integral of the PSD up to fs / 2. A flat PSD 2 sigma^2 / fs
    gives the standard deviation sigma for any n and fs.
    """
    freqs = np.fft.rfftfreq(n, 1 / fs)
    if callable(psd):
        values = np.zeros_like(freqs)
        values[1:] = psd(freqs[1:])
    else:
        values = np.array(psd, dtype=float)
        if values.shape != freqs.shape:
            raise ValueError(
                f"psd has shape {values.shape}, but rfftfreq({n}, 1/{fs}) has {freqs.shape}"
            )
        values[0] = 0.0
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("psd must be finite and >= 0 at every positive frequency")

    # Standard deviation of the real and of the imaginary part of each coefficient.
    sigma = np.sqrt(values * n * fs / 4)
    coefficients = sigma * (
        rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size)
    )
    coefficients[0] = 0.0
    if n % 2 == 0:
        # The Nyquist coefficient of a real series is real: all of its variance in one part.
        coefficients[-1] = np.sqrt(2) * coefficients[-1].real
    return np.fft.irfft(coefficients, n)
