"""
Noise for gravitational wave detector simulations.

This module provides functions that generate the kinds of noise found in
LIGO-like detectors. Every function draws from the rng it is given, a numpy Generator
or RandomState, never from the global numpy RNG (P5.3, core:MISS-3); the same seed
gives the same noise.
"""

import numpy as np


def generate_gaussian_noise(length, noise_level=1.0, *, rng):
    """
    Generate plain Gaussian (white) noise.

    White noise has a flat power spectral density over all frequencies and is
    often the basis of more complex noise models.

    Args:
        length: Number of data points
        noise_level: Standard deviation (amplitude of the noise)
        rng: numpy Generator or RandomState to draw from

    Returns:
        np.array: Array of noise values
    """
    return rng.normal(loc=0.0, scale=noise_level, size=length)


def generate_colored_noise(length, sample_rate, noise_level=1.0, *, rng):
    """
    Generate more realistic, colored noise (for example dominated by seismic noise).

    Simulates a 1/f behaviour at low frequencies, as seismic disturbances cause in
    real LIGO detectors. At higher frequencies other noise sources dominate (shot
    noise, thermal noise).

    Args:
        length: Number of data points
        sample_rate: Sampling rate in Hz (needed for the frequency analysis)
        noise_level: Base amplitude of the noise
        rng: numpy Generator or RandomState to draw from

    Returns:
        np.array: Array of colored noise
    """
    # 1. Generate white noise as the basis
    white_noise = generate_gaussian_noise(length, noise_level, rng=rng)

    # 2. Go to the frequency domain (FFT)
    freqs = np.fft.fftfreq(length, 1 / sample_rate)
    noise_fft = np.fft.fft(white_noise)

    # 3. Apply a spectral weight (simulate seismic noise)
    # Amplify low frequencies: factor ~ sqrt(1/f)
    mask = np.abs(freqs) > 0  # avoid a division by zero at 0 Hz

    # The factor 10 is chosen empirically to make the characteristic visible
    scaling_factor = (10 / np.abs(freqs[mask])) ** 0.5
    noise_fft[mask] *= scaling_factor

    # 4. Back to the time domain (inverse FFT)
    colored_noise = np.real(np.fft.ifft(noise_fft))

    return colored_noise
