"""Tests for gwd.data.preprocessing (P3.2)."""

import numpy as np
import pytest

from gwd.config import BANDPASS, DURATION, NOTCH_FREQS, SAMPLE_RATE
from gwd.data.preprocessing import (
    check_preprocessing,
    condition_strain,
    normalize,
    welch_psd_on_grid,
    whiten,
)

LINES_HZ = (60, 120, 180, 240)


def synthetic_strain(sample_rate):
    """Strain-sized noise plus power-line harmonics, as a gwpy TimeSeries."""
    timeseries = pytest.importorskip("gwpy.timeseries")
    rng = np.random.default_rng(7)
    n = int(sample_rate * DURATION)
    t = np.arange(n) / sample_rate
    values = rng.normal(0.0, 1e-21, n)
    for freq in LINES_HZ:
        values += 5e-21 * np.sin(2 * np.pi * freq * t)
    return timeseries.TimeSeries(values, sample_rate=sample_rate, t0=1262304018)


def old_fetch_chain(strain):
    """The inline chain from fetch_noise.py before P3.2, copied unchanged."""
    if strain.sample_rate.value != SAMPLE_RATE:
        strain = strain.resample(SAMPLE_RATE)
    for freq in NOTCH_FREQS:
        strain = strain.notch(freq)
    strain = strain.bandpass(*BANDPASS)
    return strain.value


def line_amplitude(values, freq):
    spectrum = np.abs(np.fft.rfft(values))
    freqs = np.fft.rfftfreq(len(values), 1 / SAMPLE_RATE)
    return spectrum[np.argmin(np.abs(freqs - freq))]


@pytest.mark.parametrize("sample_rate", [SAMPLE_RATE, 2 * SAMPLE_RATE])
def test_condition_strain_is_bit_identical_to_the_old_fetch_chain(sample_rate):
    strain = synthetic_strain(sample_rate)
    conditioned = condition_strain(strain, SAMPLE_RATE)
    assert isinstance(conditioned, np.ndarray)
    assert len(conditioned) == int(SAMPLE_RATE * DURATION)
    np.testing.assert_array_equal(conditioned, old_fetch_chain(strain))


def test_condition_strain_removes_every_power_line_including_240_hz():
    # The field test used to skip the 240 Hz notch (DATA-10, EVAL-5).
    strain = synthetic_strain(SAMPLE_RATE)
    without_240 = condition_strain(strain, SAMPLE_RATE, notches=(60, 120, 180))
    conditioned = condition_strain(strain, SAMPLE_RATE)
    assert line_amplitude(conditioned, 240) < line_amplitude(without_240, 240) / 10


def test_condition_strain_does_not_change_the_input():
    strain = synthetic_strain(SAMPLE_RATE)
    before = strain.value.copy()
    condition_strain(strain, SAMPLE_RATE)
    np.testing.assert_array_equal(strain.value, before)


def test_normalize_gives_zero_mean_and_unit_std():
    x = np.random.default_rng(1).normal(3.0, 2.5, 16384)
    z = normalize(x)
    assert np.mean(z) == pytest.approx(0.0, abs=1e-12)
    assert np.std(z) == pytest.approx(1.0, rel=1e-9)


def test_normalize_matches_the_old_formulas_bit_for_bit():
    x = np.random.default_rng(2).normal(0.0, 1.0, 16384)
    with_eps = (x - np.mean(x)) / (np.std(x) + 1e-10)
    without_eps = (x - np.mean(x)) / np.std(x)
    np.testing.assert_array_equal(normalize(x), with_eps)
    np.testing.assert_array_equal(normalize(x, eps=0.0), without_eps)


def test_normalize_keeps_the_shape_and_accepts_lists():
    x = np.random.default_rng(3).normal(0.0, 1.0, (1, 100, 1))
    assert normalize(x).shape == x.shape
    assert normalize([1.0, 2.0, 3.0]).shape == (3,)


def test_normalize_of_a_constant_is_zero_not_nan():
    assert np.array_equal(normalize(np.full(10, 4.0)), np.zeros(10))


def test_eps_dominates_on_raw_strain_so_strain_needs_eps_zero():
    # Raw LIGO strain has a std around 1e-21, far below the default eps.
    strain = np.random.default_rng(4).normal(0.0, 1e-21, 16384)
    assert np.std(normalize(strain)) < 1e-9
    assert np.std(normalize(strain, eps=0.0)) == pytest.approx(1.0, rel=1e-9)


# --- whitening (V3) ---


def test_whiten_with_the_flat_psd_of_white_noise_divides_by_sigma():
    """White noise of std sigma has the PSD 2 sigma^2 / fs; whitening gives x / sigma."""
    rng = np.random.default_rng(1)
    n, sigma = 4096, 3.0
    x = rng.normal(0.0, sigma, n)
    psd = np.full(n // 2 + 1, 2 * sigma**2 / SAMPLE_RATE)
    white = whiten(x, SAMPLE_RATE, psd)
    np.testing.assert_allclose(white, (x - x.mean()) / sigma, atol=1e-12)


def test_whiten_is_linear_and_zeroes_bins_without_power():
    rng = np.random.default_rng(2)
    n = 2048
    a, b = rng.normal(size=n), rng.normal(size=n)
    freqs = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    psd = np.where(freqs < 10, 0.0, 1e-3 * (1 + freqs / 100))
    np.testing.assert_allclose(
        whiten(a + b, SAMPLE_RATE, psd),
        whiten(a, SAMPLE_RATE, psd) + whiten(b, SAMPLE_RATE, psd),
        atol=1e-12,
    )
    spectrum = np.fft.rfft(whiten(a, SAMPLE_RATE, psd))
    assert np.all(np.abs(spectrum[freqs < 10]) < 1e-9)


def test_whiten_rejects_a_psd_on_another_grid_and_a_long_taper():
    x = np.zeros(1024)
    with pytest.raises(ValueError, match="1024 samples need 513"):
        whiten(x, SAMPLE_RATE, np.ones(512))
    with pytest.raises(ValueError, match="taper"):
        whiten(x, SAMPLE_RATE, np.ones(513), taper_s=1.0)


def test_the_taper_leaves_the_middle_and_damps_the_ends():
    n = 4 * SAMPLE_RATE
    x = np.random.default_rng(3).normal(size=n)
    psd = np.full(n // 2 + 1, 2 / SAMPLE_RATE)
    plain = whiten(x, SAMPLE_RATE, psd)
    tapered = whiten(x, SAMPLE_RATE, psd, taper_s=1.0)
    middle = slice(SAMPLE_RATE + 1, 3 * SAMPLE_RATE - 1)
    # The same up to the mean, which whitening removes (DC) and the taper changes.
    assert np.std(tapered[middle] - plain[middle]) < 1e-9
    assert np.abs(tapered[:4]).max() < 0.01 and np.abs(tapered[-4:]).max() < 0.01


def test_the_welch_psd_of_white_noise_is_flat_and_robust_to_a_burst():
    """The median average keeps a short loud burst from raising the estimate."""
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 1.0, 36 * SAMPLE_RATE)
    psd = welch_psd_on_grid(x, SAMPLE_RATE)
    freqs = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
    band = (freqs >= 20) & (freqs <= 300)
    assert psd.shape == (len(x) // 2 + 1,)
    np.testing.assert_allclose(np.median(psd[band]) * SAMPLE_RATE / 2, 1.0, rtol=0.05)
    burst = x.copy()
    burst[18 * SAMPLE_RATE : 18 * SAMPLE_RATE + 400] += 50 * rng.normal(size=400)
    loud = welch_psd_on_grid(burst, SAMPLE_RATE)
    assert np.median(loud[band] / psd[band]) < 1.3


def test_an_unknown_preprocessing_is_rejected():
    assert check_preprocessing("whiten_bandpass") == "whiten_bandpass"
    with pytest.raises(ValueError, match="whiten_bandpass"):
        check_preprocessing("whiten")
