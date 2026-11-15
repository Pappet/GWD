"""Tests for gwd.data.preprocessing (P3.2)."""

import numpy as np
import pytest

from gwd.config import BANDPASS, DURATION, NOTCH_FREQS, SAMPLE_RATE
from gwd.data.preprocessing import condition_strain, normalize

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
