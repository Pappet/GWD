"""Checks for gwd.core.interferometer: directions, order of magnitude, and the reference.

The model is a qualitative learning model (D6, CORE-10). These tests pin only the
directions that any interferometer model must get right (the ASD falls with power, arm
length, and features), its order of magnitude against the aLIGO reference, and the
reference itself (P5.4).
"""

import sys

import numpy as np
import pytest

from gwd.core import interferometer
from gwd.core.interferometer import (
    InterferometerModel,
    ReferencePSDError,
    reference_asd,
    reference_available,
    reference_psd,
)


def asd(**settings):
    model = InterferometerModel()
    for name, value in settings.items():
        setattr(model, name, value)
    return model.calculate_total_sensitivity()


def test_default_asd_is_positive_and_finite():
    model = InterferometerModel()
    sensitivity = model.calculate_total_sensitivity()
    assert sensitivity.shape == model.freq.shape
    assert np.all(np.isfinite(sensitivity))
    assert np.all(sensitivity > 0)


def test_asd_falls_with_arm_length():
    short, mid, long = (asd(arm_length=length) for length in [1000, 4000, 16000])
    assert np.all(mid < short)
    assert np.all(long < mid)


def test_asd_falls_with_laser_power_where_shot_noise_dominates():
    # Only above 500 Hz: below the standard quantum limit, radiation pressure grows with
    # the power, so a correct model may rise with power at low frequencies.
    freq = InterferometerModel().freq
    shot_noise_band = freq >= 500
    low, mid, high = (
        asd(laser_power=power)[shot_noise_band] for power in [50, 200, 800]
    )
    assert np.all(mid < low)
    assert np.all(high < mid)
    # Shot noise goes as 1/sqrt(P): four times the power halves the ASD at 3 kHz.
    assert high[-1] / mid[-1] == pytest.approx(0.5, rel=0.05)


@pytest.mark.parametrize("feature", ["power_recycling", "squeezed_light"])
def test_feature_lowers_the_asd_everywhere(feature):
    base = asd()
    assert np.all(asd(**{feature: True}) < base)


def test_signal_recycling_lowers_the_asd_around_100_hz():
    model = InterferometerModel()
    base = model.calculate_total_sensitivity()
    with_sr = asd(signal_recycling=True)
    assert np.all(with_sr <= base)
    near_100_hz = np.abs(model.freq - 100) < 20
    assert np.all(with_sr[near_100_hz] < 0.95 * base[near_100_hz])


def test_all_features_together_beat_each_one_alone():
    features = ["power_recycling", "signal_recycling", "squeezed_light"]
    combined = asd(**dict.fromkeys(features, True))
    for feature in features:
        assert np.all(combined < asd(**{feature: True}))


@pytest.mark.parametrize("reflectivity", [0.99, 0.999, 0.99999])
def test_finesse_follows_the_mirror_reflectivity(reflectivity):
    """Since P4.5 the finesse is a property of R (APP-12, CORE-11)."""
    model = InterferometerModel()
    model.mirror_reflectivity = reflectivity
    expected = np.pi * np.sqrt(reflectivity) / (1 - reflectivity)
    assert model.finesse == pytest.approx(expected)


def test_the_default_finesse_matches_the_default_mirrors():
    assert InterferometerModel().finesse == pytest.approx(3140, rel=1e-3)


def test_the_finesse_cannot_be_set_apart_from_the_mirrors():
    model = InterferometerModel()
    with pytest.raises(AttributeError):
        model.finesse = 450


def test_the_reference_asd_is_the_aligo_design_curve_at_100_hz(needs_pycbc):
    """The check of the P5.4 plan: aLIGOZeroDetHighPower is about 3.9e-24 at 100 Hz."""
    assert reference_asd(100.0) == pytest.approx(3.9e-24, rel=0.2)
    assert np.ndim(reference_asd(100.0)) == 0


def test_the_reference_psd_equals_pycbc_on_its_grid(needs_pycbc):
    import pycbc.psd

    pycbc_psd = pycbc.psd.aLIGOZeroDetHighPower(8193, 0.5, 10.0)
    freqs = pycbc_psd.sample_frequencies.numpy()
    expected = pycbc_psd.numpy()
    ours = reference_psd(freqs, low_freq_cutoff=10.0)
    # PyCBC leaves its last bin (4096 Hz here) at 0; everywhere else the values agree.
    np.testing.assert_array_equal(ours[:-1], expected[:-1])
    assert np.all(ours[freqs < 10] == 0)
    assert np.all(ours[freqs >= 10] > 0)


def test_the_reference_is_zero_below_the_cutoff_and_at_zero_hz(needs_pycbc):
    psd = reference_psd(np.array([[0.0, 5.0], [10.0, 100.0]]), low_freq_cutoff=10.0)
    assert psd.shape == (2, 2)
    assert psd[0, 0] == 0 and psd[0, 1] == 0
    assert psd[1, 0] > psd[1, 1] > 0
    assert reference_psd(0.0) == 0


def test_without_pycbc_the_reference_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "pycbc", None)
    assert not reference_available()
    with pytest.raises(ReferencePSDError, match="physics"):
        reference_asd(100.0)


def test_the_default_model_lies_above_the_reference(needs_pycbc):
    """An honest learning model (D6): never better than aLIGO, 2 to 3 orders above it
    at 100 Hz (270 times at P5.4)."""
    model = InterferometerModel()
    ratio = model.calculate_total_sensitivity() / reference_asd(model.freq)
    assert np.all(ratio > 1)
    at_100_hz = np.interp(100, model.freq, ratio)
    assert 100 < at_100_hz < 1000


AD_HOC_FACTORS = [
    "MIRROR_MASS",
    "TEMPERATURE",
    "SQUEEZING_FACTOR",
    "THERMAL_SCALE",
    "COATING_WEIGHT",
    "SEISMIC_LEVEL",
    "RADIATION_PRESSURE_SCALE",
    "BS_LOSS_WEIGHT",
    "MIRROR_LOSS_WEIGHT",
    "POWER_RECYCLING_FACTOR",
    "SIGNAL_RECYCLING_DEPTH",
    "SIGNAL_RECYCLING_CENTER",
    "SIGNAL_RECYCLING_WIDTH",
]


@pytest.mark.parametrize("name", AD_HOC_FACTORS)
def test_each_ad_hoc_factor_is_a_constant_that_the_docstring_names(name):
    """CORE-10: the learning model lists its ad hoc factors instead of hiding them."""
    assert isinstance(getattr(interferometer, name), float)
    assert name in InterferometerModel.__doc__
