"""Checks for gwd.core.interferometer: the ASD must fall with power, arm length, and features.

The model itself is a toy (CORE-10, rewritten in P5.4); these tests pin only the directions
that any interferometer model must get right.
"""

import numpy as np
import pytest

from gwd.core.interferometer import InterferometerModel


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
