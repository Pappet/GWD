"""Smoke test for gwd.core.simulation.TimeDomainSimulator on both backends (CORE-12)."""

import numpy as np
import pytest

from gwd.core.simulation import TimeDomainSimulator


@pytest.fixture
def simulator(backend):
    return TimeDomainSimulator(duration=0.5, sample_rate=4096)


def test_generate_chirp_gives_physical_strain_with_the_merger_at_80_percent(simulator):
    strain, freq = simulator.generate_chirp(30, 30, 400)
    assert strain.shape == simulator.t.shape
    assert freq.shape == simulator.t.shape
    assert np.all(np.isfinite(strain))
    # normalize=False: a 30+30 Msun binary at 400 Mpc has a peak strain of order 1e-21.
    assert 1e-23 < np.max(np.abs(strain)) < 1e-19
    t_peak = simulator.t[np.argmax(np.abs(strain))]
    assert abs(t_peak - 0.8 * simulator.duration) <= 1e-3


def test_generate_chirp_scales_with_inverse_distance(simulator):
    near, _ = simulator.generate_chirp(30, 30, 400)
    far, _ = simulator.generate_chirp(30, 30, 800)
    np.testing.assert_allclose(far, near / 2, rtol=1e-6, atol=0)


def test_detector_noise_is_added_and_scales_with_the_slider(simulator):
    signal, _ = simulator.generate_chirp(30, 30, 400)
    state = np.random.get_state()
    noisy = simulator.add_detector_noise(signal, 1.0)
    np.random.set_state(state)
    noisy_doubled = simulator.add_detector_noise(signal, 2.0)

    noise = noisy - signal
    assert noisy.shape == signal.shape
    assert np.std(noise) > 0
    np.testing.assert_allclose(noisy_doubled - signal, 2 * noise, rtol=1e-6, atol=0)


def test_response_converts_strain_to_arm_length_change_and_intensity(simulator):
    strain, _ = simulator.generate_chirp(30, 30, 400)
    delta_l, intensity = simulator.calculate_response(strain)
    np.testing.assert_allclose(delta_l, strain * simulator.arm_length)
    assert np.all((intensity >= 0) & (intensity <= 1))
    # Small phase: sin^2(x) = x^2 with x = 4 pi dL / lambda.
    phase = 4 * np.pi * delta_l / simulator.laser_wavelength
    np.testing.assert_allclose(intensity, phase**2, rtol=1e-6, atol=0)
