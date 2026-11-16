"""Smoke test for gwd.core.simulation.TimeDomainSimulator on both backends (CORE-12)."""

import numpy as np
import pytest

from gwd.core.noise import colored_noise_from_psd, pink_psd
from gwd.core.simulation import TimeDomainSimulator


@pytest.fixture
def simulator(backend, rng):
    return TimeDomainSimulator(
        duration=0.5, sample_rate=4096, backend=backend, rng=rng, noise_psd=pink_psd
    )


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
    simulator.rng = np.random.default_rng(3)
    noisy = simulator.add_detector_noise(signal, 1.0)
    simulator.rng = np.random.default_rng(3)
    noisy_doubled = simulator.add_detector_noise(signal, 2.0)

    noise = noisy - signal
    assert noisy.shape == signal.shape
    assert np.std(noise) > 0
    np.testing.assert_allclose(noisy_doubled - signal, 2 * noise, rtol=1e-6, atol=0)


def test_detector_noise_is_drawn_from_the_noise_psd(simulator):
    """CORE-12: the noise comes from colored_noise_from_psd with the simulator's PSD."""
    signal = np.zeros_like(simulator.t)
    simulator.rng = np.random.default_rng(9)
    noisy = simulator.add_detector_noise(signal)
    expected = colored_noise_from_psd(
        signal.size, simulator.sample_rate, pink_psd, rng=np.random.default_rng(9)
    )
    np.testing.assert_array_equal(noisy, expected)


def test_the_arm_length_change_is_strain_times_the_arm_length(simulator):
    strain, _ = simulator.generate_chirp(30, 30, 400)
    np.testing.assert_array_equal(
        simulator.arm_length_change(strain), strain * simulator.arm_length
    )
    assert simulator.arm_length == 4000
    # CORE-12: the unused intensity is gone with calculate_response.
    assert not hasattr(simulator, "calculate_response")


def test_the_simulator_needs_a_known_backend():
    with pytest.raises(ValueError, match="pycbc"):
        TimeDomainSimulator(
            backend="lal", rng=np.random.default_rng(), noise_psd=pink_psd
        )
