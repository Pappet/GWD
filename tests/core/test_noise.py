"""Statistical checks for gwd.core.noise with a seeded rng (see conftest.py)."""

import numpy as np
import pytest
from scipy.signal import welch

from gwd.core.noise import generate_colored_noise, generate_gaussian_noise

N = 100_000


@pytest.mark.parametrize("sigma", [1.0, 1e-22])
def test_white_noise_has_the_requested_std(sigma, rng):
    noise = generate_gaussian_noise(N, sigma, rng=rng)
    assert noise.shape == (N,)
    # The standard error of the std is sigma / sqrt(2 N) ~ 0.2 %, of the mean sigma / sqrt(N).
    assert np.std(noise) == pytest.approx(sigma, rel=0.01)
    assert abs(np.mean(noise)) < 5 * sigma / np.sqrt(N)


def test_white_noise_is_flat(rng):
    freqs, psd = welch(generate_gaussian_noise(N, 1.0, rng=rng), fs=4096, nperseg=1024)
    band = (freqs > 10) & (freqs < 2000)
    assert psd[band].max() / psd[band].min() < 2


def test_colored_noise_scales_linearly_with_noise_level():
    base = generate_colored_noise(16384, 4096, 1.0, rng=np.random.default_rng(1))
    doubled = generate_colored_noise(16384, 4096, 2.0, rng=np.random.default_rng(1))
    np.testing.assert_allclose(doubled, 2 * base, rtol=1e-12, atol=0)


def test_colored_noise_has_a_one_over_f_spectrum(rng):
    fs = 4096
    freqs, psd = welch(generate_colored_noise(N, fs, 1.0, rng=rng), fs=fs, nperseg=4096)
    low = np.mean(psd[(freqs >= 18) & (freqs <= 22)])
    high = np.mean(psd[(freqs >= 1800) & (freqs <= 2200)])
    assert low / high == pytest.approx(100, rel=0.2)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="CORE-9: the std of colored noise depends on n and fs, not only on noise_level",
)
def test_colored_noise_std_is_noise_level_for_any_n_and_fs(rng):
    stds = {
        (n, fs): np.std(generate_colored_noise(n, fs, 1.0, rng=rng))
        for n, fs in [(16384, 4096), (16384, 1024), (2048, 4096)]
    }
    assert all(std == pytest.approx(1.0, rel=0.1) for std in stds.values()), stds
