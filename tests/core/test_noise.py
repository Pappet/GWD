"""Statistical checks for gwd.core.noise with a seeded rng (see conftest.py)."""

from functools import partial

import numpy as np
import pytest
from scipy.signal import welch

from gwd.core.noise import colored_noise_from_psd, pink_psd, white_noise

N = 100_000
# The (n, fs) pairs of the P5.4 check: the normalization must not depend on them (CORE-9).
SHAPES = [(16384, 4096), (16384, 1024), (2048, 4096)]


@pytest.mark.parametrize("sigma", [1.0, 1e-22])
def test_white_noise_has_the_requested_std(sigma, rng):
    noise = white_noise(N, sigma, rng=rng)
    assert noise.shape == (N,)
    # The standard error of the std is sigma / sqrt(2 N) ~ 0.2 %, of the mean sigma / sqrt(N).
    assert np.std(noise) == pytest.approx(sigma, rel=0.01)
    assert abs(np.mean(noise)) < 5 * sigma / np.sqrt(N)


def test_white_noise_is_flat(rng):
    freqs, psd = welch(white_noise(N, 1.0, rng=rng), fs=4096, nperseg=1024)
    band = (freqs > 10) & (freqs < 2000)
    assert psd[band].max() / psd[band].min() < 2


def expected_variance(n, fs, psd):
    """The variance colored_noise_from_psd promises: sum of S df, the Nyquist bin half."""
    freqs = np.fft.rfftfreq(n, 1 / fs)[1:]
    weights = np.ones_like(freqs)
    if n % 2 == 0:
        weights[-1] = 0.5
    return np.sum(weights * psd(freqs)) * fs / n


def mean_welch_psd(n, fs, psd, rng, realizations=20, nperseg=256):
    """Welch PSD averaged over independent realizations, so a band check is tight."""
    estimates = [
        welch(colored_noise_from_psd(n, fs, psd, rng=rng), fs=fs, nperseg=nperseg)
        for _ in range(realizations)
    ]
    return estimates[0][0], np.mean([estimate for _, estimate in estimates], axis=0)


@pytest.mark.parametrize(("n", "fs"), SHAPES)
@pytest.mark.parametrize(
    "psd",
    [pink_psd, lambda f: np.full_like(f, 3e-3)],
    ids=["pink", "flat"],
)
def test_the_welch_psd_matches_the_requested_psd(n, fs, psd, rng):
    """The check of the P5.4 plan: the Welch PSD agrees within 15 % for every (n, fs)."""
    freqs, estimate = mean_welch_psd(n, fs, psd, rng)
    # Bins 4 to 124 of 129: away from DC, where the Hann window leaks, and from Nyquist.
    for band in np.array_split(np.arange(4, 125), 4):
        ratio = np.mean(estimate[band] / psd(freqs[band]))
        assert ratio == pytest.approx(1.0, rel=0.15), (n, fs, freqs[band[0]], ratio)


@pytest.mark.parametrize(("n", "fs"), SHAPES)
def test_the_variance_is_the_integral_of_the_psd(n, fs, rng):
    variances = [
        np.var(colored_noise_from_psd(n, fs, pink_psd, rng=rng)) for _ in range(50)
    ]
    assert np.mean(variances) == pytest.approx(
        expected_variance(n, fs, pink_psd), rel=0.05
    )


@pytest.mark.parametrize(("n", "fs"), SHAPES)
def test_colored_noise_std_is_the_requested_std_for_any_n_and_fs(n, fs, rng):
    """CORE-9, a strict xfail from P5.1 to P5.4: the std depended on n and fs.

    A flat one-sided PSD 2 sigma^2 / fs is white noise of standard deviation sigma.
    """
    sigma = 1.5
    flat = partial(np.full_like, fill_value=2 * sigma**2 / fs)
    stds = [np.std(colored_noise_from_psd(n, fs, flat, rng=rng)) for _ in range(20)]
    assert np.mean(stds) == pytest.approx(sigma, rel=0.03)


def test_colored_noise_has_mean_zero_and_odd_lengths_work(rng):
    for n in (4096, 4097):
        noise = colored_noise_from_psd(n, 4096, pink_psd, rng=rng)
        assert noise.shape == (n,)
        assert abs(np.mean(noise)) < 1e-12 * np.std(noise)


def test_colored_noise_scales_with_the_square_root_of_the_psd():
    base = colored_noise_from_psd(16384, 4096, pink_psd, rng=np.random.default_rng(1))
    doubled = colored_noise_from_psd(
        16384,
        4096,
        lambda f: 4 * pink_psd(f),
        rng=np.random.default_rng(1),
    )
    np.testing.assert_allclose(doubled, 2 * base, rtol=1e-12, atol=0)


def test_pink_noise_has_a_one_over_f_spectrum(rng):
    fs = 4096
    freqs, psd = welch(
        colored_noise_from_psd(N, fs, pink_psd, rng=rng), fs=fs, nperseg=4096
    )
    low = np.mean(psd[(freqs >= 18) & (freqs <= 22)])
    high = np.mean(psd[(freqs >= 1800) & (freqs <= 2200)])
    assert low / high == pytest.approx(100, rel=0.2)
    assert pink_psd(20.0, psd_at_f_ref=3.0, f_ref=10.0) == pytest.approx(1.5)


def test_a_psd_array_on_the_rfft_grid_gives_the_same_noise():
    n, fs = 2048, 4096
    freqs = np.fft.rfftfreq(n, 1 / fs)
    # The DC value is ignored, so an infinite pink PSD at 0 Hz is fine.
    values = np.concatenate([[np.inf], pink_psd(freqs[1:])])
    from_array = colored_noise_from_psd(n, fs, values, rng=np.random.default_rng(4))
    from_function = colored_noise_from_psd(
        n, fs, pink_psd, rng=np.random.default_rng(4)
    )
    np.testing.assert_array_equal(from_array, from_function)


def test_a_wrong_psd_raises(rng):
    with pytest.raises(ValueError, match="shape"):
        colored_noise_from_psd(2048, 4096, np.ones(10), rng=rng)
    with pytest.raises(ValueError, match="finite"):
        colored_noise_from_psd(2048, 4096, lambda f: -np.ones_like(f), rng=rng)
    with pytest.raises(ValueError, match="finite"):
        colored_noise_from_psd(2048, 4096, lambda f: np.full_like(f, np.nan), rng=rng)
