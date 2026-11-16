"""gwd.core.snr: peak SNR, optimal SNR, and the scaling to either (P5.5)."""

import numpy as np
import pytest

from gwd.core import snr
from gwd.core.interferometer import reference_psd
from gwd.core.noise import colored_noise_from_psd, pink_psd, white_noise
from gwd.core.timebase import time_axis
from gwd.core.waveforms import generate_chirp

FS = 4096
DURATION = 4.0
BAND = (20.0, 300.0)  # gwd.config.BANDPASS; gwd.core reads no config


def chirp(backend="pycbc", t_merger=3.0, mass1=30.0, mass2=30.0):
    h, _ = generate_chirp(
        time_axis(FS, DURATION),
        backend=backend,
        t_merger=t_merger,
        mass1=mass1,
        mass2=mass2,
        normalize=True,
    )
    return h


def aligo_psd(f):
    return reference_psd(f, low_freq_cutoff=15.0)


def test_peak_and_white_definitions():
    h = np.array([0.0, 3.0, -4.0, 0.0])
    assert snr.peak_snr(h, 2.0) == 2.0
    assert snr.optimal_snr_white(h, 2.0) == 2.5


def test_a_flat_psd_gives_the_white_optimal_snr():
    h = chirp()
    sigma = 0.7
    # The one-sided PSD of white noise with std sigma is 2 sigma^2 / fs.
    assert snr.optimal_snr_psd(h, 2 * sigma**2 / FS, FS) == pytest.approx(
        snr.optimal_snr_white(h, sigma), rel=1e-6
    )


def test_the_psd_may_be_a_function_or_an_array():
    h = chirp()
    freqs = np.fft.rfftfreq(len(h), 1 / FS)
    values = np.zeros_like(freqs)
    values[1:] = pink_psd(freqs[1:])
    assert snr.optimal_snr_psd(h, pink_psd, FS, *BAND) == pytest.approx(
        snr.optimal_snr_psd(h, values, FS, *BAND), rel=1e-12
    )


def test_a_psd_of_zero_in_the_band_is_refused():
    h = chirp()
    with pytest.raises(ValueError, match="finite and > 0"):
        snr.optimal_snr_psd(h, lambda f: np.where(f < 50, 0.0, 1.0), FS, *BAND)
    with pytest.raises(ValueError, match="shape"):
        snr.optimal_snr_psd(h, np.ones(10), FS)


def measure_kwargs(rng):
    noise = colored_noise_from_psd(len(chirp()), FS, pink_psd, rng=rng)
    return {
        "peak": {"sigma": 1.5},
        "optimal_white": {"sigma": 1.5},
        "optimal_psd": {"psd": pink_psd, "fs": FS, "f_low": BAND[0], "f_high": BAND[1]},
        "optimal_welch": {
            "noise": noise,
            "fs": FS,
            "f_low": BAND[0],
            "f_high": BAND[1],
        },
    }


@pytest.mark.parametrize("kind", list(snr.KINDS))
@pytest.mark.parametrize("target", [0.5, 8.0, 30.0])
def test_measure_of_scale_to_snr_is_the_target(kind, target, rng):
    kwargs = measure_kwargs(rng)[kind]
    scaled = snr.scale_to_snr(chirp(), target, kind, **kwargs)
    assert snr.measure(scaled, kind, **kwargs) == pytest.approx(target, rel=1e-12)


def test_measure_refuses_unknown_kinds_and_arguments():
    with pytest.raises(ValueError, match="kind"):
        snr.measure(chirp(), "matched", sigma=1.0)
    with pytest.raises(TypeError, match="psd"):
        snr.measure(chirp(), "peak", sigma=1.0, psd=1.0)
    with pytest.raises(ValueError, match="SNR is 0"):
        snr.scale_to_snr(np.zeros(16), 5.0, "peak", sigma=1.0)


@pytest.mark.parametrize("t_merger", [2.0, 2.75, 3.5])
def test_optimal_over_peak_for_30_30_in_white_noise(t_merger, needs_pycbc):
    """Reproduces the finding: peak SNR 1 of a 30/30 Msun chirp is rho_opt ~ 13.5."""
    h = chirp("pycbc", t_merger)
    ratio = snr.optimal_snr_white(h, 1.0) / snr.peak_snr(h, 1.0)
    assert ratio == pytest.approx(13.5, rel=0.10)


def test_optimal_snr_psd_matches_pycbc_sigma(needs_pycbc):
    from pycbc.filter import sigma
    from pycbc.types import FrequencySeries, TimeSeries

    h = chirp("pycbc", 3.0) * 1e-21
    htilde = TimeSeries(h, delta_t=1 / FS).to_frequencyseries()
    freqs = htilde.sample_frequencies.numpy()
    values = np.zeros_like(freqs)
    values[1:] = aligo_psd(freqs[1:])
    psd = FrequencySeries(values, delta_f=htilde.delta_f)

    expected = sigma(
        htilde, psd, low_frequency_cutoff=BAND[0], high_frequency_cutoff=BAND[1]
    )
    assert snr.optimal_snr_psd(h, aligo_psd, FS, *BAND) == pytest.approx(
        expected, rel=0.01
    )


@pytest.mark.parametrize("name", ["pink", "aligo"])
def test_the_welch_variant_finds_the_snr_of_a_known_psd(name, rng, request):
    """Colored noise from a known PSD: the Welch estimate lands within 10 %."""
    if name == "aligo":
        request.getfixturevalue("needs_pycbc")
        psd, scale = aligo_psd, 1e-21
    else:
        psd, scale = pink_psd, 1.0
    h = chirp() * scale
    analytic = snr.optimal_snr_psd(h, psd, FS, *BAND)
    for _ in range(5):
        noise = colored_noise_from_psd(len(h), FS, psd, rng=rng)
        welch = snr.optimal_snr_welch(h, noise, FS, *BAND)
        assert welch == pytest.approx(analytic, rel=0.10)


def test_the_welch_variant_of_white_noise_is_the_white_optimal_snr(rng):
    h = chirp()
    noise = white_noise(len(h), 1.0, rng=rng)
    full_band = snr.optimal_snr_welch(h, noise, FS, 1.0, FS / 2)
    assert full_band == pytest.approx(snr.optimal_snr_white(h, 1.0), rel=0.10)


def test_scale_signal_to_snr_is_the_v1_peak_scaling(rng):
    h = chirp()
    noise = white_noise(len(h), 2.0, rng=rng)
    scaled, value = snr.scale_signal_to_snr(h, noise, 4.0)
    assert value == 4.0
    assert snr.peak_snr(scaled, np.std(noise)) == pytest.approx(4.0, rel=1e-12)
    unit, value = snr.scale_signal_to_snr(h, noise, 1.5, "unit")
    assert value == 1.5 and np.array_equal(unit, h * 1.5)
    zeros, value = snr.scale_signal_to_snr(np.zeros(8), noise[:8], 3.0)
    assert value == 0.0 and not zeros.any()
