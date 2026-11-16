"""Tests for gwd.data.scenarios and gwd.data.injection (P3.5)."""

import dataclasses
import sys

import numpy as np
import pytest

from gwd.config import BANDPASS, N_SAMPLES, SAMPLE_RATE
from gwd.core.population import random_bbh_params
from gwd.core.snr import optimal_snr_welch
from gwd.data import injection
from gwd.data.scenarios import APP_V1, EVAL_V1, TRAIN_V1, Scenario

V1 = [TRAIN_V1, EVAL_V1, APP_V1]


@pytest.fixture
def noise_folder(tmp_path):
    """Five raw-strain chunks with different content, like data/noise_background."""
    rng = np.random.default_rng(5)
    for i in range(5):
        chunk = rng.normal(0.0, 1e-21, N_SAMPLES) + i * 1e-22
        np.save(tmp_path / f"real_noise_{i:04d}.npy", chunk)
    return tmp_path


def batch(scenario, seed, provider=None, n=12):
    rng = np.random.RandomState(seed)
    return [
        injection.make_sample(
            scenario, rng, has_signal=(i % 2 == 0), noise_provider=provider
        )
        for i in range(n)
    ]


def assert_same(samples_a, samples_b):
    for a, b in zip(samples_a, samples_b, strict=True):
        assert a.data.tobytes() == b.data.tobytes()
        assert a.params == b.params
        assert a.noise_kind == b.noise_kind


@pytest.mark.parametrize("scenario", V1, ids=lambda s: s.name)
def test_the_same_seed_gives_bit_identical_samples(scenario, noise_folder):
    provider = injection.real_noise_provider(noise_folder)
    assert_same(batch(scenario, 3, provider), batch(scenario, 3, provider))


def test_the_chunk_does_not_depend_on_the_file_order(noise_folder, monkeypatch):
    provider = injection.real_noise_provider(noise_folder)
    expected = batch(TRAIN_V1, 8, provider)
    assert {s.noise_kind for s in expected} == {"real", "white"}

    listing = injection.glob.glob
    monkeypatch.setattr(
        injection.glob, "glob", lambda pattern: list(reversed(sorted(listing(pattern))))
    )
    reversed_provider = injection.real_noise_provider(noise_folder)
    assert_same(batch(TRAIN_V1, 8, reversed_provider), expected)


def test_without_chunks_there_is_no_provider(tmp_path):
    assert injection.real_noise_provider(tmp_path) is None


@pytest.mark.parametrize("scenario", V1, ids=lambda s: s.name)
def test_drawn_parameters_stay_in_the_scenario_bounds(scenario):
    rng = np.random.RandomState(11)
    params = [injection.draw_params(scenario, rng) for _ in range(2000)]
    snr = np.array([p.snr for p in params])
    t_merger = np.array([p.t_merger for p in params])
    m1 = np.array([p.mass1 for p in params])
    m2 = np.array([p.mass2 for p in params])

    assert scenario.snr_range[0] <= snr.min() and snr.max() <= scenario.snr_range[1]
    low, high = scenario.t_merger_range
    assert low <= t_merger.min() and t_merger.max() <= high
    if scenario.mass_mode == "random":
        assert scenario.masses[0] <= m2.min()
        assert np.all(m2 <= m1) and m1.max() <= scenario.masses[1]
    else:
        assert set(m1) == {scenario.masses[0]} and set(m2) == {scenario.masses[1]}


def test_a_fixed_snr_is_not_drawn():
    rng = np.random.RandomState(2)
    params = injection.draw_params(EVAL_V1, rng, snr=1.25)
    assert params.snr == 1.25
    # Only the merger time was drawn.
    reference = np.random.RandomState(2)
    assert params.t_merger == reference.uniform(*EVAL_V1.t_merger_range)
    assert rng.rand() == reference.rand()


def test_random_masses_come_from_the_core_population():
    """P5.3: masses and distance from random_bbh_params, then merger time and SNR.

    Before P5.3 two spins and an inclination were drawn after the distance and dropped.
    """
    params = injection.draw_params(TRAIN_V1, np.random.RandomState(5))
    reference = np.random.RandomState(5)
    binary = random_bbh_params(reference, TRAIN_V1.masses)
    assert (params.mass1, params.mass2, params.distance) == (
        binary["mass1"],
        binary["mass2"],
        binary["distance"],
    )
    assert params.t_merger == reference.uniform(*TRAIN_V1.t_merger_range)
    assert params.snr == reference.uniform(*TRAIN_V1.snr_range)


def test_a_signal_sample_is_noise_plus_the_scaled_chirp():
    sample = injection.make_sample(EVAL_V1, np.random.RandomState(6), has_signal=True)
    assert sample.has_signal and sample.noise_kind == "white"
    assert np.array_equal(sample.data, sample.noise + sample.signal)
    # "unit" reference: the chirp has peak 1, so the signal peak is the SNR.
    assert np.isclose(np.max(np.abs(sample.signal)), sample.snr, rtol=1e-12)


def test_noise_only_samples_have_no_signal_and_snr_zero():
    sample = injection.make_sample(APP_V1, np.random.RandomState(6), has_signal=False)
    assert sample.signal is None and sample.params is None and sample.snr == 0.0
    assert sample.data is sample.noise


def test_white_noise_samples_get_the_analytic_optimal_snr():
    sample = injection.make_sample(EVAL_V1, np.random.RandomState(6), has_signal=True)
    assert sample.peak_snr == sample.snr
    expected = np.sqrt(np.sum(sample.signal**2)) / injection.WHITE_NOISE_SIGMA
    assert sample.optimal_snr == pytest.approx(expected, rel=1e-12)
    # A 30/30 Msun chirp: rho_opt is about 13.5 times the peak SNR (P5.5).
    assert sample.optimal_snr / sample.peak_snr == pytest.approx(13.5, rel=0.1)


def test_real_noise_samples_get_the_optimal_snr_in_the_band(noise_folder):
    provider = injection.real_noise_provider(noise_folder)
    scenario = dataclasses.replace(TRAIN_V1, real_noise_fraction=1.0)
    sample = injection.make_sample(
        scenario, np.random.RandomState(2), has_signal=True, noise_provider=provider
    )
    assert sample.noise_kind == "real"
    expected = optimal_snr_welch(sample.signal, sample.noise, SAMPLE_RATE, *BANDPASS)
    assert sample.optimal_snr == expected > 0
    # The test chunk is white, so the band only drops power: below the full-band value.
    full = np.sqrt(np.sum(sample.signal**2)) / np.std(sample.noise)
    assert 0.0 < sample.optimal_snr < full


def test_noise_only_samples_have_optimal_snr_zero(noise_folder):
    provider = injection.real_noise_provider(noise_folder)
    for scenario in V1:
        sample = injection.make_sample(
            scenario,
            np.random.RandomState(1),
            has_signal=False,
            noise_provider=provider,
        )
        assert sample.optimal_snr == 0.0 and sample.peak_snr == 0.0


def test_train_draws_the_noise_choice_even_without_chunks():
    rng = np.random.RandomState(9)
    sample = injection.make_sample(TRAIN_V1, rng, has_signal=False)
    reference = np.random.RandomState(9)
    reference.rand()
    assert np.array_equal(sample.data, injection.white_noise(N_SAMPLES, reference))


def test_real_chunks_keep_the_v1_scale_skew(noise_folder):
    """V1 normalizes raw strain with the default eps, so the std is near 1e-11, not 1."""
    files = injection.real_noise_files(noise_folder)
    chunk = injection.load_random_real_noise(N_SAMPLES, files, np.random.RandomState(0))
    assert 1e-12 < np.std(chunk) < 1e-10


def test_the_v1_scenarios_are_frozen():
    """A changed V1 field would change what old datasets mean; add a V2 instead.

    The fields V2 added (P6.2) are left out of the hash at their V1 value, so the V1
    hashes in existing manifests stay valid.
    """
    assert {s.name: s.hash() for s in V1} == {
        "TRAIN_V1": "479afb1a7e1c",
        "EVAL_V1": "f251721c360b",
        "APP_V1": "6cc438d4cd62",
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        TRAIN_V1.snr_range = (0.0, 1.0)


def test_invalid_scenarios_are_rejected():
    fields = TRAIN_V1.distribution() | {"name": "X"}
    for change in (
        {"mass_mode": "uniform"},
        {"peak_reference": "psd"},
        {"real_noise_fraction": 1.5},
    ):
        with pytest.raises(ValueError):
            Scenario(**fields | change)


def test_conditioning_the_signal_is_not_part_of_v1():
    """Conditioning the signal (V2) needs the optimal SNR and a pad, so a V1 scenario
    with condition_signal is rejected; V2 is tested in tests/data/test_scenario_v2.py."""
    with pytest.raises(ValueError, match="optimal"):
        dataclasses.replace(EVAL_V1, name="X", condition_signal=True)
    with pytest.raises(ValueError, match="V1"):
        injection.make_conditioned_sample(EVAL_V1, np.random.RandomState(0))


@pytest.mark.xfail(
    strict=True,
    reason="V1 skew (DATA-1, EVAL-1, EVAL-2, EVAL-3), kept as documentation of the old "
    "data; V2 is consistent since P6.2 (tests/data/test_scenario_v2.py)",
)
def test_known_skew():
    """Training and evaluation should draw from the same distribution; in V1 they do not."""
    assert TRAIN_V1.distribution() == EVAL_V1.distribution()


def test_samples_name_their_noise_chunk_and_waveform_backend(noise_folder):
    provider = injection.real_noise_provider(noise_folder)
    samples = batch(TRAIN_V1, 8, provider, n=20)
    chunks = {path.name for path in noise_folder.iterdir()}
    assert {s.noise_kind for s in samples} == {"real", "white"}
    for s in samples:
        assert (s.noise_file in chunks) == (s.noise_kind == "real")
        assert s.noise_file is None or s.noise_kind == "real"
        if s.has_signal:
            assert s.waveform_backend == "pycbc"
        else:
            assert s.waveform_backend is None


def test_a_short_chunk_falls_back_to_white_noise_and_says_so(tmp_path):
    np.save(tmp_path / "short.npy", np.zeros(10))
    files = injection.real_noise_files(tmp_path)
    with pytest.warns(UserWarning, match="short.npy has 10 samples"):
        noise, name = injection.draw_real_noise(
            N_SAMPLES, files, np.random.RandomState(0)
        )
    assert name is None and noise.shape == (N_SAMPLES,)


def test_a_provider_fallback_is_recorded_as_white_noise():
    def white_provider(length, rng=None):
        return injection.white_noise(length, rng), None

    scenario = dataclasses.replace(TRAIN_V1, name="ALL_REAL", real_noise_fraction=1.0)
    sample = injection.make_sample(
        scenario, np.random.RandomState(0), noise_provider=white_provider
    )
    assert sample.noise_kind == "white" and sample.noise_file is None


def test_the_backend_is_passed_on_and_does_not_change_the_draws():
    """Newton is only ever chosen on purpose; the random stream is the same."""
    pycbc = injection.make_sample(TRAIN_V1, np.random.RandomState(4), has_signal=True)
    newton = injection.make_sample(
        TRAIN_V1, np.random.RandomState(4), has_signal=True, backend="newton"
    )
    assert (pycbc.waveform_backend, newton.waveform_backend) == ("pycbc", "newton")
    assert pycbc.params == newton.params
    np.testing.assert_array_equal(pycbc.noise, newton.noise)
    assert not np.array_equal(pycbc.signal, newton.signal)


def test_without_pycbc_a_signal_sample_raises(monkeypatch):
    from gwd.core.waveforms import WaveformBackendError

    monkeypatch.setitem(sys.modules, "pycbc", None)
    with pytest.raises(WaveformBackendError):
        injection.make_sample(TRAIN_V1, np.random.RandomState(4), has_signal=True)
