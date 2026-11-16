"""Tests for gwd.data.scenarios and gwd.data.injection (P3.5)."""

import dataclasses

import numpy as np
import pytest

from gwd.config import N_SAMPLES
from gwd.data import injection
from gwd.data.scenarios import APP_V1, EVAL_V1, SCENARIOS, TRAIN_V1, Scenario

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


def test_the_same_seed_gives_bit_identical_test_sets():
    X1, y1 = injection.make_test_set(EVAL_V1, 10, np.random.RandomState(4))
    X2, y2 = injection.make_test_set(EVAL_V1, 10, np.random.RandomState(4))
    assert X1.shape == (10, N_SAMPLES, 1)
    assert X1.tobytes() == X2.tobytes()
    assert y1.tolist() == y2.tolist()


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


def test_train_draws_the_noise_choice_even_without_chunks():
    rng = np.random.RandomState(9)
    sample = injection.make_sample(TRAIN_V1, rng, has_signal=False)
    reference = np.random.RandomState(9)
    reference.rand()
    assert np.array_equal(sample.data, injection.white_noise(N_SAMPLES, reference))


def test_noise_std_reference_scales_to_the_measured_noise_std():
    noise = 3.0 * np.random.RandomState(1).normal(size=N_SAMPLES)
    chirp = np.sin(np.linspace(0, 40, N_SAMPLES))
    scaled, snr = injection.scale_signal_to_snr(chirp, noise, 4.0, "noise_std")
    assert snr == 4.0
    assert np.isclose(np.max(np.abs(scaled)), 4.0 * np.std(noise))


def test_real_chunks_keep_the_v1_scale_skew(noise_folder):
    """V1 normalizes raw strain with the default eps, so the std is near 1e-11, not 1."""
    files = injection.real_noise_files(noise_folder)
    chunk = injection.load_random_real_noise(N_SAMPLES, files, np.random.RandomState(0))
    assert 1e-12 < np.std(chunk) < 1e-10


def test_the_v1_scenarios_are_frozen():
    """A changed V1 field would change what old datasets mean; add a V2 instead."""
    assert {name: s.hash() for name, s in SCENARIOS.items()} == {
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
    scenario = dataclasses.replace(EVAL_V1, name="X", condition_signal=True)
    with pytest.raises(NotImplementedError):
        injection.make_sample(scenario, np.random.RandomState(0))


@pytest.mark.xfail(
    strict=True, reason="V1 skew (DATA-1, EVAL-1, EVAL-2, EVAL-3), fixed as V2 in P6.2"
)
def test_known_skew():
    """Training and evaluation should draw from the same distribution; in V1 they do not."""
    assert TRAIN_V1.distribution() == EVAL_V1.distribution()
