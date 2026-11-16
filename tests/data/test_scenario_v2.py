"""Scenario V2: inject first, then condition together, in simulated aLIGO noise (P6.2).

The acceptance tests of the plan: (a) the same seed gives a bit-identical sample;
(b) the label means the same in white and colored noise: the optimal SNR measured in
the conditioned sample is within 10 % of the drawn one; (c) no shortcut above 300 Hz:
less than 1 % of the conditioned signal's energy lies there; (d) TRAIN_V2 and INDIST_V2
draw from the same distribution (test_v2_consistent). Besides: the normalization does
not depend on the noise type, the pads keep the filter ringing out, the V2 test sets
and their extrapolation bin, a V2 dataset that validates, and the evaluation in the
optimal SNR.

V2 conditions with gwpy and draws its noise from the aLIGO PSD of PyCBC; without them
these tests skip (CI installs both and sets GWD_REQUIRE_PYCBC=1). The chirps come from
the Newton backend where the waveform does not matter, which needs no PyCBC.
"""

import dataclasses
import os

import numpy as np
import pytest

from gwd.config import BANDPASS, N_SAMPLES, SAMPLE_RATE
from gwd.core import waveforms
from gwd.core.snr import optimal_snr_welch
from gwd.data import injection
from gwd.data.scenarios import (
    APP_V2,
    INDIST_V2,
    SCENARIOS,
    STRESS_V2,
    TESTSETS,
    TRAIN_V1,
    TRAIN_V2,
    Scenario,
)

pytest.importorskip("gwpy")
if not waveforms.pycbc_available():
    if os.environ.get("GWD_REQUIRE_PYCBC") == "1":
        pytest.fail(
            "pycbc missing, but GWD_REQUIRE_PYCBC=1 requires it",
            pytrace=False,
        )
    pytest.skip("pycbc missing (aLIGO PSD)", allow_module_level=True)

TARGET = 8.0
WHITE_V2 = dataclasses.replace(TRAIN_V2, name="WHITE_V2", simulated_noise="white")
NOISES = {"aligo": TRAIN_V2, "white": WHITE_V2}
# Masses (M☉) from light, with much energy above 300 Hz unfiltered, to heavy.
MASSES = [(5.0, 5.0), (10.0, 5.0), (30.0, 30.0), (60.0, 10.0), (100.0, 100.0)]


def signal_sample(scenario, seed, mass1, mass2, backend="pycbc", snr=TARGET):
    fixed = dataclasses.replace(scenario, mass_mode="fixed", masses=(mass1, mass2))
    return injection.make_sample(
        fixed, np.random.RandomState(seed), has_signal=True, snr=snr, backend=backend
    )


def energy_above(x, f_cut=BANDPASS[1]):
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
    return power[freqs > f_cut].sum() / power.sum()


# --- (a) reproducible ---


@pytest.mark.parametrize("has_signal", [True, False])
def test_the_same_seed_gives_a_bit_identical_sample(has_signal):
    first, second = (
        injection.make_sample(TRAIN_V2, np.random.RandomState(3), has_signal=has_signal)
        for _ in range(2)
    )
    assert first.data.tobytes() == second.data.tobytes()
    assert first.noise.tobytes() == second.noise.tobytes()
    assert first.params == second.params
    other = injection.make_sample(
        TRAIN_V2, np.random.RandomState(4), has_signal=has_signal
    )
    assert not np.array_equal(first.data, other.data)


# --- (b) label invariance ---


@pytest.mark.parametrize("noise", sorted(NOISES))
@pytest.mark.parametrize("mass1, mass2", MASSES)
def test_the_measured_optimal_snr_matches_the_label(noise, mass1, mass2):
    """In the conditioned sample, rho_opt (Welch PSD of its noise, 20-300 Hz) / 8 is in
    [0.9, 1.1] for white and aLIGO noise alike."""
    for seed in range(3):
        sample = signal_sample(NOISES[noise], seed, mass1, mass2)
        assert sample.optimal_snr == TARGET == sample.params.snr
        measured = optimal_snr_welch(
            sample.signal, sample.noise, SAMPLE_RATE, *BANDPASS
        )
        assert 0.9 <= measured / TARGET <= 1.1, (seed, measured)


def test_the_optimal_snr_is_the_one_of_the_window_against_the_psd():
    sample = signal_sample(TRAIN_V2, 0, 30.0, 30.0)
    assert sample.has_signal and sample.noise_kind == "aligo"
    # The peak SNR is measured in the conditioned sample; for 30/30 Msun at rho_opt 8
    # in aLIGO noise it is far below the V1 range (peak SNR 1 is rho_opt ~ 13.5 there).
    assert 0.3 < sample.snr == sample.peak_snr < 3.0
    np.testing.assert_allclose(sample.data, sample.noise + sample.signal, atol=1e-12)


# --- (c) no shortcut above 300 Hz ---


@pytest.mark.parametrize("mass1, mass2", MASSES)
def test_no_signal_energy_above_300_hz_is_left(mass1, mass2):
    sample = signal_sample(TRAIN_V2, 1, mass1, mass2)
    assert energy_above(sample.signal) < 0.01


def test_the_unconditioned_v1_chirp_had_the_shortcut():
    """For contrast: a light V1 chirp keeps several percent of its energy above 300 Hz."""
    v1 = dataclasses.replace(TRAIN_V1, mass_mode="fixed", masses=(5.0, 5.0))
    sample = injection.make_sample(v1, np.random.RandomState(1), has_signal=True)
    assert energy_above(sample.signal) > 0.01


# --- (d) consistent ---


def test_v2_consistent():
    """TRAIN_V2 and INDIST_V2 are the same distribution; the sets differ by seed and n."""
    assert TRAIN_V2.distribution() == INDIST_V2.distribution()
    assert TESTSETS["indist_v2"].scenario is INDIST_V2
    for name in ("indist_v2", "stress_v2"):
        assert TESTSETS[name].training is TRAIN_V2


def test_stress_v2_adds_one_extrapolation_bin_below_the_training_range():
    spec = TESTSETS["stress_v2"]
    indist = TESTSETS["indist_v2"]
    assert STRESS_V2.distribution() == {
        **TRAIN_V2.distribution(),
        "snr_range": STRESS_V2.snr_range,
    }
    edges, train_edges = spec.snr_edges(), indist.snr_edges()
    assert edges[1:] == train_edges
    assert edges[1] == TRAIN_V2.snr_range[0] and edges[0] < edges[1]
    assert spec.snr_kind == indist.snr_kind == "optimal"


def test_the_v2_scenarios():
    for scenario in (TRAIN_V2, INDIST_V2, STRESS_V2, APP_V2):
        assert SCENARIOS[scenario.name] is scenario
        assert scenario.version == 2
        assert scenario.condition_signal and scenario.snr_kind == "optimal"
        assert scenario.simulated_noise == "aligo" and scenario.real_noise_fraction == 0
    # The Glitch Hunter's challenge lies inside the training ranges (APP-2).
    assert TRAIN_V2.snr_range[0] <= APP_V2.snr_range[0] < APP_V2.snr_range[1]
    assert APP_V2.snr_range[1] <= TRAIN_V2.snr_range[1]
    assert TRAIN_V2.masses[0] <= min(APP_V2.masses) <= max(APP_V2.masses)
    assert max(APP_V2.masses) <= TRAIN_V2.masses[1]


def test_invalid_v2_scenarios_are_rejected():
    fields = dataclasses.asdict(TRAIN_V2) | {"name": "X"}
    for change in (
        {"snr_kind": "peak"},
        {"peak_reference": "unit"},
        {"pad_s": 0.0},
        {"real_noise_fraction": 0.5},
        {"simulated_noise": "pink"},
        {"snr_kind": "matched"},
    ):
        with pytest.raises(ValueError):
            Scenario(**fields | change)
    v1_fields = dataclasses.asdict(TRAIN_V1)
    for change in ({"simulated_noise": "aligo"}, {"pad_s": 1.0}):
        with pytest.raises(ValueError):
            Scenario(**v1_fields | change)


# --- normalization, edges, noise ---


@pytest.mark.parametrize("noise", sorted(NOISES))
def test_every_sample_is_normalized_whatever_the_noise(noise):
    """P3.2: V1 real chunks reached the CNN at a std of 0.07-0.21, white noise at 1.
    V2 normalizes the conditioned sum with eps 0, so both noise types give std 1."""
    rng = np.random.RandomState(0)
    for has_signal in (False, True, False, True):
        sample = injection.make_sample(
            NOISES[noise], rng, has_signal=has_signal, backend="newton"
        )
        assert sample.data.shape == (N_SAMPLES,)
        assert sample.noise_kind == noise
        assert abs(np.mean(sample.data)) < 1e-12
        assert np.std(sample.data) == pytest.approx(1.0, rel=1e-12)


def test_the_simulated_noise_has_the_aligo_psd():
    n = 8 * SAMPLE_RATE
    noise = injection.draw_simulated_noise("aligo", n, np.random.RandomState(0))
    freqs = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    psd = injection.simulated_psd("aligo", n)
    assert np.all(psd[freqs < injection.SIMULATED_PSD_F_LOW] == 0)
    from gwd.core.snr import welch_psd

    f, measured = welch_psd(noise, SAMPLE_RATE, 1.0)
    band = (f >= 30) & (f <= 300)
    expected = np.interp(f[band], freqs, psd)
    np.testing.assert_allclose(np.median(measured[band] / expected), 1.0, rtol=0.1)


@pytest.mark.parametrize("noise", sorted(NOISES))
def test_the_pads_keep_the_filter_ringing_out(noise):
    """The RMS of the first and last 0.1 s against the middle 0.1 s, pooled over 20
    noise-only samples, is close to 1 (no edge transient of the conditioning)."""
    rng = np.random.RandomState(1)
    k = int(0.1 * SAMPLE_RATE)
    middle = N_SAMPLES // 2
    edges, centers = [], []
    for _ in range(20):
        x = injection.make_sample(NOISES[noise], rng, has_signal=False).data
        edges += [np.mean(x[:k] ** 2), np.mean(x[-k:] ** 2)]
        centers.append(np.mean(x[middle - k // 2 : middle + k // 2] ** 2))
    ratio = np.sqrt(np.mean(edges) / np.mean(centers))
    assert 0.9 < ratio < 1.15


def test_v1_samples_are_unchanged_by_v2():
    """V1 still draws as before: no conditioning of the signal, the same stream."""
    sample = injection.make_sample(
        TRAIN_V1, np.random.RandomState(0), has_signal=True, backend="newton"
    )
    reference = np.random.RandomState(0)
    reference.rand()
    np.testing.assert_array_equal(
        sample.noise, injection.white_noise(N_SAMPLES, reference)
    )
    np.testing.assert_array_equal(sample.data, sample.noise + sample.signal)


# --- test sets, dataset, evaluation ---


@pytest.fixture(scope="module")
def v2_sets(tmp_path_factory):
    from gwd.evaluation import testsets

    root = tmp_path_factory.mktemp("v2sets")
    return {
        name: testsets.build(
            name,
            40,
            5,
            backend="newton",
            data_dir=str(root / "eval"),
            manifest_dir=str(root / "manifests"),
            noise_dir=str(root / "noise"),
        )
        for name in ("indist_v2", "stress_v2")
    }


def test_the_v2_sets_are_stratified_in_the_optimal_snr(v2_sets):
    from gwd.evaluation import testsets
    from gwd.evaluation.metrics import bin_index

    for name, built in v2_sets.items():
        signal = built.y == 1
        edges = built.spec.snr_edges()
        counts = np.bincount(
            bin_index(built.optimal_snr[signal], edges), minlength=len(edges) - 1
        )
        assert (counts > 0).all(), (name, counts)
        bins = testsets.bins(built)
        np.testing.assert_array_equal(bins["optimal_snr"], edges)
        assert bins["peak_snr"].size == testsets.OPTIMAL_SNR_BINS + 1
        assert set(map(str, built.noise_source)) == {"aligo"}
        assert np.abs(built.X.std(axis=1) - 1).max() < 1e-3


def test_only_the_extrapolation_bin_lies_outside_training(v2_sets):
    from gwd.evaluation import testsets

    stress, indist = v2_sets["stress_v2"], v2_sets["indist_v2"]
    signal = stress.y == 1
    flags = testsets.outside_training(stress)
    expected = stress.optimal_snr[signal] < TRAIN_V2.snr_range[0]
    assert expected.any()
    np.testing.assert_array_equal(flags[signal], expected)
    assert not testsets.outside_training(indist).any()


def test_the_v2_manifest_stores_the_parameters_of_every_sample(v2_sets):
    """For the matched filter baseline (P6.4): masses, merger time, and SNRs per sample."""
    import json

    built = v2_sets["indist_v2"]
    with open(built.manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["scenario"] == "INDIST_V2"
    assert manifest["scenario_hash"] == INDIST_V2.hash()
    assert manifest["training_reference"] == "TRAIN_V2"
    assert manifest["noise"] is None
    rows = manifest["samples"]["test"]
    assert len(rows) == 40
    signal_rows = [row for row in rows if row["label"] == 1]
    for row in signal_rows:
        assert row["mass1"] >= row["mass2"] > 0
        assert TRAIN_V2.snr_range[0] <= row["optimal_snr"] <= TRAIN_V2.snr_range[1]
        assert row["peak_snr"] > 0 and row["t_merger"] > 0


def test_the_efficiency_curve_of_a_v2_set_is_over_the_optimal_snr(v2_sets):
    from gwd.evaluation import evaluate

    built = v2_sets["stress_v2"]
    scores = np.ones(built.n)
    centers, rates, peaks = evaluate.efficiency_curve(built, scores, 0.5)
    np.testing.assert_allclose(centers, np.arange(3.0, 20.0, 2.0))
    assert rates == [1.0] * len(centers)
    # The third value is the mean peak SNR of each bin, which grows with the optimal SNR.
    assert np.all(np.array(peaks) > 0)
    assert peaks[-1] > peaks[0]


def test_a_v2_dataset_validates(tmp_path, capsys):
    from gwd.data.generate import generate_dataset
    from gwd.data.validate import check_dataset

    out = tmp_path / "v2"
    generate_dataset(
        12,
        str(out),
        TRAIN_V2,
        seed=1,
        noise_dir=str(tmp_path / "none"),
        backend="newton",
    )
    printed = capsys.readouterr().out
    assert "simulated aLIGO design noise" in printed
    integrity, distribution, name = check_dataset(str(out))
    assert (integrity, distribution, name) == ([], [], "TRAIN_V2")
    import pandas as pd

    labels = pd.read_csv(out / "labels.csv")
    assert set(labels["noise_source"]) == {"aligo"}
    signal = labels[labels["has_signal"] == 1]
    assert signal["optimal_snr"].between(*TRAIN_V2.snr_range).all()
    assert (signal["peak_snr"] == signal["snr"]).all()
