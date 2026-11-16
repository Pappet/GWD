"""Scenario V3: V2 plus whitening with the PSD before the band-pass.

The checks of the step: (1) V3 is V2 except for the whitening, and the V1 and V2
scenario hashes and test set hashes did not change; (2) with the same seed, indist_v3
and stress_v3 hold the same events and noise realizations as indist_v2 and stress_v2;
(3) whitening simulated aLIGO noise gives a flat spectrum and unit variance after
normalize; (4) the optimal SNR of a chirp means the same in V3; (5) seeds reproduce;
(6) the real data path whitens with the Welch PSD of the padded data and agrees with the
known PSD path; (7) a model gets the preprocessing it was trained with.

V3 conditions with gwpy and whitens with the aLIGO PSD of PyCBC; without them these
tests skip (CI installs both and sets GWD_REQUIRE_PYCBC=1).
"""

import dataclasses
import json
import os

import numpy as np
import pytest

from gwd.config import BANDPASS, N_SAMPLES, SAMPLE_RATE
from gwd.core import waveforms
from gwd.core.snr import optimal_snr_psd, optimal_snr_welch, welch_psd
from gwd.data import injection
from gwd.data.preprocessing import (
    condition_array,
    conditioned_crop,
    normalize,
    whiten,
)
from gwd.data.scenarios import (
    APP_V2,
    APP_V3,
    INDIST_V2,
    INDIST_V3,
    SCENARIOS,
    STRESS_V2,
    STRESS_V3,
    TESTSETS,
    TRAIN_V2,
    TRAIN_V3,
    Scenario,
    counterpart_set,
    preprocessing_of,
    training_scenario,
)

pytest.importorskip("gwpy")
if not waveforms.pycbc_available():
    if os.environ.get("GWD_REQUIRE_PYCBC") == "1":
        pytest.fail(
            "pycbc missing, but GWD_REQUIRE_PYCBC=1 requires it",
            pytrace=False,
        )
    pytest.skip("pycbc missing (aLIGO PSD)", allow_module_level=True)

PAIRS = [(TRAIN_V2, TRAIN_V3), (INDIST_V2, INDIST_V3), (STRESS_V2, STRESS_V3)]
PAIRS += [(APP_V2, APP_V3)]
WHITE_V2 = dataclasses.replace(TRAIN_V2, name="WHITE_V2", simulated_noise="white")
MASSES = [(5.0, 5.0), (10.0, 5.0), (30.0, 30.0), (60.0, 10.0), (100.0, 100.0)]
TARGET = 8.0


def fixed(scenario, mass1, mass2):
    return dataclasses.replace(scenario, mass_mode="fixed", masses=(mass1, mass2))


def band_means(x, width=10, band=(20, 300)):
    """The Welch PSD (1 s segments) of x, averaged in bands of width Hz."""
    freqs, psd = welch_psd(x, SAMPLE_RATE, 1.0)
    return np.array(
        [
            np.mean(psd[(freqs >= low) & (freqs < low + width)])
            for low in range(band[0], band[1], width)
        ]
    )


def noise_spectrum(scenario, seed, k=8):
    rng = np.random.RandomState(seed)
    data = [
        injection.make_sample(scenario, rng, has_signal=False).data for _ in range(k)
    ]
    return band_means(np.concatenate(data))


# --- (1) V3 is V2 plus whitening; V1 and V2 unchanged ---


@pytest.mark.parametrize("v2, v3", PAIRS, ids=[v3.name for _, v3 in PAIRS])
def test_v3_is_v2_with_whitening(v2, v3):
    assert SCENARIOS[v3.name] is v3
    assert (v3.version, v3.whiten, v3.preprocessing) == (3, True, "whiten_bandpass")
    assert (v2.whiten, v2.preprocessing) == (False, "bandpass")
    assert dataclasses.replace(v3, name=v2.name, version=2, whiten=False) == v2


def test_the_v2_scenario_hashes_did_not_change():
    """whiten is left out of the hash at False, like the V2 fields in V1 (P6.2)."""
    assert {s.name: s.hash() for s in (TRAIN_V2, INDIST_V2, STRESS_V2, APP_V2)} == {
        "TRAIN_V2": "505fba09ba85",
        "INDIST_V2": "241d5e281d4a",
        "STRESS_V2": "2ab2f2854a8b",
        "APP_V2": "4ee22bf234f6",
    }
    assert len({s.hash() for s in SCENARIOS.values()}) == len(SCENARIOS)


# SHA-256 of test sets built before V3 (empty noise folder). The two n = 1000 sets are
# the ones in docs/architecture.md and the owner's results/testsets/.
SET_HASHES = [
    (
        "stress_v1",
        1000,
        "00a4d6967b55b2de0a67cc7ece790a0ec7aa7034263186a2481717349f91bea1",
    ),
    (
        "indist_v2",
        1000,
        "c159ad58094e81d5c79cee286fc6f30b8a0083dce38a7d6eae9fe2d89f44c5fd",
    ),
    (
        "indist_v1",
        64,
        "4854d4bbeb9253f33f296676cab468d61f2e32c71d3f20fa024cfe29a4eef733",
    ),
    (
        "stress_v2",
        16,
        "11103ebd02ac524119f0879cdab5ce09e95cd0e09899d5a38d44298e726efaf0",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize("name, n, sha256", SET_HASHES, ids=[h[0] for h in SET_HASHES])
def test_the_v1_and_v2_test_sets_did_not_change(tmp_path, name, n, sha256):
    """Seed 0, as before V3. Slow (indist_v2 with n 1000 takes about 20 s), and the
    hashes are those of this laptop's PyCBC and numpy builds."""
    from gwd.evaluation import testsets

    built = testsets.build(
        name,
        n,
        0,
        data_dir=str(tmp_path / "eval"),
        manifest_dir=str(tmp_path / "manifests"),
        noise_dir=str(tmp_path / "noise"),
    )
    assert built.sha256 == sha256


def test_whiten_needs_the_v2_chain():
    fields = dataclasses.asdict(SCENARIOS["TRAIN_V1"]) | {"name": "X", "whiten": True}
    with pytest.raises(ValueError, match="whiten needs condition_signal"):
        Scenario(**fields)


def test_the_v3_sets_draw_from_the_v2_streams():
    for name, partner in (("indist_v3", "indist_v2"), ("stress_v3", "stress_v2")):
        spec, other = TESTSETS[name], TESTSETS[partner]
        assert spec.stream_name == partner and other.stream_name == partner
        assert spec.snr_edges() == other.snr_edges()
        assert spec.training is TRAIN_V3
        assert counterpart_set(other, "whiten_bandpass") is spec
        assert counterpart_set(spec, "bandpass") is other
        assert counterpart_set(spec, "whiten_bandpass") is spec
    assert counterpart_set(TESTSETS["stress_v1"], "whiten_bandpass") is None


# --- (2) the same samples, preprocessed differently ---


@pytest.fixture(scope="module")
def paired_sets(tmp_path_factory):
    """indist and stress in V2 and V3 with the same seed and n, and the raw noise each
    drew (recorded from injection.draw_simulated_noise)."""
    from gwd.evaluation import testsets

    root = tmp_path_factory.mktemp("v3sets")
    built, raw = {}, {}
    original = injection.draw_simulated_noise
    with pytest.MonkeyPatch.context() as patch:
        for name in ("indist_v2", "indist_v3", "stress_v2", "stress_v3"):
            drawn = raw[name] = []

            def record(*args, _drawn=drawn, **kwargs):
                noise = original(*args, **kwargs)
                _drawn.append(noise.copy())
                return noise

            patch.setattr(injection, "draw_simulated_noise", record)
            built[name] = testsets.build(
                name,
                16,
                7,
                data_dir=str(root / "eval"),
                manifest_dir=str(root / "manifests"),
                noise_dir=str(root / "noise"),
            )
    return built, raw


@pytest.mark.parametrize("kind", ["indist", "stress"])
def test_v2_and_v3_sets_hold_the_same_events_and_noise(paired_sets, kind):
    """The same seed gives the same labels, parameters, and raw noise realizations;
    only the preprocessing (and so the arrays and the measured peak SNR) differs."""
    built, raw = paired_sets
    v2, v3 = built[f"{kind}_v2"], built[f"{kind}_v3"]
    for key in ("test_y", "test_optimal_snr", "test_mass1", "test_mass2"):
        np.testing.assert_array_equal(v2.arrays[key], v3.arrays[key])
    np.testing.assert_array_equal(
        v2.arrays["test_t_merger"], v3.arrays["test_t_merger"]
    )
    noise2, noise3 = raw[f"{kind}_v2"], raw[f"{kind}_v3"]
    assert len(noise2) == len(noise3) == 2 * 16  # test and calib_bg
    for a, b in zip(noise2, noise3):
        np.testing.assert_array_equal(a, b)
    assert not np.allclose(v2.X, v3.X)
    assert v2.sha256 != v3.sha256


def test_the_v3_manifest_names_the_stream_and_the_preprocessing(paired_sets):
    built, _ = paired_sets
    with open(built["indist_v3"].manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["scenario"] == "INDIST_V3"
    assert manifest["scenario_hash"] == INDIST_V3.hash()
    assert manifest["stream"] == "indist_v2"
    assert manifest["preprocessing"] == "whiten_bandpass"
    assert manifest["training_reference"] == "TRAIN_V3"


def test_a_v3_sample_draws_as_a_v2_sample():
    """make_sample consumes the same random numbers in V2 and V3."""
    for has_signal in (True, False, None):
        rng2, rng3 = np.random.RandomState(11), np.random.RandomState(11)
        v2 = injection.make_sample(TRAIN_V2, rng2, has_signal=has_signal)
        v3 = injection.make_sample(TRAIN_V3, rng3, has_signal=has_signal)
        assert (v2.params, v2.has_signal, v2.optimal_snr) == (
            v3.params,
            v3.has_signal,
            v3.optimal_snr,
        )
        assert rng2.rand() == rng3.rand()


# --- (3) flat spectrum, unit variance ---


def test_whitening_simulated_aligo_noise_gives_a_flat_spectrum():
    """The known PSD whitens the simulated noise: its Welch PSD times fs/2 is 1 within
    20 % in every 10 Hz band from 20 to 300 Hz, and its std is 1. Measured with seed 0:
    0.89 to 1.14, which is the scatter of the estimate from 32 s; the aLIGO PSD itself
    spans a factor of 33 in that band."""
    n = 32 * SAMPLE_RATE
    noise = injection.draw_simulated_noise("aligo", n, np.random.RandomState(0))
    white = whiten(noise, SAMPLE_RATE, injection.simulated_psd("aligo", n))
    levels = band_means(white) * SAMPLE_RATE / 2
    assert np.all((levels > 0.8) & (levels < 1.2)), levels
    assert np.std(white) == pytest.approx(1.0, rel=0.03)


def test_the_v3_noise_has_the_shape_of_the_band_pass_only():
    """After the whole V3 chain the noise spectrum is that of white noise through the
    same notches and band-pass (gwpy's band-pass has a ripple of its own), within 25 %
    per 10 Hz band; in V2 the aLIGO colour leaves a range of more than 5 times."""
    reference = noise_spectrum(WHITE_V2, 0)
    v3 = noise_spectrum(TRAIN_V3, 1) / reference
    v2 = noise_spectrum(TRAIN_V2, 1) / reference
    v3 /= np.median(v3)
    assert np.all((v3 > 0.75) & (v3 < 1.33)), v3
    assert v2.max() / v2.min() > 5


def test_every_v3_sample_has_unit_variance():
    rng = np.random.RandomState(2)
    for has_signal in (False, True, False, True):
        sample = injection.make_sample(TRAIN_V3, rng, has_signal=has_signal)
        assert sample.data.shape == (N_SAMPLES,)
        assert abs(np.mean(sample.data)) < 1e-12
        assert np.std(sample.data) == pytest.approx(1.0, rel=1e-12)
        if has_signal:
            np.testing.assert_allclose(
                sample.data, sample.noise + sample.signal, atol=1e-12
            )


def test_the_pads_keep_the_ringing_out_of_v3():
    rng = np.random.RandomState(1)
    k = int(0.1 * SAMPLE_RATE)
    middle = N_SAMPLES // 2
    edges, centers = [], []
    for _ in range(20):
        x = injection.make_sample(TRAIN_V3, rng, has_signal=False).data
        edges += [np.mean(x[:k] ** 2), np.mean(x[-k:] ** 2)]
        centers.append(np.mean(x[middle - k // 2 : middle + k // 2] ** 2))
    assert 0.9 < np.sqrt(np.mean(edges) / np.mean(centers)) < 1.15


# --- (4) the optimal SNR keeps its meaning ---


@pytest.mark.parametrize("mass1, mass2", MASSES)
def test_whitening_keeps_the_optimal_snr_of_a_chirp(mass1, mass2):
    """By definition: rho against the aLIGO PSD equals rho of the whitened chirp
    against the flat PSD of unit white noise, in BANDPASS."""
    n = 8 * SAMPLE_RATE
    time = np.arange(n) / SAMPLE_RATE - 2.0
    chirp, _ = waveforms.generate_chirp(
        time, backend="pycbc", t_merger=3.0, mass1=mass1, mass2=mass2
    )
    psd = injection.simulated_psd("aligo", n)
    flat = np.full_like(psd, 2.0 / SAMPLE_RATE)
    colored = optimal_snr_psd(chirp, psd, SAMPLE_RATE, *BANDPASS)
    white = optimal_snr_psd(
        whiten(chirp, SAMPLE_RATE, psd), flat, SAMPLE_RATE, *BANDPASS
    )
    assert white == pytest.approx(colored, rel=1e-9)


@pytest.mark.parametrize("mass1, mass2", MASSES)
def test_the_measured_optimal_snr_of_a_v3_sample_matches_the_label(mass1, mass2):
    """As (b) of V2: rho_opt measured in the whitened, conditioned sample against the
    Welch PSD of its noise is within 10 % of the drawn 8."""
    for seed in range(3):
        sample = injection.make_sample(
            fixed(TRAIN_V3, mass1, mass2),
            np.random.RandomState(seed),
            has_signal=True,
            snr=TARGET,
        )
        assert sample.optimal_snr == TARGET == sample.params.snr
        measured = optimal_snr_welch(
            sample.signal, sample.noise, SAMPLE_RATE, *BANDPASS
        )
        assert 0.9 <= measured / TARGET <= 1.1, (seed, measured)


# --- (5) reproducible ---


@pytest.mark.parametrize("has_signal", [True, False])
def test_the_same_seed_gives_a_bit_identical_v3_sample(has_signal):
    first, second = (
        injection.make_sample(TRAIN_V3, np.random.RandomState(3), has_signal=has_signal)
        for _ in range(2)
    )
    assert first.data.tobytes() == second.data.tobytes()
    assert first.params == second.params
    other = injection.make_sample(
        TRAIN_V3, np.random.RandomState(4), has_signal=has_signal
    )
    assert not np.array_equal(first.data, other.data)


# --- (6) the real data path ---


def low_frequency_part(rng, n, level=3e-19):
    """Raw H1 strain has a std of a few 1e-19, almost all of it below 10 Hz."""
    freqs = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    inside = (freqs >= 0.2) & (freqs <= 10.0)
    spectrum = inside * (rng.normal(size=len(freqs)) + 1j * rng.normal(size=len(freqs)))
    low = np.fft.irfft(spectrum, n)
    return low * level / np.std(low)


def test_the_real_data_path_whitens_like_the_known_psd():
    """Real data are whitened with the Welch PSD of their padded 36 s segment. On
    simulated aLIGO noise plus a strong part below 10 Hz the result agrees with the
    known PSD path (correlation > 0.97, measured about 0.99), without edge ringing."""
    from gwpy.timeseries import TimeSeries

    pad_s = 16
    n = (4 + 2 * pad_s) * SAMPLE_RATE
    psd = injection.simulated_psd("aligo", n)
    k = int(0.1 * SAMPLE_RATE)
    middle = N_SAMPLES // 2
    edges, centers = [], []
    for seed in range(5):
        rng = np.random.RandomState(seed)
        noise = injection.draw_simulated_noise("aligo", n, rng)
        strain = TimeSeries(noise + low_frequency_part(rng, n), sample_rate=SAMPLE_RATE)
        real = normalize(
            conditioned_crop(
                strain, SAMPLE_RATE, pad_s, preprocessing="whiten_bandpass"
            ),
            eps=0.0,
        )
        known = normalize(condition_array(noise, SAMPLE_RATE, pad_s, psd=psd), eps=0.0)
        assert real.shape == (N_SAMPLES,)
        assert np.corrcoef(real, known)[0, 1] > 0.97
        edges += [np.mean(real[:k] ** 2), np.mean(real[-k:] ** 2)]
        centers.append(np.mean(real[middle - k // 2 : middle + k // 2] ** 2))
    assert 0.85 < np.sqrt(np.mean(edges) / np.mean(centers)) < 1.15


def test_bandpass_is_still_the_default_of_the_real_data_path():
    from gwpy.timeseries import TimeSeries

    rng = np.random.RandomState(0)
    n = 36 * SAMPLE_RATE
    strain = TimeSeries(
        injection.draw_simulated_noise("aligo", n, rng), sample_rate=SAMPLE_RATE
    )
    default = conditioned_crop(strain, SAMPLE_RATE, 16)
    explicit = conditioned_crop(strain, SAMPLE_RATE, 16, preprocessing="bandpass")
    whitened = conditioned_crop(
        strain, SAMPLE_RATE, 16, preprocessing="whiten_bandpass"
    )
    np.testing.assert_array_equal(default, explicit)
    assert not np.allclose(normalize(default, eps=0), normalize(whitened, eps=0))


# --- (7) the model decides the preprocessing ---


@pytest.mark.parametrize(
    "metadata, expected",
    [
        (None, "bandpass"),
        ({}, "bandpass"),
        ({"dataset_scenario": "unknown (v1)"}, "bandpass"),
        ({"dataset_scenario": "TRAIN_V2"}, "bandpass"),
        ({"dataset_scenario": "TRAIN_V3"}, "whiten_bandpass"),
        (
            {"dataset_scenario": "TRAIN_V2", "preprocessing": "whiten_bandpass"},
            "whiten_bandpass",
        ),
        ({"preprocessing": "bandpass", "dataset_scenario": "TRAIN_V3"}, "bandpass"),
        ({"online": {"scenario": "TRAIN_V3"}}, "whiten_bandpass"),
    ],
)
def test_preprocessing_of_the_metadata(metadata, expected):
    assert preprocessing_of(metadata) == expected


@pytest.mark.parametrize(
    "metadata, expected",
    [
        (None, None),
        ({}, None),
        ({"dataset_scenario": "unknown (v1)"}, None),
        ({"dataset_scenario": "NO_SUCH_SCENARIO"}, None),
        ({"dataset_scenario": None, "online": None}, None),
        ({"dataset_scenario": "TRAIN_V2"}, "TRAIN_V2"),
        ({"online": {"scenario": "TRAIN_V3"}}, "TRAIN_V3"),
        # The top-level key wins over the online block.
        (
            {"dataset_scenario": "TRAIN_V2", "online": {"scenario": "TRAIN_V3"}},
            "TRAIN_V2",
        ),
    ],
)
def test_training_scenario_of_the_metadata(metadata, expected):
    scenario = training_scenario(metadata)
    assert (scenario.name if scenario else None) == expected


# --- dataset ---


def test_a_v3_dataset_validates(tmp_path, capsys):
    from gwd.data.generate import generate_dataset
    from gwd.data.validate import check_dataset

    out = tmp_path / "v3"
    generate_dataset(12, str(out), TRAIN_V3, seed=1, noise_dir=str(tmp_path / "none"))
    assert "whitened with its PSD" in capsys.readouterr().out
    assert check_dataset(str(out)) == ([], [], "TRAIN_V3")
