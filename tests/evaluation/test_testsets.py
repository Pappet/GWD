"""Fixed, hashed test sets: cache, hash, stale sets, bins, and streams (P5.7).

Every set here is built with an empty noise folder, so only with Gaussian noise; the
in-process tests pass tmp folders and the Newton backend (numpy only, fast), and the
CLI test builds with PyCBC in a tmp GWD_HOME. Nothing reads or writes the real data/
or results/.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from gwd.core import waveforms
from gwd.data.scenarios import EVAL_V1, TESTSETS, TRAIN_V1
from gwd.evaluation import testsets
from gwd.evaluation.metrics import bin_index

N = 64
SEED = 7


def folders(root):
    return {
        "data_dir": str(root / "data" / "eval"),
        "manifest_dir": str(root / "results" / "testsets"),
        "noise_dir": str(root / "data" / "noise_background"),
    }


def build(root, name="indist_v1", n=N, seed=SEED, **options):
    return testsets.build(
        name, n, seed, backend="newton", **{**folders(root), **options}
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Both standard sets, built once in a tmp folder."""
    root = tmp_path_factory.mktemp("sets")
    return {name: build(root, name) for name in TESTSETS}


def test_the_standard_sets_are_indist_and_stress():
    assert TESTSETS["indist_v1"].scenario == TRAIN_V1
    assert TESTSETS["stress_v1"].scenario == EVAL_V1
    for spec in TESTSETS.values():
        assert spec.training == TRAIN_V1


def test_a_second_build_is_a_cache_hit_with_the_same_hash(tmp_path):
    first = build(tmp_path)
    second = build(tmp_path)
    assert (first.cache_hit, second.cache_hit) == (False, True)
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64
    np.testing.assert_array_equal(first.X, second.X)


def test_the_same_seed_gives_the_same_hash_in_another_folder(tmp_path, built):
    assert build(tmp_path).sha256 == built["indist_v1"].sha256


def test_another_seed_n_or_set_gives_another_hash(tmp_path, built):
    hashes = {
        built["indist_v1"].sha256,
        built["stress_v1"].sha256,
        build(tmp_path, seed=SEED + 1).sha256,
        build(tmp_path, n=N + 2).sha256,
    }
    assert len(hashes) == 4


def test_files_are_named_after_set_seed_and_n_and_the_manifest_records_it(built):
    for name, data in built.items():
        assert data.path.endswith(
            os.path.join("data", "eval", f"{name}_{SEED}_{N}.npz")
        )
        assert data.manifest_path.endswith(
            os.path.join("results", "testsets", f"{name}_{SEED}_{N}.json")
        )
        manifest = json.loads(open(data.manifest_path, encoding="utf-8").read())
        assert manifest["sha256"] == data.sha256
        assert (manifest["name"], manifest["seed"], manifest["n"]) == (name, SEED, N)
        assert manifest["scenario"] == data.spec.scenario.name
        assert manifest["scenario_hash"] == data.spec.scenario.hash()
        assert manifest["format"] == testsets.TESTSET_FORMAT
        assert manifest["parts"] == {"test": N, "calib_bg": N}


def test_the_manifest_lists_the_parameters_of_every_sample(built):
    data = built["indist_v1"]
    manifest = json.loads(open(data.manifest_path, encoding="utf-8").read())
    rows = manifest["samples"]["test"]
    assert len(rows) == N
    expected = {"mass1", "mass2", "t_merger", "peak_snr", "optimal_snr", "noise_source"}
    assert expected <= set(rows[0])
    assert [row["peak_snr"] for row in rows] == pytest.approx(list(data.peak_snr))
    assert [row["label"] for row in rows] == list(data.y)
    assert {row["noise_source"] for row in rows} == {"gaussian"}


def test_calib_bg_holds_only_label_0(built):
    for data in built.values():
        manifest = json.loads(open(data.manifest_path, encoding="utf-8").read())
        assert {row["label"] for row in manifest["samples"]["calib_bg"]} == {0}
        assert data.calib_X.shape == (N, 16384, 1)
        assert not any(key.startswith("calib_") and "snr" in key for key in data.arrays)


def test_test_is_balanced_and_normalized(built):
    for data in built.values():
        assert data.X.shape == (N, 16384, 1)
        assert data.X.dtype == np.float32
        assert int(data.y.sum()) == N // 2
        assert np.abs(data.X.mean(axis=1)).max() < 1e-3
        assert np.abs(data.X.std(axis=1) - 1).max() < 1e-3


@pytest.mark.parametrize("name", sorted(TESTSETS))
def test_every_snr_bin_is_occupied(built, name):
    data = built[name]
    signal = data.y == 1
    edges = testsets.bins(data)
    for key, values in (
        ("peak_snr", data.peak_snr),
        ("optimal_snr", data.optimal_snr),
    ):
        index = bin_index(values[signal], edges[key])
        assert (index >= 0).all(), key
        counts = np.bincount(index, minlength=len(edges[key]) - 1)
        assert (counts > 0).all(), (key, counts)
    assert edges["optimal_snr"].size == testsets.OPTIMAL_SNR_BINS + 1


def test_the_peak_snr_is_stratified_over_the_scenario_range(built):
    data = built["indist_v1"]
    signal = data.y == 1
    low, high = TRAIN_V1.snr_range
    assert data.peak_snr[signal].min() >= low
    assert data.peak_snr[signal].max() <= high
    counts = np.bincount(bin_index(data.peak_snr[signal], data.spec.peak_snr_edges()))
    # 32 signal samples in 11 bins: 3 or 2 each.
    assert set(counts) <= {2, 3}


def test_a_fixed_mass_is_one_bin(built):
    edges = testsets.bins(built["stress_v1"])["total_mass"]
    np.testing.assert_array_equal(edges, [60.0, 60.0])


def test_stress_marks_the_peak_snr_below_the_training_minimum(built):
    stress, indist = built["stress_v1"], built["indist_v1"]
    flags = testsets.outside_training(stress)
    signal = stress.y == 1
    np.testing.assert_array_equal(flags[signal], stress.peak_snr[signal] < 1.0)
    assert not flags[~signal].any()
    assert not testsets.outside_training(indist).any()


def test_a_changed_array_raises_value_error_on_load(tmp_path):
    data = build(tmp_path)
    arrays = dict(data.arrays)
    arrays["test_X"] = arrays["test_X"].copy()
    arrays["test_X"][0, 0] += 1.0
    with open(data.path, "wb") as f:
        np.savez(f, **arrays)
    with pytest.raises(ValueError, match="changed after the build"):
        testsets.load("indist_v1", N, SEED, **folders(tmp_path))
    # build does not replace it on its own either.
    with pytest.raises(testsets.InvalidTestsetError, match="--rebuild"):
        build(tmp_path)
    rebuilt = build(tmp_path, rebuild=True)
    assert (rebuilt.cache_hit, rebuilt.sha256) == (False, data.sha256)


def test_a_changed_parameter_array_raises_too(tmp_path):
    data = build(tmp_path)
    arrays = dict(data.arrays)
    arrays["test_peak_snr"] = arrays["test_peak_snr"] * 2
    with open(data.path, "wb") as f:
        np.savez(f, **arrays)
    with pytest.raises(ValueError, match="SHA-256"):
        testsets.load("indist_v1", N, SEED, **folders(tmp_path))


def test_a_manifest_from_another_scenario_version_is_stale(tmp_path):
    data = build(tmp_path)
    manifest = json.loads(open(data.manifest_path, encoding="utf-8").read())
    manifest["scenario_hash"] = "0" * 12
    with open(data.manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with pytest.raises(testsets.InvalidTestsetError, match="stale: scenario_hash"):
        build(tmp_path)


def test_new_noise_chunks_make_a_set_with_real_noise_stale(tmp_path):
    build(tmp_path, "indist_v1")
    build(tmp_path, "stress_v1")
    noise = tmp_path / "data" / "noise_background"
    noise.mkdir(parents=True)
    np.save(noise / "real_noise_0000.npy", np.zeros(2 * 16384))
    with pytest.raises(testsets.InvalidTestsetError, match="noise chunks"):
        build(tmp_path, "indist_v1")
    # stress_v1 has no real noise, so the folder does not matter.
    assert build(tmp_path, "stress_v1").cache_hit


def test_a_missing_manifest_means_a_new_build(tmp_path):
    data = build(tmp_path)
    os.remove(data.manifest_path)
    with pytest.raises(FileNotFoundError):
        testsets.load("indist_v1", N, SEED, **folders(tmp_path))
    again = build(tmp_path)
    assert (again.cache_hit, again.sha256) == (False, data.sha256)


def test_the_global_rng_is_untouched(tmp_path):
    np.random.seed(123)
    state = np.random.get_state()
    build(tmp_path, n=8)
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_the_streams_are_disjoint_from_training_seeds():
    """No part stream equals what training draws from with the same integer seed."""
    for seed in range(20):
        first = {
            (name, part): tuple(testsets.part_rng(name, seed, part).rand(4))
            for name in TESTSETS
            for part in testsets.PARTS
        }
        assert len(set(first.values())) == len(first)
        training = {
            tuple(np.random.RandomState(seed).rand(4)),
            tuple(np.random.default_rng(seed).random(4)),
        }
        assert not training & set(first.values())


def test_invalid_n_and_seed(tmp_path):
    with pytest.raises(ValueError, match="at least 2"):
        build(tmp_path, n=1)
    with pytest.raises(ValueError, match="negative"):
        build(tmp_path, seed=-1)
    with pytest.raises(KeyError, match="indist_v1"):
        build(tmp_path, "nope")


def test_a_cached_set_loads_without_pycbc(tmp_path, monkeypatch):
    data = build(tmp_path)
    monkeypatch.setitem(sys.modules, "pycbc", None)
    assert build(tmp_path).sha256 == data.sha256
    with pytest.raises(waveforms.WaveformBackendError):
        testsets.build("indist_v1", N, SEED + 1, **folders(tmp_path))


def test_build_testset_twice_with_the_cli(tmp_path):
    """The check of the plan: same hash, cache hit, manifest under results/testsets/."""
    if not waveforms.pycbc_available():
        pytest.skip("pycbc missing")
    env = {**os.environ, "GWD_HOME": str(tmp_path), "MPLBACKEND": "Agg"}
    args = ["eval", "build-testset", "--scenario", "indist_v1", "--seed", "7"]
    outputs = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-m", "gwd.cli", *args, "--n", "64"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout)
    assert outputs[0].startswith("Built: test set indist_v1 (seed 7, n 64)")
    assert outputs[1].startswith("Cache hit: test set indist_v1 (seed 7, n 64)")
    hashes = [
        line.split()[-1]
        for text in outputs
        for line in text.splitlines()
        if "SHA-256:" in line
    ]
    assert len(hashes) == 2 and hashes[0] == hashes[1]
    manifest = tmp_path / "results" / "testsets" / "indist_v1_7_64.json"
    assert json.loads(manifest.read_text())["sha256"] == hashes[0]
    assert (tmp_path / "data" / "eval" / "indist_v1_7_64.npz").exists()

    # A changed array: exit code 1 and the reason, no traceback.
    npz = tmp_path / "data" / "eval" / "indist_v1_7_64.npz"
    with np.load(npz) as f:
        arrays = {key: f[key] for key in f.files}
    arrays["test_y"] = 1 - arrays["test_y"]
    with open(npz, "wb") as f:
        np.savez(f, **arrays)
    result = subprocess.run(
        [sys.executable, "-m", "gwd.cli", *args, "--n", "64"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "changed after the build" in result.stderr
    assert "Traceback" not in result.stderr
