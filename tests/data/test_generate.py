"""gwd.data.generate writes the backend and noise source of every sample (P4.1)."""

import json

import numpy as np
import pandas as pd
import pytest

from gwd.config import N_SAMPLES
from gwd.core import waveforms
from gwd.data import provenance
from gwd.data.generate import generate_dataset
from gwd.data.scenarios import EVAL_V1, TRAIN_V1

OLD_COLUMNS = [
    "filename",
    "has_signal",
    "merger_time",
    "snr",
    "mass1",
    "mass2",
    "distance",
]
NEW_COLUMNS = ["waveform_backend", "noise_source", "noise_file"]


@pytest.fixture
def noise_dir(tmp_path):
    folder = tmp_path / "noise"
    folder.mkdir()
    rng = np.random.default_rng(2)
    for i in range(3):
        np.save(folder / f"real_noise_{i:04d}.npy", rng.normal(0, 1e-21, N_SAMPLES))
    return folder


def generate(tmp_path, noise_dir, n=20, scenario=TRAIN_V1, seed=7):
    out = tmp_path / "out"
    generate_dataset(n, str(out), scenario, seed=seed, noise_dir=str(noise_dir))
    labels = pd.read_csv(out / "labels.csv", keep_default_na=False)
    manifest = json.loads((out / "manifest.json").read_text())
    return out, labels, manifest


def test_labels_name_backend_noise_source_and_chunk(tmp_path, noise_dir):
    _, labels, _ = generate(tmp_path, noise_dir)
    assert list(labels.columns) == OLD_COLUMNS + NEW_COLUMNS

    backend = "pycbc" if waveforms.HAS_PYCBC else "newton"
    signal = labels["has_signal"] == 1
    assert set(labels.loc[signal, "waveform_backend"]) == {backend}
    assert set(labels.loc[~signal, "waveform_backend"]) == {""}

    real = labels["noise_source"] == "real"
    assert set(labels["noise_source"]) == {"real", "gaussian"}
    chunks = {path.name for path in noise_dir.iterdir()}
    assert set(labels.loc[real, "noise_file"]) <= chunks
    assert set(labels.loc[~real, "noise_file"]) == {""}


def test_the_manifest_describes_the_dataset(tmp_path, noise_dir):
    out, labels, manifest = generate(tmp_path, noise_dir)
    real = int((labels["noise_source"] == "real").sum())
    assert manifest["scenario"] == "TRAIN_V1"
    assert manifest["scenario_hash"] == TRAIN_V1.hash()
    assert (manifest["seed"], manifest["n"]) == (7, 20)
    assert manifest["waveform_backend"] == waveforms.active_backend()
    assert sum(manifest["signal_backends"].values()) == 10
    assert manifest["real_noise_samples"] == real
    assert manifest["real_noise_fraction"] == real / 20
    assert manifest["scenario_real_noise_fraction"] == 0.8
    assert manifest["noise_chunks"] == 3
    assert manifest["versions"]["numpy"] == np.__version__
    assert manifest["labels_sha256"] == provenance.sha256_file(out / "labels.csv")
    # The commit of this checkout; None only outside Git.
    commit, dirty = provenance.git_commit()
    assert (manifest["git_commit"], manifest["git_dirty"]) == (commit, dirty)


def test_the_same_data_gives_the_same_manifest(tmp_path, noise_dir):
    first = generate(tmp_path / "a", noise_dir)[2]
    second = generate(tmp_path / "b", noise_dir)[2]
    assert first == second


def test_a_seed_draws_the_same_samples_as_the_seeded_global_rng(tmp_path, noise_dir):
    out, _, _ = generate(tmp_path, noise_dir, n=6, seed=3)
    np.random.seed(3)
    generate_dataset(6, str(tmp_path / "global"), noise_dir=str(noise_dir))
    for i in range(6):
        name = f"sample_{i:05d}.npy"
        np.testing.assert_array_equal(
            np.load(out / name), np.load(tmp_path / "global" / name)
        )
    manifest = json.loads((tmp_path / "global" / "manifest.json").read_text())
    assert manifest["seed"] is None


def test_rng_and_seed_together_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        generate_dataset(2, str(tmp_path), rng=np.random.RandomState(1), seed=1)


def test_the_start_line_names_backend_and_noise(tmp_path, noise_dir, capsys):
    generate(tmp_path, noise_dir, n=2)
    out, err = capsys.readouterr()
    backend = waveforms.active_backend()
    line = (
        f"Backend: {backend} | Noise: real for 80% of samples (3 chunks), else gaussian"
    )
    assert line in out
    assert "WARNING" not in err


def test_without_chunks_the_noise_is_gaussian_with_a_warning(tmp_path, capsys):
    _, labels, manifest = generate(tmp_path, tmp_path / "empty", n=4)
    out, err = capsys.readouterr()
    assert "Noise: gaussian only (no real chunks found)" in out
    assert "no real noise chunks" in err and "gwd data fetch-noise" in err
    assert set(labels["noise_source"]) == {"gaussian"}
    assert (manifest["real_noise_fraction"], manifest["noise_chunks"]) == (0.0, 0)


def test_a_scenario_without_real_noise_says_so(tmp_path, noise_dir, capsys):
    generate(tmp_path, noise_dir, n=2, scenario=EVAL_V1)
    out, err = capsys.readouterr()
    assert "Noise: gaussian (EVAL_V1 uses no real noise)" in out
    assert "WARNING" not in err


def test_newton_gets_a_loud_warning(tmp_path, noise_dir, capsys, monkeypatch):
    monkeypatch.setattr(waveforms, "HAS_PYCBC", False)
    _, labels, manifest = generate(tmp_path, noise_dir, n=4)
    out, err = capsys.readouterr()
    assert "Backend: newton | Noise:" in out
    assert "the waveform backend is the Newtonian approximation" in err
    assert manifest["waveform_backend"] == "newton"
    assert manifest["signal_backends"] == {"newton": 2}
    assert set(labels.loc[labels["has_signal"] == 1, "waveform_backend"]) == {"newton"}
