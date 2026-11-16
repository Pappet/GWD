"""gwd.data.generate writes the backend and noise source of every sample (P4.1)."""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from gwd.config import N_SAMPLES
from gwd.core import waveforms
from gwd.data import provenance
from gwd.data.generate import OutputFolderError, generate_dataset
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

    signal = labels["has_signal"] == 1
    assert set(labels.loc[signal, "waveform_backend"]) == {"pycbc"}
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
    assert manifest["time_axis"] == "arange(n)/fs"
    assert manifest["waveform_backend"] == waveforms.backend_label("pycbc")
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
    backend = waveforms.backend_label("pycbc")
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


def test_newton_chosen_on_purpose_gets_a_loud_warning(tmp_path, noise_dir, capsys):
    """Only tests and the golden values pass backend='newton'; gwd data generate never."""
    out = tmp_path / "out"
    generate_dataset(4, str(out), seed=7, noise_dir=str(noise_dir), backend="newton")
    labels = pd.read_csv(out / "labels.csv", keep_default_na=False)
    manifest = json.loads((out / "manifest.json").read_text())
    out, err = capsys.readouterr()
    assert "Backend: newton | Noise:" in out
    assert "the waveform backend is the Newtonian approximation" in err
    assert manifest["waveform_backend"] == "newton"
    assert manifest["signal_backends"] == {"newton": 2}
    assert set(labels.loc[labels["has_signal"] == 1, "waveform_backend"]) == {"newton"}


def test_without_pycbc_nothing_is_written_or_removed(tmp_path, noise_dir, monkeypatch):
    """P5.2: no silent Newton dataset; the old dataset stays as it was."""
    out, _, _ = generate(tmp_path, noise_dir, n=4)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    monkeypatch.setitem(sys.modules, "pycbc", None)
    with pytest.raises(waveforms.WaveformBackendError, match="physics"):
        generate_dataset(4, str(out), noise_dir=str(noise_dir), overwrite=True)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_gwd_data_generate_without_pycbc_exits_with_1_and_the_hint(tmp_path):
    code = "\n".join(
        [
            "import sys",
            "sys.modules['pycbc'] = None",
            "from gwd.cli import main",
            "sys.exit(main(['data', 'generate', '--n', '2', '--out', 'd']))",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "GWD_HOME": str(tmp_path)},
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stderr
    assert "pip install -e '.[physics]'" in result.stderr
    assert "WSL" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "d").exists()


# --- the output folder (P4.6) ---


def test_a_folder_that_is_not_empty_is_refused(tmp_path, noise_dir):
    out, _, _ = generate(tmp_path, noise_dir, n=4)
    before = sorted(p.name for p in out.iterdir())
    with pytest.raises(OutputFolderError, match="--overwrite"):
        generate_dataset(2, str(out), seed=1, noise_dir=str(noise_dir))
    assert sorted(p.name for p in out.iterdir()) == before


def test_overwrite_removes_only_samples_and_labels(tmp_path, noise_dir):
    out, _, _ = generate(tmp_path, noise_dir, n=6)
    (out / "notes.txt").write_text("mine")
    (out / "sub").mkdir()
    np.save(out / "sub" / "sample_00000.npy", np.ones(3))
    np.save(out / "other.npy", np.ones(3))

    generate_dataset(2, str(out), seed=1, noise_dir=str(noise_dir), overwrite=True)

    assert sorted(p.name for p in out.iterdir()) == [
        "labels.csv",
        "manifest.json",
        "notes.txt",
        "other.npy",
        "sample_00000.npy",
        "sample_00001.npy",
        "sub",
    ]
    assert (out / "sub" / "sample_00000.npy").exists()
    assert len(pd.read_csv(out / "labels.csv")) == 2
    assert json.loads((out / "manifest.json").read_text())["n"] == 2


def test_overwrite_in_a_parent_never_touches_the_noise_folder(tmp_path, noise_dir):
    chunks = sorted(p.name for p in noise_dir.iterdir())
    # tmp_path holds the noise folder; overwrite does not descend into it.
    generate_dataset(2, str(tmp_path), seed=1, noise_dir=str(noise_dir), overwrite=True)
    assert sorted(p.name for p in noise_dir.iterdir()) == chunks


def test_the_noise_folder_itself_is_refused(noise_dir):
    chunks = sorted(p.name for p in noise_dir.iterdir())
    with pytest.raises(OutputFolderError, match="noise folder"):
        generate_dataset(
            2, str(noise_dir), seed=1, noise_dir=str(noise_dir), overwrite=True
        )
    assert sorted(p.name for p in noise_dir.iterdir()) == chunks


def test_an_empty_or_new_folder_needs_no_overwrite(tmp_path, noise_dir):
    (tmp_path / "empty").mkdir()
    generate_dataset(2, str(tmp_path / "empty"), seed=1, noise_dir=str(noise_dir))
    generate_dataset(
        2, str(tmp_path / "new" / "deep"), seed=1, noise_dir=str(noise_dir)
    )
    assert (tmp_path / "new" / "deep" / "labels.csv").exists()


def test_the_command_stops_with_rc_1_without_overwrite(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    env = {**os.environ, "GWD_HOME": str(tmp_path)}
    args = ["data", "generate", "--n", "8", "--out", str(out)]
    result = subprocess.run(
        [sys.executable, "-m", "gwd.cli", *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "is not empty" in result.stderr and "--overwrite" in result.stderr
    assert "Traceback" not in result.stderr
    assert [p.name for p in out.iterdir()] == ["keep.txt"]

    result = subprocess.run(
        [sys.executable, "-m", "gwd.cli", *args, "--overwrite"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(list(out.glob("sample_*.npy"))) == 8
