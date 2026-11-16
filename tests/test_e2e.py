"""End to end: the whole workflow once, in a temporary GWD_HOME (P7.4).

The chain is the one of the README, each stage a gwd command line run as
``python -m gwd.cli`` in a subprocess:

    gwd data generate --n 64 -> gwd data validate -> gwd train --epochs 1
    -> gwd eval build-testset -> gwd eval thresholds --save -> gwd eval run --no-real
    -> gwd eval compare -> gwd eval leaderboard

and then the Glitch Hunter, built under Agg with the model that train wrote and the
threshold that thresholds saved, rates one simulated sample. The home starts with four
noise chunks in data/noise_background/, so generate takes the real noise path too.

Every process gets GWD_HOME set to a new temporary folder and runs in it, so nothing
reads or writes the owner's data/, models_registry/, results/, or plots/; the last
test checks that. Only structure is checked, never metric values: training on a CPU
is not deterministic. Since P7.4 this replaces the pipeline smoke tests of P0.5
(tests/smoke/test_pipeline.py); tests/smoke/ keeps the app start tests.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gwd.evaluation.leaderboard import COLUMNS as LEADERBOARD_COLUMNS
from gwd.evaluation.leaderboard import read_leaderboard

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parent.parent
OWNER_FOLDERS = ("data", "models_registry", "results", "plots")

SEED = "1"
NUM_SAMPLES = 64
NOISE_CHUNKS = 4
# 7 until P4.1, then waveform_backend, noise_source, noise_file; P5.5 peak_snr, optimal_snr
LABEL_COLUMNS = 12
SAMPLE_LENGTH = 16384
# The default test set stress_v1, with --seed SEED and the default n of 1000.
TESTSET_STEM = f"stress_v1_{SEED}_1000"

STAGES = {
    "generate": ("data", "generate", "--n", str(NUM_SAMPLES), "--seed", SEED),
    "validate": ("data", "validate"),
    "train": ("train", "--epochs", "1", "--seed", SEED),
    # P5.7: the fixed test set; thresholds, run, and compare then load it.
    "build_testset": ("eval", "build-testset", "--seed", SEED),
    # P5.8: before run, which then uses the calibrated threshold.
    "thresholds": ("eval", "thresholds", "--save", "--seed", SEED),
    "evaluate": ("eval", "run", "--no-real", "--seed", SEED),
    # compare stops without a plot when it finds one model, so it gets the same model twice.
    "compare": (
        "eval",
        "compare",
        "--model",
        "latest",
        "--model",
        "latest",
        "--seed",
        SEED,
    ),
    "leaderboard": ("eval", "leaderboard"),
}

# What gwd app glitch-hunter does before plt.show(), with the model 'latest'.
GLITCH_HUNTER = """
import json
import matplotlib
matplotlib.use("Agg")
import numpy as np
from gwd.apps.glitch_hunter import app
from gwd.apps.glitch_hunter.audio import Player
from gwd.apps.glitch_hunter.sources import SimSource
from gwd.registry import model_name, read_metadata

model, path, note = app.load_optional_model("latest")
assert model is not None, note
hunter = app.GlitchHunterApp(
    model=model,
    metadata=read_metadata(path),
    sim=SimSource(rng=np.random.RandomState(1)),
    real=None,
    player=Player(system="Linux", popen=None, which=lambda name: False),
    threshold=app.start_threshold(path),
)
hunter.ask_ai(None)
print(json.dumps({
    "model": model_name(path),
    "threshold": hunter.threshold,
    "slider": hunter.slider_thresh.val,
    "prediction": hunter.prediction,
    "noise_warning": hunter.noise_warning,
    "truth": hunter.truth_artist.get_text(),
}))
"""


def run(args, home, stdin=""):
    """Run python with args in home, with GWD_HOME set to home."""
    env = {
        **os.environ,
        "GWD_HOME": str(home),
        "MPLBACKEND": "Agg",
        "CUDA_VISIBLE_DEVICES": "",
    }
    return subprocess.run(
        [sys.executable, *args],
        input=stdin,
        env=env,
        cwd=home,
        capture_output=True,
        text=True,
    )


def assert_ran_cleanly(name, result):
    details = f"{name} (rc {result.returncode})\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    assert result.returncode == 0, details
    assert "Traceback" not in result.stderr, details


def snapshot():
    """Every folder below the owner's data folders with its mtime, and every file of
    models_registry/ and results/ with size and mtime. A file that is added, removed,
    or rewritten changes it."""
    state = {}
    for name in OWNER_FOLDERS:
        top = ROOT / name
        for folder, _, files in os.walk(top):
            state[folder] = os.stat(folder).st_mtime_ns
            if name in ("models_registry", "results"):
                for file in files:
                    stat = os.stat(os.path.join(folder, file))
                    state[os.path.join(folder, file)] = (stat.st_size, stat.st_mtime_ns)
    return state


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    """Run every stage once, in order. Returns (home, Glitch Hunter output, snapshots)."""
    before = snapshot()
    home = tmp_path_factory.mktemp("e2e")
    assert not str(home).startswith(str(ROOT))

    noise_folder = home / "data" / "noise_background"
    noise_folder.mkdir(parents=True)
    # A sine wave, because white noise has no correlation between neighbouring samples.
    sine = np.sin(2 * np.pi * 50 * np.arange(2 * SAMPLE_LENGTH) / 4096)
    for i in range(NOISE_CHUNKS):
        np.save(noise_folder / f"real_noise_{i:04d}.npy", sine)

    outputs = {}
    for name, args in STAGES.items():
        result = run(("-m", "gwd.cli", *args), home)
        assert_ran_cleanly(name, result)
        outputs[name] = result
    result = run(("-c", GLITCH_HUNTER), home)
    assert_ran_cleanly("glitch_hunter", result)
    hunter = json.loads(result.stdout.strip().splitlines()[-1])
    return home, outputs, hunter, before


def metadata_of(home):
    (config,) = (home / "models_registry").glob("*.json")
    return json.loads(config.read_text())


def test_dataset_structure(e2e):
    home = e2e[0]
    training = home / "data" / "training"
    labels = pd.read_csv(training / "labels.csv")
    assert labels.shape == (NUM_SAMPLES, LABEL_COLUMNS)
    assert labels["has_signal"].sum() == NUM_SAMPLES // 2

    sample = np.load(training / labels["filename"].iloc[0])
    assert sample.shape == (SAMPLE_LENGTH,)
    # P5.6: float32, 65 536 bytes of data per file.
    assert sample.dtype == np.float32
    assert sample.nbytes == 65536


def test_dataset_records_backend_and_noise_source(e2e):
    home = e2e[0]
    training = home / "data" / "training"
    labels = pd.read_csv(training / "labels.csv", keep_default_na=False)
    signal = labels["has_signal"] == 1
    # P5.2: the pipeline draws with PyCBC only.
    assert set(labels.loc[signal, "waveform_backend"]) == {"pycbc"}
    assert set(labels.loc[~signal, "waveform_backend"]) == {""}
    # P5.5: peak and optimal SNR, both > 0 for every signal sample.
    assert (labels.loc[signal, ["peak_snr", "optimal_snr"]] > 0).all().all()
    assert (labels["peak_snr"] == labels["snr"]).all()

    # TRAIN_V1 takes real noise for 80 % of the samples when there are chunks.
    assert set(labels["noise_source"]) == {"real", "gaussian"}
    real = labels["noise_source"] == "real"
    chunks = {f"real_noise_{i:04d}.npy" for i in range(NOISE_CHUNKS)}
    assert set(labels.loc[real, "noise_file"]) <= chunks
    assert set(labels.loc[~real, "noise_file"]) == {""}

    manifest = json.loads((training / "manifest.json").read_text())
    assert manifest["waveform_backend"].startswith("pycbc")
    assert (manifest["scenario"], manifest["n"], manifest["seed"]) == (
        "TRAIN_V1",
        NUM_SAMPLES,
        int(SEED),
    )
    assert manifest["real_noise_fraction"] == real.sum() / NUM_SAMPLES


def test_the_real_noise_chunks_reach_the_samples(e2e):
    home = e2e[0]
    training = home / "data" / "training"
    labels = pd.read_csv(training / "labels.csv", keep_default_na=False)
    noise_only = [
        np.load(training / name)
        for name in labels.loc[
            (labels["has_signal"] == 0) & (labels["noise_source"] == "real"), "filename"
        ]
    ]
    assert noise_only
    correlations = [np.corrcoef(sample[:-1], sample[1:])[0, 1] for sample in noise_only]
    assert min(correlations) > 0.9


def test_training_writes_one_model_that_names_its_dataset(e2e):
    home = e2e[0]
    training = home / "data" / "training"
    models = home / "models_registry"
    assert len(list(models.glob("*.keras"))) == 1
    assert len(list(models.glob("*.json"))) == 1

    metadata = metadata_of(home)
    manifest = json.loads((training / "manifest.json").read_text())
    assert metadata["dataset_manifest_sha256"] == sha256(training / "manifest.json")
    assert metadata["waveform_backend"] == manifest["waveform_backend"]
    assert metadata["real_noise_fraction"] == manifest["real_noise_fraction"] > 0
    assert metadata["seed"] == int(SEED)
    assert metadata["epochs_trained"] == 1


def test_the_test_set_and_its_manifest_exist(e2e):
    """P5.7: one stress_v1 set for thresholds, run, and compare."""
    home = e2e[0]
    manifest = json.loads(
        (home / "results" / "testsets" / f"{TESTSET_STEM}.json").read_text()
    )
    assert (home / "data" / "eval" / f"{TESTSET_STEM}.npz").exists()
    assert manifest["noise_sources"]["test"] == {"gaussian": 1000}
    # One report per model; the single optimal_thresholds.json is no longer written.
    (model,) = (home / "models_registry").glob("*.keras")
    report = home / "results" / "thresholds" / f"{model.stem}.json"
    thresholds = json.loads(report.read_text())
    assert thresholds["testset"]["sha256"] == manifest["sha256"]
    assert not (home / "results" / "optimal_thresholds.json").exists()


def test_thresholds_saves_the_threshold_that_run_uses(e2e):
    """P5.8: thresholds calibrates on calib_bg and saves to the model's metadata;
    run reads it from there and writes it into its leaderboard row."""
    home = e2e[0]
    manifest = json.loads(
        (home / "results" / "testsets" / f"{TESTSET_STEM}.json").read_text()
    )
    metadata = metadata_of(home)
    assert metadata["threshold_testset"]["part"] == "calib_bg"
    assert metadata["threshold_testset"]["sha256"] == manifest["sha256"]
    assert metadata["threshold_fpr_achieved"] <= metadata["threshold_fpr"] == 0.01
    row = read_leaderboard(home / "results" / "leaderboard_v2.csv").iloc[0]
    assert row["threshold"] == metadata["threshold"]
    assert row["testset_hash"] == manifest["sha256"]


def test_the_leaderboard_has_one_row(e2e):
    """P5.8: leaderboard v2 with raw numbers; the old CSV is never written."""
    home, outputs = e2e[0], e2e[1]
    leaderboard = pd.read_csv(home / "results" / "leaderboard_v2.csv")
    assert tuple(leaderboard.columns) == LEADERBOARD_COLUMNS
    assert len(leaderboard) == 1
    assert leaderboard["roc_auc"].dtype.kind == "f"
    assert not (home / "results" / "model_leaderboard.csv").exists()
    assert not (home / "analysis").exists()
    (model,) = (home / "models_registry").glob("*.keras")
    assert model.stem in outputs["leaderboard"].stdout


def test_the_comparison_plot_exists(e2e):
    home = e2e[0]
    plots = home / "plots" / "evaluation_models"
    assert not (plots / "optimal_thresholds.json").exists()
    # The comparison plot and its zoom, since P4.2 under two names (EVAL-18, HYG-24).
    (zoom,) = plots.glob("plot_compare_roc_*_zoom.png")
    comparison = plots / zoom.name.replace("_zoom.png", ".png")
    assert sorted(plots.glob("plot_compare_roc_*.png")) == sorted([comparison, zoom])


def test_the_glitch_hunter_rates_with_the_model_and_its_saved_threshold(e2e):
    home, hunter = e2e[0], e2e[2]
    metadata = metadata_of(home)
    (model,) = (home / "models_registry").glob("*.keras")
    assert hunter["model"] == model.stem
    assert hunter["threshold"] == metadata["threshold"]
    assert hunter["slider"] == pytest.approx(min(max(metadata["threshold"], 0.0), 1.0))
    assert 0.0 <= hunter["prediction"] <= 1.0
    assert hunter["truth"].startswith("TRUTH:")
    # The model was trained with real noise, so the app does not warn about it.
    assert hunter["noise_warning"] is None


def test_the_owner_data_is_untouched(e2e):
    assert snapshot() == e2e[3]
