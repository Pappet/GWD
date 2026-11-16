import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from gwd.evaluation.leaderboard import COLUMNS as LEADERBOARD_COLUMNS
from gwd.evaluation.leaderboard import read_leaderboard

from .conftest import assert_ran_cleanly, run_stage
from .entrypoints import NUM_SAMPLES, PIPELINE, SEED

pytestmark = pytest.mark.slow

# 7 until P4.1, then waveform_backend, noise_source, noise_file; P5.5 peak_snr, optimal_snr
LABEL_COLUMNS = 12
SAMPLE_LENGTH = 16384


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dataset_structure(pipeline_home):
    training = pipeline_home / "data" / "training"
    labels = pd.read_csv(training / "labels.csv")
    assert labels.shape == (NUM_SAMPLES, LABEL_COLUMNS)
    assert labels["has_signal"].sum() == NUM_SAMPLES // 2

    sample = np.load(training / labels["filename"].iloc[0])
    assert sample.shape == (SAMPLE_LENGTH,)
    # P5.6: float32, 65 536 bytes of data per file.
    assert sample.dtype == np.float32
    assert sample.nbytes == 65536


def test_dataset_records_backend_and_noise_source(pipeline_home):
    training = pipeline_home / "data" / "training"
    labels = pd.read_csv(training / "labels.csv", keep_default_na=False)
    signal = labels["has_signal"] == 1
    assert set(labels.loc[signal, "waveform_backend"]) <= {"pycbc", "newton"}
    assert set(labels.loc[~signal, "waveform_backend"]) == {""}
    # The pipeline home has no noise chunks.
    assert set(labels["noise_source"]) == {"gaussian"}
    assert set(labels["noise_file"]) == {""}
    # P5.5: peak and optimal SNR, both > 0 for every signal sample.
    assert (labels.loc[signal, ["peak_snr", "optimal_snr"]] > 0).all().all()
    assert (labels["peak_snr"] == labels["snr"]).all()

    manifest = json.loads((training / "manifest.json").read_text())
    assert manifest["waveform_backend"]
    assert (manifest["scenario"], manifest["n"], manifest["seed"]) == (
        "TRAIN_V1",
        NUM_SAMPLES,
        int(SEED),
    )
    assert manifest["real_noise_fraction"] == 0.0


def test_the_model_metadata_names_its_dataset(pipeline_home):
    training = pipeline_home / "data" / "training"
    (config,) = (pipeline_home / "models_registry").glob("*.json")
    metadata = json.loads(config.read_text())
    manifest = json.loads((training / "manifest.json").read_text())
    assert metadata["dataset_manifest_sha256"] == sha256(training / "manifest.json")
    assert metadata["waveform_backend"] == manifest["waveform_backend"]
    assert metadata["real_noise_fraction"] == 0.0
    assert metadata["seed"] == int(SEED)
    assert metadata["epochs_trained"] == 1


def test_training_writes_one_model_and_one_config(pipeline_home):
    models = pipeline_home / "models_registry"
    assert len(list(models.glob("*.keras"))) == 1
    assert len(list(models.glob("*.json"))) == 1


def test_leaderboard_has_one_row(pipeline_home):
    """P5.8: leaderboard v2 with raw numbers; the old CSV is never written."""
    leaderboard = pd.read_csv(pipeline_home / "results" / "leaderboard_v2.csv")
    assert tuple(leaderboard.columns) == LEADERBOARD_COLUMNS
    assert len(leaderboard) == 1
    assert leaderboard["roc_auc"].dtype.kind == "f"
    assert not (pipeline_home / "results" / "model_leaderboard.csv").exists()
    assert not (pipeline_home / "analysis").exists()


def test_evaluate_uses_the_threshold_that_thresholds_saved(pipeline_home):
    """P5.8: thresholds calibrates on calib_bg and saves to the model's metadata;
    evaluate then reads it from there."""
    stem = f"stress_v1_{SEED}_1000"
    manifest = json.loads(
        (pipeline_home / "results" / "testsets" / f"{stem}.json").read_text()
    )
    (config,) = (pipeline_home / "models_registry").glob("*.json")
    metadata = json.loads(config.read_text())
    assert metadata["threshold_testset"]["part"] == "calib_bg"
    assert metadata["threshold_testset"]["sha256"] == manifest["sha256"]
    assert metadata["threshold_fpr_achieved"] <= metadata["threshold_fpr"] == 0.01
    row = read_leaderboard(pipeline_home / "results" / "leaderboard_v2.csv").iloc[0]
    assert row["threshold"] == metadata["threshold"]
    assert row["testset_hash"] == manifest["sha256"]


def test_the_test_set_and_its_manifest_exist(pipeline_home):
    """P5.7: one stress_v1 set for evaluate, thresholds, and compare."""
    stem = f"stress_v1_{SEED}_1000"
    manifest = json.loads(
        (pipeline_home / "results" / "testsets" / f"{stem}.json").read_text()
    )
    assert (pipeline_home / "data" / "eval" / f"{stem}.npz").exists()
    assert manifest["noise_sources"]["test"] == {"gaussian": 1000}
    thresholds = json.loads(
        (pipeline_home / "results" / "optimal_thresholds.json").read_text()
    )
    assert thresholds["testset"]["sha256"] == manifest["sha256"]


def test_thresholds_and_comparison_plot_exist(pipeline_home):
    plots = pipeline_home / "plots" / "evaluation_models"
    assert (pipeline_home / "results" / "optimal_thresholds.json").exists()
    assert not (plots / "optimal_thresholds.json").exists()
    # The comparison plot and its zoom, since P4.2 under two names (EVAL-18, HYG-24).
    (zoom,) = plots.glob("plot_compare_roc_*_zoom.png")
    comparison = plots / zoom.name.replace("_zoom.png", ".png")
    assert sorted(plots.glob("plot_compare_roc_*.png")) == sorted([comparison, zoom])


def test_real_noise_path_runs(tmp_path):
    noise_folder = tmp_path / "data" / "noise_background"
    noise_folder.mkdir(parents=True)
    # A sine wave, because white noise has no correlation between neighbouring samples.
    sine = np.sin(2 * np.pi * 50 * np.arange(2 * SAMPLE_LENGTH) / 4096)
    for i in range(4):
        np.save(noise_folder / f"real_noise_{i:04d}.npy", sine)

    assert_ran_cleanly("generate", run_stage(PIPELINE["generate"], tmp_path))

    training = tmp_path / "data" / "training"
    labels = pd.read_csv(training / "labels.csv")
    assert labels.shape == (NUM_SAMPLES, LABEL_COLUMNS)
    assert set(labels["noise_source"]) == {"real", "gaussian"}
    real = labels["noise_source"] == "real"
    assert set(labels.loc[real, "noise_file"]) <= {
        f"real_noise_{i:04d}.npy" for i in range(4)
    }
    manifest = json.loads((training / "manifest.json").read_text())
    assert manifest["real_noise_fraction"] == real.sum() / NUM_SAMPLES
    noise_only = [
        np.load(training / name)
        for name in labels.loc[labels["has_signal"] == 0, "filename"]
    ]
    correlations = [np.corrcoef(sample[:-1], sample[1:])[0, 1] for sample in noise_only]
    assert max(correlations) > 0.9
