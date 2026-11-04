import numpy as np
import pandas as pd
import pytest

from .conftest import assert_ran_cleanly, run_stage
from .entrypoints import NUM_SAMPLES, PIPELINE

pytestmark = pytest.mark.slow

LABEL_COLUMNS = 7
LEADERBOARD_COLUMNS = 12
SAMPLE_LENGTH = 16384


def test_dataset_structure(pipeline_home):
    training = pipeline_home / "data" / "training"
    labels = pd.read_csv(training / "labels.csv")
    assert labels.shape == (NUM_SAMPLES, LABEL_COLUMNS)
    assert labels["has_signal"].sum() == NUM_SAMPLES // 2

    sample = np.load(training / labels["filename"].iloc[0])
    assert sample.shape == (SAMPLE_LENGTH,)
    assert sample.dtype == np.float64


def test_training_writes_one_model_and_one_config(pipeline_home):
    models = pipeline_home / "models_registry"
    assert len(list(models.glob("*.keras"))) == 1
    assert len(list(models.glob("*.json"))) == 1


def test_leaderboard_has_one_row(pipeline_home):
    leaderboard = pd.read_csv(pipeline_home / "results" / "model_leaderboard.csv", dtype=str)
    assert leaderboard.shape == (1, LEADERBOARD_COLUMNS)
    assert not (pipeline_home / "analysis").exists()


def test_thresholds_and_comparison_plot_exist(pipeline_home):
    plots = pipeline_home / "plots" / "evaluation_models"
    assert (pipeline_home / "results" / "optimal_thresholds.json").exists()
    assert not (plots / "optimal_thresholds.json").exists()
    assert len(list(plots.glob("plot_compare_roc_*.png"))) == 1


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
    noise_only = [np.load(training / name) for name in labels.loc[labels["has_signal"] == 0, "filename"]]
    correlations = [np.corrcoef(sample[:-1], sample[1:])[0, 1] for sample in noise_only]
    assert max(correlations) > 0.9
