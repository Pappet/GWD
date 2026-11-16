"""Training: no side effects on import, precision only with a GPU or on request (P3.7)."""

import datetime
import json
import os
import subprocess
import sys

import pytest

tf = pytest.importorskip("tensorflow")

from gwd import registry  # noqa: E402
from gwd.data import provenance  # noqa: E402
from gwd.training import train  # noqa: E402


def test_import_keeps_float32_and_defines_no_run_globals():
    # The check from the plan, in a fresh process without a GPU.
    code = "\n".join(
        [
            "import tensorflow as tf, gwd.training.train as train",
            "assert tf.keras.mixed_precision.global_policy().name == 'float32'",
            "names = ('TIMESTAMP', 'MODEL_NAME', 'MODEL_SAVE_PATH', 'DATA_FOLDER')",
            "print([name for name in names if hasattr(train, name)])",
        ]
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TF_CPP_MIN_LOG_LEVEL": "2"}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


@pytest.mark.parametrize(
    "mixed_precision, gpus, policy",
    [
        (False, [], "float32"),
        (False, ["GPU:0"], "mixed_float16"),
        (True, [], "mixed_float16"),
        (True, ["GPU:0"], "mixed_float16"),
    ],
)
def test_choose_precision(mixed_precision, gpus, policy):
    assert train.choose_precision(mixed_precision, gpus)[0] == policy


def test_new_run_names_the_files_as_the_registry_expects(tmp_path):
    run = train.new_run(str(tmp_path), now=datetime.datetime(2026, 1, 2, 3, 4, 5))
    assert run.name == "gwd_model_20260102-030405"
    assert run.model_path == str(tmp_path / "gwd_model_20260102-030405.keras")
    assert run.config_path == registry.metadata_path(run.model_path)
    assert registry.model_time(run.model_path) == datetime.datetime(2026, 1, 2, 3, 4, 5)


def test_config_defaults_come_from_gwd_config():
    from gwd import config

    cfg = train.TrainConfig()
    assert (cfg.epochs, cfg.batch_size, cfg.learning_rate, cfg.patience) == (
        config.EPOCHS,
        config.BATCH_SIZE,
        config.LEARNING_RATE,
        config.EARLY_STOPPING_PATIENCE,
    )
    assert cfg.data_folder == config.DATA_DIR_TRAIN
    assert cfg.mixed_precision is False


@pytest.mark.slow
def test_train_restores_the_policy_and_records_it(tmp_path, monkeypatch):
    import numpy as np

    from gwd.data.generate import generate_dataset
    from gwd.data.scenarios import EVAL_V1

    # EVAL_V1 has white noise only, so the test does not read data/noise_background/.
    data = tmp_path / "data"
    generate_dataset(16, str(data), EVAL_V1, rng=np.random.RandomState(1))
    manifest = json.loads((data / "manifest.json").read_text())
    cfg = train.TrainConfig(
        data_folder=str(data),
        models_dir=str(tmp_path / "models"),
        plots_dir=str(tmp_path / "plots"),
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        patience=2,
        seed=3,
    )
    # Without a GPU on purpose, so the run takes float32 on any machine.
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda kind=None: [])

    run = train.train(cfg)

    assert tf.keras.mixed_precision.global_policy().name == "float32"
    assert os.path.exists(run.model_path)
    assert os.path.exists(tmp_path / "plots" / f"training_history_{run.timestamp}.png")
    metadata = json.loads(open(run.config_path).read())
    assert metadata["model_name"] == run.name
    assert metadata["precision_policy"] == "float32"
    assert (metadata["epochs"], metadata["batch_size"]) == (1, 8)
    assert metadata["learning_rate"] == 1e-3
    assert metadata["early_stopping_patience"] == 2
    # P4.1: what was trained on, and how long.
    assert (metadata["epochs_trained"], metadata["best_epoch"]) == (1, 1)
    assert metadata["seed"] == 3
    assert metadata["dataset_scenario"] == "EVAL_V1"
    assert metadata["waveform_backend"] == manifest["waveform_backend"]
    assert metadata["real_noise_fraction"] == 0.0
    digest = provenance.sha256_file(data / "manifest.json")
    assert metadata["dataset_manifest_sha256"] == digest
    assert registry.read_metadata(run.model_path) == metadata


def test_a_dataset_without_manifest_is_recorded_as_unknown(tmp_path):
    assert train.dataset_provenance(str(tmp_path)) == {
        "dataset_manifest_sha256": "unknown (v1)",
        "dataset_scenario": "unknown (v1)",
        "waveform_backend": "unknown (v1)",
        "real_noise_fraction": "unknown (v1)",
    }


def test_the_provenance_comes_from_the_manifest(tmp_path):
    manifest = {
        "scenario": "TRAIN_V1",
        "waveform_backend": "pycbc 2.10.0",
        "real_noise_fraction": 0.75,
    }
    provenance.write_manifest(str(tmp_path), manifest)
    assert train.dataset_provenance(str(tmp_path)) == {
        "dataset_manifest_sha256": provenance.sha256_file(tmp_path / "manifest.json"),
        "dataset_scenario": "TRAIN_V1",
        "waveform_backend": "pycbc 2.10.0",
        "real_noise_fraction": 0.75,
    }


def test_load_data_without_labels_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run 'gwd data generate' first"):
        train.load_data(str(tmp_path))
