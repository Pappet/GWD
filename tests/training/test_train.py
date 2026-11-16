"""Training: no side effects on import, precision only with a GPU or on request (P3.7).

Since P5.6 also: the three-way split, the seed, float32 loading, and the metadata.
"""

import datetime
import json
import os
import subprocess
import sys

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from gwd import registry  # noqa: E402
from gwd.data import provenance  # noqa: E402
from gwd.data.scenarios import training_scenario  # noqa: E402
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
    generate_dataset(
        16, str(data), EVAL_V1, rng=np.random.RandomState(1), noise_dir=str(tmp_path)
    )
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
    # The BatchNorm statistics come from a subset of the training part.
    assert metadata["bn_recalibration_samples"] == metadata["split"]["train"]
    assert metadata["training_data"] == "files"
    assert metadata["seed"] == 3
    # P5.6: the seed, the split, the data, and val and test kept apart.
    assert (metadata["seed_drawn"], metadata["deterministic"]) == (False, False)
    assert metadata["dataset_hash"] == train.load_data(str(data)).sha256
    assert metadata["split_grouped"] is True
    split = metadata["split"]
    assert split["train"] + split["val"] + split["test"] == 16
    assert min(split["train"], split["val"], split["test"]) >= 1
    assert (split["val_fraction"], split["test_fraction"]) == (0.15, 0.15)
    assert split["grouped_by"] == "noise_file"
    for key in ("val_accuracy", "val_loss", "test_accuracy", "test_loss"):
        assert isinstance(metadata[key], float), key
    # One epoch: the best epoch is the last one.
    assert metadata["val_loss"] == metadata["metrics"]["final_val_loss"]
    assert "test_set_accuracy" not in metadata["metrics"]
    assert metadata["dataset_scenario"] == "EVAL_V1"
    # The evaluation reads it back to choose the default test set.
    assert training_scenario(metadata).name == "EVAL_V1"
    assert metadata["waveform_backend"] == manifest["waveform_backend"]
    assert metadata["real_noise_fraction"] == 0.0
    digest = provenance.sha256_file(data / "manifest.json")
    assert metadata["dataset_manifest_sha256"] == digest
    assert registry.read_metadata(run.model_path) == metadata


@pytest.mark.slow
def test_a_seed_with_deterministic_ops_repeats_the_run(tmp_path, monkeypatch):
    """P5.6 on the CPU: the same seed gives the same val_loss; no seed draws one."""
    from gwd.data.generate import generate_dataset
    from gwd.data.scenarios import EVAL_V1

    data = tmp_path / "data"
    generate_dataset(16, str(data), EVAL_V1, seed=1, noise_dir=str(tmp_path))
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda kind=None: [])

    def run(seed, name):
        cfg = train.TrainConfig(
            data_folder=str(data),
            models_dir=str(tmp_path / name),
            plots_dir=str(tmp_path / "plots"),
            epochs=1,
            batch_size=8,
            seed=seed,
            deterministic=True,
        )
        return json.loads(open(train.train(cfg).config_path).read())

    first, second, drawn = run(1, "a"), run(1, "b"), run(None, "c")
    assert abs(first["val_loss"] - second["val_loss"]) < 1e-6
    assert first["test_accuracy"] == second["test_accuracy"]
    assert first["deterministic"] is True
    assert (drawn["seed_drawn"], isinstance(drawn["seed"], int)) == (True, True)


def test_a_dataset_without_manifest_is_recorded_as_unknown(tmp_path):
    assert train.dataset_provenance(str(tmp_path)) == {
        "dataset_manifest_sha256": "unknown (v1)",
        "dataset_scenario": "unknown (v1)",
        "waveform_backend": "unknown (v1)",
        "real_noise_fraction": "unknown (v1)",
        "preprocessing": "bandpass",
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
        "preprocessing": "bandpass",
    }


@pytest.mark.parametrize(
    "scenario, expected",
    [("TRAIN_V2", "bandpass"), ("TRAIN_V3", "whiten_bandpass"), ("OTHER", "bandpass")],
)
def test_the_metadata_records_the_preprocessing_of_the_scenario(
    tmp_path, scenario, expected
):
    """V3: the model's metadata says what its input went through, so evaluation and
    the Glitch Hunter preprocess real data the same way."""
    manifest = {
        "scenario": scenario,
        "waveform_backend": "pycbc",
        "real_noise_fraction": 0,
    }
    provenance.write_manifest(str(tmp_path), manifest)
    assert train.dataset_provenance(str(tmp_path))["preprocessing"] == expected


def test_load_data_without_labels_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run 'gwd data generate' first"):
        train.load_data(str(tmp_path))


@pytest.fixture
def small_dataset(tmp_path):
    from gwd.data.generate import generate_dataset
    from gwd.data.scenarios import EVAL_V1

    data = tmp_path / "data"
    generate_dataset(8, str(data), EVAL_V1, seed=2, noise_dir=str(tmp_path))
    return data


def test_load_data_loads_every_sample_in_order(small_dataset):
    from gwd.data.preprocessing import normalize

    data = train.load_data(str(small_dataset))
    assert data.X.shape == (8, 16384, 1)
    assert data.X.dtype == np.float32
    assert list(data.y) == [1, 0] * 4
    assert list(data.labels["filename"]) == [f"sample_{i:05d}.npy" for i in range(8)]
    sample = np.load(small_dataset / "sample_00005.npy")
    expected = normalize(sample.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(data.X[5, :, 0], expected)


def test_load_data_reads_float64_datasets_from_before_p56(small_dataset):
    from gwd.data.preprocessing import normalize

    for path in small_dataset.glob("sample_*.npy"):
        np.save(path, np.load(path).astype(np.float64))
    data = train.load_data(str(small_dataset))
    assert data.X.dtype == np.float32
    sample = np.load(small_dataset / "sample_00002.npy")
    np.testing.assert_array_equal(data.X[2, :, 0], normalize(sample).astype(np.float32))


def test_the_dataset_hash_follows_the_data(small_dataset):
    first = train.load_data(str(small_dataset)).sha256
    assert train.load_data(str(small_dataset)).sha256 == first
    path = small_dataset / "sample_00004.npy"
    sample = np.load(path)
    sample[100] += 1.0
    np.save(path, sample)
    assert train.load_data(str(small_dataset)).sha256 != first


def test_a_drawn_seed_fits_numpy():
    seeds = {train.draw_seed() for _ in range(20)}
    assert len(seeds) > 1
    assert all(0 <= seed < 2**32 for seed in seeds)


@pytest.mark.parametrize(
    "damage, check",
    [
        (lambda d: (d / "sample_00003.npy").unlink(), "missing"),
        (lambda d: np.save(d / "sample_00003.npy", np.ones(100)), "shape"),
        (lambda d: np.save(d / "sample_00003.npy", np.ones(20000)), "shape"),
    ],
    ids=["missing", "short", "long"],
)
def test_load_data_stops_instead_of_skipping_or_padding(small_dataset, damage, check):
    """LOAD-SILENT: a bad file used to be skipped, cut, or zero-padded."""
    from gwd.data.validate import InvalidDatasetError

    damage(small_dataset)
    with pytest.raises(InvalidDatasetError, match="sample_00003.npy") as error:
        train.load_data(str(small_dataset))
    assert [f.check for f in error.value.findings] == [check]


def test_load_data_stops_on_an_unreadable_labels_file(small_dataset):
    from gwd.data.validate import InvalidDatasetError

    (small_dataset / "labels.csv").write_bytes(b"\xff\xfe\x00broken")
    with pytest.raises(InvalidDatasetError, match="cannot be read"):
        train.load_data(str(small_dataset))


def test_gwd_train_reports_a_broken_dataset_without_traceback(small_dataset, tmp_path):
    (small_dataset / "sample_00003.npy").unlink()
    env = {**os.environ, "GWD_HOME": str(tmp_path), "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-m", "gwd.cli", "train", "--data", str(small_dataset)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "sample_00003.npy: listed in labels.csv, but missing" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "models_registry").exists()


def online_config(tmp_path, **options):
    defaults = dict(
        models_dir=str(tmp_path / "models"),
        plots_dir=str(tmp_path / "plots"),
        scenario="train_v3",
        samples_per_epoch=6,
        val_samples=4,
        test_samples=2,
        batch_size=4,
        workers=0,
        bn_samples=4,
    )
    return train.TrainConfig(**{**defaults, **options})


def test_online_data_records_the_scenario_and_no_dataset(tmp_path):
    """Online: no dataset_hash, a clear marker, and the preprocessing of the
    scenario, which the evaluation and the Glitch Hunter read back."""
    import contextlib

    from gwd.apps.glitch_hunter.sources import start_version
    from gwd.data.scenarios import TRAIN_V3, preprocessing_of

    with contextlib.ExitStack() as stack:
        data = train.prepare_online(online_config(tmp_path), 11, stack)
        X, y = next(data.fit["x"])
    # 6 samples per epoch round up to 2 batches of 4.
    assert (data.steps_per_epoch, data.fit["steps_per_epoch"]) == (2, 2)
    assert X.shape == (4, 16384, 1) and list(y) == [1, 0, 1, 0]
    assert [len(part[1]) for part in (data.validation, data.test)] == [4, 2]
    assert data.recalibration.shape == (4, 16384, 1)
    record = data.record
    assert record["training_data"] == "online"
    assert record["dataset_hash"] is None and "online" in record["dataset_hash_note"]
    assert record["dataset_folder"] is None
    assert record["dataset_scenario"] == "TRAIN_V3"
    assert record["preprocessing"] == "whiten_bandpass"
    assert preprocessing_of(record) == "whiten_bandpass"
    assert training_scenario(record) is TRAIN_V3
    assert record["online"]["scenario"] == "TRAIN_V3"
    assert start_version(record) == 3
    online = record["online"]
    assert online["scenario_hash"] == TRAIN_V3.hash()
    assert (online["seed"], online["samples_per_epoch"]) == (11, 8)
    assert (online["val_samples"], online["test_samples"], online["workers"]) == (
        4,
        2,
        0,
    )
    assert "0x6777646f" in online["stream"]


def test_online_training_refuses_real_noise(tmp_path):
    import contextlib

    with contextlib.ExitStack() as stack, pytest.raises(ValueError, match="real noise"):
        train.prepare_online(online_config(tmp_path, scenario="TRAIN_V1"), 1, stack)


def test_default_workers_leave_two_cpus_and_stay_at_most_the_config():
    from gwd.config import ONLINE_WORKERS

    assert train.default_workers(24) == ONLINE_WORKERS
    assert train.default_workers(6) == 4
    assert train.default_workers(1) == 1


@pytest.mark.slow
def test_gwd_train_online_runs_and_records_the_run(tmp_path):
    """A short CLI run in a tmp GWD_HOME: 3 epochs of 8 fresh samples, 2 workers."""
    env = {**os.environ, "GWD_HOME": str(tmp_path), "CUDA_VISIBLE_DEVICES": ""}
    args = "train --online --scenario train_v2 --samples-per-epoch 8 --batch-size 4"
    args += " --val-samples 8 --test-samples 4 --workers 2 --epochs 3 --seed 4"
    result = subprocess.run(
        [sys.executable, "-m", "gwd.cli", *args.split()],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    (path,) = (tmp_path / "models_registry").glob("*.json")
    metadata = json.loads(path.read_text())
    assert metadata["training_data"] == "online"
    assert metadata["dataset_hash"] is None
    assert metadata["preprocessing"] == "bandpass"
    assert (metadata["epochs_trained"], metadata["steps_per_epoch"]) == (3, 2)
    assert 1 <= metadata["best_epoch"] <= 3
    assert metadata["bn_recalibration_samples"] == 1024
    assert len(metadata["epoch_seconds"]) == 3
    online = metadata["online"]
    assert online["train_samples_seen"] == 24
    assert online["workers"] == 2 and 1 <= online["worker_processes_seen"] <= 2
    assert len(online["wait_seconds"]) == 3
    assert registry.read_metadata(str(path.with_suffix(".keras"))) == metadata
    assert (tmp_path / "models_registry" / (path.stem + ".keras")).exists()
