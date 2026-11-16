"""gwd.registry: model search, selection by timestamp in the name, metadata (P3.3)."""

import os
import subprocess
import sys
import time

import pytest

from gwd import registry

OLD = "gwd_model_20251124-234144"
NEW = "gwd_model_20260301-120000"


def touch(folder, name, mtime=None):
    path = folder / name
    path.write_bytes(b"")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return str(path)


@pytest.fixture
def two_models(tmp_path):
    old = touch(tmp_path, OLD + ".keras")
    new = touch(tmp_path, NEW + ".keras")
    return tmp_path, old, new


def test_latest_is_the_newest_timestamp_in_the_name(two_models):
    folder, old, new = two_models
    assert registry.latest_model(str(folder)) == new


def test_latest_ignores_mtime_and_ctime_of_an_older_model(two_models):
    folder, old, new = two_models
    future = time.time() + 3600
    os.utime(old, (future, future))  # newest mtime
    os.chmod(old, 0o600)  # newest ctime
    assert os.path.getctime(old) >= os.path.getctime(new)
    assert registry.latest_model(str(folder)) == new


def test_list_models_is_oldest_first_and_only_keras(two_models):
    folder, old, new = two_models
    touch(folder, OLD + ".json")
    touch(folder, "physics_baseline.json")
    assert registry.list_models(str(folder)) == [old, new]


def test_missing_or_empty_folder_has_no_models(tmp_path):
    assert registry.list_models(str(tmp_path / "missing")) == []
    assert registry.latest_model(str(tmp_path)) is None


def test_name_without_timestamp_falls_back_to_mtime_with_a_warning(two_models):
    folder, old, new = two_models
    future = time.time() + 3600
    legacy = touch(folder, "dummy.keras", mtime=future)
    with pytest.warns(UserWarning, match="dummy.keras"):
        assert registry.latest_model(str(folder)) == legacy


def test_invalid_timestamp_falls_back_to_mtime(tmp_path):
    path = touch(tmp_path, "gwd_model_20251399-000000.keras", mtime=0)
    with pytest.warns(UserWarning, match="20251399"):
        assert registry.list_models(str(tmp_path)) == [path]


def test_resolve_latest_name_and_path(two_models):
    folder, old, new = two_models
    assert registry.resolve("latest", str(folder)) == new
    assert registry.resolve(OLD, str(folder)) == old
    assert registry.resolve(OLD + ".keras", str(folder)) == old
    assert registry.resolve(old, str(folder / "elsewhere")) == old


def test_resolve_raises_for_unknown_model_and_empty_folder(two_models, tmp_path):
    folder, old, new = two_models
    with pytest.raises(FileNotFoundError, match="gwd_model_19990101-000000"):
        registry.resolve("gwd_model_19990101-000000", str(folder))
    with pytest.raises(FileNotFoundError, match="No models"):
        registry.resolve("latest", str(tmp_path / "empty"))


def test_model_name():
    assert registry.model_name(os.path.join("a", NEW + ".keras")) == NEW


def test_metadata_round_trip(two_models):
    folder, old, new = two_models
    assert registry.read_metadata(new) is None
    written = registry.write_metadata(new, {"model_name": NEW, "epochs": 40})
    assert written == str(folder / (NEW + ".json"))
    assert registry.read_metadata(new) == {"model_name": NEW, "epochs": 40}


TESTSET = {
    "name": "stress_v1",
    "seed": 0,
    "n": 1000,
    "part": "calib_bg",
    "sha256": "ab",
}


def test_set_threshold_keeps_the_other_metadata(two_models):
    """P5.8: the threshold, its FPR, strategy, and test set go into the model's .json."""
    folder, old, new = two_models
    registry.write_metadata(new, {"model_name": NEW, "epochs": 40})
    registry.set_threshold(new, 0.73, 0.01, 0.009, 1000, "fixed_fpr_1", TESTSET)
    metadata = registry.read_metadata(new)
    assert metadata["model_name"] == NEW and metadata["epochs"] == 40
    assert metadata["threshold"] == 0.73
    assert metadata["threshold_fpr"] == 0.01
    assert metadata["threshold_fpr_achieved"] == 0.009
    assert metadata["threshold_n_background"] == 1000
    assert metadata["threshold_strategy"] == "fixed_fpr_1"
    assert metadata["threshold_testset"] == TESTSET
    assert registry.get_threshold(new) == 0.73
    assert registry.threshold_record(new)["threshold_testset"]["sha256"] == "ab"


def test_set_threshold_creates_metadata_for_a_model_without_one(two_models):
    folder, old, new = two_models
    registry.set_threshold(old, 0.6, 0.01, 0.01, 10, "fixed_fpr_1", TESTSET)
    assert set(registry.read_metadata(old)) == set(registry.THRESHOLD_KEYS)


def test_without_a_threshold_the_fallback_is_used_with_a_warning(two_models):
    folder, old, new = two_models
    registry.write_metadata(new, {"model_name": NEW})
    for path in (old, new):  # no metadata, and metadata from before P5.8
        with pytest.warns(registry.MissingThresholdWarning, match="--save"):
            assert registry.get_threshold(path) == registry.THRESHOLD_FALLBACK == 0.5
        assert registry.threshold_record(path) is None


def test_import_does_not_load_tensorflow():
    code = "import sys, gwd.registry; print('tensorflow' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


@pytest.mark.parametrize("script", ["evaluate", "thresholds", "compare"])
def test_evaluation_scripts_take_model_and_import_no_tensorflow(script, tmp_path):
    code = "\n".join(
        [
            "import sys",
            f"from gwd.evaluation import {script} as module",
            "rc = module.run(models=['gwd_model_19990101-000000'])"
            if script == "compare"
            else "rc = module.run(model='gwd_model_19990101-000000')",
            "print(rc, 'tensorflow' in sys.modules)",
        ]
    )
    env = {**os.environ, "GWD_HOME": str(tmp_path), "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Model 'gwd_model_19990101-000000' is neither" in result.stdout
    assert result.stdout.splitlines()[-1] == "1 False"


@pytest.mark.slow
def test_load_model_without_compiling(tmp_path):
    tf = pytest.importorskip("tensorflow")
    inputs = tf.keras.Input((8, 1))
    outputs = tf.keras.layers.Dense(1)(tf.keras.layers.Flatten()(inputs))
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    path = str(tmp_path / (NEW + ".keras"))
    model.save(path)

    loaded = registry.load_model(path)
    assert loaded.output_shape == (None, 1)
    assert not loaded.compiled
