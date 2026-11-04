import os
import subprocess
import sys

import pytest

from .entrypoints import PIPELINE, Stage


def run_stage(stage, home):
    """Run one stage as a subprocess. GWD_HOME keeps all of its files inside home."""
    env = {**os.environ, "GWD_HOME": str(home), "MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run([sys.executable, *stage.args], input=stage.stdin, env=env, cwd=home,
                          capture_output=True, text=True)


def assert_ran_cleanly(name, result):
    details = f"{name} (rc {result.returncode})\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    assert result.returncode == 0, details
    assert "Traceback" not in result.stderr, details


@pytest.fixture(scope="session")
def pipeline_home(tmp_path_factory):
    """Run every pipeline stage once, in order, and return the temporary project root."""
    home = tmp_path_factory.mktemp("pipeline")
    for name, stage in PIPELINE.items():
        assert_ran_cleanly(name, run_stage(stage, home))
    return home


@pytest.fixture(scope="session")
def app_home(tmp_path_factory):
    """Temporary project root with a dummy model, because Glitch Hunter exits with rc 1 without one."""
    home = tmp_path_factory.mktemp("apps")
    (home / "models_registry").mkdir()
    model_path = home / "models_registry" / "dummy.keras"
    code = "\n".join([
        "import tensorflow as tf",
        "inputs = tf.keras.Input((16384, 1))",
        "x = tf.keras.layers.GlobalAveragePooling1D()(inputs)",
        "outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)",
        f"tf.keras.Model(inputs, outputs).save({str(model_path)!r})",
    ])
    assert_ran_cleanly("dummy model", run_stage(Stage(("-c", code)), home))
    return home
