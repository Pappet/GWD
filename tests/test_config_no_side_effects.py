"""Importing config, or a script that imports it, creates nothing (P2.3, DATA-7)."""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
DATASET = os.path.join(SRC, "dataset")

# Modules without TensorFlow, so the fast CI job can import them.
LIGHT_MODULES = [
    "config",
    "gwd.config",
    "generate_chirp_dataset",
    "validate_dataset",
    "fetch_real_noise",
    "utils.evaluation",
    "show_leaderboard",
]
TF_MODULES = [
    "train_cnn",
    "evaluate_model_with_roc",
    "optimize_threshold",
    "compare_models_roc",
    "glitch_hunter_app",
]


def import_in_empty_home(home, modules):
    code = "\n".join(
        [
            "import sys",
            f"sys.path[:0] = [{ROOT!r}, {SRC!r}, {DATASET!r}]",
            *(f"import {module}" for module in modules),
        ]
    )
    env = {
        **os.environ,
        "GWD_HOME": str(home),
        "MPLBACKEND": "Agg",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=home,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return sorted(os.listdir(home))


def test_ensure_dirs_creates_nested_folders_and_is_idempotent(tmp_path):
    from gwd.config import ensure_dirs

    first = tmp_path / "a" / "b"
    second = tmp_path / "c"
    ensure_dirs(str(first), str(second))
    ensure_dirs(str(first))
    assert first.is_dir() and second.is_dir()


def test_importing_light_modules_creates_nothing(tmp_path):
    pytest.importorskip("gwpy")  # fetch_real_noise imports it at module level.
    assert import_in_empty_home(tmp_path, LIGHT_MODULES) == []


@pytest.mark.slow
def test_importing_tensorflow_scripts_creates_nothing(tmp_path):
    pytest.importorskip("tensorflow")
    assert import_in_empty_home(tmp_path, TF_MODULES) == []
