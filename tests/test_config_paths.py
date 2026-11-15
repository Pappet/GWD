"""Where gwd.config and the root config shim put the project root (P2.3)."""

import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_SOURCE = os.path.join(ROOT, "src", "gwd", "config.py")
WARNING = "Using GWD_HOME="

PATH_NAMES = [
    "ROOT_DIR",
    "DATA_DIR_TRAIN",
    "DATA_DIR_NOISE",
    "MODELS_DIR",
    "RESULTS_DIR",
    "PLOTS_DIR",
    "VALIDATION_DATASET_PLOTS_DIR",
    "TRAINING_PLOTS_DIR",
    "EVALUATION_MODELS_PLOTS_DIR",
    "LEADERBOARD_FILE",
    "LEGACY_LEADERBOARD_FILES",
    "OPTIMAL_THRESHOLDS_FILE",
]


def expected_paths(root):
    """The paths exactly as the old root config.py built them from its own folder."""
    return {
        "ROOT_DIR": root,
        "DATA_DIR_TRAIN": os.path.join(root, "data/training"),
        "DATA_DIR_NOISE": os.path.join(root, "data/noise_background"),
        "MODELS_DIR": os.path.join(root, "models_registry"),
        "RESULTS_DIR": os.path.join(root, "results"),
        "PLOTS_DIR": os.path.join(root, "plots"),
        "VALIDATION_DATASET_PLOTS_DIR": os.path.join(root, "plots/validation_dataset"),
        "TRAINING_PLOTS_DIR": os.path.join(root, "plots/training"),
        "EVALUATION_MODELS_PLOTS_DIR": os.path.join(root, "plots/evaluation_models"),
        "LEADERBOARD_FILE": os.path.join(root, "results", "model_leaderboard.csv"),
        "LEGACY_LEADERBOARD_FILES": [
            os.path.join(root, "analysis/model_leaderboard.csv"),
            os.path.join(root, "model_leaderboard.csv"),
        ],
        "OPTIMAL_THRESHOLDS_FILE": os.path.join(
            root, "results", "optimal_thresholds.json"
        ),
    }


def environment(gwd_home=None):
    env = {k: v for k, v in os.environ.items() if k != "GWD_HOME"}
    if gwd_home is not None:
        env["GWD_HOME"] = gwd_home
    return env


def import_both(cwd, env):
    """Import the root shim and gwd.config in a fresh process; return both path sets and stderr."""
    code = f"""
import json, sys
sys.path.insert(0, {ROOT!r})
import config, gwd.config
names = {PATH_NAMES!r}
print(json.dumps([{{n: getattr(m, n) for n in names}} for m in (config, gwd.config)]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    shim, package = json.loads(result.stdout)
    return shim, package, result.stderr


def test_without_gwd_home_paths_are_unchanged(tmp_path):
    # cwd is a temp dir on purpose: the root must not depend on the working directory.
    shim, package, stderr = import_both(tmp_path, environment())
    assert shim == package == expected_paths(ROOT)
    assert WARNING not in stderr


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_blank_gwd_home_counts_as_unset(tmp_path, value):
    shim, package, stderr = import_both(tmp_path, environment(value))
    assert shim == package == expected_paths(ROOT)
    assert WARNING not in stderr


def test_gwd_home_moves_every_path_and_warns_once(tmp_path):
    home = str(tmp_path / "home")
    shim, package, stderr = import_both(tmp_path, environment(f"  {home} "))
    assert shim == package == expected_paths(home)
    assert stderr.count(WARNING) == 1, stderr
    assert f"{WARNING}{home}" in stderr


def test_without_pyproject_and_gwd_home_import_fails(tmp_path):
    # Like an install into site-packages: no pyproject.toml anywhere above the module.
    if any(
        os.path.isfile(os.path.join(parent, "pyproject.toml"))
        for parent in [str(tmp_path), *map(str, tmp_path.parents)]
    ):
        pytest.skip("The temp dir lies below a pyproject.toml")
    shutil.copy(CONFIG_SOURCE, tmp_path / "standalone_config.py")
    result = subprocess.run(
        [sys.executable, "-c", "import standalone_config"],
        env=environment(),
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Set GWD_HOME" in result.stderr
