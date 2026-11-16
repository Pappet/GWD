import os
import subprocess
import sys

import numpy as np
import pytest

from .regen import measure_in_temp_home
from .values import GOLDEN

pytestmark = pytest.mark.golden

EXACT_RTOL = 1e-9
DATASET_RTOL = 1e-6
REGEN = os.path.join(os.path.dirname(__file__), "regen.py")


@pytest.fixture(scope="module")
def measured():
    return measure_in_temp_home()


def assert_matches(observed, expected, path="", rtol=EXACT_RTOL):
    """Numbers must match within rtol, everything else exactly."""
    if isinstance(expected, dict):
        assert list(observed) == list(expected), path
        for key in expected:
            assert_matches(observed[key], expected[key], f"{path}.{key}", rtol)
    elif isinstance(expected, list):
        assert len(observed) == len(expected), path
        for i, (obs, exp) in enumerate(zip(observed, expected)):
            assert_matches(obs, exp, f"{path}[{i}]", rtol)
    elif isinstance(expected, float):
        assert np.isclose(observed, expected, rtol=rtol, atol=0.0), (
            f"{path}: {observed} != {expected}"
        )
    else:
        assert observed == expected, path


@pytest.mark.parametrize("section", ["noise", "scaling", "interferometer"])
def test_numpy_only_values_are_unchanged(measured, section):
    assert_matches(measured[section], GOLDEN[section], section)


def test_dataset_values_are_unchanged(measured):
    """The dataset is drawn with PyCBC since P6.1, pinned by the lockfile; DATASET_RTOL
    leaves room for rounding differences between machines, not for a changed draw."""
    assert_matches(measured["dataset"], GOLDEN["dataset"], "dataset", DATASET_RTOL)


def test_pycbc_values_are_within_tolerance(measured):
    observed, expected = measured["pycbc"], GOLDEN["pycbc"]
    assert abs(observed["peak_index"] - expected["peak_index"]) <= 1
    assert np.isclose(observed["max_abs_h"], expected["max_abs_h"], rtol=1e-3, atol=0.0)
    assert np.isclose(observed["f_at_peak"], expected["f_at_peak"], rtol=1e-2, atol=0.0)


def run_measure(tmp_path, env):
    output = tmp_path / "measured.json"
    result = subprocess.run(
        [sys.executable, REGEN, "--measure", str(output)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    return result, output


def test_measuring_refuses_without_gwd_home(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "GWD_HOME"}
    result, output = run_measure(tmp_path, env)
    assert result.returncode != 0
    assert "GWD_HOME" in result.stderr
    assert not output.exists()


def test_measuring_refuses_when_real_noise_is_present(tmp_path):
    noise_folder = tmp_path / "data" / "noise_background"
    noise_folder.mkdir(parents=True)
    (noise_folder / "real_noise_0000.npy").touch()
    result, output = run_measure(tmp_path, {**os.environ, "GWD_HOME": str(tmp_path)})
    assert result.returncode != 0
    assert "real noise" in result.stderr
    assert not output.exists()
    assert not (tmp_path / "data" / "training" / "labels.csv").exists()
