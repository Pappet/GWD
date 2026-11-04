"""Measure the golden values, and write them to values.py when run as a script.

Change values.py only in 'golden:' commits, and state the reason (see AGENTS.md).

All measuring runs in a subprocess with GWD_HOME set to a new temp dir. In the repo
directory, generate_dataset would mix in the real noise from data/noise_background,
and the values would differ on every machine without those files.
"""
import contextlib
import glob
import importlib.metadata
import io
import json
import os
import pprint
import random
import subprocess
import sys
import tempfile
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent
REPO_ROOT = GOLDEN_DIR.parents[1]
VALUES_FILE = GOLDEN_DIR / "values.py"

SEED = 1
NUM_DATASET_SAMPLES = 8
ASD_FREQUENCIES = [10.0, 30.0, 100.0, 300.0, 1000.0]
CHIRP = {"t_merger": 3.0, "mass1": 36, "mass2": 29, "distance_mpc": 400, "normalize": False}
CHIRP_SAMPLES = [1000, 4000, 8000, 11000, 12000]
NOISE_SAMPLES = [0, 1, 100, 8192, 16383]
FIXED_INPUT_SAMPLES = [0, 100, 250, 500, 999]


def measure_in_temp_home():
    """Return the measured values, always from a new temp dir as GWD_HOME."""
    with tempfile.TemporaryDirectory(prefix="gwd-golden-") as home:
        output = Path(home) / "measured.json"
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--measure", str(output)],
            env={**os.environ, "GWD_HOME": home}, cwd=home, capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Measuring failed (rc {result.returncode}):\n{result.stderr[-2000:]}")
        return json.loads(output.read_text())


def _summary(values, samples):
    import numpy as np

    values = np.asarray(values)
    return {
        "sum_of_squares": float(np.sum(values**2)),
        "argmax_abs": int(np.argmax(np.abs(values))),
        "samples": [float(values[i]) for i in samples],
    }


def _seed():
    import numpy as np

    random.seed(SEED)
    np.random.seed(SEED)


def _measure_newton(time, waveforms):
    strain, freq = waveforms.generate_astrophysical_chirp(time, **CHIRP)
    summary = _summary(strain, CHIRP_SAMPLES)
    summary["f_at_peak"] = float(freq[summary["argmax_abs"]])
    return summary


def _measure_noise(noise):
    _seed()
    gaussian = noise.generate_gaussian_noise(16384, 1.0)
    _seed()
    colored = noise.generate_colored_noise(16384, 4096, 1.0)
    return {"gaussian": _summary(gaussian, NOISE_SAMPLES), "colored": _summary(colored, NOISE_SAMPLES)}


def _measure_scaling(dataset):
    import numpy as np

    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 30 * t) * t
    noise = 0.5 * np.cos(2 * np.pi * 7 * t) + 0.25 * np.sin(2 * np.pi * 53 * t)

    scaled, actual_snr = dataset.scale_signal_to_snr(signal, noise, 5.0)

    # The formula from evaluate_model_with_roc.py is inline there, so it is repeated here. P3.5 centralizes it.
    data = noise + (signal * 1.5)
    data = (data - np.mean(data)) / np.std(data)

    return {
        "scaled": _summary(scaled, FIXED_INPUT_SAMPLES),
        "actual_snr": float(actual_snr),
        "eval_formula": _summary(data, FIXED_INPUT_SAMPLES),
    }


def _measure_dataset(dataset):
    import numpy as np
    import pandas as pd

    _seed()
    dataset.generate_dataset(NUM_DATASET_SAMPLES, dataset.DATA_DIR_TRAIN)
    labels = pd.read_csv(os.path.join(dataset.DATA_DIR_TRAIN, "labels.csv"))
    samples = [np.load(os.path.join(dataset.DATA_DIR_TRAIN, name)) for name in labels["filename"].iloc[:2]]
    return {
        "labels": labels.to_dict(orient="records"),
        "first_samples": [_summary(sample, NOISE_SAMPLES) for sample in samples],
    }


def _measure_interferometer(interferometer):
    import numpy as np

    model = interferometer.InterferometerModel()
    model.freq = np.array(ASD_FREQUENCIES)
    return {"frequencies": ASD_FREQUENCIES, "asd": [float(v) for v in model.calculate_total_sensitivity()]}


def _measure_pycbc(time, waveforms):
    import numpy as np

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        strain, freq = waveforms.generate_astrophysical_chirp(time, **CHIRP)
    # The waveform code falls back to Newton on any PyCBC error and only prints a message.
    if "PyCBC" in output.getvalue():
        raise RuntimeError(f"PyCBC path did not run: {output.getvalue()}")
    peak = int(np.argmax(np.abs(strain)))
    return {"peak_index": peak, "max_abs_h": float(np.abs(strain[peak])), "f_at_peak": float(freq[peak])}


def _measure(output):
    home = os.environ.get("GWD_HOME", "")
    if not home or Path(home).resolve() == REPO_ROOT:
        sys.exit("Refusing to measure: GWD_HOME must point to a temp dir.")

    sys.path.insert(0, str(REPO_ROOT / "src" / "dataset"))
    import generate_chirp_dataset as dataset  # Also puts src/ and the project root on sys.path.
    import numpy as np
    from gwd_core import interferometer, noise, waveforms

    if glob.glob(os.path.join(dataset.REAL_NOISE_DIR, "*.npy")):
        sys.exit(f"Refusing to measure: real noise files in {dataset.REAL_NOISE_DIR} would change the values.")

    time = np.linspace(0, 4.0, 16384)
    has_pycbc = waveforms.HAS_PYCBC
    waveforms.HAS_PYCBC = False  # Everything except the PyCBC section uses the numpy-only path.
    measured = {
        "newton": _measure_newton(time, waveforms),
        "noise": _measure_noise(noise),
        "scaling": _measure_scaling(dataset),
        "dataset": _measure_dataset(dataset),
        "interferometer": _measure_interferometer(interferometer),
    }
    waveforms.HAS_PYCBC = has_pycbc
    measured["pycbc"] = _measure_pycbc(time, waveforms) if has_pycbc else None

    Path(output).write_text(json.dumps(measured))


def _write_values():
    measured = measure_in_temp_home()
    if measured["pycbc"] is None:
        sys.exit("Refusing to write values.py: pycbc is missing, so the PyCBC values would be lost.")
    versions = ", ".join(f"{name} {importlib.metadata.version(name)}" for name in ("pycbc", "lalsuite", "numpy"))
    VALUES_FILE.write_text(
        '"""Golden values. Generated by tests/golden/regen.py; change only in \'golden:\' commits."""\n'
        f"# Recorded with {versions}.\n\n"
        f"GOLDEN = {pprint.pformat(measured, sort_dicts=False, width=100)}\n"
    )
    print(f"Wrote {VALUES_FILE}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--measure":
        _measure(sys.argv[2])
    else:
        _write_values()
