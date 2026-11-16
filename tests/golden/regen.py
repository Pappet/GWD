"""Measure the golden values, and write them to values.py when run as a script.

Change values.py only in 'golden:' commits, and state the reason (see AGENTS.md).

All measuring runs in a subprocess with GWD_HOME set to a new temp dir. In the repo
directory, generate_dataset would mix in the real noise from data/noise_background,
and the values would differ on every machine without those files.
"""

import glob
import importlib.metadata
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
CHIRP = {
    "t_merger": 3.0,
    "mass1": 36,
    "mass2": 29,
    "distance_mpc": 400,
    "normalize": False,
}
CHIRP_SAMPLES = [1000, 4000, 8000, 11000, 12000]
NOISE_SAMPLES = [0, 1, 100, 8192, 16383]
FIXED_INPUT_SAMPLES = [0, 100, 250, 500, 999]


def measure_in_temp_home():
    """Return the measured values, always from a new temp dir as GWD_HOME."""
    with tempfile.TemporaryDirectory(prefix="gwd-golden-") as home:
        output = Path(home) / "measured.json"
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--measure", str(output)],
            env={**os.environ, "GWD_HOME": home},
            cwd=home,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Measuring failed (rc {result.returncode}):\n{result.stderr[-2000:]}"
            )
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
    strain, freq = waveforms.generate_chirp(time, backend="newton", **CHIRP)
    summary = _summary(strain, CHIRP_SAMPLES)
    summary["f_at_peak"] = float(freq[summary["argmax_abs"]])
    return summary


def _measure_noise(noise):
    import numpy as np

    # RandomState(SEED) draws what the seeded global RNG drew before P5.3.
    gaussian = noise.white_noise(16384, 1.0, rng=np.random.RandomState(SEED))
    # Since P5.4 the 1/f spectrum of generate_colored_noise is pink_psd, drawn with
    # the one-sided normalization of colored_noise_from_psd (CORE-9).
    colored = noise.colored_noise_from_psd(
        16384, 4096, noise.pink_psd, rng=np.random.RandomState(SEED)
    )
    return {
        "gaussian": _summary(gaussian, NOISE_SAMPLES),
        "colored": _summary(colored, NOISE_SAMPLES),
    }


def _measure_scaling(snr):
    import numpy as np

    from gwd.data.preprocessing import normalize

    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 30 * t) * t
    noise = 0.5 * np.cos(2 * np.pi * 7 * t) + 0.25 * np.sin(2 * np.pi * 53 * t)

    scaled, actual_snr = snr.scale_signal_to_snr(signal, noise, 5.0)

    # gwd.core.snr since P5.5, gwd.data.injection before; the arithmetic is the same.
    # The evaluation path since P3.5: the "unit" reference (noise + signal * snr) and
    # normalize with its default eps. The values were recorded with a plain z-score;
    # the eps changes them by a relative 1e-10, inside EXACT_RTOL.
    unit_scaled, _ = snr.scale_signal_to_snr(signal, noise, 1.5, "unit")
    data = normalize(noise + unit_scaled)

    return {
        "scaled": _summary(scaled, FIXED_INPUT_SAMPLES),
        "actual_snr": float(actual_snr),
        "eval_formula": _summary(data, FIXED_INPUT_SAMPLES),
    }


def _measure_dataset(dataset):
    import numpy as np
    import pandas as pd

    _seed()
    # Newton on purpose: the dataset values stay numpy-only (P5.2). The pipeline itself
    # always draws with PyCBC.
    dataset.generate_dataset(
        NUM_DATASET_SAMPLES, dataset.DATA_DIR_TRAIN, backend="newton"
    )
    # keep_default_na: empty cells (no backend, no chunk) stay "" instead of NaN.
    labels = pd.read_csv(
        os.path.join(dataset.DATA_DIR_TRAIN, "labels.csv"), keep_default_na=False
    )
    samples = [
        np.load(os.path.join(dataset.DATA_DIR_TRAIN, name))
        for name in labels["filename"].iloc[:2]
    ]
    return {
        "labels": labels.to_dict(orient="records"),
        "first_samples": [_summary(sample, NOISE_SAMPLES) for sample in samples],
    }


def _measure_interferometer(interferometer):
    import numpy as np

    model = interferometer.InterferometerModel()
    model.freq = np.array(ASD_FREQUENCIES)
    return {
        "frequencies": ASD_FREQUENCIES,
        "asd": [float(v) for v in model.calculate_total_sensitivity()],
    }


def _measure_pycbc(time, waveforms):
    import numpy as np

    # Raises WaveformBackendError on any PyCBC error; nothing falls back (P5.2).
    strain, freq = waveforms.generate_chirp(time, backend="pycbc", **CHIRP)
    peak = int(np.argmax(np.abs(strain)))
    return {
        "peak_index": peak,
        "max_abs_h": float(np.abs(strain[peak])),
        "f_at_peak": float(freq[peak]),
    }


def _measure(output):
    home = os.environ.get("GWD_HOME", "")
    if not home or Path(home).resolve() == REPO_ROOT:
        sys.exit("Refusing to measure: GWD_HOME must point to a temp dir.")

    import gwd.data.generate as dataset
    from gwd.config import DATA_DIR_NOISE
    from gwd.core import interferometer, noise, snr, waveforms
    from gwd.core.timebase import time_axis

    if glob.glob(os.path.join(DATA_DIR_NOISE, "*.npy")):
        sys.exit(
            f"Refusing to measure: real noise files in {DATA_DIR_NOISE} would change the values."
        )

    time = time_axis(4096, 4.0)
    # Everything except the PyCBC section passes backend="newton", the numpy-only path.
    measured = {
        "newton": _measure_newton(time, waveforms),
        "noise": _measure_noise(noise),
        "scaling": _measure_scaling(snr),
        "dataset": _measure_dataset(dataset),
        "interferometer": _measure_interferometer(interferometer),
    }
    pycbc = waveforms.pycbc_available()
    measured["pycbc"] = _measure_pycbc(time, waveforms) if pycbc else None

    Path(output).write_text(json.dumps(measured))


def _write_values():
    measured = measure_in_temp_home()
    if measured["pycbc"] is None:
        sys.exit(
            "Refusing to write values.py: pycbc is missing, so the PyCBC values would be lost."
        )
    versions = ", ".join(
        f"{name} {importlib.metadata.version(name)}"
        for name in ("pycbc", "lalsuite", "numpy")
    )
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
