"""Where a dataset comes from: the manifest and the backend and noise line (P4.1).

Two things decide what the data contains (hygiene:HYG-M1): the waveform backend, and
whether real noise chunks exist (otherwise every sample gets Gaussian noise). Since
P5.2 the backend is no longer a switch in the environment: the pipeline always uses
gwd.config.WAVEFORM_BACKEND ('pycbc') and stops without PyCBC. generate and evaluate
print both in one line when they start, with a loud warning on stderr if a caller
chose Newton on purpose (only tests and the golden values do). generate also writes
them to <out>/manifest.json, and training copies them into the model's metadata.
"""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys

from gwd.config import WAVEFORM_BACKEND
from gwd.core.waveforms import backend_label, check_backend

MANIFEST_FILE = "manifest.json"
MANIFEST_FORMAT = 1
# The time axis of the samples (gwd.core.timebase.time_axis). Manifests without this
# key were written before P5.3, on np.linspace from 0 to duration (CORE-19).
TIME_AXIS = "arange(n)/fs"
# What training records for a dataset without a manifest (written before P4.1).
UNKNOWN_V1 = "unknown (v1)"

NEWTON_WARNING = (
    "!!! WARNING: the waveform backend is the Newtonian approximation, not PyCBC.\n"
    "!!! Its chirps have no merger and ringdown and known errors in frequency and\n"
    "!!! amplitude (CORE-2, CORE-3, CORE-8). Install PyCBC with the physics extra:\n"
    "!!!     pip install -e '.[physics]'"
)


def package_version(distribution):
    """The installed version of distribution, or None if it is not installed."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit():
    """(commit, dirty) of the Git checkout this code runs from, or (None, None).

    dirty is True if tracked files have uncommitted changes. Outside a Git checkout,
    for example in an installed wheel, or without git, both are None.
    """
    here = os.path.dirname(os.path.abspath(__file__))

    def git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=here,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit, dirty


def sha256_file(path):
    """The SHA-256 of a file's bytes as hex, the same as sha256sum prints."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    scenario,
    n,
    seed,
    labels_path,
    real_noise_samples,
    signal_backends,
    noise_chunks,
    noise_dir,
    backend=WAVEFORM_BACKEND,
):
    """The manifest of a dataset of n samples drawn from scenario with backend.

    seed is the seed the samples were drawn with, None if unseeded.
    real_noise_samples counts the samples with a real chunk, signal_backends the signal
    samples per waveform backend, noise_chunks the chunks that were available.
    The manifest has no timestamp, so the same data gives the same manifest.
    """
    commit, dirty = git_commit()
    return {
        "format": MANIFEST_FORMAT,
        "scenario": scenario.name,
        "scenario_hash": scenario.hash(),
        "seed": seed,
        "n": n,
        "time_axis": TIME_AXIS,
        "waveform_backend": backend_label(backend),
        "signal_backends": dict(sorted(signal_backends.items())),
        "real_noise_fraction": real_noise_samples / n if n else 0.0,
        "real_noise_samples": real_noise_samples,
        "scenario_real_noise_fraction": scenario.real_noise_fraction,
        "noise_chunks": noise_chunks,
        "noise_dir": noise_dir,
        "versions": {
            name: package_version(name) for name in ("gwd", "numpy", "pycbc", "pandas")
        },
        "git_commit": commit,
        "git_dirty": dirty,
        "labels_sha256": sha256_file(labels_path),
    }


def write_manifest(folder, manifest):
    """Write manifest to folder/manifest.json and return its path."""
    path = os.path.join(folder, MANIFEST_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
        f.write("\n")
    return path


def read_manifest(folder):
    """(manifest, SHA-256 of manifest.json) of a dataset, or (None, None) without one."""
    path = os.path.join(folder, MANIFEST_FILE)
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest, sha256_file(path)


def source_line(noise, backend=WAVEFORM_BACKEND):
    """The one line that names backend and noise: 'Backend: ... | Noise: ...'."""
    return f"Backend: {backend_label(backend)} | Noise: {noise}"


def announce(noise, notes=(), backend=WAVEFORM_BACKEND):
    """Print source_line(noise, backend), framed, and the notes as warnings on stderr.

    The backend is checked first, so a missing PyCBC raises WaveformBackendError
    before anything is printed. The Newton warning is added for backend 'newton'.
    """
    check_backend(backend)
    print("=" * 70)
    print(source_line(noise, backend))
    print("=" * 70)
    # Flush first, so the warnings follow the line when both streams share a terminal.
    sys.stdout.flush()
    messages = list(notes)
    if backend == "newton":
        messages.insert(0, NEWTON_WARNING)
    for message in messages:
        print(message, file=sys.stderr)
    sys.stderr.flush()
