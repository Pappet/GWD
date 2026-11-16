"""Where a dataset comes from: the manifest and the backend and noise line (P4.1).

Two switches in the environment decide what the data contains (hygiene:HYG-M1):
whether PyCBC can be imported (otherwise the Newtonian approximation draws the
chirps), and whether real noise chunks exist (otherwise every sample gets Gaussian
noise). generate and evaluate print both in one line when they start, with a loud
warning on stderr for Newton. generate also writes them to <out>/manifest.json, and
training copies them into the model's metadata.
"""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys

from gwd.core.waveforms import active_backend

MANIFEST_FILE = "manifest.json"
MANIFEST_FORMAT = 1
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
):
    """The manifest of a dataset of n samples drawn from scenario.

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
        "waveform_backend": active_backend(),
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


def source_line(noise):
    """The one line that names both switches: 'Backend: ... | Noise: ...'."""
    return f"Backend: {active_backend()} | Noise: {noise}"


def announce(noise, notes=()):
    """Print source_line(noise), framed, and the notes as warnings on stderr.

    The Newton warning is added by itself when PyCBC is not the backend.
    """
    print("=" * 70)
    print(source_line(noise))
    print("=" * 70)
    # Flush first, so the warnings follow the line when both streams share a terminal.
    sys.stdout.flush()
    messages = list(notes)
    if active_backend() == "newton":
        messages.insert(0, NEWTON_WARNING)
    for message in messages:
        print(message, file=sys.stderr)
    sys.stderr.flush()
