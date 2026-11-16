"""Write a training dataset: samples, labels.csv, and manifest.json (P4.1).

labels.csv has one row per sample. Since P4.1 it also has the columns
waveform_backend ('pycbc' or 'newton', empty without signal), noise_source ('real' or
'gaussian'), and noise_file (the chunk's file name, empty for Gaussian noise). The new
columns come after the old seven, so readers that select columns by name are not
affected. manifest.json records how the dataset was made, see gwd.data.provenance.
"""

import glob
import os
from collections import Counter

import numpy as np
import pandas as pd

from gwd.config import (
    DATA_DIR_NOISE,
    DATA_DIR_TRAIN,
    LABELS_FILE,
    SAMPLES_FOR_TRAINING,
    ensure_dirs,
)
from gwd.data.injection import make_sample, real_noise_provider
from gwd.data.provenance import announce, build_manifest, write_manifest
from gwd.data.scenarios import TRAIN_V1

# noise_kind of gwd.data.injection -> noise_source in labels.csv
NOISE_SOURCES = {"real": "real", "white": "gaussian"}
# The only files that overwrite=True removes from the output folder.
SAMPLE_PATTERN = "sample_*.npy"


class OutputFolderError(ValueError):
    """The output folder cannot take a new dataset; nothing was written or removed."""


def prepare_output_folder(folder, overwrite=False, noise_dir=DATA_DIR_NOISE):
    """Make sure folder can take a new dataset, and return the files removed.

    A missing or empty folder is fine. A folder with any entry raises
    OutputFolderError, unless overwrite is true: then its sample_*.npy files and its
    labels.csv are removed, and nothing else. Other files and subfolders stay, and
    manifest.json is replaced when the new dataset is written. The noise folder is
    refused even with overwrite, because its chunks would be mixed with samples.
    """
    if os.path.realpath(folder) == os.path.realpath(noise_dir):
        raise OutputFolderError(
            f"'{folder}' is the noise folder; choose another output folder."
        )
    if not os.path.isdir(folder):
        if os.path.exists(folder):
            raise OutputFolderError(f"'{folder}' exists and is not a folder.")
        return []
    entries = os.listdir(folder)
    if not entries:
        return []
    if not overwrite:
        raise OutputFolderError(
            f"'{folder}' is not empty: it holds {', '.join(sorted(entries)[:3])}"
            f"{', ...' if len(entries) > 3 else ''}. Pass --overwrite to "
            f"replace its {SAMPLE_PATTERN} and {LABELS_FILE}, or choose another --out."
        )
    doomed = sorted(glob.glob(os.path.join(folder, SAMPLE_PATTERN)))
    labels = os.path.join(folder, LABELS_FILE)
    if os.path.isfile(labels):
        doomed.append(labels)
    for path in doomed:
        if os.path.isfile(path):
            os.remove(path)
    return doomed


def describe_noise(scenario, noise_provider, noise_dir=DATA_DIR_NOISE):
    """(text for the noise part of the start line, warnings) for a dataset."""
    if scenario.real_noise_fraction == 0:
        return f"gaussian ({scenario.name} uses no real noise)", []
    share = f"{scenario.real_noise_fraction:.0%}"
    if noise_provider is None:
        warning = (
            f"!!! WARNING: no real noise chunks in '{noise_dir}', so every sample gets "
            f"Gaussian noise\n!!! ({scenario.name} asks for {share} real noise). "
            "Run 'gwd data fetch-noise' first to download them."
        )
        return "gaussian only (no real chunks found)", [warning]
    chunks = len(noise_provider.files)
    return f"real for {share} of samples ({chunks} chunks), else gaussian", []


def generate_dataset(
    num_samples=SAMPLES_FOR_TRAINING,
    output_folder=DATA_DIR_TRAIN,
    scenario=TRAIN_V1,
    rng=None,
    seed=None,
    noise_dir=DATA_DIR_NOISE,
    overwrite=False,
):
    """Write num_samples samples of scenario, labels.csv, and manifest.json to output_folder.

    output_folder must be missing or empty, see prepare_output_folder(); with
    overwrite, its old samples and labels.csv are removed first.

    Even indices are signal samples, odd ones noise only (strictly 50/50). The noise
    is real where scenario and the chunks in noise_dir allow it. seed draws the samples
    from np.random.RandomState(seed), which gives the same samples as seeding the
    global RNG with it; rng is used instead if given. Without either, the global numpy
    RNG is used (see gwd.data.injection), and the manifest records the seed as None.
    Returns the path of manifest.json.
    """
    if rng is not None and seed is not None:
        raise ValueError("Pass rng or seed, not both")
    if seed is not None:
        rng = np.random.RandomState(seed)
    removed = prepare_output_folder(output_folder, overwrite, noise_dir)
    if removed:
        print(f"Removed {len(removed)} old files from '{output_folder}' (--overwrite).")

    noise_provider = real_noise_provider(noise_dir)
    noise_text, notes = describe_noise(scenario, noise_provider, noise_dir)
    announce(noise_text, notes)
    print(
        f"🚀 Generating data ({scenario.name}, {scenario.snr_kind} SNR). Target: {num_samples}"
    )

    ensure_dirs(output_folder)
    labels = []
    signal_backends = Counter()

    for i in range(num_samples):
        sample = make_sample(
            scenario, rng, has_signal=(i % 2 == 0), noise_provider=noise_provider
        )
        params = sample.params

        filename = f"sample_{i:05d}.npy"
        np.save(os.path.join(output_folder, filename), sample.data)

        if sample.waveform_backend:
            signal_backends[sample.waveform_backend] += 1
        labels.append(
            {
                "filename": filename,
                "has_signal": int(sample.has_signal),
                "merger_time": float(params.t_merger) if params else 0.0,
                "snr": float(sample.snr),
                "mass1": float(params.mass1) if params else 0.0,
                "mass2": float(params.mass2) if params else 0.0,
                "distance": float(params.distance) if params else 0.0,
                "waveform_backend": sample.waveform_backend or "",
                "noise_source": NOISE_SOURCES[sample.noise_kind],
                "noise_file": sample.noise_file or "",
            }
        )

        if i % 1000 == 0:
            print(f"   ... {i}/{num_samples}")

    labels_path = os.path.join(output_folder, LABELS_FILE)
    pd.DataFrame(labels).to_csv(labels_path, index=False)

    real_samples = sum(row["noise_source"] == "real" for row in labels)
    manifest = build_manifest(
        scenario,
        num_samples,
        seed,
        labels_path,
        real_samples,
        signal_backends,
        len(noise_provider.files) if noise_provider else 0,
        noise_dir,
    )
    manifest_path = write_manifest(output_folder, manifest)

    share = f" ({real_samples / num_samples:.0%})" if num_samples else ""
    print(f"   Real noise: {real_samples} of {num_samples} samples{share}")
    backends = ", ".join(f"{n} {name}" for name, n in sorted(signal_backends.items()))
    print(f"   Signal chirps: {backends or 'none'}")
    print(f"✅ Done. Manifest: {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    generate_dataset()
