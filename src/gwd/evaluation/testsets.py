"""Fixed, hashed test sets (P5.7).

build(name, n, seed) draws a named test set once, writes it to
data/eval/<name>_<seed>_<n>.npz, and its manifest to results/testsets/<name>_<seed>_<n>.json;
every later call with the same name, n, and seed loads it from there (a cache hit).
The sets are defined in gwd.data.scenarios.TESTSETS (default of D4):

- indist_v1: the training distribution TRAIN_V1, with its noise mix (80% real noise
  where data/noise_background/ has chunks, else white);
- stress_v1: EVAL_V1, the distribution every simulated number came from before P5.7;
- indist_v2 and stress_v2 (P6.2): TRAIN_V2 (simulated aLIGO noise, conditioned with
  the signal, optimal SNR 4-20), and the same with an extrapolation bin of optimal
  SNR 2-4 below it. Their training reference is TRAIN_V2.

A set has two parts, drawn from separate random streams:

- test: n samples, balanced (even indices signal, odd ones noise only), for ROC/AUC,
  the FPR at the operating point, and the efficiency curve of its signal samples. The
  k-th signal sample draws its SNR (the scenario's snr_kind: peak in V1, optimal in
  V2) uniform within bin k mod (number of bins) of EvalSet.snr_edges(), so the SNR is
  still uniform over the scenario's range and every bin is occupied once there are as
  many signal samples as bins;
- calib_bg: n noise-only samples, only for calibrating thresholds (P5.8).

Every sample is z-score normalized and stored as float32. The npz also holds the
parameters of every sample (peak and optimal SNR, masses, merger time, noise source,
noise chunk); the manifest repeats them, with the scenario, seed, waveform backend,
package versions, Git commit, and the SHA-256 of the set.

The SHA-256 covers a header (format, set name, scenario name and hash, seed, n) and
every array (name, dtype, shape, bytes), not the npz file, whose zip entries carry the
time they were written. load() recomputes it and raises InvalidTestsetError, a
ValueError, if it differs from the manifest: an array was changed after the build.
It also raises if the set is stale: written by another TESTSET_FORMAT, from a scenario
whose fields changed since (Scenario.hash()), or, for a scenario with real noise, from
other noise chunks than data/noise_background/ holds now. build() never replaces a
cached set on its own; rebuild=True (gwd eval build-testset --rebuild) does.

Random streams: each part draws from np.random.RandomState(MT19937(SeedSequence(seed,
spawn_key=(SEED_STREAM, crc32(name), part)))). Training seeds the legacy MT19937 with
the integer itself (gwd data generate, np.random.seed in gwd train) or SeedSequence(seed)
without a spawn key (the split, TensorFlow), so no training seed gives a test set
stream. Nothing here touches the global numpy RNG.

Limits: as soon as real noise is drawn, the hash is reproducible only on a machine with
the same chunks (the owner's). Tests and CI build the sets with an empty noise folder,
so only with Gaussian noise. In V1 the real chunks of indist_v1 can be the ones the
training data used; P6.3 gives each split its own pool.
"""

import hashlib
import json
import os
import tempfile
import zlib
from collections import Counter
from dataclasses import dataclass

import numpy as np

from gwd.config import (
    DATA_DIR_NOISE,
    DEFAULT_TESTSET,
    N_SAMPLES,
    SAMPLE_DTYPE,
    TESTSET_DIR,
    TESTSET_MANIFEST_DIR,
    TESTSET_SEED,
    TESTSET_SIZE,
    WAVEFORM_BACKEND,
    ensure_dirs,
)
from gwd.core.waveforms import backend_label, check_backend
from gwd.data.generate import NOISE_SOURCES
from gwd.data.injection import make_sample, real_noise_files, real_noise_provider
from gwd.data.preprocessing import normalize
from gwd.data.provenance import TIME_AXIS, git_commit, package_version, sha256_file
from gwd.data.scenarios import TESTSETS

# Raise it whenever build() draws or stores differently, so old caches count as stale.
TESTSET_FORMAT = 1
# The first word of every spawn key: 'gwdt' in ASCII.
SEED_STREAM = 0x67776474
PARTS = ("test", "calib_bg")
# Quantile bins of the SNR a set is not drawn in (the optimal SNR in V1, the peak SNR
# in V2) and of the total mass of the signal samples.
OPTIMAL_SNR_BINS = 5
TOTAL_MASS_BINS = 4
# Relative slack for the training range check of stored floats.
RANGE_SLACK = 1e-9


class InvalidTestsetError(ValueError):
    """A cached test set does not match its manifest, or it is stale."""


@dataclass
class BuiltSet:
    """A test set as built or loaded: its EvalSet, seed, n, hash, files, and arrays."""

    spec: object  # gwd.data.scenarios.EvalSet
    seed: int
    n: int
    sha256: str
    path: str
    manifest_path: str
    cache_hit: bool
    arrays: dict

    @property
    def X(self):
        """The test part as the CNN takes it: (n, N_SAMPLES, 1), float32."""
        return self.arrays["test_X"].reshape(self.n, N_SAMPLES, 1)

    @property
    def y(self):
        """The labels of the test part: 1 signal, 0 noise only."""
        return self.arrays["test_y"].astype(int)

    @property
    def calib_X(self):
        """The noise-only part calib_bg: (n, N_SAMPLES, 1), float32."""
        return self.arrays["calib_X"].reshape(-1, N_SAMPLES, 1)

    def __getattr__(self, name):
        # test_peak_snr, test_noise_source, ... as attributes: peak_snr, noise_source.
        arrays = self.__dict__.get("arrays", {})
        if f"test_{name}" in arrays:
            return arrays[f"test_{name}"]
        raise AttributeError(name)

    @property
    def total_mass(self):
        return self.arrays["test_mass1"] + self.arrays["test_mass2"]

    def label(self):
        return f"{self.spec.name} (seed {self.seed}, n {self.n})"

    def short_hash(self):
        return self.sha256[:12]


def get(name):
    """The EvalSet called name; KeyError names the known sets."""
    try:
        return TESTSETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown test set {name!r}; known: {', '.join(sorted(TESTSETS))}"
        ) from None


def file_stem(name, seed, n):
    return f"{name}_{seed}_{n}"


def paths(name, seed, n, data_dir=None, manifest_dir=None):
    """(npz path, manifest path) of a set; None means the folder of gwd.config."""
    stem = file_stem(name, seed, n)
    return (
        os.path.join(data_dir or TESTSET_DIR, stem + ".npz"),
        os.path.join(manifest_dir or TESTSET_MANIFEST_DIR, stem + ".json"),
    )


def part_rng(name, seed, part):
    """The RandomState of one part of a set, disjoint from every training stream."""
    key = (SEED_STREAM, zlib.crc32(name.encode()), PARTS.index(part))
    sequence = np.random.SeedSequence(seed, spawn_key=key)
    return np.random.RandomState(np.random.MT19937(sequence))


def header(spec, seed, n):
    """What the hash covers besides the arrays."""
    return {
        "format": TESTSET_FORMAT,
        "name": spec.name,
        "scenario": spec.scenario.name,
        "scenario_hash": spec.scenario.hash(),
        "seed": seed,
        "n": n,
    }


def content_hash(arrays, head):
    """SHA-256 of head (as sorted JSON) and every array: name, dtype, shape, bytes."""
    digest = hashlib.sha256(json.dumps(head, sort_keys=True).encode())
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(array.dtype.str.encode())
        digest.update(repr(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def noise_fingerprint(spec, noise_dir=None):
    """The chunks a set with real noise draws from: count and a hash of names and bytes.

    None for a scenario without real noise, which does not depend on the folder.
    """
    if spec.scenario.real_noise_fraction == 0:
        return None
    files = real_noise_files(noise_dir or DATA_DIR_NOISE)
    digest = hashlib.sha256()
    for path in files:
        digest.update(os.path.basename(path).encode())
        digest.update(sha256_file(path).encode())
    return {"chunks": len(files), "sha256": digest.hexdigest()}


def _check_arguments(n, seed):
    if n < 2:
        raise ValueError(f"A test set needs at least 2 samples, got n={n}")
    if seed < 0:
        raise ValueError(f"The seed must not be negative, got {seed}")


def _draw(spec, seed, n, noise_dir, backend):
    """The arrays of both parts of a set, and the per-sample rows of the manifest."""
    provider = None
    if spec.scenario.real_noise_fraction > 0:
        provider = real_noise_provider(noise_dir or DATA_DIR_NOISE)

    edges = spec.snr_edges()
    rng = part_rng(spec.name, seed, "test")
    X = np.empty((n, N_SAMPLES), dtype=SAMPLE_DTYPE)
    columns = {key: [] for key in ("y", "peak_snr", "optimal_snr", "mass1", "mass2")}
    columns.update(t_merger=[], noise_source=[], noise_file=[])
    signals = 0
    for i in range(n):
        has_signal = i % 2 == 0
        snr = None
        if has_signal:
            # Stratified: signal k draws within bin k mod bins.
            b = signals % (len(edges) - 1)
            snr = rng.uniform(edges[b], edges[b + 1])
            signals += 1
        sample = make_sample(
            spec.scenario,
            rng,
            has_signal=has_signal,
            snr=snr,
            noise_provider=provider,
            backend=backend,
        )
        X[i] = normalize(sample.data)
        params = sample.params
        columns["y"].append(int(sample.has_signal))
        columns["peak_snr"].append(float(sample.peak_snr))
        columns["optimal_snr"].append(float(sample.optimal_snr))
        columns["mass1"].append(params.mass1 if params else 0.0)
        columns["mass2"].append(params.mass2 if params else 0.0)
        columns["t_merger"].append(params.t_merger if params else 0.0)
        columns["noise_source"].append(NOISE_SOURCES[sample.noise_kind])
        columns["noise_file"].append(sample.noise_file or "")

    rng = part_rng(spec.name, seed, "calib_bg")
    calib = np.empty((n, N_SAMPLES), dtype=SAMPLE_DTYPE)
    calib_source, calib_file = [], []
    for i in range(n):
        sample = make_sample(
            spec.scenario, rng, has_signal=False, noise_provider=provider
        )
        calib[i] = normalize(sample.data)
        calib_source.append(NOISE_SOURCES[sample.noise_kind])
        calib_file.append(sample.noise_file or "")

    arrays = {"test_X": X, "calib_X": calib}
    arrays["test_y"] = np.array(columns.pop("y"), dtype=np.int8)
    for key in ("peak_snr", "optimal_snr", "mass1", "mass2", "t_merger"):
        arrays[f"test_{key}"] = np.array(columns[key], dtype=np.float64)
    for key in ("noise_source", "noise_file"):
        arrays[f"test_{key}"] = np.array(columns[key], dtype=str)
    arrays["calib_noise_source"] = np.array(calib_source, dtype=str)
    arrays["calib_noise_file"] = np.array(calib_file, dtype=str)
    return arrays


def sample_rows(arrays):
    """The per-sample table of the manifest, for both parts."""
    test = [
        {
            "index": i,
            "label": int(arrays["test_y"][i]),
            "peak_snr": float(arrays["test_peak_snr"][i]),
            "optimal_snr": float(arrays["test_optimal_snr"][i]),
            "mass1": float(arrays["test_mass1"][i]),
            "mass2": float(arrays["test_mass2"][i]),
            "t_merger": float(arrays["test_t_merger"][i]),
            "noise_source": str(arrays["test_noise_source"][i]),
            "noise_file": str(arrays["test_noise_file"][i]),
        }
        for i in range(len(arrays["test_y"]))
    ]
    calib = [
        {
            "index": i,
            "label": 0,
            "noise_source": str(source),
            "noise_file": str(arrays["calib_noise_file"][i]),
        }
        for i, source in enumerate(arrays["calib_noise_source"])
    ]
    return {"test": test, "calib_bg": calib}


def build_manifest(spec, seed, n, arrays, sha256, path, noise_dir, backend):
    commit, dirty = git_commit()
    built = BuiltSet(spec, seed, n, sha256, path, "", False, arrays)
    return {
        **header(spec, seed, n),
        "description": spec.description,
        "parts": {"test": n, "calib_bg": len(arrays["calib_noise_source"])},
        "sha256": sha256,
        "hash_of": "header fields format..n as sorted JSON, then every array: "
        "name, dtype, shape, bytes",
        "file": os.path.abspath(path),
        "rng": "RandomState(MT19937(SeedSequence(seed, spawn_key=("
        f"{SEED_STREAM:#x}, crc32(name), part))))",
        "time_axis": TIME_AXIS,
        "waveform_backend": backend_label(backend),
        "scenario_fields": spec.scenario.distribution(),
        "training_reference": spec.training.name,
        "noise_dir": os.path.abspath(noise_dir or DATA_DIR_NOISE),
        "noise": noise_fingerprint(spec, noise_dir),
        "noise_sources": {
            "test": dict(
                sorted(Counter(map(str, arrays["test_noise_source"])).items())
            ),
            "calib_bg": dict(
                sorted(Counter(map(str, arrays["calib_noise_source"])).items())
            ),
        },
        "bins": {key: [float(e) for e in edges] for key, edges in bins(built).items()},
        "versions": {name: package_version(name) for name in ("gwd", "numpy", "pycbc")},
        "git_commit": commit,
        "git_dirty": dirty,
        "samples": sample_rows(arrays),
    }


def _write_atomic(path, write):
    """Write to a temporary file next to path, then move it into place."""
    folder = os.path.dirname(path)
    ensure_dirs(folder)
    handle, temporary = tempfile.mkstemp(dir=folder, suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as f:
            write(f)
        # mkstemp creates the file for the owner only; give it the usual mode.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(temporary, 0o666 & ~umask)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def stale_reasons(manifest, spec, seed, n, noise_dir=None):
    """Why a manifest does not describe the set build() would draw now; [] if it does."""
    reasons = []
    expected = header(spec, seed, n)
    for key in ("format", "name", "seed", "n", "scenario", "scenario_hash"):
        if manifest.get(key) != expected[key]:
            reasons.append(
                f"{key} is {manifest.get(key)!r}, expected {expected[key]!r}"
            )
    if manifest.get("noise") != noise_fingerprint(spec, noise_dir):
        reasons.append("the noise chunks in the noise folder changed since the build")
    return reasons


def _rebuild_hint(name, seed, n):
    return f"gwd eval build-testset --testset {name} --seed {seed} --n {n} --rebuild"


def load(
    name=DEFAULT_TESTSET,
    n=TESTSET_SIZE,
    seed=TESTSET_SEED,
    data_dir=None,
    manifest_dir=None,
    noise_dir=None,
):
    """Load a built set and check it against its manifest.

    Raises FileNotFoundError if the npz or the manifest is missing, and
    InvalidTestsetError if the set is stale or its arrays do not match the hash.
    """
    spec = get(name)
    path, manifest_path = paths(name, seed, n, data_dir, manifest_dir)
    for file in (path, manifest_path):
        if not os.path.exists(file):
            raise FileNotFoundError(f"No test set file {file}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    reasons = stale_reasons(manifest, spec, seed, n, noise_dir)
    if reasons:
        raise InvalidTestsetError(
            f"The test set {path} is stale: {'; '.join(reasons)}. Rebuild it with "
            f"'{_rebuild_hint(name, seed, n)}'."
        )
    with np.load(path, allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in npz.files}
    actual = content_hash(arrays, header(spec, seed, n))
    if actual != manifest.get("sha256"):
        raise InvalidTestsetError(
            f"The test set {path} does not match the SHA-256 in {manifest_path} "
            f"(found {actual[:12]}, recorded {str(manifest.get('sha256'))[:12]}): "
            "its arrays were changed after the build. Rebuild it with "
            f"'{_rebuild_hint(name, seed, n)}'."
        )
    return BuiltSet(spec, seed, n, actual, path, manifest_path, True, arrays)


def build(
    name=DEFAULT_TESTSET,
    n=TESTSET_SIZE,
    seed=TESTSET_SEED,
    rebuild=False,
    data_dir=None,
    manifest_dir=None,
    noise_dir=None,
    backend=WAVEFORM_BACKEND,
):
    """Return the set name with n test samples and seed; draw and store it only once.

    If its npz and manifest exist, the set is loaded and checked (load(); a stale or
    changed set raises InvalidTestsetError instead of being replaced). Otherwise, or
    with rebuild=True, it is drawn, written, and returned with cache_hit False. The
    folders default to TESTSET_DIR, TESTSET_MANIFEST_DIR, and DATA_DIR_NOISE.
    backend is checked before drawing.
    """
    spec = get(name)
    _check_arguments(n, seed)
    path, manifest_path = paths(name, seed, n, data_dir, manifest_dir)
    if not rebuild and os.path.exists(path) and os.path.exists(manifest_path):
        return load(name, n, seed, data_dir, manifest_dir, noise_dir)

    check_backend(backend)
    arrays = _draw(spec, seed, n, noise_dir, backend)
    sha256 = content_hash(arrays, header(spec, seed, n))
    manifest = build_manifest(spec, seed, n, arrays, sha256, path, noise_dir, backend)
    _write_atomic(path, lambda f: np.savez(f, **arrays))
    text = json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"
    _write_atomic(manifest_path, lambda f: f.write(text.encode("utf-8")))
    return BuiltSet(spec, seed, n, sha256, path, manifest_path, False, arrays)


# --- bins and the training range ---


def quantile_edges(values, bins):
    """Edges that split values into bins of about equal counts; one bin if all equal."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.array([0.0, 0.0])
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    return edges if len(edges) > 1 else np.array([edges[0], edges[0]])


def snr_values(built, kind):
    """The SNR of kind ('peak' or 'optimal') of every sample of the part test."""
    return built.peak_snr if kind == "peak" else built.optimal_snr


def bins(built):
    """The bin edges of the signal samples: peak SNR, optimal SNR, total mass (M☉).

    The SNR the set was drawn in (peak in V1, optimal in V2): its equal bins. The other
    SNR and the total mass: quantile bins of the set's signal samples, so every bin is
    occupied; a fixed mass is one bin.
    """
    signal = built.y == 1
    edges = {}
    for kind in ("peak", "optimal"):
        if kind == built.spec.snr_kind:
            edges[f"{kind}_snr"] = np.array(built.spec.snr_edges())
        else:
            values = snr_values(built, kind)[signal]
            edges[f"{kind}_snr"] = quantile_edges(values, OPTIMAL_SNR_BINS)
    edges["total_mass"] = quantile_edges(built.total_mass[signal], TOTAL_MASS_BINS)
    return edges


def _within(values, low, high):
    slack = RANGE_SLACK * max(abs(low), abs(high), 1.0)
    return (values >= low - slack) & (values <= high + slack)


def outside_training(built):
    """True for each signal sample whose parameters lie outside the training ranges.

    The ranges are those of built.spec.training (TRAIN_V1, or TRAIN_V2 for the V2
    sets): the SNR of its snr_kind (peak or optimal), merger time, and the masses
    (random: both in the mass range and mass2 <= mass1; fixed: equal to them).
    Noise-only samples are False.
    """
    training = built.spec.training
    m1, m2 = built.mass1, built.mass2
    inside = _within(snr_values(built, training.snr_kind), *training.snr_range)
    inside &= _within(built.t_merger, *training.t_merger_range)
    if training.mass_mode == "random":
        inside &= _within(m1, *training.masses) & _within(m2, *training.masses)
        inside &= m2 <= m1 * (1 + RANGE_SLACK)
    else:
        inside &= _within(m1, training.masses[0], training.masses[0])
        inside &= _within(m2, training.masses[1], training.masses[1])
    return (built.y == 1) & ~inside


def describe(built):
    """Lines for the console: what was built or loaded, the hash, and the counts."""
    status = "Cache hit" if built.cache_hit else "Built"
    test_sources = Counter(map(str, built.noise_source))
    calib_sources = Counter(map(str, built.arrays["calib_noise_source"]))

    def counts(counter):
        return ", ".join(f"{k} {v}" for k, v in sorted(counter.items()))

    signal = int((built.y == 1).sum())
    return [
        f"{status}: test set {built.label()}, {built.spec.description}",
        f"   SHA-256:  {built.sha256}",
        f"   Arrays:   {built.path}",
        f"   Manifest: {built.manifest_path}",
        f"   test:     {built.n} samples ({signal} signal, {built.n - signal} noise); "
        f"noise {counts(test_sources)}",
        f"   calib_bg: {len(built.arrays['calib_noise_source'])} noise samples; "
        f"noise {counts(calib_sources)}",
    ]
