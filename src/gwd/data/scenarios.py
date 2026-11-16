"""Named injection scenarios (P3.5) and the fixed test sets drawn from them (P5.7).

A Scenario says from which distribution a sample is drawn. The three V1 scenarios
copy what the code did before P3.5, including the differences between training,
evaluation, and the Glitch Hunter. They are kept as they are, so the skew is visible
here and not fixed.

P6.2 adds V2 next to them (TRAIN_V2, INDIST_V2, STRESS_V2, APP_V2): the signal is
injected into simulated noise with the aLIGO design PSD first, and signal and noise are
then conditioned and normalized together, as real data is (see gwd.data.injection).
The SNR of V2 is the optimal (matched filter) SNR against that PSD in BANDPASS (D2 = B),
so the same value means the same detectability for every mass and noise type. V2 uses
no real noise yet (D3 = B; real noise for V2 is P6.3). V1 stays the default of
gwd data generate and of the evaluation until P6.5.

V3 (TRAIN_V3, INDIST_V3, STRESS_V3, APP_V3) is V2 with one change: the conditioning
whitens the padded sum of noise and signal with the known PSD of the noise before the
notches and the band-pass (whiten=True, preprocessing "whiten_bandpass"). The ranges,
the noise, and the draws are those of V2, so with the same seed indist_v3 and
stress_v3 hold the same events and noise realizations as indist_v2 and stress_v2
(EvalSet.stream); only the preprocessing differs.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Optional

MASS_MODES = ("random", "fixed")
PEAK_REFERENCES = ("noise_std", "unit")
SNR_KINDS = ("peak", "optimal")
# The simulated noise of a sample without a real chunk: white (std 1), or Gaussian
# noise with the aLIGO design PSD (gwd.core.interferometer.reference_psd).
SIMULATED_NOISES = ("white", "aligo")
# Fields that came with V2 (P6.2), with the value every V1 scenario has. hash() leaves a
# field out while it has this value, so the V1 hashes in existing manifests stay valid.
V2_FIELD_DEFAULTS = {"simulated_noise": "white", "pad_s": 0.0}
# The field that came with V3, with the value every V1 and V2 scenario has; hash()
# leaves it out at that value too, so the V1 and V2 hashes stay valid.
V3_FIELD_DEFAULTS = {"whiten": False}
# What a model's input goes through before the CNN, on simulated and real data alike:
# "bandpass" (V1, V2) notches the power line and band-passes; "whiten_bandpass" (V3)
# whitens with the PSD of the noise first (gwd.data.preprocessing). A model records it
# in its metadata (key "preprocessing"); preprocessing_of() reads it back.
PREPROCESSINGS = ("bandpass", "whiten_bandpass")
DEFAULT_PREPROCESSING = "bandpass"


@dataclass(frozen=True)
class Scenario:
    """The parameter distribution of one kind of sample.

    snr_kind: what the SNR means. "peak" (V1): the peak of the injected signal divided
        by the noise level, not a matched filter SNR. "optimal" (V2): the optimal SNR
        of the signal in the sample window against the PSD of the simulated noise, in
        BANDPASS (gwd.core.snr.optimal_snr_psd).
    snr_range: uniform range of the SNR of kind snr_kind.
    mass_mode: "random" draws m1 log-uniform in masses, then m2 log-uniform between
        masses[0] and m1. "fixed" uses masses as (m1, m2).
    t_merger_range: uniform range of the merger time in seconds.
    real_noise_fraction: share of samples with a real noise chunk, if chunks exist.
        Without chunks, every sample gets white noise.
    peak_reference: the noise level the peak SNR refers to. "noise_std" measures the
        std of the noise of each sample; "unit" multiplies the chirp (peak 1) by the
        SNR, so the noise std is taken as 1. None for snr_kind "optimal".
    condition_signal: whether the signal is conditioned like the noise. V1 adds the
        unfiltered signal to the (conditioned) real noise. V2 (True) adds it to the raw
        simulated noise and conditions the sum (gwd.data.preprocessing.condition_array).
    simulated_noise: "white" or "aligo", the noise of a sample without a real chunk.
        V1 knows only "white".
    pad_s: seconds of extra noise and signal on each side of the window, conditioned
        with it and cut off afterwards (V2); 0 in V1.
    whiten: whether the conditioning first whitens the padded sum with the known PSD
        of the simulated noise (V3, gwd.data.preprocessing.whiten). Needs
        condition_signal; False in V1 and V2.
    """

    name: str
    version: int
    snr_range: tuple
    mass_mode: str
    masses: tuple
    t_merger_range: tuple
    real_noise_fraction: float
    peak_reference: Optional[str]
    snr_kind: str = "peak"
    condition_signal: bool = False
    simulated_noise: str = "white"
    pad_s: float = 0.0
    whiten: bool = False

    def __post_init__(self):
        if self.mass_mode not in MASS_MODES:
            raise ValueError(f"mass_mode must be one of {MASS_MODES}")
        if not 0.0 <= self.real_noise_fraction <= 1.0:
            raise ValueError("real_noise_fraction must be between 0 and 1")
        if self.snr_kind not in SNR_KINDS:
            raise ValueError(f"snr_kind must be one of {SNR_KINDS}")
        if self.simulated_noise not in SIMULATED_NOISES:
            raise ValueError(f"simulated_noise must be one of {SIMULATED_NOISES}")
        if self.pad_s < 0:
            raise ValueError("pad_s must not be negative")
        if self.condition_signal:
            self._check_v2()
        else:
            self._check_v1()
        if self.whiten and not self.condition_signal:
            raise ValueError("whiten needs condition_signal (V3 builds on V2)")

    @property
    def preprocessing(self):
        """What a model trained on this scenario expects of its input (PREPROCESSINGS).

        "whiten_bandpass" for V3 (whiten, then notches and band-pass), else "bandpass":
        V1 and V2 run the notch and band-pass chain only, also on real data.
        """
        return "whiten_bandpass" if self.whiten else DEFAULT_PREPROCESSING

    def _check_v1(self):
        """V1: a peak SNR, white simulated noise, no pad (gwd.data.injection, V1)."""
        if self.peak_reference not in PEAK_REFERENCES:
            raise ValueError(f"peak_reference must be one of {PEAK_REFERENCES}")
        if self.snr_kind != "peak" or self.simulated_noise != "white" or self.pad_s:
            raise ValueError(
                "Without condition_signal (V1) a scenario has snr_kind 'peak', "
                "simulated_noise 'white', and pad_s 0"
            )

    def _check_v2(self):
        """V2: an optimal SNR, no peak reference, a pad, no real noise until P6.3."""
        if self.snr_kind != "optimal" or self.peak_reference is not None:
            raise ValueError(
                "With condition_signal (V2) a scenario has snr_kind 'optimal' and "
                "peak_reference None (D2 = B)"
            )
        if self.pad_s <= 0:
            raise ValueError("With condition_signal (V2) pad_s must be positive")
        if self.real_noise_fraction:
            raise ValueError(
                "Real noise for V2 comes with P6.3; a V2 scenario has "
                "real_noise_fraction 0"
            )

    def noise_text(self):
        """What noise the scenario draws, for the start lines (without real chunks)."""
        if self.simulated_noise == "aligo":
            how = (
                "whitened with its PSD and conditioned with the signal"
                if self.whiten
                else "conditioned with the signal"
            )
            return f"simulated aLIGO design noise (aLIGOZeroDetHighPower), {how}"
        if self.condition_signal:
            return "white gaussian, conditioned with the signal"
        return "gaussian"

    def distribution(self):
        """All fields except the name, to compare what two scenarios draw."""
        fields = asdict(self)
        del fields["name"]
        return fields

    def hash(self):
        """A short hash of all fields that is the same in every process.

        The built-in hash() of a dataclass changes between processes, so it cannot be
        stored with a dataset.
        """
        fields = asdict(self)
        for key, default in {**V2_FIELD_DEFAULTS, **V3_FIELD_DEFAULTS}.items():
            if fields[key] == default:
                del fields[key]
        text = json.dumps(fields, sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:12]


# Training data (gwd.data.generate).
TRAIN_V1 = Scenario(
    name="TRAIN_V1",
    version=1,
    snr_range=(1.0, 12.0),
    mass_mode="random",
    masses=(5.0, 100.0),
    t_merger_range=(2.0, 3.6),
    real_noise_fraction=0.8,
    peak_reference="noise_std",
)

# Simulated test sets of evaluate, thresholds, and compare.
EVAL_V1 = Scenario(
    name="EVAL_V1",
    version=1,
    snr_range=(0.5, 2.5),
    mass_mode="fixed",
    masses=(30.0, 30.0),
    t_merger_range=(2.0, 3.5),
    real_noise_fraction=0.0,
    peak_reference="unit",
)

# Simulation mode of the Glitch Hunter.
APP_V1 = Scenario(
    name="APP_V1",
    version=1,
    snr_range=(0.5, 2.0),
    mass_mode="fixed",
    masses=(30.0, 30.0),
    t_merger_range=(2.0, 3.8),
    real_noise_fraction=0.0,
    peak_reference="unit",
)

# --- V2 (P6.2): inject first, then condition together; optimal SNR (D2 = B) ---

# Training data of V2. The optimal SNR range 4-20 lies around the usual detection
# threshold of 8; for comparison, TRAIN_V1 gave optimal SNRs of 19 to 916 (P5.5).
TRAIN_V2 = Scenario(
    name="TRAIN_V2",
    version=2,
    snr_range=(4.0, 20.0),
    mass_mode="random",
    masses=(5.0, 100.0),
    t_merger_range=(2.0, 3.6),
    real_noise_fraction=0.0,
    peak_reference=None,
    snr_kind="optimal",
    condition_signal=True,
    simulated_noise="aligo",
    pad_s=2.0,
)

# The in-distribution test set: TRAIN_V2 under its own name, so the two can only
# differ by seed and n (tests/data/test_scenario_v2.py, test_v2_consistent).
INDIST_V2 = replace(TRAIN_V2, name="INDIST_V2")

# The stress test set: the training range in bins, plus one explicit extrapolation bin
# below it (optimal SNR 2-4).
STRESS_V2 = replace(TRAIN_V2, name="STRESS_V2", snr_range=(2.0, 20.0))

# The Glitch Hunter's 'Challenge' mode in V2: 30/30 M☉ at the low end of the training
# range, so it is hard but inside TRAIN_V2 (APP-2).
APP_V2 = replace(
    TRAIN_V2,
    name="APP_V2",
    snr_range=(4.0, 8.0),
    mass_mode="fixed",
    masses=(30.0, 30.0),
)

# --- V3: V2 plus whitening with the known PSD before the notches and band-pass ---

# Training data of V3: TRAIN_V2 with whitening. Everything else, and so every draw,
# is that of TRAIN_V2 (tests/data/test_scenario_v3.py).
TRAIN_V3 = replace(TRAIN_V2, name="TRAIN_V3", version=3, whiten=True)
INDIST_V3 = replace(INDIST_V2, name="INDIST_V3", version=3, whiten=True)
STRESS_V3 = replace(STRESS_V2, name="STRESS_V3", version=3, whiten=True)
APP_V3 = replace(APP_V2, name="APP_V3", version=3, whiten=True)

SCENARIOS = {
    s.name: s
    for s in (
        TRAIN_V1,
        EVAL_V1,
        APP_V1,
        TRAIN_V2,
        INDIST_V2,
        STRESS_V2,
        APP_V2,
        TRAIN_V3,
        INDIST_V3,
        STRESS_V3,
        APP_V3,
    )
}


@dataclass(frozen=True)
class EvalSet:
    """A named, fixed test set drawn from a scenario (P5.7, gwd.evaluation.testsets).

    The name carries the version (indist_v1), and it names the cached files; a set
    whose drawing changes gets a new name. snr_bin_width splits the scenario's SNR
    range (of its snr_kind: peak in V1, optimal in V2) into equal bins, which the
    signal samples fill in turn, so every bin is occupied. training is the scenario
    the models were trained on: signal samples outside its parameter ranges count as
    extrapolation. stream names the set whose random streams this set draws from; None
    means its own name. A V3 set draws from the streams of its V2 counterpart, so with
    the same seed it holds the same events and noise, preprocessed differently.
    """

    name: str
    scenario: Scenario
    snr_bin_width: float
    description: str
    training: Scenario = TRAIN_V1
    stream: Optional[str] = None

    @property
    def stream_name(self):
        """The name that keys the random streams of the set (gwd.evaluation.testsets)."""
        return self.stream or self.name

    @property
    def preprocessing(self):
        """The preprocessing of the set's samples: that of its scenario."""
        return self.scenario.preprocessing

    @property
    def snr_kind(self):
        """The SNR the set is drawn and binned in: 'peak' (V1) or 'optimal' (V2)."""
        return self.scenario.snr_kind

    def snr_edges(self):
        """The edges of the SNR bins, from the low to the high end of the range."""
        low, high = self.scenario.snr_range
        bins = round((high - low) / self.snr_bin_width)
        if bins < 1 or abs(low + bins * self.snr_bin_width - high) > 1e-9:
            raise ValueError(
                f"{self.name}: the bin width {self.snr_bin_width} does not split "
                f"the {self.snr_kind} SNR range {self.scenario.snr_range} evenly"
            )
        return tuple(low + i * self.snr_bin_width for i in range(bins + 1))


# The standard test sets (D4 = A): the training distribution, and the old evaluation
# distribution. OOD events stay in the field test, reported on their own.
INDIST_V1 = EvalSet(
    name="indist_v1",
    scenario=TRAIN_V1,
    snr_bin_width=1.0,
    description="the training distribution TRAIN_V1, with its noise mix",
)
STRESS_V1 = EvalSet(
    name="stress_v1",
    scenario=EVAL_V1,
    snr_bin_width=0.25,
    description="EVAL_V1 as before P5.7: white noise, 30/30 M☉, peak SNR 0.5–2.5",
)
# The V2 sets (P6.2): the training distribution TRAIN_V2, and the same with the
# extrapolation bin of optimal SNR 2-4 below it. Models trained on TRAIN_V2 (P6.5).
INDIST_V2_SET = EvalSet(
    name="indist_v2",
    scenario=INDIST_V2,
    snr_bin_width=2.0,
    description="the training distribution TRAIN_V2: simulated aLIGO noise, "
    "optimal SNR 4–20",
    training=TRAIN_V2,
)
STRESS_V2_SET = EvalSet(
    name="stress_v2",
    scenario=STRESS_V2,
    snr_bin_width=2.0,
    description="TRAIN_V2 with an extrapolation bin: optimal SNR 2–20, 2–4 below the "
    "training range",
    training=TRAIN_V2,
)

# The V3 sets: the V2 sets with whitening, drawn from the V2 streams (same events).
INDIST_V3_SET = replace(
    INDIST_V2_SET,
    name="indist_v3",
    scenario=INDIST_V3,
    description="indist_v2 whitened: the same events and noise as indist_v2 with the "
    "same seed, whitened with the aLIGO PSD before the band-pass",
    training=TRAIN_V3,
    stream="indist_v2",
)
STRESS_V3_SET = replace(
    STRESS_V2_SET,
    name="stress_v3",
    scenario=STRESS_V3,
    description="stress_v2 whitened: the same events and noise as stress_v2 with the "
    "same seed, whitened with the aLIGO PSD before the band-pass",
    training=TRAIN_V3,
    stream="stress_v2",
)

TESTSETS = {
    s.name: s
    for s in (
        INDIST_V1,
        STRESS_V1,
        INDIST_V2_SET,
        STRESS_V2_SET,
        INDIST_V3_SET,
        STRESS_V3_SET,
    )
}


def preprocessing_of(metadata):
    """The preprocessing a model expects, from its metadata (gwd.registry.read_metadata).

    The key "preprocessing" (written by gwd train since V3); else that of the scenario
    in "dataset_scenario", if it is a known one; else DEFAULT_PREPROCESSING, which is
    what every model got before V3 (no metadata, or a model from before P4.1).
    """
    metadata = metadata or {}
    value = metadata.get("preprocessing")
    if value in PREPROCESSINGS:
        return value
    scenario = SCENARIOS.get(metadata.get("dataset_scenario"))
    return scenario.preprocessing if scenario else DEFAULT_PREPROCESSING


def counterpart_set(spec, preprocessing):
    """The test set with the draws of spec and the given preprocessing, or None.

    spec itself if its preprocessing is that one; else the set that draws from the same
    streams with the same scenario fields apart from the whitening (indist_v2 and
    indist_v3, stress_v2 and stress_v3). None if there is none (the V1 sets have no
    whitened counterpart).
    """
    if spec.preprocessing == preprocessing:
        return spec

    def draws(s):
        fields = s.scenario.distribution()
        for key in ("version", "whiten"):
            fields.pop(key)
        return s.stream_name, s.snr_bin_width, fields

    for other in TESTSETS.values():
        if other.preprocessing == preprocessing and draws(other) == draws(spec):
            return other
    return None
