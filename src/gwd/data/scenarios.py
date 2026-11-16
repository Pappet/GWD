"""Named injection scenarios (P3.5).

A Scenario says from which distribution a sample is drawn. The three V1 scenarios
copy what the code did before P3.5, including the differences between training,
evaluation, and the Glitch Hunter. They are kept as they are, so the skew is visible
here and not fixed; P6.2 adds V2 next to them.
"""

import hashlib
import json
from dataclasses import asdict, dataclass

MASS_MODES = ("random", "fixed")
PEAK_REFERENCES = ("noise_std", "unit")


@dataclass(frozen=True)
class Scenario:
    """The parameter distribution of one kind of sample.

    snr_kind: what the SNR means. V1 only knows "peak": the peak of the injected
        signal divided by the noise level, not a matched filter SNR.
    snr_range: uniform range of the peak SNR.
    mass_mode: "random" draws m1 log-uniform in masses, then m2 log-uniform between
        masses[0] and m1. "fixed" uses masses as (m1, m2).
    t_merger_range: uniform range of the merger time in seconds.
    real_noise_fraction: share of samples with a real noise chunk, if chunks exist.
        Without chunks, every sample gets white noise.
    peak_reference: the noise level the peak SNR refers to. "noise_std" measures the
        std of the noise of each sample; "unit" multiplies the chirp (peak 1) by the
        SNR, so the noise std is taken as 1.
    condition_signal: whether the signal is conditioned like the noise. V1 adds the
        unfiltered signal to the (conditioned) real noise.
    """

    name: str
    version: int
    snr_range: tuple
    mass_mode: str
    masses: tuple
    t_merger_range: tuple
    real_noise_fraction: float
    peak_reference: str
    snr_kind: str = "peak"
    condition_signal: bool = False

    def __post_init__(self):
        if self.mass_mode not in MASS_MODES:
            raise ValueError(f"mass_mode must be one of {MASS_MODES}")
        if self.peak_reference not in PEAK_REFERENCES:
            raise ValueError(f"peak_reference must be one of {PEAK_REFERENCES}")
        if not 0.0 <= self.real_noise_fraction <= 1.0:
            raise ValueError("real_noise_fraction must be between 0 and 1")

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
        text = json.dumps(asdict(self), sort_keys=True)
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

SCENARIOS = {s.name: s for s in (TRAIN_V1, EVAL_V1, APP_V1)}


@dataclass(frozen=True)
class EvalSet:
    """A named, fixed test set drawn from a scenario (P5.7, gwd.evaluation.testsets).

    The name carries the version (indist_v1), and it names the cached files; a set
    whose drawing changes gets a new name. peak_snr_bin_width splits the scenario's
    peak SNR range into equal bins, which the signal samples fill in turn, so every
    bin is occupied. training is the scenario the models were trained on: signal
    samples outside its parameter ranges count as extrapolation.
    """

    name: str
    scenario: Scenario
    peak_snr_bin_width: float
    description: str
    training: Scenario = TRAIN_V1

    def peak_snr_edges(self):
        """The edges of the peak SNR bins, from the low to the high end of the range."""
        low, high = self.scenario.snr_range
        bins = round((high - low) / self.peak_snr_bin_width)
        if bins < 1 or abs(low + bins * self.peak_snr_bin_width - high) > 1e-9:
            raise ValueError(
                f"{self.name}: the bin width {self.peak_snr_bin_width} does not split "
                f"the peak SNR range {self.scenario.snr_range} evenly"
            )
        return tuple(low + i * self.peak_snr_bin_width for i in range(bins + 1))


# The standard test sets (D4 = A): the training distribution, and the old evaluation
# distribution. OOD events stay in the field test, reported on their own.
INDIST_V1 = EvalSet(
    name="indist_v1",
    scenario=TRAIN_V1,
    peak_snr_bin_width=1.0,
    description="the training distribution TRAIN_V1, with its noise mix",
)
STRESS_V1 = EvalSet(
    name="stress_v1",
    scenario=EVAL_V1,
    peak_snr_bin_width=0.25,
    description="EVAL_V1 as before P5.7: white noise, 30/30 M☉, peak SNR 0.5–2.5",
)

TESTSETS = {s.name: s for s in (INDIST_V1, STRESS_V1)}
