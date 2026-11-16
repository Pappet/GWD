"""Where the Glitch Hunter's samples come from: the simulation and GWOSC (P4.4).

Nothing here draws or touches matplotlib, so a worker thread may call any of it.

- SimSource draws a sample through gwd.data.injection from the scenario of a Mode:
  'Training' draws from TRAIN_V1, the training distribution, and is the start mode;
  'Challenge' draws from APP_V1, which is deliberately hard (decision D4).
- GwoscSource downloads a known event or a random stretch of O3 noise with a pad of
  PAD_S seconds on each side, conditions it as a whole, and crops the pads
  (conditioned_crop), like the field test of gwd.evaluation.real_data (P4.3).
"""

import importlib.util
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from gwd.config import DURATION, N_SAMPLES, SAMPLE_RATE
from gwd.data.gwosc import EVENTS, StrainError, fetch_strain
from gwd.data.injection import make_sample, real_noise_provider
from gwd.data.preprocessing import conditioned_crop, normalize
from gwd.data.scenarios import APP_V1, TRAIN_V1

DETECTOR = "H1"

# Seconds of extra data on each side of a window, cut off after conditioning, as in
# the field test (gwd.evaluation.real_data.PAD_S).
PAD_S = 16

# GPS span of the O3 run (O3a and O3b). It includes the break of October 2019 and the
# times a detector was off; a random time there has no data, the download fails, and
# the app falls back to the simulation (APP-19).
O3_START = 1238112018
O3_END = 1253923218

# A random noise window keeps this distance in seconds from every known event, so
# neither the window nor its pads contain the event.
EVENT_CLEARANCE_S = DURATION + 2 * PAD_S


@dataclass(frozen=True)
class Mode:
    """A simulation mode: its name in the GUI, its scenario, and what it means."""

    name: str
    scenario: object  # a gwd.data.scenarios.Scenario
    note: str


TRAINING = Mode(
    "Training",
    TRAIN_V1,
    "the training distribution (TRAIN_V1): peak SNR 1-12, masses 5-100 Msun",
)
CHALLENGE = Mode(
    "Challenge",
    APP_V1,
    "deliberately hard, extrapolation (APP_V1): peak SNR 0.5-2, "
    "partly below the training range",
)
MODES = (TRAINING, CHALLENGE)


@dataclass(frozen=True)
class LoadedSample:
    """One sample the GUI shows and the model rates.

    data: N_SAMPLES values, as the model gets them before normalize().
    has_signal: the truth: a chirp was injected, or the window holds a known event.
    name: what the GUI calls the sample.
    source: "simulation" or "gwosc".
    snr: the peak SNR the signal was scaled to (simulation); 0.0 otherwise.
    signal: the injected chirp in the units of data (simulation), else None.
    noise_kind: "white" or "real" (simulation), "gwosc" for downloaded data.
    event: the gwd.data.gwosc.Event in the window, else None.
    note: the mode's note, or why the simulation stands in for a failed download.
    """

    data: np.ndarray
    has_signal: bool
    name: str
    source: str
    snr: float = 0.0
    signal: Optional[np.ndarray] = None
    noise_kind: str = "white"
    event: Optional[object] = None
    note: str = ""

    @property
    def out_of_distribution(self):
        """True for a known event outside the training masses (Event.in_training_range)."""
        return self.event is not None and not self.event.in_training_range


class SimSource:
    """Simulated samples from the scenario of a mode, through gwd.data.injection.

    A scenario with real noise (TRAIN_V1) draws real chunks from data/noise_background
    when there are any, as gwd data generate does; otherwise all noise is white.
    """

    def __init__(self, mode=TRAINING, noise_provider="default", rng=None):
        self.mode = mode
        if noise_provider == "default":
            noise_provider = real_noise_provider()
        self.noise_provider = noise_provider
        self.rng = rng

    def draw(self, note=None):
        """Draw one sample. note replaces the mode's note (used for a fallback)."""
        sample = make_sample(
            self.mode.scenario, self.rng, noise_provider=self.noise_provider
        )
        return LoadedSample(
            data=sample.data,
            has_signal=sample.has_signal,
            name=f"Simulation ({self.mode.name})",
            source="simulation",
            snr=sample.snr,
            signal=sample.signal,
            noise_kind=sample.noise_kind,
            note=self.mode.note if note is None else note,
        )


def gwpy_available():
    """True if gwpy can be imported, without importing it."""
    return importlib.util.find_spec("gwpy") is not None


class GwoscSource:
    """Real H1 data from GWOSC: a known event or random O3 noise, 50/50.

    rng is a random.Random (default: the module random). fetch is the download
    function, gwd.data.gwosc.fetch_strain; tests pass a fake.
    """

    def __init__(self, detector=DETECTOR, pad_s=PAD_S, rng=None, fetch=None):
        self.detector = detector
        self.pad_s = pad_s
        self.rng = rng or random
        self.fetch = fetch or fetch_strain

    def pick(self):
        """Choose (start, end, event): a known event's window or a noise window.

        The event window is the field test's: int(gps - 2) to int(gps + 2). event is
        None for noise.
        """
        if self.rng.choice([True, False]):
            event = self.rng.choice(EVENTS)
            return int(event.gps - DURATION / 2), int(event.gps + DURATION / 2), event
        return (*self.noise_window(), None)

    def noise_window(self):
        """A random window in O3 that keeps EVENT_CLEARANCE_S from every known event."""
        while True:
            start = int(self.rng.uniform(O3_START, O3_END - DURATION))
            if all(abs(start - e.gps) > EVENT_CLEARANCE_S for e in EVENTS):
                return start, start + int(DURATION)

    def load(self, start, end):
        """Download [start, end) with the pads, condition it whole, crop the pads.

        Returns N_SAMPLES values normalized with eps=0 (the conditioned strain has a std
        near 1e-21, so an eps would dominate). Raises StrainError instead of resizing.
        """
        strain = self.fetch(self.detector, start, end, pad_s=self.pad_s)
        data = conditioned_crop(strain, SAMPLE_RATE, self.pad_s)
        if len(data) != N_SAMPLES:
            raise StrainError(
                f"{self.detector} {start}-{end}: {len(data)} samples after the crop, "
                f"expected {N_SAMPLES}"
            )
        return normalize(data, eps=0.0)

    def draw(self):
        """Download one window. Raises whatever the download or the check raises."""
        start, end, event = self.pick()
        data = self.load(start, end)
        if event is None:
            name, note = f"Real noise, O3, GPS {start}", "no known event in the window"
        else:
            name = event.name
            note = f"{event.kind}, {event.mass1:g} + {event.mass2:g} Msun" + (
                "" if event.in_training_range else ": outside the training masses (OOD)"
            )
        return LoadedSample(
            data=data,
            has_signal=event is not None,
            name=name,
            source="gwosc",
            noise_kind="gwosc",
            event=event,
            note=note,
        )
