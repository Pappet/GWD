"""Field test on real LIGO data: known events and a stretch of real H1 noise (P4.3).

The field test is optional (D10): ``gwd eval run`` runs it only with ``--real``, because
it needs network access. Every window is downloaded with a pad of PAD_S seconds on each
side, conditioned as a whole, and cropped afterwards (conditioned_crop), so the filter
ringing at the ends of the data stays out of the scored windows (core:MISS-2, EVAL-4).

The counts are honest about what was measured:

- An event counts as found if its score is >= the threshold, the rule of
  gwd.evaluation.metrics.predict_labels.
- Events inside and outside the training mass range (Event.in_training_range) are
  counted separately (core:MISS-4).
- An event whose download fails or is too short is skipped, not counted as missed. If
  any event of a group is missing, the group reads 'n/a (k of n loaded)' (EVAL-13).
- The false alarms are given as 'k false alarms in T h' of analysed noise, without a
  rate per year (EVAL-21).

fetch_strain keeps a local copy of every downloaded file (cache=True), so a second run
reads the data from disk.
"""

from dataclasses import dataclass

import numpy as np

from gwd.config import N_SAMPLES, SAMPLE_RATE
from gwd.data.gwosc import StrainError, event, fetch_strain
from gwd.data.preprocessing import conditioned_crop, normalize
from gwd.evaluation.metrics import predict_labels

DETECTOR = "H1"

# Seconds of extra data on each side of every window, cut off after conditioning
PAD_S = 16

# Field test events from gwd.data.gwosc.EVENTS, in the order they are reported
FIELD_TEST_EVENTS = tuple(
    event(name) for name in ("GW150914", "GW170817", "GW190521", "GW170104", "GW151226")
)

# A stretch of real noise (Hanford, O3 run), cut into windows of N_SAMPLES
NOISE_START = 1253326755
NOISE_DURATION = 200


@dataclass(frozen=True)
class EventCount:
    """Events found among those that loaded, out of all events of one group."""

    found: int
    loaded: int
    total: int

    @property
    def text(self):
        """'found/total' if every event loaded, else 'n/a (loaded of total loaded)'."""
        if self.loaded < self.total:
            return f"n/a ({self.loaded} of {self.total} loaded)"
        return f"{self.found}/{self.total}"


def far_text(false_alarms, seconds):
    """'k false alarms in T h', or 'n/a (noise not loaded)' if no noise was analysed."""
    if false_alarms is None:
        return "n/a (noise not loaded)"
    return f"{false_alarms} false alarms in {seconds / 3600:.3g} h"


def events_text(in_range, ood):
    """The leaderboard text: the events in the training range, then the OOD events."""
    return f"{in_range.text}; OOD {ood.text}"


# The field test result when it is skipped (--no-real)
NO_FIELD_TEST = {
    "events_text": "n/a",
    "far_text": "n/a",
    "loaded": 0,
    "total": 0,
    "event_scores": {},
}


def load_window(start, end, pad_s=PAD_S, detector=DETECTOR):
    """Download [start, end) with pad_s on each side, condition it, crop the pads.

    Returns (end - start) * SAMPLE_RATE samples. Raises StrainError (from fetch_strain,
    or here if the cropped data have another length) instead of resizing.
    """
    strain = fetch_strain(detector, start, end, pad_s=pad_s)
    data = conditioned_crop(strain, SAMPLE_RATE, pad_s)
    expected = round((end - start) * SAMPLE_RATE)
    if len(data) != expected:
        raise StrainError(
            f"{detector} {start}-{end}: {len(data)} samples after the crop, "
            f"expected {expected}"
        )
    return data


def event_window(ev):
    """The 4 s window of one event, as before P4.3: from int(gps - 2) to int(gps + 2)."""
    return int(ev.gps - 2.0), int(ev.gps + 2.0)


def score_events(model, events=FIELD_TEST_EVENTS):
    """Score each event's window. Returns {name: score}, None where loading failed."""
    scores = {}
    for ev in events:
        try:
            window = load_window(*event_window(ev))
        except Exception as error:
            print(f"   ⚠️ {ev.name} skipped: {error}")
            scores[ev.name] = None
            continue
        # eps=0: the conditioned strain has a std near 1e-21, eps would dominate
        x = normalize(window, eps=0.0).reshape(1, N_SAMPLES, 1)
        scores[ev.name] = float(model.predict(x, verbose=0)[0][0])
    return scores


def count_events(scores, threshold, events=FIELD_TEST_EVENTS):
    """(in training range, out of distribution) as two EventCounts."""
    counts = []
    for in_range in (True, False):
        group = [ev.name for ev in events if ev.in_training_range == in_range]
        loaded = [scores[name] for name in group if scores.get(name) is not None]
        found = int(predict_labels(loaded, threshold).sum()) if loaded else 0
        counts.append(EventCount(found, len(loaded), len(group)))
    return tuple(counts)


def count_false_alarms(model, threshold, start=NOISE_START, duration=NOISE_DURATION):
    """Score the noise in windows of N_SAMPLES. Returns (false alarms, seconds analysed).

    Returns (None, 0) if the noise cannot be loaded.
    """
    try:
        data = load_window(start, start + duration)
    except Exception as error:
        print(f"   ⚠️ Noise skipped: {error}")
        return None, 0
    n_windows = len(data) // N_SAMPLES
    windows = data[: n_windows * N_SAMPLES].reshape(n_windows, N_SAMPLES)
    x = np.array([normalize(w, eps=0.0) for w in windows]).reshape(-1, N_SAMPLES, 1)
    scores = model.predict(x, verbose=0).flatten()
    seconds = n_windows * N_SAMPLES / SAMPLE_RATE
    return int(predict_labels(scores, threshold).sum()), seconds


def run_field_test(model, threshold, events=FIELD_TEST_EVENTS):
    """Score the events and count the false alarms on real noise (network)."""
    print(f"\n🌍 [REAL WORLD] Field test on real {DETECTOR} data (pad {PAD_S} s)...")

    scores = score_events(model, events)
    in_range, ood = count_events(scores, threshold, events)

    print(f"\n   Event scores (detected at score >= {threshold}):")
    for ev in events:
        score = scores[ev.name]
        label = "" if ev.in_training_range else "  (outside the training range)"
        if score is None:
            print(f"      {ev.name:12s}: not loaded{label}")
        else:
            status = "detected" if score >= threshold else "missed"
            print(f"      {ev.name:12s}: {score:.3f} {status}{label}")
    print(f"   In the training range: {in_range.text}")
    print(f"   Outside (OOD):         {ood.text}")

    print(f"\n   Scoring {NOISE_DURATION} s of background noise...")
    false_alarms, seconds = count_false_alarms(model, threshold)
    far = far_text(false_alarms, seconds)
    print(f"   -> {far}")

    return {
        "events_text": events_text(in_range, ood),
        "far_text": far,
        "in_range": in_range,
        "ood": ood,
        "loaded": in_range.loaded + ood.loaded,
        "total": in_range.total + ood.total,
        "false_alarms": false_alarms,
        "seconds": seconds,
        "event_scores": scores,
    }
