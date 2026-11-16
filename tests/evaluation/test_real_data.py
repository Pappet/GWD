"""The real data field test (P4.3): padding, honest counting, only with --real.

fetch_strain is mocked; nothing here needs the network.
"""

import numpy as np
import pytest

from gwd.config import N_SAMPLES, SAMPLE_RATE
from gwd.data.gwosc import StrainError
from gwd.data.preprocessing import condition_strain, conditioned_crop
from gwd.evaluation import real_data
from gwd.evaluation.real_data import EventCount

THRESHOLD = 0.15
EDGE = int(0.1 * SAMPLE_RATE)  # 0.1 s


class ScoreModel:
    """Stands in for a Keras model: scores in call order, then the last one again."""

    def __init__(self, *scores):
        self.scores = list(scores)
        self.shapes = []

    def predict(self, x, verbose=0):
        self.shapes.append(x.shape)
        score = self.scores.pop(0) if len(self.scores) > 1 else self.scores[0]
        return np.full((len(x), 1), score)


@pytest.fixture
def timeseries():
    return pytest.importorskip("gwpy.timeseries")


@pytest.fixture
def fake_fetch(monkeypatch, timeseries):
    """Replace fetch_strain in real_data with white noise; returns the recorded calls.

    Names in calls.fail raise StrainError for the event window that contains them;
    calls.short removes that many seconds from every download.
    """
    calls = type("Calls", (list,), {"fail": (), "short": 0, "noise_error": None})()

    def fake(detector, start, end, pad_s=0, **kwargs):
        calls.append((detector, start, end, pad_s))
        for ev in real_data.FIELD_TEST_EVENTS:
            if ev.name in calls.fail and start <= ev.gps < end:
                raise StrainError(f"{detector} {start}-{end}: 3 NaN samples")
        if calls.noise_error and start == real_data.NOISE_START:
            raise calls.noise_error
        seconds = end - start + 2 * pad_s - calls.short
        values = np.random.default_rng(5).normal(0.0, 1e-21, seconds * SAMPLE_RATE)
        return timeseries.TimeSeries(values, sample_rate=SAMPLE_RATE, t0=start - pad_s)

    monkeypatch.setattr(real_data, "fetch_strain", fake)
    return calls


def red_noise(rng, seconds, level=3e-19, band=(0.2, 10.0), white=1e-21):
    """White noise plus a low frequency part from 0.2 to 10 Hz with a std of level.

    Raw H1 strain looks like this: its std is a few 1e-19 and almost all of its power
    lies below 10 Hz, while the band-pass passes 20-300 Hz.
    """
    n = int(seconds * SAMPLE_RATE)
    freqs = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    inside = (freqs >= band[0]) & (freqs <= band[1])
    spectrum = inside * (rng.normal(size=len(freqs)) + 1j * rng.normal(size=len(freqs)))
    low = np.fft.irfft(spectrum, n)
    return low * level / np.std(low) + rng.normal(0.0, white, n)


def edge_to_centre_rms(windows):
    """RMS of the first and last 0.1 s against the RMS of the middle 0.1 s, pooled."""
    edges = np.concatenate([np.r_[w[:EDGE], w[-EDGE:]] for w in windows])
    middle = len(windows[0]) // 2
    centres = np.concatenate(
        [w[middle - EDGE // 2 : middle + EDGE // 2] for w in windows]
    )
    return np.sqrt(np.mean(edges**2)) / np.sqrt(np.mean(centres**2))


# --- Padding ---


def test_a_padded_36_s_segment_gives_exactly_one_window(fake_fetch):
    window = real_data.load_window(1000, 1004)
    assert window.shape == (N_SAMPLES,)
    assert fake_fetch == [("H1", 1000, 1004, real_data.PAD_S)]
    assert real_data.PAD_S == 16  # 4 s + 2 * 16 s = 36 s


def test_padding_removes_the_edge_transients(timeseries):
    pad = real_data.PAD_S * SAMPLE_RATE
    padded, unpadded = [], []
    for seed in range(20):
        values = red_noise(np.random.default_rng(seed), 36)
        strain = timeseries.TimeSeries(values, sample_rate=SAMPLE_RATE)
        padded.append(conditioned_crop(strain, SAMPLE_RATE, real_data.PAD_S))
        # Counter-check: only the 4 s window, conditioned on its own as before P4.3
        alone = strain[pad : pad + N_SAMPLES]
        unpadded.append(condition_strain(alone, SAMPLE_RATE))

    assert 0.9 <= edge_to_centre_rms(padded) <= 1.1
    assert edge_to_centre_rms(unpadded) > 2


def test_conditioned_crop_rejects_data_shorter_than_the_pads(timeseries):
    strain = timeseries.TimeSeries(np.zeros(2 * SAMPLE_RATE), sample_rate=SAMPLE_RATE)
    with pytest.raises(ValueError, match="too short"):
        conditioned_crop(strain, SAMPLE_RATE, 1)
    with pytest.raises(ValueError, match="negative"):
        conditioned_crop(strain, SAMPLE_RATE, -1)


def test_too_short_data_are_skipped_not_resized(fake_fetch, capsys):
    fake_fetch.short = 1
    with pytest.raises(StrainError, match="expected 16384"):
        real_data.load_window(1000, 1004)

    result = real_data.run_field_test(ScoreModel(0.9), THRESHOLD)
    assert result["loaded"] == 0
    assert set(result["event_scores"].values()) == {None}
    assert result["far_text"] == "n/a (noise not loaded)"


# --- Events ---


def test_field_test_events_are_the_old_five_in_order():
    assert [e.name for e in real_data.FIELD_TEST_EVENTS] == [
        "GW150914",
        "GW170817",
        "GW190521",
        "GW170104",
        "GW151226",
    ]


def test_events_and_noise_are_fetched_with_the_pad(fake_fetch):
    model = ScoreModel(0.9)
    real_data.run_field_test(model, THRESHOLD)

    pad = real_data.PAD_S
    windows = [
        (int(e.gps - 2.0), int(e.gps + 2.0)) for e in real_data.FIELD_TEST_EVENTS
    ]
    noise = (real_data.NOISE_START, real_data.NOISE_START + real_data.NOISE_DURATION)
    assert fake_fetch == [("H1", a, b, pad) for a, b in [*windows, noise]]
    noise_windows = real_data.NOISE_DURATION * SAMPLE_RATE // N_SAMPLES
    assert model.shapes == [(1, N_SAMPLES, 1)] * 5 + [(noise_windows, N_SAMPLES, 1)]


def test_failed_downloads_are_not_counted_as_missed(fake_fetch, capsys):
    fake_fetch.fail = ("GW150914", "GW190521")
    result = real_data.run_field_test(ScoreModel(0.0), THRESHOLD)
    out = capsys.readouterr().out

    assert result["loaded"] == 3
    assert result["total"] == 5
    assert result["event_scores"]["GW150914"] is None
    assert result["event_scores"]["GW190521"] is None
    # Both failures are in the training range: 2 of its 4 events loaded.
    assert result["in_range"] == EventCount(found=0, loaded=2, total=4)
    assert result["events_text"] == "n/a (2 of 4 loaded); OOD 0/1"
    assert "0/5" not in result["events_text"]
    assert "0/5" not in out
    assert "3 NaN samples" in out


def test_out_of_distribution_events_are_counted_separately(fake_fetch):
    # GW170817 (second) is the only OOD event of the field test.
    result = real_data.run_field_test(ScoreModel(0.9, 0.0, 0.9, 0.9, 0.9, 0.0), 0.5)
    assert result["in_range"] == EventCount(found=4, loaded=4, total=4)
    assert result["ood"] == EventCount(found=0, loaded=1, total=1)
    assert result["events_text"] == "4/4; OOD 0/1"


def test_a_score_equal_to_the_threshold_is_a_detection(fake_fetch):
    """The rule of gwd.evaluation.metrics.predict_labels: score >= threshold."""
    result = real_data.run_field_test(ScoreModel(0.25), 0.25)
    assert result["events_text"] == "4/4; OOD 1/1"
    assert result["false_alarms"] == 50


# --- False alarms ---


def test_false_alarms_are_counted_in_hours_without_a_rate_per_year(fake_fetch):
    result = real_data.run_field_test(ScoreModel(0.0), THRESHOLD)
    assert result["false_alarms"] == 0
    assert result["seconds"] == real_data.NOISE_DURATION == 200
    assert result["far_text"] == "0 false alarms in 0.0556 h"
    assert "/y" not in result["far_text"]

    result = real_data.run_field_test(ScoreModel(1.0), THRESHOLD)
    assert result["far_text"] == "50 false alarms in 0.0556 h"


def test_a_failed_noise_download_is_reported(fake_fetch, capsys):
    fake_fetch.noise_error = ConnectionError("GWOSC unreachable")
    result = real_data.run_field_test(ScoreModel(0.9), THRESHOLD)
    assert result["false_alarms"] is None
    assert result["far_text"] == "n/a (noise not loaded)"
    assert "GWOSC unreachable" in capsys.readouterr().out


def test_event_count_text():
    assert EventCount(3, 4, 4).text == "3/4"
    assert EventCount(0, 0, 4).text == "n/a (0 of 4 loaded)"
    assert EventCount(2, 3, 4).text == "n/a (3 of 4 loaded)"
