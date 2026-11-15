"""The field test downloads through gwd.data.gwosc (P3.4); fetch_strain is mocked."""

import numpy as np
import pytest

from gwd.config import N_SAMPLES, SAMPLE_RATE
from gwd.data.gwosc import StrainError
from gwd.evaluation import evaluate


class FixedModel:
    """Stands in for a Keras model: the same score for every input."""

    def __init__(self, score):
        self.score = score
        self.shapes = []

    def predict(self, x, verbose=0):
        self.shapes.append(x.shape)
        return np.full((len(x), 1), self.score)


@pytest.fixture
def fake_fetch(monkeypatch):
    """Replace fetch_strain in evaluate; returns the recorded calls.

    Names in calls.fail raise StrainError for the event window that contains them.
    """
    timeseries = pytest.importorskip("gwpy.timeseries")
    calls = type("Calls", (list,), {"fail": ()})()

    def fake(detector, start, end, **kwargs):
        calls.append((detector, start, end, kwargs))
        for ev in evaluate.FIELD_TEST_EVENTS:
            if ev.name in calls.fail and start <= ev.gps < end:
                raise StrainError(f"{detector} {start}-{end}: 3 NaN samples")
        values = np.random.default_rng(5).normal(
            0.0, 1e-21, (end - start) * SAMPLE_RATE
        )
        return timeseries.TimeSeries(values, sample_rate=SAMPLE_RATE, t0=start)

    monkeypatch.setattr(evaluate, "fetch_strain", fake)
    return calls


def test_field_test_events_are_the_old_five_in_order():
    assert [e.name for e in evaluate.FIELD_TEST_EVENTS] == [
        "GW150914",
        "GW170817",
        "GW190521",
        "GW170104",
        "GW151226",
    ]


def test_events_are_fetched_as_four_second_windows(fake_fetch):
    model = FixedModel(0.9)
    result = evaluate.evaluate_real_world(model)

    windows = [(int(e.gps - 2.0), int(e.gps + 2.0)) for e in evaluate.FIELD_TEST_EVENTS]
    noise = (
        evaluate.REAL_NOISE_START,
        evaluate.REAL_NOISE_START + evaluate.REAL_NOISE_DUR,
    )
    # No pad_s: the default pad_s=0 keeps the 4 s windows of before P3.4.
    assert fake_fetch == [("H1", a, b, {}) for a, b in [*windows, noise]]
    assert result["found"] == result["total"] == 5
    noise_cuts = evaluate.REAL_NOISE_DUR * SAMPLE_RATE // N_SAMPLES
    assert model.shapes == [(1, N_SAMPLES, 1)] * (5 + noise_cuts)


def test_a_rejected_download_scores_none(fake_fetch, capsys):
    fake_fetch.fail = ("GW170817",)
    result = evaluate.evaluate_real_world(FixedModel(0.9))
    assert result["event_scores"]["GW170817"] is None
    assert result["found"] == 4
    assert result["total"] == 5
    assert "3 NaN samples" in capsys.readouterr().out
