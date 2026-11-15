"""The Glitch Hunter downloads through gwd.data.gwosc (P3.4); no window, no network."""

import numpy as np
import pytest

from gwd.apps.glitch_hunter import app as app_module
from gwd.config import DURATION, N_SAMPLES, SAMPLE_RATE
from gwd.data.gwosc import EVENTS, StrainError


@pytest.fixture
def hunter(monkeypatch):
    """A GlitchHunterApp without model and window, with fetch_strain mocked."""
    timeseries = pytest.importorskip("gwpy.timeseries")
    app = app_module.GlitchHunterApp.__new__(app_module.GlitchHunterApp)
    app.fs = SAMPLE_RATE
    app.time = np.linspace(0, DURATION, N_SAMPLES)
    app.plots = []
    app.update_plot_threadsafe = lambda **kwargs: app.plots.append(kwargs)
    app.calls = []
    app.error = None

    def fake(detector, start, end, **kwargs):
        app.calls.append((detector, start, end, kwargs))
        if app.error:
            raise app.error
        values = np.random.default_rng(9).normal(
            0.0, 1e-21, (end - start) * SAMPLE_RATE
        )
        return timeseries.TimeSeries(values, sample_rate=SAMPLE_RATE, t0=start)

    monkeypatch.setattr(app_module, "fetch_strain", fake)
    # Always "signal", and always the first event.
    monkeypatch.setattr(app_module.random, "choice", lambda options: options[0])
    return app


def test_an_event_is_fetched_as_a_four_second_window(hunter):
    hunter.load_random_real_thread()
    t0 = int(EVENTS[0].gps)
    assert hunter.calls == [("H1", t0 - 2, t0 + 2, {})]
    assert hunter.event_name == EVENTS[0].name
    assert hunter.is_real_data is True
    assert hunter.current_data.shape == (N_SAMPLES,)
    assert hunter.plots == [{"title": f"ECHTE DATEN: {EVENTS[0].name} (H1)"}]


def test_a_rejected_download_falls_back_to_simulation(hunter):
    hunter.error = StrainError("H1: 3 NaN samples")
    hunter.load_random_real_thread()
    assert hunter.plots == [{"fallback": True}]
