"""Tests for gwd.data.gwosc (P3.4). fetch_open_data is mocked; GWOSC only under network."""

import os
import subprocess
import sys

import numpy as np
import pytest

from gwd.config import DURATION, N_SAMPLES, SAMPLE_RATE
from gwd.data.gwosc import EVENTS, StrainError, event, fetch_strain

GPS = 1126259462

# The two dicts that EVENTS replaces, copied unchanged from before P3.4.
OLD_REAL_EVENTS = {
    "GW150914": 1126259462.4,
    "GW170817": 1187008882.4,
    "GW190521": 1242442967.4,
    "GW170104": 1167559936.6,
    "GW151226": 1135136350.6,
}
OLD_KNOWN_EVENTS = {
    "GW150914": 1126259462.4,
    "GW151226": 1135136350.6,
    "GW170104": 1167559936.6,
    "GW170608": 1180922494.5,
    "GW170814": 1186741861.5,
    "GW170817": 1187008882.4,
    "GW190412": 1239082262.1,
    "GW190521": 1242442967.4,
    "GW190814": 1249852257.0,
}
# Mass range of gwd.core.waveforms.generate_random_bbh_parameters.
TRAINING_MASSES = (5.0, 100.0)


@pytest.fixture
def fake_gwosc(monkeypatch):
    """Replace TimeSeries.fetch_open_data; returns the list of recorded calls.

    The fake returns noise for the requested span at SAMPLE_RATE. Set calls.drop to
    remove samples from the end and calls.nan to put NaN into the data.
    """
    timeseries = pytest.importorskip("gwpy.timeseries")

    class Calls(list):
        drop = 0
        nan = 0

    calls = Calls()

    def fake(ifo, start, end, **kwargs):
        calls.append((ifo, start, end, kwargs))
        n = round((end - start) * SAMPLE_RATE) - calls.drop
        values = np.random.default_rng(1).normal(0.0, 1e-21, n)
        values[: calls.nan] = np.nan
        return timeseries.TimeSeries(values, sample_rate=SAMPLE_RATE, t0=start)

    monkeypatch.setattr(timeseries.TimeSeries, "fetch_open_data", fake)
    return calls


def test_pad_zero_requests_exactly_the_window(fake_gwosc):
    strain = fetch_strain("H1", GPS - 2, GPS + 2)
    assert fake_gwosc == [("H1", GPS - 2, GPS + 2, {"cache": True, "verbose": False})]
    assert len(strain) == N_SAMPLES
    assert strain.t0.value == GPS - 2


def test_pad_widens_the_window_on_both_sides(fake_gwosc):
    strain = fetch_strain("L1", GPS - 2, GPS + 2, pad_s=16, cache=False, verbose=True)
    assert fake_gwosc == [("L1", GPS - 18, GPS + 18, {"cache": False, "verbose": True})]
    assert len(strain) == round((DURATION + 32) * SAMPLE_RATE)


def test_short_data_raises_instead_of_resizing(fake_gwosc):
    fake_gwosc.drop = 1
    with pytest.raises(StrainError, match="got 16383 samples, expected 16384"):
        fetch_strain("H1", GPS - 2, GPS + 2)


def test_nan_raises(fake_gwosc):
    fake_gwosc.nan = 3
    with pytest.raises(StrainError, match="3 NaN samples"):
        fetch_strain("H1", GPS - 2, GPS + 2)


def test_negative_pad_raises(fake_gwosc):
    with pytest.raises(ValueError, match="pad_s"):
        fetch_strain("H1", GPS - 2, GPS + 2, pad_s=-1)
    assert fake_gwosc == []


def test_events_replace_both_old_lists():
    gps = {e.name: e.gps for e in EVENTS}
    assert len(gps) == len(EVENTS)
    assert gps == OLD_KNOWN_EVENTS
    # Same order as before, so random.choice in the Glitch Hunter picks the same event.
    assert list(gps) == list(OLD_KNOWN_EVENTS)
    assert {name: event(name).gps for name in OLD_REAL_EVENTS} == OLD_REAL_EVENTS


def test_out_of_distribution_events():
    assert event("GW170817").in_training_range is False
    assert event("GW190814").in_training_range is False
    assert {e.name for e in EVENTS if not e.in_training_range} == {
        "GW170817",
        "GW190814",
    }


@pytest.mark.parametrize("ev", EVENTS, ids=lambda e: e.name)
def test_event_fields_are_consistent(ev):
    low, high = TRAINING_MASSES
    assert ev.kind in {"BBH", "BNS", "NSBH"}
    assert ev.mass1 >= ev.mass2 > 0
    assert ev.in_training_range == (low <= ev.mass2 and ev.mass1 <= high)


def test_unknown_event_raises():
    with pytest.raises(KeyError, match="GW000000"):
        event("GW000000")


@pytest.mark.parametrize("module", ["gwd.data.gwosc", "gwd.data.fetch_noise"])
def test_import_does_not_load_gwpy(module):
    code = f"import sys, {module}; print('gwpy' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert result.stdout.splitlines()[-1] == "False"


@pytest.mark.network
def test_fetch_strain_from_gwosc():
    strain = fetch_strain("H1", GPS - 2, GPS + 2, cache=False)
    assert len(strain) == N_SAMPLES
    assert strain.sample_rate.value == SAMPLE_RATE
    assert np.isfinite(strain.value).all()
