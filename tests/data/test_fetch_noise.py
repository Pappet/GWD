"""The noise download goes through gwd.data.gwosc.fetch_strain (P3.4); no network.

Since P4.1 a failing download ends gwd data fetch-noise with rc 1 (DATA-6).
"""

import numpy as np
import pytest

from gwd.config import N_SAMPLES, NOISE_FETCH_SECONDS, SAMPLE_RATE
from gwd.data import fetch_noise
from gwd.data.gwosc import StrainError


@pytest.fixture
def fake_fetch(monkeypatch, tmp_path):
    """Replace fetch_strain with 40 s of noise and write the chunks to tmp_path."""
    timeseries = pytest.importorskip("gwpy.timeseries")
    calls = type("Calls", (list,), {"error": None})()

    def fake(detector, start, end, **kwargs):
        calls.append((detector, start, end, kwargs))
        if calls.error:
            raise calls.error
        values = np.random.default_rng(3).normal(0.0, 1e-21, 40 * SAMPLE_RATE)
        return timeseries.TimeSeries(values, sample_rate=SAMPLE_RATE, t0=start)

    monkeypatch.setattr(fetch_noise, "fetch_strain", fake)
    monkeypatch.setattr(fetch_noise, "OUTPUT_DIR", str(tmp_path / "noise"))
    return calls


def test_downloads_the_segment_and_writes_chunks(fake_fetch, tmp_path):
    assert fetch_noise.fetch_and_slice() == 0
    start = fetch_noise.START_GPS
    assert fake_fetch == [("H1", start, start + NOISE_FETCH_SECONDS, {"verbose": True})]
    files = sorted((tmp_path / "noise").iterdir())
    assert [f.name for f in files] == [f"real_noise_{i:04d}.npy" for i in range(10)]
    assert all(np.load(f).shape == (N_SAMPLES,) for f in files)


def test_a_rejected_download_writes_nothing(fake_fetch, tmp_path, capsys):
    fake_fetch.error = StrainError("H1: 3 NaN samples")
    assert fetch_noise.fetch_and_slice() == 1
    assert "3 NaN samples" in capsys.readouterr().err
    assert not (tmp_path / "noise").exists()


def test_detector_start_and_duration_are_parameters(fake_fetch, tmp_path):
    assert fetch_noise.fetch_and_slice("L1", 1000, 40) == 0
    assert fake_fetch == [("L1", 1000, 1040, {"verbose": True})]


def test_a_failing_download_ends_the_command_with_a_nonzero_exit(
    monkeypatch, tmp_path, capsys
):
    """The error comes from gwpy itself, below fetch_strain, as on a network failure."""
    timeseries = pytest.importorskip("gwpy.timeseries")
    from gwd import cli

    def unreachable(*args, **kwargs):
        raise ConnectionError("GWOSC unreachable")

    monkeypatch.setattr(timeseries.TimeSeries, "fetch_open_data", unreachable)
    monkeypatch.setattr(fetch_noise, "OUTPUT_DIR", str(tmp_path / "noise"))

    assert cli.main(["data", "fetch-noise"]) != 0
    assert "GWOSC unreachable" in capsys.readouterr().err
    assert not (tmp_path / "noise").exists()
