"""The noise download goes through gwd.data.gwosc.fetch_strain (P3.4); no network.

Since P4.1 a failing download ends gwd data fetch-noise with rc 1 (DATA-6). Since P4.3
the first and the last chunk, which hold the filter edge transients, are not written
(evaluation:MISS-4).
"""

import numpy as np
import pytest

from gwd.config import N_SAMPLES, NOISE_FETCH_SECONDS, SAMPLE_RATE
from gwd.data import fetch_noise
from gwd.data.gwosc import StrainError


@pytest.fixture
def fake_fetch(monkeypatch, tmp_path):
    """Replace fetch_strain with 40 s of noise (calls.seconds) and write to tmp_path."""
    timeseries = pytest.importorskip("gwpy.timeseries")
    calls = type("Calls", (list,), {"error": None, "seconds": 40})()

    def fake(detector, start, end, **kwargs):
        calls.append((detector, start, end, kwargs))
        if calls.error:
            raise calls.error
        values = np.random.default_rng(3).normal(
            0.0, 1e-21, calls.seconds * SAMPLE_RATE
        )
        return timeseries.TimeSeries(values, sample_rate=SAMPLE_RATE, t0=start)

    monkeypatch.setattr(fetch_noise, "fetch_strain", fake)
    monkeypatch.setattr(fetch_noise, "OUTPUT_DIR", str(tmp_path / "noise"))
    return calls


def test_downloads_the_segment_and_writes_chunks_without_the_edges(
    fake_fetch, tmp_path
):
    """40 s are 10 chunks; the first and the last are dropped, 8 are written."""
    assert fetch_noise.fetch_and_slice() == 0
    start = fetch_noise.START_GPS
    assert fake_fetch == [("H1", start, start + NOISE_FETCH_SECONDS, {"verbose": True})]
    files = sorted((tmp_path / "noise").iterdir())
    # Each file keeps the position of its chunk in the segment.
    assert [f.name for f in files] == [f"real_noise_{i:04d}.npy" for i in range(1, 9)]
    assert all(np.load(f).shape == (N_SAMPLES,) for f in files)


def test_the_written_chunks_are_the_middle_of_the_conditioned_segment(
    fake_fetch, tmp_path
):
    timeseries = pytest.importorskip("gwpy.timeseries")
    from gwd.data.preprocessing import condition_strain

    assert fetch_noise.fetch_and_slice() == 0
    values = np.random.default_rng(3).normal(0.0, 1e-21, 40 * SAMPLE_RATE)
    whole = condition_strain(
        timeseries.TimeSeries(values, sample_rate=SAMPLE_RATE), SAMPLE_RATE
    )
    chunk_1 = np.load(tmp_path / "noise" / "real_noise_0001.npy")
    np.testing.assert_array_equal(chunk_1, whole[N_SAMPLES : 2 * N_SAMPLES])


def test_older_files_are_kept_and_named(fake_fetch, tmp_path, capsys):
    folder = tmp_path / "noise"
    folder.mkdir()
    for i in (0, 9, 1023):
        np.save(folder / f"real_noise_{i:04d}.npy", np.zeros(3))
    assert fetch_noise.fetch_and_slice() == 0

    names = sorted(f.name for f in folder.iterdir())
    assert names == [f"real_noise_{i:04d}.npy" for i in (*range(10), 1023)]
    assert np.load(folder / "real_noise_0000.npy").shape == (3,)
    err = capsys.readouterr().err
    assert "3 older files" in err
    assert "real_noise_0000.npy, real_noise_0009.npy, real_noise_1023.npy" in err


def test_a_segment_of_two_chunks_writes_nothing(fake_fetch, tmp_path, capsys):
    fake_fetch.seconds = 8
    assert fetch_noise.fetch_and_slice("H1", 1000, 8) == 1
    assert "leave none" in capsys.readouterr().err
    assert not (tmp_path / "noise").exists()


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
