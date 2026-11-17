"""gwd eval lens: what training changed in a chirp telescope (gwd.evaluation.lens)."""

import json
from dataclasses import replace

import numpy as np
import pytest

from gwd.evaluation import lens as report


def test_gauge_free_removes_a_constant_a_shift_and_a_stretch():
    u = np.linspace(-1.0, -0.01, 400)
    phase = -40.0 * (-u) ** 0.625
    slope = np.gradient(phase, u)
    weight = np.ones_like(u)
    hidden = 0.7 + 3e-3 * slope - 0.02 * u * slope
    assert np.abs(report.gauge_free(hidden, phase, u, weight)).max() < 1e-9
    # What they cannot explain stays, and nodes of weight 0 take no part in the fit.
    bump = np.exp(-(((u + 0.5) / 0.05) ** 2))
    left = report.gauge_free(hidden + bump, phase, u, weight)
    assert left.max() - left.min() == pytest.approx(1.0, abs=0.15)
    weight[u > -0.2] = 0.0
    spoiled = hidden + bump + np.where(u > -0.2, 50.0, 0.0)
    again = report.gauge_free(spoiled, phase, u, weight)
    assert again[u <= -0.2] == pytest.approx(
        report.gauge_free((hidden + bump), phase, u, weight)[u <= -0.2], abs=1e-9
    )


def test_rescaled_takes_out_a_common_factor():
    u = np.linspace(-1.0, -0.01, 50)
    start = (-u) ** -0.25
    weight = np.gradient(u)
    assert report.rescaled(3.0 * start, start, weight) == pytest.approx(start)
    # A bump stays where it is, on a curve that lies as close to the start as a factor
    # allows: what is left has no part along the curve.
    bump = np.exp(-(((u + 0.5) / 0.05) ** 2))
    left = report.rescaled(2.0 * (start + bump), start, weight) - start
    assert np.argmax(left) == np.argmax(bump) and 0.5 < left.max() < 1.0
    assert abs((weight * left * (start + bump)).sum()) < 1e-9
    # Nodes of weight 0 take no part in the fit.
    weight[:10] = 0.0
    spoiled = 3.0 * start + np.where(np.arange(50) < 10, 7.0, 0.0)
    assert report.rescaled(spoiled, start, weight)[10:] == pytest.approx(start[10:])


def test_summarize_splits_light_and_heavy_signals():
    factors = np.array([0.5, 0.7, 0.9, 0.8])
    masses = np.array([20.0, 30.0, 100.0, 60.0])
    median, low, light, heavy = report.summarize(factors, masses)
    assert median == pytest.approx(0.75)
    assert low == pytest.approx(np.percentile(factors, 10))
    assert (light, heavy) == (pytest.approx(0.6), pytest.approx(0.9))
    assert np.isnan(report.summarize(factors, np.full(4, 50.0))[2:]).all()


def test_an_unknown_model_exits_with_one(tmp_path, capsys):
    assert report.run(model=str(tmp_path / "missing.keras")) == 1
    assert "neither a file nor a model" in capsys.readouterr().out


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A small telescope after two training steps, saved with metadata."""
    pytest.importorskip("tensorflow")
    pytest.importorskip("gwpy")
    pytest.importorskip("pycbc")
    from gwd import registry
    from gwd.data import scenarios
    from gwd.data.injection import make_sample
    from gwd.data.preprocessing import normalize
    from gwd.training import telescope

    small = replace(scenarios.TRAIN_V3, name="SMALL_V3", masses=(30.0, 40.0))
    model = telescope.build_telescope(
        small, 1e-2, seed=3, start="newton", mass_ratios=(1.0,), noise_samples=32
    )
    samples = [
        make_sample(small, np.random.RandomState(i), has_signal=i % 2 == 0)
        for i in range(8)
    ]
    x = np.array([normalize(s.data) for s in samples], dtype=np.float32)[:, :, None]
    model.fit(x, np.arange(8) % 2 == 0, batch_size=4, epochs=1, verbose=0)
    folder = tmp_path_factory.mktemp("models")
    path = str(folder / "gwd_model_20260101-000000.keras")
    model.save(path)
    registry.write_metadata(
        path,
        {
            "dataset_scenario": "SMALL_V3",
            "telescope": {"lens_start": "newton", "trained": "all"},
        },
    )
    return path, small


def test_the_report_names_the_fitting_factor_and_writes_the_plot(
    trained, tmp_path, monkeypatch, capsys
):
    path, small = trained
    from gwd.data import scenarios

    monkeypatch.setitem(scenarios.SCENARIOS, "SMALL_V3", small)
    monkeypatch.setattr(report, "EVALUATION_MODELS_PLOTS_DIR", str(tmp_path))
    assert report.run(model=path, signals=6, seed=1) == 0
    out = capsys.readouterr().out
    assert "Fitting factor on 6 signals of SMALL_V3 (seed 1)" in out
    assert "start 'newton'; trained: all" in out
    assert "Phase left to PyCBC (rad, rms)" in out
    assert (tmp_path / "lens_gwd_model_20260101-000000.png").stat().st_size > 10_000
    # The start values are back in place after the report measured with them.
    model = report.load_model(path)
    lens, _ = report.telescope_layers(model)
    assert np.abs(lens.phase_delta.numpy()).max() > 0


def test_the_fitting_factor_of_the_start_values_ignores_what_was_trained(trained):
    path, small = trained
    lens, _ = report.telescope_layers(report.load_model(path))
    signals, masses = report.draw_signals(small, 4, seed=2)
    assert signals.shape == (4, 16384) and np.all((masses >= 60) & (masses <= 80))
    added = lens.phase_delta.numpy().copy()
    with report.start_values(lens):
        assert np.all(lens.phase_delta.numpy() == 0)
        before = report.fitting_factors(lens, signals)
    assert np.array_equal(lens.phase_delta.numpy(), added)
    assert np.all((before > 0.3) & (before < 1.05))
    # The same draw again: the signals are a function of the seed.
    assert np.array_equal(report.draw_signals(small, 4, seed=2)[0], signals)


def test_a_model_without_a_lens_is_refused(tmp_path, capsys):
    pytest.importorskip("tensorflow")
    from gwd.training.model import build_model

    path = tmp_path / "gwd_model_20260102-000000.keras"
    build_model((64, 1)).save(path)
    assert report.run(model=str(path)) == 1
    assert "is not a chirp telescope" in capsys.readouterr().out


def test_the_metadata_of_the_fixture_is_what_gwd_train_writes(trained):
    path, _ = trained
    with open(path.replace(".keras", ".json"), encoding="utf-8") as f:
        assert json.load(f)["telescope"]["lens_start"] == "newton"
