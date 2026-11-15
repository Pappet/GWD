"""The waveform explorer and the interferometer designer without a window (P4.5).

build_figure() returns (fig, state), so the tests move the sliders under Agg and read the
plots and the model back.
"""

import os
import subprocess
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from gwd.apps import interferometer_designer, waveform_explorer
from gwd.apps._backend import EXIT_NO_BACKEND, HEADLESS_OK

pytestmark = pytest.mark.apps


@pytest.fixture(autouse=True)
def agg():
    """Draw with Agg even where a display exists, and close every figure afterwards."""
    previous = matplotlib.get_backend()
    plt.switch_backend("Agg")
    yield
    plt.close("all")
    if previous.lower() != "agg":
        plt.switch_backend(previous)


def plotted_range(lines):
    data = np.concatenate([line.get_ydata() for line in lines])
    return data.min(), data.max()


def test_the_noise_plot_rescales_after_the_sliders_move():
    """APP-11: the slider axes no longer hide the noise plot from update()."""
    np.random.seed(5)
    fig, state = waveform_explorer.build_figure()
    start_ylim = state.ax_noise.get_ylim()
    state.s_dist.set_val(1000)
    state.s_noise.set_val(0.1)

    low, high = plotted_range([state.line_noise, state.line_clean_ref])
    bottom, top = state.ax_noise.get_ylim()
    assert state.ax_noise.get_ylim() != start_ylim
    assert bottom == pytest.approx(low, rel=0.2)
    assert top == pytest.approx(high, rel=0.2)


def test_the_finesse_follows_the_mirror_slider():
    fig, state = interferometer_designer.build_figure()
    for r in (0.999, 0.9999, 0.995):
        state.slider_mirror.set_val(r)
        assert state.sim.mirror_reflectivity == pytest.approx(r)
        assert state.sim.finesse == pytest.approx(np.pi * np.sqrt(r) / (1 - r))


def test_the_curve_does_not_jump_when_the_mirror_slider_keeps_its_value():
    """APP-12: before P4.5 the first touch set the finesse from 450 to 3140."""
    fig, state = interferometer_designer.build_figure()
    finesse = state.sim.finesse
    # plot_sensitivity() clears the axes, so lines[0] is always the current curve.
    before = state.ax_sensitivity.lines[0].get_ydata().copy()

    state.slider_mirror.set_val(state.slider_mirror.val)

    assert state.sim.finesse == finesse
    np.testing.assert_array_equal(state.ax_sensitivity.lines[0].get_ydata(), before)


def draw_without_warnings(fig):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.canvas.draw()
    return [str(w.message) for w in caught]


def test_the_waveform_explorer_draws_every_glyph():
    fig, state = waveform_explorer.build_figure()
    assert draw_without_warnings(fig) == []


def test_the_interferometer_designer_draws_every_glyph():
    """APP-15: with all features on, every label of the schema and info box is drawn."""
    fig, state = interferometer_designer.build_figure()
    for index in range(3):
        state.checks.set_active(index)
    assert state.sim.power_recycling
    assert state.sim.signal_recycling
    assert state.sim.squeezed_light
    assert draw_without_warnings(fig) == []


@pytest.mark.parametrize("app", ["waveform", "interferometer", "glitch-hunter"])
def test_an_app_without_a_window_exits_with_2_and_a_hint(app, tmp_path):
    """apps:APPX-2: before P4.5 the apps ended with rc 0 and showed nothing."""
    env = {**os.environ, "GWD_HOME": str(tmp_path), "MPLBACKEND": "Agg"}
    env.pop(HEADLESS_OK, None)
    result = subprocess.run(
        [sys.executable, "-m", "gwd.cli", "app", app],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == EXIT_NO_BACKEND, result.stderr
    assert "cannot open a window" in result.stderr
    assert "python3-tk" in result.stderr
    assert "WSLg" in result.stderr
    assert "Traceback" not in result.stderr
