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
    fig, state = waveform_explorer.build_figure(rng=np.random.default_rng(5))
    start_ylim = state.ax_noise.get_ylim()
    state.s_dist.set_val(1000)
    state.s_noise.set_val(0.1)

    low, high = plotted_range([state.line_noise, state.line_clean_ref])
    bottom, top = state.ax_noise.get_ylim()
    assert state.ax_noise.get_ylim() != start_ylim
    assert bottom == pytest.approx(low, rel=0.2)
    assert top == pytest.approx(high, rel=0.2)


def test_without_pycbc_the_explorer_uses_newton_and_says_so(monkeypatch):
    """P5.2: the demo app chooses Newton on purpose and names it in the window."""
    monkeypatch.setitem(sys.modules, "pycbc", None)
    fig, state = waveform_explorer.build_figure()
    assert state.backend == "newton"
    assert state.sim.backend == "newton"
    assert "Newton" in state.backend_text.get_text()
    assert "physically inaccurate" in state.backend_text.get_text()
    assert state.backend_text in fig.texts
    # P5.4: the noise comes from the 1/f stand-in, and the noise plot says so.
    assert state.sim.noise_psd is waveform_explorer.STAND_IN_PSD
    assert "stand-in" in state.ax_noise.get_title()
    state.s_mass1.set_val(50)
    assert np.all(np.isfinite(state.line_strain.get_ydata()))
    assert draw_without_warnings(fig) == []


def test_with_pycbc_the_explorer_uses_pycbc():
    pytest.importorskip("pycbc")
    fig, state = waveform_explorer.build_figure()
    assert state.backend == "pycbc"
    assert state.backend_text.get_text().startswith("Waveforms: pycbc ")


def test_with_pycbc_the_explorer_draws_noise_from_the_aligo_psd():
    """P5.4: the noise has the aLIGO design PSD above 10 Hz, times the noise slider."""
    pytest.importorskip("pycbc")
    from gwd.core.interferometer import reference_psd

    fig, state = waveform_explorer.build_figure(rng=np.random.default_rng(2))
    assert "aLIGO design PSD (aLIGOZeroDetHighPower)" in state.ax_noise.get_title()
    freqs = np.array([5.0, 10.0, 100.0])
    np.testing.assert_array_equal(
        state.sim.noise_psd(freqs), reference_psd(freqs, low_freq_cutoff=10.0)
    )
    # The default noise is of the order of the aLIGO strain noise, below the chirp.
    noise = state.line_noise.get_ydata() - state.line_clean_ref.get_ydata()
    chirp_peak = np.max(np.abs(state.line_clean_ref.get_ydata()))
    assert 1e-23 < np.std(noise) < chirp_peak


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
