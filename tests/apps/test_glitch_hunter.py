"""The Glitch Hunter's GUI under Agg, with a dummy model and a slow GWOSC (P4.4).

fetch_open_data is mocked with a delay of 0.5 s, so a download is still running when the
test clicks again. The timer of the figure does not fire under Agg; the tests call
poll_results() where the GUI thread would.
"""

import os
import subprocess
import sys
import threading
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from gwd import registry
from gwd.apps.glitch_hunter import app as app_module
from gwd.apps.glitch_hunter import sources
from gwd.apps.glitch_hunter.audio import Player
from gwd.config import SAMPLE_RATE
from gwd.data.gwosc import EVENTS, event

pytestmark = pytest.mark.apps

DELAY_S = 0.5


@pytest.fixture(autouse=True)
def agg():
    previous = matplotlib.get_backend()
    plt.switch_backend("Agg")
    yield
    plt.close("all")
    if previous.lower() != "agg":
        plt.switch_backend(previous)


class DummyModel:
    def __init__(self, score=0.9):
        self.score = score
        self.inputs = []

    def predict(self, x, verbose=0):
        self.inputs.append(x)
        return np.array([[self.score]])


class Picks:
    """random.Random stand-in: always a known event (or noise if signal is False)."""

    def __init__(self, signal=True, event=EVENTS[0]):
        self.signal = signal
        self.event = event

    def choice(self, options):
        return self.signal if options == [True, False] else self.event

    def uniform(self, low, high):
        return low + 5000.5


class FakePopen:
    def __init__(self):
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        return self

    def poll(self):
        return 0


@pytest.fixture
def gwosc(monkeypatch):
    """fetch_open_data with a delay; records every call and the calling thread."""
    timeseries = pytest.importorskip("gwpy.timeseries")
    state = {"calls": [], "error": None}

    def fetch_open_data(cls, detector, start, end, **kwargs):
        state["calls"].append((detector, start, end))
        time.sleep(DELAY_S)
        if state["error"]:
            raise state["error"]
        values = np.random.default_rng(9).normal(
            0.0, 1e-21, int(end - start) * SAMPLE_RATE
        )
        return timeseries.TimeSeries(values, sample_rate=SAMPLE_RATE, t0=start)

    monkeypatch.setattr(
        timeseries.TimeSeries, "fetch_open_data", classmethod(fetch_open_data)
    )
    return state


def make_app(model="dummy", sim=True, picks=None, tmp_path=None, threshold=None):
    if model == "dummy":
        model = DummyModel()
    sim_source = (
        sources.SimSource(noise_provider=None, rng=np.random.RandomState(1))
        if sim
        else None
    )
    real = sources.GwoscSource(rng=picks or Picks())
    popen = FakePopen()
    player = Player(system="Linux", popen=popen, which=lambda name: True)
    extra = {} if threshold is None else {"threshold": threshold}
    app = app_module.GlitchHunterApp(
        model=model, metadata=None, sim=sim_source, real=real, player=player, **extra
    )
    app.popen = popen
    return app


def finish_download(app):
    app.thread.join(timeout=10)
    assert not app.thread.is_alive()
    app.poll_results()


def test_ask_ai_right_after_ligo_random_rates_no_mixed_sample(gwosc):
    """APP-5: the old thread set has_signal before the data came, so 'Ask AI' showed
    'TRUTH: SIGNAL (SNR: 0.00)' for the simulated sample still on screen."""
    app = make_app()
    while app.sample.has_signal:  # a simulated noise sample, as in the bug
        app.generate_new_sample(None)
    old = app.sample

    app.start_background_loading(None)
    app.ask_ai(None)
    assert app.sample is old
    assert app.prediction is None and app.truth_artist is None
    assert app.model.inputs == []
    assert not app.btn_ask.active and not app.btn_real.active

    finish_download(app)
    assert app.sample.source == "gwosc" and app.sample.name == EVENTS[0].name
    assert app.btn_ask.active and app.btn_real.active
    app.ask_ai(None)
    text = app.truth_artist.get_text()
    assert text == f"TRUTH: {EVENTS[0].name}"
    assert "SNR" not in text


def test_a_double_click_starts_no_second_download(gwosc):
    app = make_app()
    app.start_background_loading(None)
    first = app.thread
    app.start_background_loading(None)
    assert app.thread is first
    finish_download(app)
    assert len(gwosc["calls"]) == 1


def test_a_download_finished_after_a_newer_sample_is_dropped(gwosc):
    app = make_app()
    app.start_background_loading(None)
    app.generate_new_sample(None)  # 'Simulation' during the download
    newer = app.sample
    finish_download(app)
    assert app.sample is newer
    assert app.results.empty()
    assert not app.loading and app.btn_real.active


def test_the_worker_thread_never_calls_matplotlib(gwosc, monkeypatch):
    """APP-6: all drawing happens in the thread that runs poll_results()."""
    app = make_app()
    threads = []
    render = app._render_current_sample
    draw_idle = app.fig.canvas.draw_idle

    def record(function):
        def wrapper(*args, **kwargs):
            threads.append(threading.current_thread())
            return function(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(app, "_render_current_sample", record(render))
    monkeypatch.setattr(app.fig.canvas, "draw_idle", record(draw_idle))
    app.start_background_loading(None)
    app.thread.join(timeout=10)
    drawn_before_poll = len(threads)
    app.poll_results()
    assert len(threads) > drawn_before_poll
    assert set(threads) == {threading.main_thread()}


def test_a_failed_download_falls_back_to_the_simulation(gwosc):
    gwosc["error"] = ConnectionError("no route to host")
    app = make_app()
    app.start_background_loading(None)
    finish_download(app)
    assert app.sample.source == "simulation"
    assert "GWOSC download failed: no route to host" in app.sample.note
    assert app.ax_signal.get_title().startswith("Simulated data: Simulation (Training)")
    assert "simulated" in app.ax_spec.get_title()


def test_without_a_simulation_a_failed_download_shows_why(gwosc):
    gwosc["error"] = ConnectionError("offline")
    app = make_app(sim=False)
    assert app.sample is None and not app.btn_sim.active
    app.start_background_loading(None)
    finish_download(app)
    assert app.sample is None
    assert "offline" in app.ax_signal.texts[0].get_text()
    assert app.btn_real.active


def test_the_titles_follow_the_source(gwosc):
    app = make_app()
    assert "simulated, white noise" in app.ax_spec.get_title()
    app.start_background_loading(None)
    finish_download(app)
    assert app.ax_signal.get_title().startswith(f"REAL DATA: {EVENTS[0].name} (H1)")
    assert app.ax_spec.get_title() == "Spectrogram (GWOSC H1, notched and band-passed)"


def test_without_a_model_there_is_no_system_exit():
    """APP-8: the app starts, and 'Ask AI' is off instead of sys.exit(1)."""
    app = make_app(model=None)
    assert app.sample is not None
    assert not app.btn_ask.active
    assert app.btn_ask.label.get_text() == "Ask AI (no model)"
    app.ask_ai(None)
    assert app.prediction is None


def test_the_app_starts_without_any_model_in_a_subprocess(tmp_path):
    env = {
        **os.environ,
        "GWD_HOME": str(tmp_path),
        "MPLBACKEND": "Agg",
        "GWD_HEADLESS_OK": "1",
    }
    code = (
        "import sys\nfrom gwd.cli import main\nrc = main(['app', 'glitch-hunter'])\n"
        "print(rc, 'tensorflow' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines()[-1] == "0 False", result.stderr
    assert "'Ask AI' is off" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "mode, low, high",
    [(sources.TRAINING, 1.0, 12.0), (sources.CHALLENGE, 0.5, 2.0)],
    ids=["Training", "Challenge"],
)
def test_each_mode_shows_its_peak_snr(mode, low, high):
    app = make_app()
    assert app.sim.mode is sources.TRAINING  # the start mode
    assert app.radio_mode.value_selected == app_module.mode_label(sources.TRAINING)
    app.radio_mode.set_active(sources.MODES.index(mode))
    assert app.sim.mode is mode
    snrs = []
    for _ in range(20):
        app.generate_new_sample(None)
        if app.sample.has_signal:
            snrs.append(app.sample.snr)
    assert snrs and all(low <= snr <= high for snr in snrs)
    assert mode.name in app.ax_signal.get_title()


def test_the_scenario_switch_draws_v2_and_keeps_the_mode():
    """P6.2: V1 at the start; V2 draws the same mode from its V2 scenario."""
    pytest.importorskip("gwpy")
    app = make_app()
    assert app.sim.mode is sources.TRAINING
    assert app.radio_version.value_selected == "V1"
    assert app.radio_version.active
    app.radio_mode.set_active(sources.MODES.index(sources.CHALLENGE))
    app.radio_version.set_active(1)
    assert app.sim.mode is sources.CHALLENGE_V2
    assert app.radio_mode.value_selected == app_module.mode_label(sources.CHALLENGE_V2)
    assert app.sample.name == "Simulation (V2 Challenge)"
    assert "V2 Challenge" in app.ax_signal.get_title()
    app.radio_version.set_active(0)
    assert app.sim.mode is sources.CHALLENGE
    assert app.radio_mode.value_selected == app_module.mode_label(sources.CHALLENGE)
    assert app.sample.name == "Simulation (Challenge)"


def test_without_gwpy_the_scenario_switch_is_off():
    sim = sources.SimSource(noise_provider=None, rng=np.random.RandomState(1))
    app = app_module.GlitchHunterApp(
        sim=sim, player=Player(system="Linux", popen=FakePopen()), v2_available=False
    )
    assert not app.radio_version.active
    app.set_version("V2")
    assert app.sim.mode is sources.TRAINING


def test_the_slider_starts_at_the_threshold_and_rejudges_the_truth():
    """APP-4: the slider showed 0.50 while 0.46 decided; the box kept its color."""
    app = make_app(model=DummyModel(score=0.6))
    assert app.slider_thresh.val == app_module.THRESHOLD_FALLBACK == app.threshold
    assert app.slider_thresh.valmin <= app.threshold <= app.slider_thresh.valmax
    while not app.sample.has_signal:
        app.generate_new_sample(None)
    app.ask_ai(None)
    green = matplotlib.colors.to_rgba("lightgreen", 0.8)
    red = matplotlib.colors.to_rgba("lightcoral", 0.8)
    assert app.truth_artist.get_bbox_patch().get_facecolor() == green
    app.slider_thresh.set_val(0.65)
    assert app.threshold == 0.65
    assert app.truth_artist.get_bbox_patch().get_facecolor() == red
    texts = [t.get_text() for t in app.ax_confidence.texts]
    assert "UNSURE: just below the threshold" in texts


def test_real_data_answers_are_judged_and_ood_is_marked(gwosc):
    """apps:APPX-3: the AI's answer on real data was never judged."""
    app = make_app(model=DummyModel(score=0.1), picks=Picks(event=event("GW170817")))
    app.start_background_loading(None)
    finish_download(app)
    app.ask_ai(None)
    assert app.truth_artist.get_text() == (
        "TRUTH: GW170817 (OOD: outside the training masses)"
    )
    red = matplotlib.colors.to_rgba("lightcoral", 0.8)
    assert app.truth_artist.get_bbox_patch().get_facecolor() == red
    assert "OOD" in app.ax_signal.get_title()


def test_play_sound_creates_nothing_in_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = make_app()
    app.play_sound(None)
    assert os.listdir(tmp_path) == []
    assert app.popen.commands == [["paplay", app.player.path]]
    assert os.path.isfile(app.player.path)
    app.close()
    assert not os.path.exists(app.player.path)


def test_the_glitch_hunter_uses_the_registry_threshold(monkeypatch):
    """P5.8: main() starts the app at exactly registry.get_threshold of its model."""
    monkeypatch.setattr(registry, "get_threshold", lambda path: 0.123)
    monkeypatch.setattr(
        app_module,
        "load_optional_model",
        lambda model: (DummyModel(score=0.2), "gwd_model_x.keras", None),
    )
    monkeypatch.setattr(app_module, "read_metadata", lambda path: None)
    monkeypatch.setattr(app_module, "require_interactive_backend", lambda name: None)
    monkeypatch.setattr(app_module, "GwoscSource", lambda: None)
    monkeypatch.setattr(app_module, "gwpy_available", lambda: False)
    monkeypatch.setattr(
        app_module,
        "SimSource",
        lambda: sources.SimSource(noise_provider=None, rng=np.random.RandomState(1)),
    )
    started = []
    real_app = app_module.GlitchHunterApp

    def record(**kwargs):
        app = real_app(**kwargs)
        started.append(app)
        return app

    monkeypatch.setattr(app_module, "GlitchHunterApp", record)
    monkeypatch.setattr(app_module.plt, "show", lambda: None)
    assert app_module.main("latest") == 0
    (app,) = started
    assert app.threshold == 0.123
    assert app.slider_thresh.val == 0.123
    app.ask_ai(None)
    texts = [t.get_text() for t in app.ax_confidence.texts]
    assert "12.3%" in texts
    assert app_module.decision(0.123, app.threshold)[1] == "red"


def test_a_v3_model_starts_in_v3_and_gets_whitened_real_data(monkeypatch):
    """V3: the simulation starts in the model's scenario version, real data get its
    preprocessing, and a sample of another version says it does not match."""
    pytest.importorskip("gwpy")
    metadata = {"dataset_scenario": "TRAIN_V3", "preprocessing": "whiten_bandpass"}
    monkeypatch.setattr(registry, "get_threshold", lambda path: 0.5)
    monkeypatch.setattr(
        app_module,
        "load_optional_model",
        lambda model: (DummyModel(score=0.2), "gwd_model_x.keras", None),
    )
    monkeypatch.setattr(app_module, "read_metadata", lambda path: metadata)
    monkeypatch.setattr(app_module, "require_interactive_backend", lambda name: None)
    monkeypatch.setattr(
        app_module, "GwoscSource", lambda: sources.GwoscSource(rng=Picks())
    )
    monkeypatch.setattr(
        app_module,
        "SimSource",
        lambda: sources.SimSource(noise_provider=None, rng=np.random.RandomState(1)),
    )
    started = []
    real_app = app_module.GlitchHunterApp

    def record(**kwargs):
        app = real_app(**kwargs)
        started.append(app)
        return app

    monkeypatch.setattr(app_module, "GlitchHunterApp", record)
    monkeypatch.setattr(app_module.plt, "show", lambda: None)
    assert app_module.main("latest") == 0
    (app,) = started
    assert app.sim.mode is sources.TRAINING_V3
    assert app.radio_version.value_selected == "V3"
    assert app.real.preprocessing == "whiten_bandpass"
    assert "trained with" not in app.ax_signal.get_title()
    app.set_version("V1")
    assert app.sim.mode is sources.TRAINING
    assert app.real.preprocessing == "whiten_bandpass"
    assert "the model was trained with 'whiten_bandpass'" in app.ax_signal.get_title()


def test_without_a_model_real_data_follow_the_scenario_switch():
    pytest.importorskip("gwpy")
    app = make_app(model=None)
    assert app.real.preprocessing == "bandpass"
    app.set_version("V3")
    assert app.real.preprocessing == "whiten_bandpass"
    assert "whitened" in app.ax_spec.get_title()
    app.set_version("V2")
    assert app.real.preprocessing == "bandpass"
