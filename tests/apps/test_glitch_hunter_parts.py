"""The Glitch Hunter's sources, audio, and rating rules, without a figure (P4.4).

No window, no network, no TensorFlow, so the fast CI job runs these. The GUI itself is
tested in test_glitch_hunter.py under the apps marker.
"""

import os

import numpy as np
import pytest

from gwd.apps.glitch_hunter import app as app_module
from gwd.apps.glitch_hunter import audio, sources
from gwd.config import N_SAMPLES, SAMPLE_RATE
from gwd.data.gwosc import EVENTS, StrainError, event
from gwd.data.scenarios import APP_V1, TRAIN_V1


class Picks:
    """A stand-in for random.Random that picks what a test asks for."""

    def __init__(self, signal=True, event=EVENTS[0], times=()):
        self.signal = signal
        self.event = event
        self.times = list(times)

    def choice(self, options):
        return self.signal if options == [True, False] else self.event

    def uniform(self, low, high):
        return self.times.pop(0) if self.times else (low + high) / 2


def fake_fetch(calls, error=None, short=0):
    """A fetch_strain that returns white noise of the right length as a TimeSeries."""
    timeseries = pytest.importorskip("gwpy.timeseries")

    def fetch(detector, start, end, pad_s=0, **kwargs):
        calls.append((detector, start, end, pad_s))
        if error:
            raise error
        n = (end - start + 2 * pad_s) * SAMPLE_RATE - short
        values = np.random.default_rng(3).normal(0.0, 1e-21, n)
        return timeseries.TimeSeries(values, sample_rate=SAMPLE_RATE, t0=start - pad_s)

    return fetch


# --- modes and the simulation ---


def test_training_is_the_start_mode_and_challenge_is_marked_as_hard():
    """D4: 'Training' draws from TRAIN_V1, 'Challenge' from APP_V1."""
    assert sources.MODES == (sources.TRAINING, sources.CHALLENGE)
    assert sources.TRAINING.scenario is TRAIN_V1
    assert sources.CHALLENGE.scenario is APP_V1
    assert sources.SimSource(noise_provider=None).mode is sources.TRAINING
    assert "deliberately hard" in sources.CHALLENGE.note
    assert "extrapolation" in app_module.mode_label(sources.CHALLENGE)


@pytest.mark.parametrize(
    "mode, low, high",
    [(sources.TRAINING, 1.0, 12.0), (sources.CHALLENGE, 0.5, 2.0)],
    ids=["Training", "Challenge"],
)
def test_each_mode_draws_the_peak_snr_of_its_scenario(mode, low, high):
    source = sources.SimSource(mode, noise_provider=None, rng=np.random.RandomState(5))
    seen = set()
    for _ in range(16):
        sample = source.draw()
        seen.add(sample.has_signal)
        assert sample.data.shape == (N_SAMPLES,)
        assert sample.source == "simulation" and sample.noise_kind == "white"
        assert sample.name == f"Simulation ({mode.name})"
        if sample.has_signal:
            assert low <= sample.snr <= high
        else:
            assert sample.snr == 0.0 and sample.signal is None
    assert seen == {True, False}


def test_training_draws_real_noise_chunks_when_there_are_any(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(3):
        np.save(tmp_path / f"real_noise_{i:04d}.npy", rng.normal(0, 1e-21, N_SAMPLES))
    provider = sources.real_noise_provider(str(tmp_path))
    source = sources.SimSource(noise_provider=provider, rng=np.random.RandomState(2))
    kinds = {source.draw().noise_kind for _ in range(20)}
    assert kinds == {"real", "white"}  # TRAIN_V1: 80 % real


def test_a_fallback_note_replaces_the_mode_note():
    source = sources.SimSource(noise_provider=None, rng=np.random.RandomState(1))
    assert source.draw().note == sources.TRAINING.note
    assert source.draw(note="download failed").note == "download failed"


# --- GWOSC ---


def test_an_event_is_downloaded_padded_and_cropped():
    """P4.3: the window is conditioned with 16 s on each side, then cropped."""
    calls = []
    source = sources.GwoscSource(rng=Picks(event=EVENTS[0]), fetch=fake_fetch(calls))
    sample = source.draw()
    t0 = EVENTS[0].gps
    assert calls == [("H1", int(t0 - 2), int(t0 + 2), sources.PAD_S)]
    assert sources.PAD_S == 16
    assert sample.data.shape == (N_SAMPLES,)
    assert np.isclose(np.std(sample.data), 1.0)
    assert sample.has_signal and sample.event is EVENTS[0]
    assert sample.source == "gwosc" and sample.name == "GW150914"
    assert not sample.out_of_distribution


def test_an_event_outside_the_training_masses_is_marked():
    """apps:APPX-3: GW170817 is out of distribution, in the note and the truth."""
    bns = event("GW170817")
    source = sources.GwoscSource(rng=Picks(event=bns), fetch=fake_fetch([]))
    sample = source.draw()
    assert sample.out_of_distribution
    assert "OOD" in sample.note
    assert app_module.truth_text(sample) == (
        "TRUTH: GW170817 (OOD: outside the training masses)"
    )


def test_a_noise_window_keeps_clear_of_every_event():
    """APP-19: a time near an event is drawn again; the pads stay clear too."""
    near = EVENTS[3].gps + 10
    calls = []
    picks = Picks(signal=False, times=[near, sources.O3_START + 5000.5])
    sample = sources.GwoscSource(rng=picks, fetch=fake_fetch(calls)).draw()
    start = sources.O3_START + 5000
    assert calls == [("H1", start, start + 4, sources.PAD_S)]
    assert not sample.has_signal and sample.event is None
    assert app_module.truth_text(sample) == "TRUTH: real noise (no known event)"
    assert sources.EVENT_CLEARANCE_S == 4 + 2 * sources.PAD_S


def test_a_rejected_download_raises():
    error = StrainError("H1: 3 NaN samples")
    source = sources.GwoscSource(rng=Picks(), fetch=fake_fetch([], error=error))
    with pytest.raises(StrainError, match="NaN"):
        source.draw()


def test_gwpy_available_does_not_import_gwpy(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "gwpy", None)
    assert sources.gwpy_available() is False


# --- rating ---


def test_the_start_threshold_lies_inside_the_slider_range():
    """APP-4: the slider started at 0.50 while the decision used 0.46."""
    assert app_module.DEFAULT_THRESHOLD == 0.46
    assert (
        app_module.THRESHOLD_MIN
        <= app_module.DEFAULT_THRESHOLD
        <= app_module.THRESHOLD_MAX
    )


@pytest.mark.parametrize(
    "score, threshold, expected",
    [
        (0.46, 0.46, "red"),  # >= is an alarm, as in gwd.evaluation.metrics
        (0.40, 0.46, "orange"),
        (0.35, 0.46, "blue"),
        (0.85, 0.90, "orange"),  # the band moves with the threshold
        (0.75, 0.90, "blue"),
        (0.30, 0.20, "red"),
    ],
)
def test_the_unsure_band_lies_just_below_the_threshold(score, threshold, expected):
    assert app_module.decision(score, threshold)[1] == expected


def test_an_answer_is_correct_if_the_alarm_matches_the_truth():
    assert app_module.is_correct(0.5, 0.46, True)
    assert not app_module.is_correct(0.5, 0.46, False)
    assert app_module.is_correct(0.1, 0.46, False)
    assert not app_module.is_correct(0.5, 0.6, True)


def test_the_truth_of_a_simulated_sample_names_its_peak_snr():
    sample = sources.LoadedSample(
        np.zeros(N_SAMPLES), True, "Simulation (Training)", "simulation", snr=3.25
    )
    assert app_module.truth_text(sample) == "TRUTH: SIGNAL (peak SNR 3.25)"
    noise = sources.LoadedSample(np.zeros(N_SAMPLES), False, "x", "simulation")
    assert app_module.truth_text(noise) == "TRUTH: NOISE"


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({"real_noise_fraction": 0.0}, app_module.NO_REAL_NOISE_WARNING),
        ({"real_noise_fraction": 0.79}, None),
        ({"real_noise_fraction": "unknown (v1)"}, app_module.UNKNOWN_NOISE_NOTE),
        ({"model_name": "gwd_model_20251205-232013"}, app_module.UNKNOWN_NOISE_NOTE),
        (None, app_module.UNKNOWN_NOISE_NOTE),
    ],
    ids=["gaussian only", "real noise", "unknown v1", "old metadata", "no metadata"],
)
def test_the_training_noise_warning_follows_the_metadata(metadata, expected):
    """APPX-1: a model trained without real noise is warned about."""
    assert app_module.training_noise_warning(metadata) == expected


def test_without_any_model_ask_ai_is_off_but_a_named_model_must_exist(monkeypatch):
    """APP-8: no sys.exit; 'latest' without models gives a note instead."""

    def missing(model):
        raise FileNotFoundError(f"No model {model}.")

    monkeypatch.setattr(app_module, "resolve", missing)
    model, path, note = app_module.load_optional_model("latest")
    assert model is None and path is None
    assert "'Ask AI' is off" in note
    with pytest.raises(FileNotFoundError):
        app_module.load_optional_model("gwd_model_20990101-000000")


def test_without_tensorflow_ask_ai_is_off(monkeypatch):
    def no_tf(path):
        raise ModuleNotFoundError("No module named 'tensorflow'")

    monkeypatch.setattr(app_module, "resolve", lambda model: "m.keras")
    monkeypatch.setattr(app_module, "load_model", no_tf)
    model, path, note = app_module.load_optional_model()
    assert model is None and path == "m.keras"
    assert "pip install -e '.[ml]'" in note


# --- audio ---


class FakeProcess:
    def __init__(self, command, **kwargs):
        self.command = command
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True


def test_to_audio_scales_to_the_peak_and_keeps_silence_silent():
    out = audio.to_audio(np.array([0.0, -2.0, 1.0]))
    assert out.dtype == np.float32
    assert np.isclose(np.max(np.abs(out)), audio.PEAK)
    assert not audio.to_audio(np.zeros(4)).any()


@pytest.mark.parametrize(
    "system, installed, expected",
    [
        ("Darwin", set(), "afplay"),
        ("Linux", {"paplay", "aplay"}, "paplay"),
        ("Linux", {"aplay"}, "aplay"),
        ("Linux", set(), None),
        ("Windows", set(), None),  # winsound, no command
        ("Plan9", set(), None),
    ],
)
def test_each_platform_has_its_own_player(system, installed, expected):
    command = audio.player_command("x.wav", system, lambda name: name in installed)
    assert (command[0] if command else None) == expected


def test_a_sound_is_played_from_a_temporary_folder_without_blocking(
    tmp_path, monkeypatch
):
    """APP-9: no temp_sound.wav in the working directory, Popen instead of run."""
    monkeypatch.chdir(tmp_path)
    started = []

    def popen(command, **kwargs):
        started.append(FakeProcess(command))
        return started[-1]

    player = audio.Player(system="Linux", popen=popen, which=lambda name: True)
    assert player.play(np.sin(np.arange(N_SAMPLES)), SAMPLE_RATE) is None
    assert player.play(np.sin(np.arange(N_SAMPLES)), SAMPLE_RATE) is None
    assert os.listdir(tmp_path) == []
    assert [p.command for p in started] == [["paplay", player.path]] * 2
    assert started[0].terminated  # the second sound stops the first
    assert os.path.dirname(player.path) != str(tmp_path)
    rate, data = audio.wavfile.read(player.path)
    assert rate == SAMPLE_RATE * audio.PLAYBACK_SPEED and len(data) == N_SAMPLES
    player.close()
    assert not os.path.exists(os.path.dirname(player.path))


def test_a_missing_player_is_reported_not_raised():
    player = audio.Player(system="Linux", which=lambda name: None)
    assert "paplay" in player.play(np.ones(8), SAMPLE_RATE)

    def broken(command, **kwargs):
        raise FileNotFoundError(command[0])

    player = audio.Player(system="Darwin", popen=broken)
    assert player.play(np.ones(8), SAMPLE_RATE).startswith("Cannot start afplay")
    player.close()
