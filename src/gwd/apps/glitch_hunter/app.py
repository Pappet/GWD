#!/usr/bin/env python3
"""The Glitch Hunter: a CNN rates simulated or real LIGO data (P4.4).

This module is only the GUI. The samples come from gwd.apps.glitch_hunter.sources and
the sound from gwd.apps.glitch_hunter.audio.

Threads: 'LIGO random' downloads in a worker thread. The worker only fills local
variables and puts (generation, sample, message) into a queue.Queue; it never calls
matplotlib and never changes the app's state. A timer of the figure canvas runs
poll_results() in the GUI thread, which takes the result over and draws it. Every new
sample raises the generation counter, so a download that finishes after a newer sample
was shown is dropped (APP-5, APP-6).

The model is optional: without one, 'Ask AI' is off. TensorFlow loads only when a model
is loaded (gwd.registry.load_model). Without PyCBC the simulation is off and the app
shows real data only; without gwpy it shows simulated data only.

The alarm threshold starts at the model's own threshold (gwd.registry.get_threshold,
calibrated by gwd eval thresholds at FPR 1 %, P5.8); a model without one starts at
THRESHOLD_FALLBACK, with a warning, and so does the app without a model.

The switch 'Scenario' chooses V1 or V2 for the simulation (P6.2): the same two modes,
drawn from the V1 or the V2 scenarios. V1 is the start value until P6.5; V2 needs gwpy
for its conditioning and is off without it.
"""

import queue
import sys
import threading

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

from gwd import registry
from gwd.apps._backend import require_interactive_backend
from gwd.apps.glitch_hunter.audio import Player
from gwd.apps.glitch_hunter.sources import (
    DETECTOR,
    TRAINING,
    VERSIONS,
    GwoscSource,
    SimSource,
    counterpart,
    gwpy_available,
)
from gwd.config import (
    DURATION,
    N_SAMPLES,
    SAMPLE_RATE,
    THRESHOLD_FALLBACK,
    WAVEFORM_BACKEND,
)
from gwd.core.timebase import time_axis
from gwd.core.waveforms import WaveformBackendError, check_backend
from gwd.data.preprocessing import normalize
from gwd.data.provenance import UNKNOWN_V1
from gwd.registry import load_model, model_name, read_metadata, resolve

# The slider's range. The start threshold comes from the registry (start_threshold);
# the whole score range fits, so the slider shows the threshold that decides (APP-4).
THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.0
# A score at most this far below the threshold reads 'UNSURE'.
UNSURE_BAND = 0.10
# How often the GUI thread looks for a finished download, in milliseconds.
POLL_INTERVAL_MS = 100

GWDATA_HINT = "Install gwpy for real data: pip install -e '.[gwdata]'"
ML_HINT = "Install TensorFlow to rate samples: pip install -e '.[ml]'"

NO_REAL_NOISE_WARNING = (
    "WARNING: this model was trained without real noise (Gaussian only). "
    "Real LIGO data lies outside its training data, so its scores there mean little "
    "(APPX-1)."
)
UNKNOWN_NOISE_NOTE = (
    "Note: the model's metadata does not say whether it was trained with real noise "
    f"('{UNKNOWN_V1}' or no metadata: trained before P4.1)."
)

DISABLED_COLOR = "0.85"


def training_noise_warning(metadata):
    """What to tell the player about the model's training noise (APPX-1).

    NO_REAL_NOISE_WARNING if the metadata says the real noise share was 0,
    UNKNOWN_NOISE_NOTE if it does not say (no metadata, or from before P4.1), and
    None if the model saw real noise.
    """
    fraction = (metadata or {}).get("real_noise_fraction", UNKNOWN_V1)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        return UNKNOWN_NOISE_NOTE
    if fraction == 0:
        return NO_REAL_NOISE_WARNING
    return None


def decision(score, threshold, band=UNSURE_BAND):
    """(text, color) for a score.

    An alarm at score >= threshold, the rule of gwd.evaluation.metrics.predict_labels;
    'UNSURE' up to band below the threshold, so the band moves with the slider.
    """
    if score >= threshold:
        return "ALARM: SIGNAL DETECTED!", "red"
    if score >= threshold - band:
        return "UNSURE: just below the threshold", "orange"
    return "QUIET: noise only", "blue"


def truth_text(sample):
    """The truth about a sample, as the GUI shows it after 'Ask AI'."""
    if sample.source == "simulation":
        if sample.has_signal:
            return f"TRUTH: SIGNAL ({sample.snr_kind} SNR {sample.snr:.2f})"
        return "TRUTH: NOISE"
    if sample.event is None:
        return "TRUTH: real noise (no known event)"
    text = f"TRUTH: {sample.event.name}"
    if sample.out_of_distribution:
        text += " (OOD: outside the training masses)"
    return text


def is_correct(score, threshold, has_signal):
    """True if the model's answer at this threshold matches the truth."""
    return (score >= threshold) == has_signal


def start_threshold(path):
    """The alarm threshold at the start: the model's own from the registry (P5.8).

    Without a model (path None) it is THRESHOLD_FALLBACK; a model without a calibrated
    threshold also gets it, and gwd.registry.get_threshold warns.
    """
    if path is None:
        return float(THRESHOLD_FALLBACK)
    return registry.get_threshold(path)


def load_optional_model(model="latest"):
    """Load a model for 'Ask AI'. Returns (model, path, note); model is None if off.

    'latest' without any model, or without TensorFlow, turns 'Ask AI' off and says why
    in note. A model that was named explicitly but does not exist raises
    FileNotFoundError.
    """
    try:
        path = resolve(model)
    except FileNotFoundError as error:
        if model != "latest":
            raise
        return None, None, f"{error} 'Ask AI' is off."
    try:
        return load_model(path), path, None
    except ImportError as error:
        return None, path, f"{ML_HINT} ({error}). 'Ask AI' is off."


class GlitchHunterApp:
    """The GUI. sim and real are the sources; None turns that kind of sample off."""

    def __init__(
        self,
        model=None,
        metadata=None,
        sim=None,
        real=None,
        player=None,
        threshold=THRESHOLD_FALLBACK,
        v2_available=None,
    ):
        self.model = model
        # V2 conditions its samples with gwpy (gwd.data.preprocessing.condition_array).
        self.v2_available = gwpy_available() if v2_available is None else v2_available
        self.noise_warning = (
            training_noise_warning(metadata) if model is not None else None
        )
        self.sim = sim
        self.real = real
        self.player = player or Player()

        self.time = time_axis(SAMPLE_RATE, DURATION)
        self.threshold = threshold
        self.sample = None
        self.prediction = None
        self.truth_artist = None

        # Raised for every new sample; a download of an older generation is dropped.
        self.generation = 0
        self.loading = False
        self.results = queue.Queue()
        self.thread = None

        self.setup_gui()
        if self.sim is not None:
            self.generate_new_sample(None)
        else:
            self.show_message(
                "The simulation needs PyCBC (pip install -e . on Linux or WSL).\n"
                "Press 'LIGO random' for real data."
            )

    # --- layout ---

    def setup_gui(self):
        self.fig = plt.figure(figsize=(13, 10))
        self.fig.canvas.manager.set_window_title("Glitch Hunter")

        gs = self.fig.add_gridspec(4, 1, height_ratios=[1.5, 1.5, 1, 0.5], hspace=0.5)
        if self.noise_warning == NO_REAL_NOISE_WARNING:
            # Above the plots, so it stays visible in every mode.
            self.fig.text(
                0.5, 0.985, self.noise_warning, ha="center", va="top", color="red"
            )

        self.ax_signal = self.fig.add_subplot(gs[0])
        self.ax_spec = self.fig.add_subplot(gs[1], sharex=self.ax_signal)
        self.ax_confidence = self.fig.add_subplot(gs[2])
        self.fig.subplots_adjust(bottom=0.2)

        ax_thresh = self.fig.add_axes([0.2, 0.11, 0.41, 0.03])
        self.slider_thresh = Slider(
            ax_thresh,
            "Alarm threshold",
            THRESHOLD_MIN,
            THRESHOLD_MAX,
            # A threshold above every score (it detects nothing) shows as the maximum.
            valinit=float(np.clip(self.threshold, THRESHOLD_MIN, THRESHOLD_MAX)),
            valfmt="%.3f",
            color="orange",
        )
        self.slider_thresh.on_changed(self.update_threshold)

        mode = self.sim.mode if self.sim else TRAINING
        modes = VERSIONS[mode.version]
        ax_mode = self.fig.add_axes([0.68, 0.085, 0.23, 0.08])
        ax_mode.set_title("Simulation mode", fontsize=9)
        self.radio_mode = RadioButtons(
            ax_mode, [mode_label(m) for m in modes], active=modes.index(mode)
        )
        self.radio_mode.on_clicked(self.set_mode)

        ax_version = self.fig.add_axes([0.92, 0.085, 0.06, 0.08])
        ax_version.set_title("Scenario", fontsize=9)
        versions = sorted(VERSIONS)
        self.radio_version = RadioButtons(
            ax_version,
            [version_label(v) for v in versions],
            active=versions.index(mode.version),
        )
        self.radio_version.on_clicked(self.set_version)

        self.btn_sim = self._button(
            [0.05, 0.02, 0.20, 0.05], "Simulation", "lightblue", "skyblue"
        )
        self.btn_sim.on_clicked(self.generate_new_sample)
        self.btn_real = self._button(
            [0.28, 0.02, 0.20, 0.05], "LIGO random", "mediumpurple", "violet"
        )
        self.btn_real.on_clicked(self.start_background_loading)
        self.btn_sound = self._button(
            [0.51, 0.02, 0.20, 0.05], "Listen", "gold", "yellow"
        )
        self.btn_sound.on_clicked(self.play_sound)
        ask = "Ask AI" if self.model is not None else "Ask AI (no model)"
        self.btn_ask = self._button([0.74, 0.02, 0.20, 0.05], ask, "lightgreen", "lime")
        self.btn_ask.on_clicked(self.ask_ai)

        self.timer = self.fig.canvas.new_timer(interval=POLL_INTERVAL_MS)
        self.timer.add_callback(self.poll_results)
        self.timer.start()
        self.fig.canvas.mpl_connect("close_event", lambda event: self.close())
        self._update_buttons()

    def _button(self, rect, label, color, hovercolor):
        button = Button(
            self.fig.add_axes(rect), label, color=color, hovercolor=hovercolor
        )
        button.enabled_color = color
        return button

    @staticmethod
    def _enable(button, enabled):
        button.active = enabled
        button.color = button.enabled_color if enabled else DISABLED_COLOR
        button.ax.set_facecolor(button.color)
        button.label.set_alpha(1.0 if enabled else 0.4)

    def _update_buttons(self):
        """Lock what needs a loaded sample while a download runs.

        'Simulation' and the mode stay usable during a download; a new simulated sample
        makes the download stale, so its result is dropped.
        """
        self._enable(self.btn_sim, self.sim is not None)
        self.radio_mode.active = self.sim is not None
        self.radio_version.active = self.sim is not None and self.v2_available
        self._enable(self.btn_real, self.real is not None and not self.loading)
        ready = self.sample is not None and not self.loading
        self._enable(self.btn_sound, ready)
        self._enable(self.btn_ask, ready and self.model is not None)

    # --- samples ---

    def generate_new_sample(self, event):
        """Show a simulated sample of the current mode; a pending download goes stale."""
        if self.sim is None:
            return
        self.generation += 1
        self.loading = False
        self.sample = self.sim.draw()
        self._render_current_sample()
        self._update_buttons()

    def set_mode(self, label):
        if self.sim is None:
            return
        modes = VERSIONS[self.sim.mode.version]
        self.sim.mode = next(mode for mode in modes if mode_label(mode) == label)
        self.generate_new_sample(None)

    def set_version(self, label):
        """Switch the simulation to scenario V1 or V2, keep the mode, draw a new sample."""
        if self.sim is None:
            return
        version = next(v for v in VERSIONS if version_label(v) == label)
        if version == self.sim.mode.version:
            return
        if version != 1 and not self.v2_available:
            return
        self.sim.mode = counterpart(self.sim.mode, version)
        modes = VERSIONS[version]
        for text, mode in zip(self.radio_mode.labels, modes):
            text.set_text(mode_label(mode))
        # Selecting the mode again updates value_selected and draws the new sample.
        self.radio_mode.set_active(modes.index(self.sim.mode))

    def start_background_loading(self, event):
        """Start a download in a worker thread, so the GUI does not freeze."""
        if self.loading or self.real is None:
            return  # a second click while loading starts no second download
        self.generation += 1
        self.loading = True
        self._update_buttons()
        self.ax_signal.set_title(
            "Loading data from GWOSC... (please wait)",
            fontsize=12,
            fontweight="bold",
            color="orange",
        )
        self.fig.canvas.draw_idle()
        self.thread = threading.Thread(
            target=self._load_real, args=(self.generation,), daemon=True
        )
        self.thread.start()

    def _load_real(self, generation):
        """Worker thread: download, or simulate if that fails; no matplotlib here.

        Puts (generation, sample, message) into the queue; sample is None if nothing
        could be loaded, and message says what went wrong.
        """
        sample, message = None, None
        try:
            sample = self.real.draw()
        except Exception as error:  # network, GWOSC, gwpy, StrainError: all fall back
            message = f"GWOSC download failed: {error}"
        sim = self.sim
        if sample is None and sim is not None:
            try:
                sample = sim.draw(note=f"{message}; simulated instead")
            except Exception as error:  # reported in the GUI, not lost in the thread
                message = f"{message}; the simulation failed too: {error}"
        self.results.put((generation, sample, message))

    def poll_results(self):
        """GUI thread (timer): take over a finished download, drop older generations."""
        while True:
            try:
                generation, sample, message = self.results.get_nowait()
            except queue.Empty:
                return
            if generation != self.generation:
                continue
            self.loading = False
            if message:
                print(message, file=sys.stderr)
            self.sample = sample
            if sample is None:
                self.show_message(message)
            else:
                self._render_current_sample()
            self._update_buttons()

    # --- drawing (GUI thread only) ---

    def _clear_rating(self):
        if self.truth_artist is not None:
            self.truth_artist.remove()
            self.truth_artist = None
        self.prediction = None
        self.ax_confidence.clear()
        self.ax_confidence.axis("off")
        if self.model is not None:
            hint = "Press 'Ask AI' for an analysis"
        else:
            hint = "No model loaded: 'Ask AI' is off"
        self.ax_confidence.text(0.5, 0.5, hint, ha="center", va="center", color="gray")

    def show_message(self, text):
        self._clear_rating()
        self.ax_signal.clear()
        self.ax_spec.clear()
        self.ax_signal.text(
            0.5, 0.5, text, ha="center", va="center", transform=self.ax_signal.transAxes
        )
        self.fig.canvas.draw_idle()

    def _titles(self):
        """(plot title, spectrogram title) for the shown sample, by its source."""
        sample = self.sample
        if sample.source == "simulation":
            title = f"Simulated data: {sample.name}"
            spec = f"Spectrogram (simulated, {sample.noise_kind} noise)"
        else:
            title = f"REAL DATA: {sample.name} ({DETECTOR})"
            spec = f"Spectrogram (GWOSC {DETECTOR}, notched and band-passed)"
        return f"{title}\n{sample.note}", spec

    def _render_current_sample(self):
        """Draw self.sample. Only the GUI thread calls this."""
        sample = self.sample
        self._clear_rating()
        title, spec_title = self._titles()
        color = "purple" if sample.source == "gwosc" else "black"

        # Unit std for the plot; the model gets normalize(sample.data) in ask_ai.
        display = normalize(sample.data, eps=0.0)
        self.ax_signal.clear()
        self.ax_signal.plot(self.time, display, color="#333", lw=0.5)
        self.ax_signal.set_title(title, fontsize=11, fontweight="bold", color=color)
        self.ax_signal.set_ylabel("Strain (normalized)")
        self.ax_signal.set_xlim(0, DURATION)
        self.ax_signal.grid(True, alpha=0.3)

        self.ax_spec.clear()
        self.ax_spec.specgram(
            display, NFFT=256, Fs=SAMPLE_RATE, noverlap=200, cmap="inferno"
        )
        self.ax_spec.set_title(spec_title, fontsize=11, fontweight="bold")
        self.ax_spec.set_ylim(0, 600)
        self.fig.canvas.draw_idle()

    def _draw_truth(self):
        if self.truth_artist is not None:
            self.truth_artist.remove()
        correct = is_correct(self.prediction, self.threshold, self.sample.has_signal)
        self.truth_artist = self.fig.text(
            0.5,
            0.955,
            truth_text(self.sample),
            ha="center",
            fontsize=12,
            bbox=dict(
                facecolor="lightgreen" if correct else "lightcoral",
                alpha=0.8,
                boxstyle="round",
            ),
        )

    def update_confidence_display(self):
        self.ax_confidence.clear()
        self.ax_confidence.axis("off")
        score, threshold = self.prediction, self.threshold
        text, bar_color = decision(score, threshold)

        self.ax_confidence.axvline(threshold, color="black", linestyle="--", alpha=0.5)
        self.ax_confidence.text(
            threshold, -0.6, f"{threshold:.1%}", ha="center", fontsize=8
        )
        self.ax_confidence.barh([0], [score], color=bar_color, height=0.5, alpha=0.8)
        self.ax_confidence.barh([0], [1.0], color="gray", height=0.5, alpha=0.1)
        self.ax_confidence.set_xlim(0, 1)
        self.ax_confidence.set_ylim(-0.5, 0.5)
        self.ax_confidence.text(
            0.5, 0.6, text, ha="center", fontsize=16, fontweight="bold", color=bar_color
        )
        self.ax_confidence.text(
            0.5, -0.3, f"AI score: {score:.2%}", ha="center", fontsize=12
        )

    # --- actions ---

    def update_threshold(self, value):
        """Move the threshold and judge the shown answer again."""
        self.threshold = value
        if self.prediction is not None:
            self.update_confidence_display()
            self._draw_truth()
            self.fig.canvas.draw_idle()

    def ask_ai(self, event):
        if (
            self.model is None
            or self.loading
            or self.sample is None
            or self.prediction is not None
        ):
            return
        sample = self.sample
        x = normalize(sample.data).reshape(1, N_SAMPLES, 1)
        self.prediction = float(self.model.predict(x, verbose=0)[0][0])
        self.update_confidence_display()

        if sample.signal is not None:
            # In the units of the plot, which shows the data with unit std.
            scale = np.std(sample.data) or 1.0
            self.ax_signal.plot(
                self.time,
                sample.signal / scale,
                color="cyan",
                lw=2,
                alpha=0.8,
                label="Injected signal",
            )
            self.ax_signal.legend(loc="upper right")
        self._draw_truth()
        self.fig.canvas.draw_idle()

    def play_sound(self, event):
        if self.sample is None or self.loading:
            return
        problem = self.player.play(normalize(self.sample.data, eps=0.0), SAMPLE_RATE)
        if problem:
            print(problem, file=sys.stderr)

    def close(self):
        self.timer.stop()
        self.player.close()


def mode_label(mode):
    """The radio button label of a mode; 'Challenge' says that it is hard on purpose.

    The V1 challenge lies partly below the training range; the V2 one inside it.
    """
    if mode.name == TRAINING.name:
        return f"{mode.name} (training data)"
    if mode.version == 1:
        return f"{mode.name} (hard: extrapolation)"
    return f"{mode.name} (hard: low SNR)"


def version_label(version):
    """The radio button label of a scenario version: V1 or V2."""
    return f"V{version}"


def main(model="latest"):
    """Start the Glitch Hunter with model ('latest', a name, or a path); return the rc."""
    require_interactive_backend("Glitch Hunter")

    sim, sim_problem = None, None
    try:
        check_backend(WAVEFORM_BACKEND)
        sim = SimSource()
    except WaveformBackendError as error:
        sim_problem = str(error)
    real = GwoscSource() if gwpy_available() else None
    if sim is None and real is None:
        print(f"ERROR: {sim_problem}\n{GWDATA_HINT}", file=sys.stderr)
        return 1
    if sim is None:
        print(f"Note: the simulation is off. {sim_problem}", file=sys.stderr)
    if real is None:
        print(f"Note: real data are off. {GWDATA_HINT}", file=sys.stderr)

    print(">>> Loading the model...")
    try:
        loaded, path, note = load_optional_model(model)
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    metadata = None
    if loaded is None:
        print(f"Note: {note}", file=sys.stderr)
        threshold = start_threshold(None)
    else:
        print(f"Model: {model_name(path)}")
        metadata = read_metadata(path)
        warning = training_noise_warning(metadata)
        if warning:
            print(warning)
        threshold = start_threshold(path)
        print(f"Alarm threshold: {threshold:.4g}")
    print(">>> Ready to hunt!")

    app = GlitchHunterApp(
        model=loaded, metadata=metadata, sim=sim, real=real, threshold=threshold
    )
    plt.show()
    app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
