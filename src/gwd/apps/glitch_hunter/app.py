#!/usr/bin/env python3
import os
import platform
import random
import subprocess
import sys
import threading

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
from matplotlib.widgets import Button, Slider

from gwd.apps._backend import require_interactive_backend
from gwd.config import (
    DURATION,
    MODELS_DIR,
    N_SAMPLES,
    SAMPLE_RATE,
)
from gwd.data.gwosc import EVENTS, fetch_strain
from gwd.data.injection import make_sample
from gwd.data.preprocessing import condition_strain, normalize
from gwd.data.provenance import UNKNOWN_V1
from gwd.data.scenarios import APP_V1
from gwd.registry import list_models, load_model, read_metadata

O3_START = 1238112018
O3_END = 1253923218

NO_REAL_NOISE_WARNING = (
    "WARNING: this model was trained without real noise (Gaussian only). "
    "Real LIGO data lies outside its training data, so its scores there mean little "
    "(APPX-1)."
)
UNKNOWN_NOISE_NOTE = (
    "Note: the model's metadata does not say whether it was trained with real noise "
    f"('{UNKNOWN_V1}' or no metadata: trained before P4.1)."
)


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


class GlitchHunterApp:
    def __init__(self):
        print(">>> Lade Gehirn (Neuronales Netz)...")

        # One check covers a missing and an empty folder; rc 1 stays until P4.4 (APP-8).
        files = list_models()
        if not files:
            print(f"ERROR: No models (.keras) in '{MODELS_DIR}'.")
            print("   Run 'gwd train' first to train a model.")
            sys.exit(1)

        # The newest timestamp in the name, not ctime (apps:APPX-4).
        latest_model_path = files[-1]
        print(f"{len(files)} Modelle gefunden.")
        print(f"Lade neuestes Modell: {os.path.basename(latest_model_path)}")

        self.model = load_model(latest_model_path)
        self.noise_warning = training_noise_warning(read_metadata(latest_model_path))
        if self.noise_warning:
            print(self.noise_warning)
        print(">>> Bereit zum Jagen!")

        self.fs = SAMPLE_RATE
        self.duration = DURATION
        self.time = np.linspace(0, self.duration, N_SAMPLES)

        self.current_data = None
        self.pure_signal = None
        self.has_signal = False
        self.is_real_data = False
        self.event_name = ""
        self.prediction = None
        self.true_snr = 0
        self.detection_threshold = 0.46  # Statt 0.75
        self.truth_text_artist = None

        self.setup_gui()
        self.generate_new_sample(None)

    def setup_gui(self):
        self.fig = plt.figure(figsize=(13, 10))
        self.fig.canvas.manager.set_window_title("Glitch Hunter - ULTIMATE Edition")

        gs = self.fig.add_gridspec(4, 1, height_ratios=[1.5, 1.5, 1, 0.5], hspace=0.4)
        if self.noise_warning == NO_REAL_NOISE_WARNING:
            # Above the plots, so it stays visible in every mode.
            self.fig.text(
                0.5, 0.985, self.noise_warning, ha="center", va="top", color="red"
            )

        self.ax_signal = self.fig.add_subplot(gs[0])
        self.ax_spec = self.fig.add_subplot(gs[1], sharex=self.ax_signal)
        self.ax_confidence = self.fig.add_subplot(gs[2])

        plt.subplots_adjust(bottom=0.2)

        ax_thresh = plt.axes([0.25, 0.11, 0.5, 0.03])
        self.slider_thresh = Slider(
            ax_thresh,
            "Alarm-Schwelle",
            0.5,
            0.99,
            valinit=self.detection_threshold,
            color="orange",
        )
        self.slider_thresh.on_changed(self.update_threshold)

        ax_sim = plt.axes([0.05, 0.03, 0.20, 0.05])
        self.btn_sim = Button(
            ax_sim, "Simulation", color="lightblue", hovercolor="skyblue"
        )
        self.btn_sim.on_clicked(self.generate_new_sample)

        ax_real = plt.axes([0.28, 0.03, 0.20, 0.05])
        self.btn_real = Button(
            ax_real, "LIGO Zufall", color="mediumpurple", hovercolor="violet"
        )
        # Ruft jetzt den Background-Starter auf
        self.btn_real.on_clicked(self.start_background_loading)

        ax_sound = plt.axes([0.51, 0.03, 0.20, 0.05])
        self.btn_sound = Button(ax_sound, "Anhören", color="gold", hovercolor="yellow")
        self.btn_sound.on_clicked(self.play_sound)

        ax_ask = plt.axes([0.74, 0.03, 0.20, 0.05])
        self.btn_ask = Button(
            ax_ask, "KI fragen", color="lightgreen", hovercolor="lime"
        )
        self.btn_ask.on_clicked(self.ask_ai)

    def start_background_loading(self, event):
        """Startet den Download in einem separaten Thread, damit die GUI nicht einfriert."""
        self.ax_signal.set_title(
            "Lade Daten von LIGO... (Bitte warten)",
            fontsize=12,
            fontweight="bold",
            color="orange",
        )
        self.fig.canvas.draw_idle()

        # Thread starten
        thread = threading.Thread(target=self.load_random_real_thread)
        thread.daemon = True  # Thread stirbt, wenn Hauptprogramm beendet wird
        thread.start()

    def load_random_real_thread(self):
        """Die eigentliche Arbeit passiert hier im Hintergrund."""
        print("Background Thread: Wähle zufällige LIGO-Daten...")

        want_signal = random.choice([True, False])
        t0 = 0
        detector = "H1"

        if want_signal:
            known = random.choice(EVENTS)
            name, time = known.name, known.gps
            print(f"Ziel: Bekanntes Event {name}")
            t0 = int(time)
            self.event_name = name
            self.has_signal = True
        else:
            print("Ziel: Zufälliges Rauschen (O3 Run)")
            self.event_name = "Random Noise (O3)"
            self.has_signal = False

            found_data = False
            for _ in range(5):
                t_candidate = random.uniform(O3_START, O3_END)
                is_event = False
                for known in EVENTS:
                    if abs(t_candidate - known.gps) < 16:
                        is_event = True
                        break
                if not is_event:
                    t0 = int(t_candidate)
                    found_data = True
                    break

            if not found_data:
                print("Kein Slot gefunden, fallback.")
                self.update_plot_threadsafe(fallback=True)
                return

        try:
            print(f"Download H1 Daten um {t0}...")
            # Raises on a wrong length or NaN instead of trimming or padding
            strain = fetch_strain(detector, t0 - 2, t0 + 2)

            # Notch the power line and its harmonics, then band-pass
            raw_array = condition_strain(strain, self.fs)

            # eps=0: the conditioned strain has a std near 1e-21, eps would dominate
            self.current_data = normalize(raw_array, eps=0.0)

            self.is_real_data = True
            self.pure_signal = None
            self.true_snr = 0
            self.prediction = None

            # GUI Update muss vom Haupt-Thread gemacht werden?
            # Matplotlib mit TkAgg ist da manchmal zickig, aber draw_idle ist meist thread-safe genug für einfache updates.
            self.update_plot_threadsafe(
                title=f"ECHTE DATEN: {self.event_name} ({detector})"
            )
            print("Daten geladen und bereinigt!")

        except Exception as e:
            print(f"Fehler im Thread: {e}")
            self.update_plot_threadsafe(fallback=True)

    def update_plot_threadsafe(self, title="", fallback=False):
        """Hilfsmethode, um Plot aus dem Thread zu aktualisieren"""
        if fallback:
            # Wenn was schief ging, generieren wir schnell eine Simulation
            # Aber wir müssen aufpassen, nicht rekursiv Threads zu starten
            # Wir rufen einfach die Logik direkt auf
            self.is_real_data = False
            self.event_name = "Simulation (Fallback)"
            # Noise only, from the same scenario as the simulation mode
            noise = make_sample(APP_V1, has_signal=False).data
            self.current_data = noise
            self.has_signal = False
            self.pure_signal = np.zeros_like(noise)
            title = "Simulierte Daten (Fallback)"

        # Reset UI
        if self.truth_text_artist is not None:
            self.truth_text_artist.remove()
            self.truth_text_artist = None

        self.prediction = None
        self.ax_confidence.clear()
        self.ax_confidence.axis("off")
        self.ax_confidence.text(
            0.5,
            0.5,
            "Drücke 'KI fragen' für Analyse",
            ha="center",
            va="center",
            color="gray",
        )

        # PLOT 1
        self.ax_signal.clear()
        self.ax_signal.plot(self.time, self.current_data, color="#333", lw=0.5)
        self.ax_signal.set_title(title, fontsize=12, fontweight="bold", color="purple")
        self.ax_signal.set_ylabel("Strain")
        self.ax_signal.set_xlim(0, self.duration)
        self.ax_signal.grid(True, alpha=0.3)

        # PLOT 2
        self.ax_spec.clear()
        Pxx, freqs, bins, im = self.ax_spec.specgram(
            self.current_data, NFFT=256, Fs=self.fs, noverlap=200, cmap="inferno"
        )
        self.ax_spec.set_title(
            "2. Spektrogramm (Bereinigt)", fontsize=12, fontweight="bold"
        )
        self.ax_spec.set_ylim(0, 600)

        self.fig.canvas.draw_idle()

    def play_sound(self, event):
        if self.current_data is None:
            return
        audio_data = self.current_data / np.max(np.abs(self.current_data)) * 0.8
        playback_rate = int(self.fs * 2.0)
        filename = "temp_sound.wav"
        wavfile.write(filename, playback_rate, audio_data.astype(np.float32))
        try:
            if platform.system() == "Darwin":
                subprocess.run(["afplay", filename], check=False)
            elif platform.system() == "Linux":
                try:
                    subprocess.run(["aplay", filename], check=False)
                except Exception:
                    subprocess.run(["paplay", filename], check=False)
            elif platform.system() == "Windows":
                import winsound

                winsound.PlaySound(filename, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            print(f"Audio Fehler: {e}")

    def update_threshold(self, val):
        self.detection_threshold = val
        if self.prediction is not None:
            self.update_confidence_display(self.prediction)

    def generate_new_sample(self, event):
        self.is_real_data = False
        self.event_name = "Simulation"
        sample = make_sample(APP_V1)
        self.has_signal = sample.has_signal
        self.current_data = sample.data
        if sample.has_signal:
            self.pure_signal = sample.signal
        else:
            self.pure_signal = np.zeros_like(sample.data)
        self.true_snr = sample.snr

        self.update_plot_threadsafe(title="Simulierte Daten")

    def ask_ai(self, event):
        if self.prediction is not None:
            return

        input_data = self.current_data.reshape(1, len(self.current_data), 1)
        input_data = normalize(input_data)

        self.prediction = self.model.predict(input_data)[0][0]
        self.update_confidence_display(self.prediction)

        if not self.is_real_data and self.has_signal:
            self.ax_signal.plot(
                self.time,
                self.pure_signal,
                color="cyan",
                lw=2,
                alpha=0.8,
                label="Signal",
            )
            self.ax_signal.legend(loc="upper right")

        if self.is_real_data:
            if self.has_signal:
                truth_text = f"WAHRHEIT: {self.event_name}"
                box_color = "gold"
            else:
                truth_text = "WAHRHEIT: Echtes Rauschen"
                box_color = "lightgray"
        else:
            truth_text = "WAHRHEIT: " + ("SIGNAL" if self.has_signal else "RAUSCHEN")
            if self.has_signal:
                truth_text += f" (SNR: {self.true_snr:.2f})"
            is_detection = self.prediction > self.detection_threshold
            success = is_detection == self.has_signal
            box_color = "lightgreen" if success else "lightcoral"

        self.truth_text_artist = self.fig.text(
            0.5,
            0.95,
            truth_text,
            ha="center",
            fontsize=12,
            bbox=dict(facecolor=box_color, alpha=0.8, boxstyle="round"),
        )
        self.fig.canvas.draw_idle()

    def update_confidence_display(self, pred_prob):
        self.ax_confidence.clear()
        self.ax_confidence.axis("off")
        threshold = self.detection_threshold

        if pred_prob > threshold:
            decision = "ALARM: SIGNAL ENTDECKT!"
            bar_color = "red"
        elif pred_prob > 0.5:
            decision = "UNSICHER (Ignoriert)"
            bar_color = "orange"
        else:
            decision = "RUHIG: Nur Rauschen"
            bar_color = "blue"

        self.ax_confidence.axvline(threshold, color="black", linestyle="--", alpha=0.5)
        self.ax_confidence.text(
            threshold, -0.6, f"{threshold:.0%}", ha="center", fontsize=8
        )

        self.ax_confidence.barh(
            [0], [pred_prob], color=bar_color, height=0.5, alpha=0.8
        )
        self.ax_confidence.barh([0], [1.0], color="gray", height=0.5, alpha=0.1)
        self.ax_confidence.set_xlim(0, 1)
        self.ax_confidence.set_ylim(-0.5, 0.5)

        percent = pred_prob * 100
        self.ax_confidence.text(
            0.5,
            0.6,
            f"{decision}",
            ha="center",
            fontsize=16,
            fontweight="bold",
            color=bar_color,
        )
        self.ax_confidence.text(
            0.5, -0.3, f"KI Konfidenz: {percent:.2f}%", ha="center", fontsize=12
        )
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    require_interactive_backend("Glitch Hunter")
    app = GlitchHunterApp()
    plt.show()
