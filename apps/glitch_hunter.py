#!/usr/bin/env python3
import os
import sys
import glob 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import tensorflow as tf
import scipy.io.wavfile as wavfile
import subprocess
import platform
import random
import threading 
from gwpy.timeseries import TimeSeries

# Imports aus deinem Core
sys.path.append(os.path.dirname(__file__))
from gwd_core.waveforms import generate_astrophysical_chirp
from gwd_core.noise import generate_gaussian_noise

KNOWN_EVENTS = {
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

O3_START = 1238112018
O3_END = 1253923218

class GlitchHunterApp:
    def __init__(self):
        print(">>> Lade Gehirn (Neuronales Netz)...")
        
        # --- NEU: Intelligente Modell-Suche ---
        models_dir = "models_registry"
        
        if not os.path.exists(models_dir):
            print(f"FEHLER: Ordner '{models_dir}' nicht gefunden.")
            print("   Bitte führe zuerst 'python dataset/train_cnn.py' aus, um ein Modell zu trainieren.")
            sys.exit(1)
            
        # Suche alle .keras Dateien
        files = glob.glob(os.path.join(models_dir, "*.keras"))
        if not files:
            print(f"FEHLER: Keine Modelle (.keras) in '{models_dir}' gefunden.")
            sys.exit(1)
            
        # Wähle die neueste Datei basierend auf dem Erstellungsdatum (ctime)
        latest_model_path = max(files, key=os.path.getctime)
        print(f"{len(files)} Modelle gefunden.")
        print(f"Lade neuestes Modell: {os.path.basename(latest_model_path)}")
            
        self.model = tf.keras.models.load_model(latest_model_path)
        print(">>> Bereit zum Jagen!")

        self.fs = 4096
        self.duration = 4.0
        self.time = np.linspace(0, self.duration, int(self.fs * self.duration))
        
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
        self.fig.canvas.manager.set_window_title('Glitch Hunter - ULTIMATE Edition')
        
        gs = self.fig.add_gridspec(4, 1, height_ratios=[1.5, 1.5, 1, 0.5], hspace=0.4)

        self.ax_signal = self.fig.add_subplot(gs[0])
        self.ax_spec = self.fig.add_subplot(gs[1], sharex=self.ax_signal)
        self.ax_confidence = self.fig.add_subplot(gs[2])
        
        plt.subplots_adjust(bottom=0.2)
        
        ax_thresh = plt.axes([0.25, 0.11, 0.5, 0.03])
        self.slider_thresh = Slider(ax_thresh, 'Alarm-Schwelle', 0.5, 0.99, valinit=self.detection_threshold, color='orange')
        self.slider_thresh.on_changed(self.update_threshold)
        
        ax_sim = plt.axes([0.05, 0.03, 0.20, 0.05])
        self.btn_sim = Button(ax_sim, 'Simulation', color='lightblue', hovercolor='skyblue')
        self.btn_sim.on_clicked(self.generate_new_sample)
        
        ax_real = plt.axes([0.28, 0.03, 0.20, 0.05])
        self.btn_real = Button(ax_real, 'LIGO Zufall', color='mediumpurple', hovercolor='violet')
        # Ruft jetzt den Background-Starter auf
        self.btn_real.on_clicked(self.start_background_loading)
        
        ax_sound = plt.axes([0.51, 0.03, 0.20, 0.05])
        self.btn_sound = Button(ax_sound, 'Anhören', color='gold', hovercolor='yellow')
        self.btn_sound.on_clicked(self.play_sound)
        
        ax_ask = plt.axes([0.74, 0.03, 0.20, 0.05])
        self.btn_ask = Button(ax_ask, 'KI fragen', color='lightgreen', hovercolor='lime')
        self.btn_ask.on_clicked(self.ask_ai)

    def start_background_loading(self, event):
        """Startet den Download in einem separaten Thread, damit die GUI nicht einfriert."""
        self.ax_signal.set_title("Lade Daten von LIGO... (Bitte warten)", fontsize=12, fontweight='bold', color='orange')
        self.fig.canvas.draw_idle()
        
        # Thread starten
        thread = threading.Thread(target=self.load_random_real_thread)
        thread.daemon = True # Thread stirbt, wenn Hauptprogramm beendet wird
        thread.start()

    def load_random_real_thread(self):
        """Die eigentliche Arbeit passiert hier im Hintergrund."""
        print("Background Thread: Wähle zufällige LIGO-Daten...")
        
        want_signal = random.choice([True, False])
        t0 = 0
        detector = 'H1'
        
        if want_signal:
            name, time = random.choice(list(KNOWN_EVENTS.items()))
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
                for e_time in KNOWN_EVENTS.values():
                    if abs(t_candidate - e_time) < 16:
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
            strain = TimeSeries.fetch_open_data(detector, t0 - 2, t0 + 2, verbose=False)
            
            if strain.sample_rate.value != self.fs:
                strain = strain.resample(self.fs)
            
            # NEU: CLEANING / NOTCH FILTER
            # Wir entfernen die 60Hz Netzbrummen und Obertöne (120, 180...)
            # Das entfernt die horizontalen Linien, die die KI verwirren!
            strain = strain.notch(60).notch(120).notch(180).notch(240)
            
            white_data = strain.bandpass(20, 300)
            raw_array = white_data.value
            
            expected_len = len(self.time)
            if len(raw_array) > expected_len:
                raw_array = raw_array[:expected_len]
            elif len(raw_array) < expected_len:
                raw_array = np.pad(raw_array, (0, expected_len - len(raw_array)))

            # Normalisierung
            self.current_data = (raw_array - np.mean(raw_array)) / np.std(raw_array)
            
            self.is_real_data = True
            self.pure_signal = None
            self.true_snr = 0
            self.prediction = None
            
            # GUI Update muss vom Haupt-Thread gemacht werden? 
            # Matplotlib mit TkAgg ist da manchmal zickig, aber draw_idle ist meist thread-safe genug für einfache updates.
            self.update_plot_threadsafe(title=f"ECHTE DATEN: {self.event_name} ({detector})")
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
            noise = generate_gaussian_noise(len(self.time), noise_level=1.0)
            # ... (Rest der Simulation Logik vereinfacht)
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
        self.ax_confidence.axis('off')
        self.ax_confidence.text(0.5, 0.5, "Drücke 'KI fragen' für Analyse", ha='center', va='center', color='gray')

        # PLOT 1
        self.ax_signal.clear()
        self.ax_signal.plot(self.time, self.current_data, color='#333', lw=0.5)
        self.ax_signal.set_title(title, fontsize=12, fontweight='bold', color='purple')
        self.ax_signal.set_ylabel("Strain")
        self.ax_signal.set_xlim(0, self.duration)
        self.ax_signal.grid(True, alpha=0.3)
        
        # PLOT 2
        self.ax_spec.clear()
        Pxx, freqs, bins, im = self.ax_spec.specgram(self.current_data, NFFT=256, Fs=self.fs, 
                                                    noverlap=200, cmap='inferno')
        self.ax_spec.set_title("2. Spektrogramm (Bereinigt)", fontsize=12, fontweight='bold')
        self.ax_spec.set_ylim(0, 600) 
        
        self.fig.canvas.draw_idle()

    def play_sound(self, event):
        if self.current_data is None: return
        audio_data = self.current_data / np.max(np.abs(self.current_data)) * 0.8
        playback_rate = int(self.fs * 2.0) 
        filename = "temp_sound.wav"
        wavfile.write(filename, playback_rate, audio_data.astype(np.float32))
        try:
            if platform.system() == "Darwin": subprocess.run(['afplay', filename], check=False)
            elif platform.system() == "Linux":
                try: subprocess.run(['aplay', filename], check=False)
                except: subprocess.run(['paplay', filename], check=False)
            elif platform.system() == "Windows":
                import winsound
                winsound.PlaySound(filename, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e: print(f"Audio Fehler: {e}")

    def update_threshold(self, val):
        self.detection_threshold = val
        if self.prediction is not None: self.update_confidence_display(self.prediction)

    def generate_new_sample(self, event):
        self.is_real_data = False
        self.event_name = "Simulation"
        noise = generate_gaussian_noise(len(self.time), noise_level=1.0)
        self.has_signal = np.random.choice([True, False])
        
        if self.has_signal:
            t_merger = np.random.uniform(2.0, 3.8)
            snr = np.random.uniform(0.5, 2.0) 
            sig_raw, _ = generate_astrophysical_chirp(self.time, t_merger=t_merger)
            self.pure_signal = sig_raw * snr
            self.current_data = noise + self.pure_signal
            self.true_snr = snr
        else:
            self.pure_signal = np.zeros_like(noise)
            self.current_data = noise
            self.true_snr = 0
            
        self.update_plot_threadsafe(title="Simulierte Daten")

    def ask_ai(self, event):
        if self.prediction is not None: return
        
        input_data = self.current_data.reshape(1, len(self.current_data), 1)
        input_data = (input_data - np.mean(input_data)) / np.std(input_data)
        
        self.prediction = self.model.predict(input_data)[0][0]
        self.update_confidence_display(self.prediction)
        
        if not self.is_real_data and self.has_signal:
            self.ax_signal.plot(self.time, self.pure_signal, color='cyan', lw=2, alpha=0.8, label='Signal')
            self.ax_signal.legend(loc='upper right')
        
        if self.is_real_data:
            if self.has_signal:
                truth_text = f"WAHRHEIT: {self.event_name}"
                box_color = 'gold'
            else:
                truth_text = "WAHRHEIT: Echtes Rauschen"
                box_color = 'lightgray'
        else:
            truth_text = "WAHRHEIT: " + ("SIGNAL" if self.has_signal else "RAUSCHEN")
            if self.has_signal: truth_text += f" (SNR: {self.true_snr:.2f})"
            is_detection = self.prediction > self.detection_threshold
            success = (is_detection == self.has_signal)
            box_color = 'lightgreen' if success else 'lightcoral'
        
        self.truth_text_artist = self.fig.text(0.5, 0.95, truth_text, ha='center', fontsize=12, 
                                              bbox=dict(facecolor=box_color, alpha=0.8, boxstyle='round'))
        self.fig.canvas.draw_idle()

    def update_confidence_display(self, pred_prob):
        self.ax_confidence.clear()
        self.ax_confidence.axis('off')
        threshold = self.detection_threshold
        
        if pred_prob > threshold:
            decision = "ALARM: SIGNAL ENTDECKT!"
            bar_color = 'red'
        elif pred_prob > 0.5:
            decision = "UNSICHER (Ignoriert)"
            bar_color = 'orange'
        else:
            decision = "RUHIG: Nur Rauschen"
            bar_color = 'blue'
            
        self.ax_confidence.axvline(threshold, color='black', linestyle='--', alpha=0.5)
        self.ax_confidence.text(threshold, -0.6, f'{threshold:.0%}', ha='center', fontsize=8)

        self.ax_confidence.barh([0], [pred_prob], color=bar_color, height=0.5, alpha=0.8)
        self.ax_confidence.barh([0], [1.0], color='gray', height=0.5, alpha=0.1)
        self.ax_confidence.set_xlim(0, 1)
        self.ax_confidence.set_ylim(-0.5, 0.5)
        
        percent = pred_prob * 100
        self.ax_confidence.text(0.5, 0.6, f"{decision}", ha='center', fontsize=16, fontweight='bold', color=bar_color)
        self.ax_confidence.text(0.5, -0.3, f"KI Konfidenz: {percent:.2f}%", ha='center', fontsize=12)
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    app = GlitchHunterApp()
    plt.show()