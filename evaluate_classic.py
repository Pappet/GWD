#!/usr/bin/env python3
import os
import sys
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pycbc.types import TimeSeries
from pycbc.filter import matched_filter
from pycbc.waveform import get_td_waveform

# Imports aus dem Core
sys.path.append(os.path.dirname(__file__))
from gwd_core.noise import generate_gaussian_noise
from gwd_core.waveforms import generate_astrophysical_chirp

# ==========================================
# KONFIGURATION
# ==========================================
MODELS_DIR = "models_registry"
BASELINE_FILE = os.path.join(MODELS_DIR, "physics_baseline.json")

FS = 4096
DURATION = 4.0
F_LOWER = 20.0  # Untere Grenzfrequenz für den Filter

def get_template(mass1=30, mass2=30):
    """
    Erstellt die physikalische Schablone (Template) für die Suche.
    Nutzt IMRPhenomD, passend zur Simulation.
    """
    # Template generieren
    hp, hc = get_td_waveform(approximant="IMRPhenomD",
                             mass1=mass1,
                             mass2=mass2,
                             delta_t=1.0/FS,
                             f_lower=F_LOWER)
    
    # Nur Plus-Polarisation nutzen und auf Buffer-Größe anpassen
    template = hp
    template.resize(int(DURATION * FS))
    
    # Zyklisches Verschieben (PyCBC Standard für FFT-Effizienz)
    # Wir schieben den Start der Welle an den Anfang des Arrays
    template = template.cyclic_time_shift(template.start_time)
    
    return template

def run_matched_filter(data_array, template):
    """
    Führt den Matched Filter durch.
    Gibt das maximale SNR zurück (wie stark passt die Schablone?).
    """
    # Daten in PyCBC TimeSeries wandeln
    strain = TimeSeries(data_array, delta_t=1.0/FS)
    
    # Template an Datenlänge anpassen (falls abweichend)
    template.resize(len(strain))
    
    # Filter ausführen (Wir nehmen hier weißes Rauschen an -> psd=None)
    # Bei farbigem Rauschen müsste man hier ein PSD übergeben.
    snr = matched_filter(template, strain,
                         psd=None, 
                         low_frequency_cutoff=F_LOWER)
    
    # Ränder abschneiden (Einschwing-Artefakte)
    crop_sec = 0.5
    snr = snr.crop(crop_sec, crop_sec)
    
    # Das stärkste Signal im gesamten Zeitraum finden
    peak_snr = abs(snr).max()
    return peak_snr

def evaluate_sensitivity_classic():
    print("\n🧪 [PHYSIK-BASELINE] Starte Matched Filter Analyse...")
    print("   (Das kann einen Moment dauern, da wir echte Physik berechnen...)")
    
    time = np.linspace(0, DURATION, int(FS * DURATION))
    
    # Wir nutzen ein Standard-Template (30+30 M_sol)
    # Das passt perfekt zu den Standard-Parametern unserer Simulation.
    template = get_template(30, 30)
    
    # Test-Parameter
    snr_levels = np.linspace(0.0, 2.5, 10) 
    detection_rates = []
    MF_THRESHOLD = 5.0  # Physik-Standard für "sichere Entdeckung"
    
    # Loop über verschiedene Signal-Stärken
    for signal_snr in snr_levels:
        detected = 0
        n_batch = 25 # Anzahl Tests pro Level
        
        print(f"   ... Prüfe Injected SNR {signal_snr:.1f}", end="\r")
        
        for _ in range(n_batch):
            # 1. Rauschen erzeugen
            noise = generate_gaussian_noise(len(time), 1.0)
            
            # 2. Signal injizieren (mit zufälliger Position, aber passender Masse)
            t_m = np.random.uniform(1.5, 2.5)
            sig, _ = generate_astrophysical_chirp(time, t_merger=t_m, mass1=30, mass2=30)
            
            # Signal skalieren und addieren
            data = noise + (sig * signal_snr)
            
            # 3. Physik anwenden (Matched Filter)
            found_snr = run_matched_filter(data, template)
            
            if found_snr >= MF_THRESHOLD:
                detected += 1
                
        detection_rates.append(detected / n_batch)
    
    print("\n✅ Berechnung abgeschlossen.")
    
    # --- Metriken berechnen (SNR50 / SNR90) ---
    arr_rates = np.array(detection_rates)
    baseline_snr50 = -1.0
    baseline_snr90 = -1.0
    
    try:
        # Finde den ersten Index, wo Rate > 0.5 bzw 0.9 ist
        idx_50 = np.where(arr_rates >= 0.5)[0][0]
        baseline_snr50 = float(snr_levels[idx_50])
        
        idx_90 = np.where(arr_rates >= 0.9)[0][0]
        baseline_snr90 = float(snr_levels[idx_90])
    except IndexError:
        print("⚠️ Warnung: Konnte SNR50/90 nicht voll bestimmen (Bereich zu klein?).")

    # --- Speichern für das Leaderboard ---
    baseline_data = {
        "method": "Matched Filter (IMRPhenomD)",
        "threshold": MF_THRESHOLD,
        "SNR50": baseline_snr50,
        "SNR90": baseline_snr90,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "note": "Theoretisches Limit bei Gaussian Noise (White)"
    }
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(BASELINE_FILE, "w") as f:
        json.dump(baseline_data, f, indent=4)
        
    print(f"💾 Baseline gespeichert in: {BASELINE_FILE}")
    print(f"   -> Physik SNR90 Limit: {baseline_snr90:.2f}")

    return snr_levels, detection_rates

if __name__ == "__main__":
    levels, rates = evaluate_sensitivity_classic()
    
    # Kurzer Plot zur Bestätigung
    plt.figure(figsize=(8, 5))
    plt.plot(levels, rates, 'o-', color='black', label='Matched Filter (Physik)')
    plt.axhline(0.5, color='gray', linestyle='--')
    plt.xlabel("Injected SNR")
    plt.ylabel("Detection Probability")
    plt.title("Physikalische Sensitivität (Baseline)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()