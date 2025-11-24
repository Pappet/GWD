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
from pycbc.psd import welch, interpolate

# Imports aus dem Core
sys.path.append(os.path.dirname(__file__))
from gwd_core.noise import generate_colored_noise  # FIXED: Farbiges Rauschen!
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
    hp, hc = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=mass1,
        mass2=mass2,
        delta_t=1.0/FS,
        f_lower=F_LOWER,
        distance=1  # Explizit (obwohl Standard)
    )
    
    # Nur Plus-Polarisation nutzen
    template = hp
    
    # FIXED: Bessere Längen-Anpassung
    target_len = int(DURATION * FS)
    if len(template) < target_len:
        # Mit Nullen auffüllen
        template.resize(target_len)
    elif len(template) > target_len:
        # Abschneiden
        template = template[:target_len]
    
    # Zyklisches Verschieben (PyCBC Standard für FFT-Effizienz)
    template = template.cyclic_time_shift(template.start_time)
    
    return template

def estimate_psd(data_array, sample_rate):
    """
    FIXED: Schätzt das PSD aus den Daten für korrektes Matched Filter Weighting.
    
    Args:
        data_array: Numpy Array mit Strain-Daten
        sample_rate: Abtastrate in Hz
    
    Returns:
        PyCBC FrequencySeries mit PSD
    """
    # In TimeSeries konvertieren
    strain = TimeSeries(data_array, delta_t=1.0/sample_rate)
    
    # Welch-Methode für PSD-Schätzung
    # seg_len: Länge der Segmente (in Samples)
    # seg_stride: Überlappung zwischen Segmenten
    seg_len = int(sample_rate * 0.5)  # 0.5 Sekunden pro Segment
    seg_stride = int(seg_len / 2)  # 50% Überlappung
    
    psd = welch(
        strain,
        seg_len=seg_len,
        seg_stride=seg_stride,
        avg_method='median'
    )
    
    # Interpoliere PSD auf die benötigte Frequenzauflösung
    delta_f = 1.0 / strain.duration
    psd = interpolate(psd, delta_f)
    
    return psd

def run_matched_filter(data_array, template, use_psd=True):
    """
    FIXED: Führt den Matched Filter mit korrektem PSD durch.
    
    Args:
        data_array: Numpy Array mit Daten
        template: PyCBC TimeSeries Template
        use_psd: Falls True, wird PSD aus Daten geschätzt
    
    Returns:
        Maximales SNR
    """
    # Daten in PyCBC TimeSeries wandeln
    strain = TimeSeries(data_array, delta_t=1.0/FS)
    
    # Template an Datenlänge anpassen
    if len(template) != len(strain):
        if len(template) < len(strain):
            template.resize(len(strain))
        else:
            template = template[:len(strain)]
    
    # FIXED: PSD berechnen oder None verwenden
    if use_psd:
        # Für farbiges Rauschen: PSD schätzen
        psd = estimate_psd(data_array, FS)
    else:
        # Für weißes Rauschen: kein PSD (nur zum Test)
        psd = None
    
    # Filter ausführen
    snr = matched_filter(
        template, 
        strain,
        psd=psd,  # FIXED: Jetzt mit PSD!
        low_frequency_cutoff=F_LOWER
    )
    
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
    
    # FIXED: Template-Bank statt nur einem Template
    # Wir testen mehrere Massen-Kombinationen
    template_bank = [
        (15, 15),
        (30, 30),
        (50, 50)
    ]
    
    templates = [get_template(m1, m2) for m1, m2 in template_bank]
    print(f"   Template-Bank: {len(templates)} Templates")
    
    # Test-Parameter
    snr_levels = np.linspace(0.0, 2.5, 10) 
    detection_rates = []
    MF_THRESHOLD = 5.0  # Physik-Standard für "sichere Entdeckung"
    
    # Loop über verschiedene Signal-Stärken
    for signal_snr in snr_levels:
        detected = 0
        n_batch = 25  # Anzahl Tests pro Level
        
        print(f"   ... Prüfe Injected SNR {signal_snr:.1f}", end="\r")
        
        for _ in range(n_batch):
            # 1. FIXED: Farbiges Rauschen erzeugen (realistischer!)
            noise = generate_colored_noise(len(time), FS, 1.0)
            
            # 2. Signal injizieren mit zufälligen Parametern
            t_m = np.random.uniform(1.5, 2.5)
            # FIXED: Zufällige Massen aus der Template-Bank wählen
            m1, m2 = template_bank[np.random.randint(0, len(template_bank))]
            
            sig, _ = generate_astrophysical_chirp(
                time, 
                t_merger=t_m, 
                mass1=m1, 
                mass2=m2,
                normalize=True  # Für konsistente SNR-Berechnung
            )
            
            # Signal skalieren und addieren
            data = noise + (sig * signal_snr)
            
            # 3. FIXED: Mit allen Templates testen (Template-Bank)
            best_snr = 0
            for template in templates:
                found_snr = run_matched_filter(data, template, use_psd=True)
                if found_snr > best_snr:
                    best_snr = found_snr
            
            if best_snr >= MF_THRESHOLD:
                detected += 1
                
        detection_rates.append(detected / n_batch)
    
    print("\n✅ Berechnung abgeschlossen.")
    
    # --- Metriken berechnen (SNR50 / SNR90) ---
    arr_rates = np.array(detection_rates)
    baseline_snr50 = -1.0
    baseline_snr90 = -1.0
    
    try:
        # Finde den ersten Index, wo Rate > 0.5 bzw 0.9 ist
        idx_50 = np.where(arr_rates >= 0.5)[0]
        if len(idx_50) > 0:
            baseline_snr50 = float(snr_levels[idx_50[0]])
        
        idx_90 = np.where(arr_rates >= 0.9)[0]
        if len(idx_90) > 0:
            baseline_snr90 = float(snr_levels[idx_90[0]])
    except IndexError:
        print("⚠️ Warnung: Konnte SNR50/90 nicht voll bestimmen (Bereich zu klein?).")

    # --- Speichern für das Leaderboard ---
    baseline_data = {
        "method": "Matched Filter (IMRPhenomD + PSD)",
        "threshold": MF_THRESHOLD,
        "template_bank": template_bank,
        "noise_type": "colored (seismic)",
        "SNR50": baseline_snr50,
        "SNR90": baseline_snr90,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "note": "Verbessert: Template-Bank, farbiges Rauschen, PSD-Weighting"
    }
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(BASELINE_FILE, "w") as f:
        json.dump(baseline_data, f, indent=4)
        
    print(f"💾 Baseline gespeichert in: {BASELINE_FILE}")
    print(f"   -> Physik SNR50 Limit: {baseline_snr50:.2f}")
    print(f"   -> Physik SNR90 Limit: {baseline_snr90:.2f}")

    return snr_levels, detection_rates

def compare_with_without_psd():
    """
    Zusätzlicher Test: Vergleich mit/ohne PSD um den Effekt zu zeigen.
    """
    print("\n📊 Bonus: Vergleich mit/ohne PSD...")
    
    time = np.linspace(0, DURATION, int(FS * DURATION))
    template = get_template(30, 30)
    
    snr_levels = np.linspace(0.5, 2.0, 8)
    rates_with_psd = []
    rates_without_psd = []
    
    for signal_snr in snr_levels:
        detected_with = 0
        detected_without = 0
        n_test = 20
        
        for _ in range(n_test):
            noise = generate_colored_noise(len(time), FS, 1.0)
            sig, _ = generate_astrophysical_chirp(time, t_merger=2.0, mass1=30, mass2=30)
            data = noise + (sig * signal_snr)
            
            # Mit PSD
            snr_with = run_matched_filter(data, template, use_psd=True)
            if snr_with >= 5.0:
                detected_with += 1
            
            # Ohne PSD
            snr_without = run_matched_filter(data, template, use_psd=False)
            if snr_without >= 5.0:
                detected_without += 1
        
        rates_with_psd.append(detected_with / n_test)
        rates_without_psd.append(detected_without / n_test)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(snr_levels, rates_with_psd, 'o-', label='Mit PSD (korrekt)', color='green', lw=2)
    plt.plot(snr_levels, rates_without_psd, 's--', label='Ohne PSD (suboptimal)', color='red', lw=2)
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(0.9, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel("Injected SNR")
    plt.ylabel("Detection Probability")
    plt.title("Effekt von PSD-Weighting bei farbigem Rauschen")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Hauptanalyse
    levels, rates = evaluate_sensitivity_classic()
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(levels, rates, 'o-', color='black', label='Matched Filter (Physik)', lw=2)
    plt.axhline(0.5, color='gray', linestyle='--', label='50% Detection')
    plt.axhline(0.9, color='blue', linestyle='--', label='90% Detection')
    plt.xlabel("Injected SNR")
    plt.ylabel("Detection Probability")
    plt.title("Physikalische Sensitivität (Baseline) - VERBESSERT")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Bonus-Analyse
    print("\n❓ Möchtest du den PSD-Effekt sehen? (Drücke Enter)")
    input()
    compare_with_without_psd()