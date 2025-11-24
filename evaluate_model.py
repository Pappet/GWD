#!/usr/bin/env python3
import os
import sys
import glob
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from gwpy.timeseries import TimeSeries

# Imports aus dem Core
sys.path.append(os.path.dirname(__file__))
from gwd_core.waveforms import generate_astrophysical_chirp
from gwd_core.noise import generate_gaussian_noise

# ==========================================
# KONFIGURATION
# ==========================================
MODELS_DIR = "models_registry"
LEADERBOARD_FILE = "model_leaderboard.csv"
BASELINE_FILE = os.path.join(MODELS_DIR, "physics_baseline.json")

# Parameter für die Tests
FS = 4096
DURATION = 4.0
THRESHOLD = 0.75  # Ab wann schlägt der Alarm an?

# Echte Events für den "Feld-Test"
REAL_EVENTS = {
    "GW150914": 1126259462.4, # Stark
    "GW170817": 1187008882.4, # Lang & anders
    "GW190521": 1242442967.4, # Kurz
    "GW170104": 1167559936.6, # Mittel
    "GW151226": 1135136350.6, # Schwächer
}

# Ein Stück echtes Rauschen (Hanford, O3 Run)
REAL_NOISE_START = 1253326755 
REAL_NOISE_DUR = 200 # Sekunden

# ==========================================
# HELFER-FUNKTIONEN
# ==========================================

def get_latest_model():
    """Sucht das neueste Modell im Registry-Ordner."""
    if not os.path.exists(MODELS_DIR):
        print(f"❌ Ordner '{MODELS_DIR}' nicht gefunden.")
        return None
    
    files = glob.glob(os.path.join(MODELS_DIR, "*.keras"))
    if not files:
        print("❌ Keine Modelle (.keras) gefunden.")
        return None
        
    # Neueste Datei basierend auf Erstellungsdatum
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def preprocess_real_data(strain):
    """WICHTIG: Die Signalverarbeitung für echte LIGO-Daten."""
    if strain.sample_rate.value != FS:
        strain = strain.resample(FS)
    
    # Cleaning: Notch & Bandpass
    strain = strain.notch(60).notch(120).notch(180)
    white_data = strain.bandpass(20, 300)
    
    return white_data.value

def update_leaderboard(model_name, sim_metrics, real_metrics):
    """Schreibt alle Ergebnisse in die CSV-Historie."""
    
    # --- NEU: Baseline Vergleich laden ---
    phys_gap = "-"
    try:
        with open(BASELINE_FILE, "r") as f:
            base_data = json.load(f)
            base_90 = base_data.get("SNR90", -1)
            
            # Wenn wir beide Werte haben, berechnen wir den Abstand
            if base_90 > 0 and sim_metrics['snr90'] is not None:
                gap = sim_metrics['snr90'] - base_90
                # Formatierung: +0.50 heißt "Modell braucht 0.5 mehr SNR als Physik"
                phys_gap = f"+{gap:.2f}"
    except FileNotFoundError:
        pass # Noch keine Baseline berechnet

    entry = {
        "Model": model_name,
        "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        
        # Simulation Metrics
        "Sim_Accuracy": f"{sim_metrics['acc']:.1%}",
        "Sim_SNR50": f"{sim_metrics['snr50']:.2f}" if sim_metrics['snr50'] else "-",
        "Sim_SNR90": f"{sim_metrics['snr90']:.2f}" if sim_metrics['snr90'] else "-",
        
        # NEU: Der wissenschaftliche Vergleichswert
        "Physik_Gap": phys_gap,
        
        "Sim_FalseAlarm": f"{sim_metrics['far']:.2%}",
        
        # Real World Metrics
        "Real_Events_Found": f"{real_metrics['found']}/{real_metrics['total']}",
        "Real_Noise_FAR": real_metrics['far_text']
    }
    
    df_new = pd.DataFrame([entry])
    
    if os.path.exists(LEADERBOARD_FILE):
        df_old = pd.read_csv(LEADERBOARD_FILE)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.to_csv(LEADERBOARD_FILE, index=False)
    else:
        df_new.to_csv(LEADERBOARD_FILE, index=False)
        
    print(f"\n🏆 Leaderboard aktualisiert: {LEADERBOARD_FILE}")
    if phys_gap != "-":
        print(f"   -> Abstand zur Physik-Grenze: {phys_gap} SNR-Einheiten")

# ==========================================
# TEST 1: SIMULATION (Das Labor)
# ==========================================
def evaluate_simulation(model):
    print("\n🧪 [LABOR] Starte Simulationstests...")
    time = np.linspace(0, DURATION, int(FS * DURATION))
    
    # --- A. Efficiency Curve (Sensitivität) ---
    snr_levels = np.linspace(0.0, 2.5, 15)
    detection_rates = []
    
    for snr in snr_levels:
        detected = 0
        n_batch = 50
        X_batch = []
        
        for _ in range(n_batch):
            noise = generate_gaussian_noise(len(time), 1.0)
            t_m = np.random.uniform(2.0, 3.5)
            sig, _ = generate_astrophysical_chirp(time, t_merger=t_m)
            data = noise + (sig * snr)
            data = (data - np.mean(data)) / np.std(data) # Standardize
            X_batch.append(data)
            
        X = np.array(X_batch).reshape(n_batch, len(time), 1)
        preds = model.predict(X, verbose=0).flatten()
        detection_rates.append(np.mean(preds > THRESHOLD))

    # Metriken extrahieren
    snr_50 = None
    snr_90 = None
    try:
        arr_rates = np.array(detection_rates)
        # Finde ersten Index über 0.5 bzw 0.9
        idx50 = np.where(arr_rates >= 0.5)[0]
        idx90 = np.where(arr_rates >= 0.9)[0]
        
        if len(idx50) > 0: snr_50 = snr_levels[idx50[0]]
        if len(idx90) > 0: snr_90 = snr_levels[idx90[0]]
    except:
        pass
        
    # --- B. Stress Test (False Alarms) ---
    n_stress = 500
    y_true = []
    X_stress = []
    
    for _ in range(n_stress):
        has_sig = np.random.rand() > 0.5
        noise = generate_gaussian_noise(len(time), 1.0)
        if has_sig:
            snr = np.random.uniform(0.8, 2.0)
            sig, _ = generate_astrophysical_chirp(time, t_merger=3.0)
            data = noise + (sig * snr)
        else:
            data = noise
            
        data = (data - np.mean(data)) / np.std(data)
        X_stress.append(data)
        y_true.append(1 if has_sig else 0)
        
    preds_stress = model.predict(np.array(X_stress).reshape(n_stress, len(time), 1), verbose=0).flatten()
    y_pred = (preds_stress > THRESHOLD).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / n_stress
    far = fp / (fp + tn) # False Positive Rate
    
    # Plotting (Labor)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(snr_levels, detection_rates, 'o-')
    plt.axhline(0.5, color='gray', linestyle='--')
    plt.title("Sensitivity (Sim)")
    plt.xlabel("SNR")
    
    plt.subplot(1, 2, 2)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=plt.gca(), colorbar=False, cmap='Blues')
    plt.title("Confusion Matrix (Sim)")
    plt.tight_layout()
    plt.show()
    
    return {
        "acc": acc,
        "far": far,
        "snr50": snr_50,
        "snr90": snr_90
    }

# ==========================================
# TEST 2: REAL DATA (Die Wildnis)
# ==========================================
def evaluate_real_world(model):
    print("\n🌍 [REAL WORLD] Starte Test auf echten LIGO-Daten...")
    
    # --- A. Events finden ---
    found_events = 0
    print("   Suche bekannte Events...", end=" ")
    for name, gps in REAL_EVENTS.items():
        try:
            t_start = int(gps - 2.0)
            t_end = int(gps + 2.0)
            
            strain = TimeSeries.fetch_open_data('H1', t_start, t_end, verbose=False)
            data = preprocess_real_data(strain)
            
            # Fix Length & Standardize
            expected = int(FS * DURATION)
            if len(data) != expected: data = np.resize(data, expected)
            data = (data - np.mean(data)) / np.std(data)
            
            pred = model.predict(data.reshape(1, expected, 1), verbose=0)[0][0]
            if pred > THRESHOLD:
                found_events += 1
        except:
            pass
    print(f"-> {found_events}/{len(REAL_EVENTS)} gefunden.")

    # --- B. Gespenster jagen (Noise Test) ---
    print(f"   Prüfe {REAL_NOISE_DUR}s Hintergrundrauschen...", end=" ")
    false_alarms = 0
    total_samples = 0
    far_text = "N/A"
    
    try:
        long_strain = TimeSeries.fetch_open_data('H1', REAL_NOISE_START, REAL_NOISE_START + REAL_NOISE_DUR, verbose=False)
        clean_data = preprocess_real_data(long_strain)
        
        step = int(FS * DURATION)
        num_cuts = len(clean_data) // step
        
        for i in range(num_cuts):
            segment = clean_data[i*step : (i+1)*step]
            segment = (segment - np.mean(segment)) / np.std(segment)
            
            pred = model.predict(segment.reshape(1, step, 1), verbose=0)[0][0]
            if pred > THRESHOLD:
                false_alarms += 1
        
        total_samples = num_cuts
        
        # Berechnung Events pro Jahr
        analyzed_years = (total_samples * DURATION) / (365.25 * 24 * 3600)
        if false_alarms == 0:
            far_text = f"< {1/analyzed_years:.1f}/y"
        else:
            far_text = f"{false_alarms/analyzed_years:.1f}/y"
            
        print(f"-> {false_alarms} Fehler ({far_text})")
        
    except Exception as e:
        print(f"\n⚠️ Konnte LIGO-Daten nicht laden (Internet?): {e}")
    
    return {
        "found": found_events,
        "total": len(REAL_EVENTS),
        "far_text": far_text
    }

# ==========================================
# MAIN
# ==========================================
def main():
    model_path = get_latest_model()
    if not model_path: return

    print("="*60)
    print(f"🤖 EVALUATION REPORT FÜR: {os.path.basename(model_path)}")
    print("="*60)
    
    model = tf.keras.models.load_model(model_path)
    
    # 1. Simulation
    sim_res = evaluate_simulation(model)
    
    # 2. Real Data
    real_res = evaluate_real_world(model)
    
    # 3. Reporting
    print("\n" + "="*60)
    print("ZUSAMMENFASSUNG:")
    print(f"   Simulierte Genauigkeit:  {sim_res['acc']:.1%}")
    print(f"   Sensitivität (SNR 90%):  {sim_res['snr90']}")
    print(f"   Echte Events erkannt:    {real_res['found']} / {real_res['total']}")
    print(f"   False Alarm Rate (Real): {real_res['far_text']}")
    print("="*60)
    
    update_leaderboard(os.path.basename(model_path), sim_res, real_res)

if __name__ == "__main__":
    main()