#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODELS_DIR, LEADERBOARD_FILE, EVALUATION_MODELS_PLOTS_DIR

import glob
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, 
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from gwpy.timeseries import TimeSeries

# Imports aus dem Core
sys.path.append(os.path.dirname(__file__))
from gwd_core.waveforms import generate_astrophysical_chirp
from gwd_core.noise import generate_gaussian_noise

# ==========================================
# KONFIGURATION
# ==========================================
BASELINE_FILE = os.path.join(MODELS_DIR, "physics_baseline.json")
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Parameter für die Tests
FS = 4096
DURATION = 4.0
THRESHOLD = 0.15  # Nur für Vergleich, ROC ist threshold-unabhängig

# Echte Events für den "Feld-Test"
REAL_EVENTS = {
    "GW150914": 1126259462.4,
    "GW170817": 1187008882.4,
    "GW190521": 1242442967.4,
    "GW170104": 1167559936.6,
    "GW151226": 1135136350.6,
}

# Ein Stück echtes Rauschen (Hanford, O3 Run)
REAL_NOISE_START = 1253326755 
REAL_NOISE_DUR = 200

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
        
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def preprocess_real_data(strain):
    """Signalverarbeitung für echte LIGO-Daten."""
    if strain.sample_rate.value != FS:
        strain = strain.resample(FS)
    
    # Cleaning: Notch & Bandpass
    strain = strain.notch(60).notch(120).notch(180)
    white_data = strain.bandpass(20, 300)
    
    return white_data.value

def calculate_operating_point(y_true, y_scores, threshold):
    """
    Berechnet TPR, FPR für einen spezifischen Threshold.
    Nützlich um den gewählten Arbeitspunkt in der ROC-Kurve zu markieren.
    """
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return tpr, fpr

# ==========================================
# TEST 1: SIMULATION MIT ROC-ANALYSE
# ==========================================
def evaluate_simulation_with_roc(model):
    print("\n🧪 [LABOR] Starte ROC-Analyse auf Simulationsdaten...")
    time = np.linspace(0, DURATION, int(FS * DURATION))
    
    # --- A. ROC-Kurve erstellen (Threshold-unabhängig!) ---
    print("   📊 Generiere Test-Set für ROC-Analyse...")
    n_roc_samples = 1000  # Große Stichprobe für stabile ROC
    
    X_roc = []
    y_true_roc = []
    
    for _ in range(n_roc_samples):
        has_sig = np.random.rand() > 0.5
        noise = generate_gaussian_noise(len(time), 1.0)
        
        if has_sig:
            snr = np.random.uniform(0.5, 2.5)  # Breiter Bereich
            t_m = np.random.uniform(2.0, 3.5)
            sig, _ = generate_astrophysical_chirp(time, t_merger=t_m)
            data = noise + (sig * snr)
        else:
            data = noise
            
        data = (data - np.mean(data)) / np.std(data)
        X_roc.append(data)
        y_true_roc.append(1 if has_sig else 0)
    
    X_roc = np.array(X_roc).reshape(n_roc_samples, len(time), 1)
    y_true_roc = np.array(y_true_roc)
    
    # Predictions (Wahrscheinlichkeiten, nicht binär!)
    print("   🧠 Berechne Modell-Predictions...")
    y_scores = model.predict(X_roc, verbose=0).flatten()
    
    # ROC-Kurve berechnen
    fpr, tpr, thresholds = roc_curve(y_true_roc, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall Kurve (Alternative Darstellung)
    precision, recall, pr_thresholds = precision_recall_curve(y_true_roc, y_scores)
    avg_precision = average_precision_score(y_true_roc, y_scores)
    
    # Aktueller Operating Point (bei THRESHOLD=0.75)
    op_tpr, op_fpr = calculate_operating_point(y_true_roc, y_scores, THRESHOLD)
    
    print(f"   ✅ ROC AUC: {roc_auc:.4f}")
    print(f"   ✅ Average Precision: {avg_precision:.4f}")
    print(f"   📍 Operating Point (Threshold={THRESHOLD}): TPR={op_tpr:.2%}, FPR={op_fpr:.2%}")
    
    # --- B. Efficiency Curve (wie vorher, aber ergänzend) ---
    print("   📈 Berechne Sensitivity Curve...")
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
            data = (data - np.mean(data)) / np.std(data)
            X_batch.append(data)
            
        X = np.array(X_batch).reshape(n_batch, len(time), 1)
        preds = model.predict(X, verbose=0).flatten()
        detection_rates.append(np.mean(preds > THRESHOLD))

    # SNR50 / SNR90 bestimmen
    snr_50 = None
    snr_90 = None
    try:
        arr_rates = np.array(detection_rates)
        idx50 = np.where(arr_rates >= 0.5)[0]
        idx90 = np.where(arr_rates >= 0.9)[0]
        
        if len(idx50) > 0: snr_50 = snr_levels[idx50[0]]
        if len(idx90) > 0: snr_90 = snr_levels[idx90[0]]
    except:
        pass
    
    # --- C. Confusion Matrix bei gewähltem Threshold ---
    y_pred = (y_scores > THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_roc, y_pred).ravel()
    acc = (tp + tn) / n_roc_samples
    far = fp / (fp + tn)
    
    # --- PLOTTING ---
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. ROC-Kurve (HAUPTPLOT!)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Zufall')
    ax1.plot(op_fpr, op_tpr, 'ro', markersize=10, label=f'Operating Point (τ={THRESHOLD})')
    ax1.set_xlabel('False Positive Rate', fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontweight='bold')
    ax1.set_title('ROC-Kurve (Threshold-unabhängig)', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision-Recall Kurve
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(recall, precision, color='green', lw=2, 
             label=f'PR (AP = {avg_precision:.3f})')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Recall (TPR)', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Precision-Recall Kurve', fontweight='bold')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 2])
    ConfusionMatrixDisplay.from_predictions(y_true_roc, y_pred, ax=ax3, 
                                           colorbar=False, cmap='Blues')
    ax3.set_title(f'Confusion Matrix (τ={THRESHOLD})', fontweight='bold')
    
    # 4. Sensitivity Curve
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(snr_levels, detection_rates, 'o-', color='purple', lw=2)
    ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax4.axhline(0.9, color='blue', linestyle='--', alpha=0.5, label='90%')
    if snr_50: ax4.axvline(snr_50, color='orange', linestyle=':', label=f'SNR50={snr_50:.2f}')
    if snr_90: ax4.axvline(snr_90, color='red', linestyle=':', label=f'SNR90={snr_90:.2f}')
    ax4.set_xlabel('Injected SNR', fontweight='bold')
    ax4.set_ylabel('Detection Probability', fontweight='bold')
    ax4.set_title('Sensitivity Curve', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Score Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    scores_signal = y_scores[y_true_roc == 1]
    scores_noise = y_scores[y_true_roc == 0]
    ax5.hist(scores_noise, bins=30, alpha=0.5, label='Noise', color='blue', density=True)
    ax5.hist(scores_signal, bins=30, alpha=0.5, label='Signal', color='red', density=True)
    ax5.axvline(THRESHOLD, color='black', linestyle='--', lw=2, label=f'Threshold={THRESHOLD}')
    ax5.set_xlabel('Model Score', fontweight='bold')
    ax5.set_ylabel('Density', fontweight='bold')
    ax5.set_title('Score Distributions', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Metriken-Tabelle
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    metrics_text = f"""
    ╔═══════════════════════════╗
    ║   PERFORMANCE METRICS     ║
    ╠═══════════════════════════╣
    ║ ROC AUC:        {roc_auc:.4f}   ║
    ║ Avg Precision:  {avg_precision:.4f}   ║
    ║                           ║
    ║ Accuracy:       {acc:.2%}    ║
    ║ TPR (Recall):   {op_tpr:.2%}    ║
    ║ FPR:            {op_fpr:.2%}    ║
    ║ False Alarms:   {far:.2%}    ║
    ║                           ║
    ║ SNR50:          {snr_50 if snr_50 else 'N/A'}      ║
    ║ SNR90:          {snr_90 if snr_90 else 'N/A'}      ║
    ╚═══════════════════════════╝
    """
    
    ax6.text(0.1, 0.5, metrics_text, fontfamily='monospace', fontsize=10,
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Comprehensive Model Evaluation', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(EVALUATION_MODELS_PLOTS_DIR, f'plot_evaluate_model_with_roc_{TIMESTAMP}.png'), dpi=150)
    print(f'📊 Plot: {os.path.join(EVALUATION_MODELS_PLOTS_DIR, f"plot_evaluate_model_with_roc_{TIMESTAMP}.png")}')
    plt.close()
    
    return {
        'acc': acc,
        'far': far,
        'snr50': snr_50,
        'snr90': snr_90,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'tpr': op_tpr,
        'fpr': op_fpr
    }

# ==========================================
# TEST 2: REAL DATA
# ==========================================
def evaluate_real_world(model):
    print("\n🌍 [REAL WORLD] Starte Test auf echten LIGO-Daten...")
    
    # --- A. Events finden ---
    found_events = 0
    event_scores = {}
    
    print("   Suche bekannte Events...", end=" ")
    for name, gps in REAL_EVENTS.items():
        try:
            t_start = int(gps - 2.0)
            t_end = int(gps + 2.0)
            
            strain = TimeSeries.fetch_open_data('H1', t_start, t_end, verbose=False)
            data = preprocess_real_data(strain)
            
            expected = int(FS * DURATION)
            if len(data) != expected: 
                data = np.resize(data, expected)
            data = (data - np.mean(data)) / np.std(data)
            
            pred = model.predict(data.reshape(1, expected, 1), verbose=0)[0][0]
            event_scores[name] = pred
            
            if pred > THRESHOLD:
                found_events += 1
        except Exception as e:
            print(f"\n⚠️ Fehler bei {name}: {e}")
            event_scores[name] = None
    
    print(f"-> {found_events}/{len(REAL_EVENTS)} gefunden.")
    
    # Ausgabe der Scores
    print("\n   Event Detection Scores:")
    for name, score in event_scores.items():
        if score is not None:
            status = "✓ DETECTED" if score > THRESHOLD else "✗ Missed"
            print(f"      {name:12s}: {score:.3f} {status}")

    # --- B. False Alarm Rate auf Rauschen ---
    print(f"\n   Prüfe {REAL_NOISE_DUR}s Hintergrundrauschen...", end=" ")
    false_alarms = 0
    total_samples = 0
    far_text = "N/A"
    
    try:
        long_strain = TimeSeries.fetch_open_data('H1', REAL_NOISE_START, 
                                                 REAL_NOISE_START + REAL_NOISE_DUR, 
                                                 verbose=False)
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
        print(f"\n⚠️ Konnte LIGO-Daten nicht laden: {e}")
    
    return {
        "found": found_events,
        "total": len(REAL_EVENTS),
        "far_text": far_text,
        "event_scores": event_scores
    }

# ==========================================
# LEADERBOARD UPDATE
# ==========================================
def update_leaderboard(model_name, sim_metrics, real_metrics):
    """Schreibt erweiterte Metriken in die CSV."""
    
    # Baseline Vergleich laden
    phys_gap = "-"
    try:
        with open(BASELINE_FILE, "r") as f:
            base_data = json.load(f)
            base_90 = base_data.get("SNR90", -1)
            
            if base_90 > 0 and sim_metrics['snr90'] is not None:
                gap = sim_metrics['snr90'] - base_90
                phys_gap = f"+{gap:.2f}"
    except FileNotFoundError:
        pass

    entry = {
        "Model": model_name,
        "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        
        # ROC Metriken (NEU!)
        "ROC_AUC": f"{sim_metrics['roc_auc']:.4f}",
        "Avg_Precision": f"{sim_metrics['avg_precision']:.4f}",
        
        # Klassische Metriken
        "Sim_Accuracy": f"{sim_metrics['acc']:.1%}",
        "TPR": f"{sim_metrics['tpr']:.2%}",
        "FPR": f"{sim_metrics['fpr']:.2%}",
        
        # Sensitivity
        "Sim_SNR50": f"{sim_metrics['snr50']:.2f}" if sim_metrics['snr50'] else "-",
        "Sim_SNR90": f"{sim_metrics['snr90']:.2f}" if sim_metrics['snr90'] else "-",
        "Physik_Gap": phys_gap,
        
        # Real World
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
    print(f"   -> ROC AUC: {sim_metrics['roc_auc']:.4f}")
    if phys_gap != "-":
        print(f"   -> Abstand zur Physik: {phys_gap} SNR")

# ==========================================
# MAIN
# ==========================================
def main():
    model_path = get_latest_model()
    if not model_path: 
        return

    print("="*70)
    print(f"🤖 COMPREHENSIVE EVALUATION: {os.path.basename(model_path)}")
    print("="*70)
    
    model = tf.keras.models.load_model(model_path)
    
    # 1. Simulation mit ROC
    sim_res = evaluate_simulation_with_roc(model)
    
    # 2. Real Data
    real_res = evaluate_real_world(model)
    
    # 3. Final Report
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY:")
    print("="*70)
    print(f"  ROC AUC:              {sim_res['roc_auc']:.4f}")
    print(f"  Average Precision:    {sim_res['avg_precision']:.4f}")
    print(f"  Accuracy:             {sim_res['acc']:.1%}")
    print(f"  TPR (Sensitivity):    {sim_res['tpr']:.2%}")
    print(f"  FPR:                  {sim_res['fpr']:.2%}")
    print(f"  SNR90:                {sim_res['snr90']}")
    print(f"  Real Events Found:    {real_res['found']}/{real_res['total']}")
    print(f"  False Alarm Rate:     {real_res['far_text']}")
    print("="*70)
    
    update_leaderboard(os.path.basename(model_path), sim_res, real_res)

if __name__ == "__main__":
    main()