#!/usr/bin/env python3
"""
ROC Comparison Tool - Vergleicht mehrere Modelle in einer ROC-Kurve
Nützlich um zu sehen, welches Modell threshold-unabhängig besser ist.
"""

import os
import sys
import glob
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_curve, auc

sys.path.append(os.path.dirname(__file__))
from gwd_core.waveforms import generate_astrophysical_chirp
from gwd_core.noise import generate_gaussian_noise

MODELS_DIR = "models_registry"
FS = 4096
DURATION = 4.0
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def load_all_models():
    """Lädt alle verfügbaren Modelle."""
    if not os.path.exists(MODELS_DIR):
        print(f"❌ Ordner '{MODELS_DIR}' nicht gefunden.")
        return []
    
    files = glob.glob(os.path.join(MODELS_DIR, "*.keras"))
    if not files:
        print("❌ Keine Modelle gefunden.")
        return []
    
    models = []
    for f in sorted(files, key=os.path.getctime):
        try:
            model = tf.keras.models.load_model(f)
            name = os.path.basename(f).replace('.keras', '')
            models.append((name, model))
            print(f"✓ Geladen: {name}")
        except Exception as e:
            print(f"✗ Fehler bei {f}: {e}")
    
    return models

def generate_test_set(n_samples=1000):
    """Generiert einen festen Test-Set für faire Vergleiche."""
    print(f"\n📊 Generiere {n_samples} Test-Samples...")
    
    time = np.linspace(0, DURATION, int(FS * DURATION))
    X_test = []
    y_true = []
    
    for i in range(n_samples):
        has_sig = np.random.rand() > 0.5
        noise = generate_gaussian_noise(len(time), 1.0)
        
        if has_sig:
            snr = np.random.uniform(0.5, 2.5)
            t_m = np.random.uniform(2.0, 3.5)
            sig, _ = generate_astrophysical_chirp(time, t_merger=t_m)
            data = noise + (sig * snr)
        else:
            data = noise
        
        data = (data - np.mean(data)) / np.std(data)
        X_test.append(data)
        y_true.append(1 if has_sig else 0)
        
        if (i+1) % 200 == 0:
            print(f"   ... {i+1}/{n_samples}")
    
    return np.array(X_test).reshape(n_samples, len(time), 1), np.array(y_true)

def compare_models_roc(models, X_test, y_true):
    """Vergleicht mehrere Modelle in einer ROC-Kurve."""
    
    print("\n🧠 Berechne ROC-Kurven für alle Modelle...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Farben für die Modelle
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    auc_scores = []
    
    for (name, model), color in zip(models, colors):
        print(f"   → {name}...")
        
        # Predictions
        y_scores = model.predict(X_test, verbose=0).flatten()
        
        # ROC berechnen
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        auc_scores.append((name, roc_auc))
        
        # Plotten
        ax1.plot(fpr, tpr, color=color, lw=2, 
                label=f'{name[:30]}... (AUC={roc_auc:.3f})')
    
    # Zufall-Linie
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random Classifier')
    
    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # --- AUC Bar Chart ---
    auc_scores_sorted = sorted(auc_scores, key=lambda x: x[1], reverse=True)
    names = [x[0][:30] for x in auc_scores_sorted]
    aucs = [x[1] for x in auc_scores_sorted]
    
    bars = ax2.barh(range(len(names)), aucs, color=colors[:len(names)])
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel('ROC AUC', fontsize=12, fontweight='bold')
    ax2.set_title('Model Ranking by AUC', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Werte in die Bars schreiben
    for i, (bar, auc_val) in enumerate(zip(bars, aucs)):
        ax2.text(auc_val - 0.05, i, f'{auc_val:.4f}', 
                va='center', ha='right', fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(f'plot_compare_roc_{TIMESTAMP}.png', dpi=150)
    print(f'📊 Plot: plot_compare_roc_{TIMESTAMP}.png')
    plt.close()
    
    return auc_scores_sorted

def zoom_to_relevant_region(models, X_test, y_true):
    """
    BONUS: Zoomed ROC Plot
    Im GW-Bereich sind wir meist bei sehr niedriger FPR interessiert.
    Ein Zoom auf FPR < 0.1 zeigt die relevanten Unterschiede besser.
    """
    print("\n🔍 Erstelle Zoom auf relevante Region (FPR < 0.1)...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for (name, model), color in zip(models, colors):
        y_scores = model.predict(X_test, verbose=0).flatten()
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Nur FPR < 0.1 plotten
        mask = fpr <= 0.1
        ax.plot(fpr[mask], tpr[mask], color=color, lw=2, marker='o', 
                markersize=3, label=f'{name[:30]}... (AUC={roc_auc:.3f})')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve - Zoom auf relevante Region (FPR < 0.1)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'plot_compare_roc_{TIMESTAMP}.png', dpi=150)
    print(f'📊 Plot: plot_compare_roc_{TIMESTAMP}.png')
    plt.close()

def main():
    print("="*70)
    print("🏆 ROC CURVE COMPARISON TOOL")
    print("="*70)
    
    # 1. Modelle laden
    models = load_all_models()
    
    if len(models) == 0:
        print("\n❌ Keine Modelle zum Vergleichen gefunden.")
        return
    
    if len(models) == 1:
        print("\n⚠️ Nur ein Modell gefunden. Trainiere mehr Modelle für Vergleiche!")
        return
    
    print(f"\n✅ {len(models)} Modelle geladen.")
    
    # 2. Test-Set generieren
    X_test, y_true = generate_test_set(n_samples=1000)
    
    # 3. Vergleich durchführen
    rankings = compare_models_roc(models, X_test, y_true)
    
    # 4. Ranking ausgeben
    print("\n" + "="*70)
    print("📊 FINAL RANKING:")
    print("="*70)
    for i, (name, auc_val) in enumerate(rankings, 1):
        print(f"  {i}. {name[:50]:<50s} | AUC: {auc_val:.4f}")
    print("="*70)
    
    # 5. Bonus: Zoom
    if len(models) > 1:
        print("\n💡 Erstelle zusätzlichen Zoom-Plot für niedrige FPR...")
        zoom_to_relevant_region(models, X_test, y_true)

if __name__ == "__main__":
    main()