#!/usr/bin/env python3
"""
Threshold Optimizer - Findet den optimalen Schwellenwert basierend auf ROC-Kurve

Verschiedene Optimierungs-Strategien:
1. Youden's J Statistic: Maximiert (TPR - FPR)
2. F1-Score: Harmonisches Mittel von Precision und Recall
3. Cost-based: Gewichtet Fehlertypen unterschiedlich
4. Fixed FPR: Maximiert TPR bei gegebenem FPR-Limit
"""

import datetime
import json
import os
import sys

import matplotlib
import numpy as np

from gwd.config import (
    EVALUATION_MODELS_PLOTS_DIR,
    OPTIMAL_THRESHOLDS_FILE,
    ensure_dirs,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve

from gwd.data.injection import make_test_set
from gwd.data.scenarios import EVAL_V1
from gwd.registry import load_model, resolve

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def find_optimal_thresholds(y_true, y_scores):
    """
    Berechnet optimale Thresholds nach verschiedenen Kriterien.

    Returns:
        dict mit verschiedenen Threshold-Vorschlägen
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)

    results = {}

    # 1. Youden's J Statistic (Maximiert TPR - FPR)
    j_scores = tpr - fpr
    j_max_idx = np.argmax(j_scores)
    results["youden"] = {
        "threshold": thresholds[j_max_idx],
        "tpr": tpr[j_max_idx],
        "fpr": fpr[j_max_idx],
        "j_stat": j_scores[j_max_idx],
        "description": "Maximiert (TPR - FPR) - Balanced approach",
    }

    # 2. Maximaler F1-Score
    # F1 = 2 * (precision * recall) / (precision + recall)
    f1_scores = []
    for t in pr_thresholds:
        y_pred = (y_scores >= t).astype(int)
        f1_scores.append(f1_score(y_true, y_pred))

    f1_scores = np.array(f1_scores)
    f1_max_idx = np.argmax(f1_scores)

    # Finde entsprechende TPR/FPR für diesen Threshold
    opt_threshold = pr_thresholds[f1_max_idx]
    opt_idx = np.argmin(np.abs(thresholds - opt_threshold))

    results["f1"] = {
        "threshold": opt_threshold,
        "tpr": tpr[opt_idx],
        "fpr": fpr[opt_idx],
        "f1_score": f1_scores[f1_max_idx],
        "description": "Maximiert F1-Score - Gut bei unbalancierten Daten",
    }

    # 3. Fixed FPR = 0.05 (5% False Alarm Rate)
    target_fpr = 0.05
    idx_fpr = np.argmin(np.abs(fpr - target_fpr))
    results["fixed_fpr_5"] = {
        "threshold": thresholds[idx_fpr],
        "tpr": tpr[idx_fpr],
        "fpr": fpr[idx_fpr],
        "description": f"Maximiert TPR bei FPR ≈ {target_fpr:.0%}",
    }

    # 4. Fixed FPR = 0.01 (1% False Alarm Rate - Konservativ)
    target_fpr = 0.01
    idx_fpr = np.argmin(np.abs(fpr - target_fpr))
    results["fixed_fpr_1"] = {
        "threshold": thresholds[idx_fpr],
        "tpr": tpr[idx_fpr],
        "fpr": fpr[idx_fpr],
        "description": f"Maximiert TPR bei FPR ≈ {target_fpr:.0%} - Konservativ",
    }

    # 5. Cost-based (Beispiel: False Negatives sind 5x schlimmer als False Positives)
    # Cost = FN_cost * FNR + FP_cost * FPR
    fn_cost = 5.0
    fp_cost = 1.0

    fnr = 1 - tpr  # False Negative Rate
    costs = fn_cost * fnr + fp_cost * fpr
    cost_min_idx = np.argmin(costs)

    results["cost_based"] = {
        "threshold": thresholds[cost_min_idx],
        "tpr": tpr[cost_min_idx],
        "fpr": fpr[cost_min_idx],
        "cost": costs[cost_min_idx],
        "description": f"Minimiert Kosten (FN:{fn_cost}x, FP:{fp_cost}x)",
    }

    # 6. TPR = 0.9 (90% Detection Rate garantiert)
    target_tpr = 0.90
    idx_tpr = np.argmin(np.abs(tpr - target_tpr))
    results["fixed_tpr_90"] = {
        "threshold": thresholds[idx_tpr],
        "tpr": tpr[idx_tpr],
        "fpr": fpr[idx_tpr],
        "description": f"Minimiert FPR bei TPR ≈ {target_tpr:.0%}",
    }

    return results, (fpr, tpr, thresholds)


def visualize_thresholds(y_true, y_scores, optimal_thresholds, roc_data):
    """Visualisiert verschiedene Threshold-Optionen in der ROC-Kurve."""

    fpr, tpr, thresholds = roc_data

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. ROC mit allen optimalen Punkten
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(fpr, tpr, "b-", lw=2, label="ROC Curve")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

    colors = ["red", "orange", "green", "cyan", "purple", "magenta"]
    markers = ["o", "s", "^", "v", "D", "P"]

    for (name, opt), color, marker in zip(optimal_thresholds.items(), colors, markers):
        ax1.plot(
            opt["fpr"],
            opt["tpr"],
            marker=marker,
            markersize=12,
            color=color,
            label=f"{name}: τ={opt['threshold']:.3f}",
        )

    ax1.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax1.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax1.set_title("ROC Curve mit optimalen Thresholds", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Threshold vs Metrics
    ax2 = fig.add_subplot(gs[1, 0])

    # TPR und FPR über Threshold
    ax2.plot(thresholds, tpr, "g-", lw=2, label="TPR (Sensitivity)")
    ax2.plot(thresholds, fpr, "r-", lw=2, label="FPR (False Alarms)")
    ax2.plot(thresholds, tpr - fpr, "b--", lw=2, label="Youden's J (TPR-FPR)")

    # Markiere optimale Punkte
    for name, opt in optimal_thresholds.items():
        ax2.axvline(opt["threshold"], color="gray", linestyle=":", alpha=0.5)

    ax2.set_xlabel("Threshold", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Rate", fontsize=12, fontweight="bold")
    ax2.set_title("Metriken vs Threshold", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # 3. Vergleichs-Tabelle
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")

    table_text = "╔════════════════════════════════════════════════╗\n"
    table_text += "║  OPTIMALE THRESHOLDS - VERGLEICH              ║\n"
    table_text += "╠════════════════════════════════════════════════╣\n"

    for name, opt in optimal_thresholds.items():
        table_text += f"║ {name.upper():<15s}                          ║\n"
        table_text += (
            f"║   Threshold:  {opt['threshold']:.4f}                       ║\n"
        )
        table_text += f"║   TPR:        {opt['tpr']:.2%}                          ║\n"
        table_text += f"║   FPR:        {opt['fpr']:.2%}                          ║\n"
        table_text += f"║   {opt['description']:<45s}║\n"
        table_text += "╠════════════════════════════════════════════════╣\n"

    table_text += "╚════════════════════════════════════════════════╝"

    ax3.text(
        0.05,
        0.95,
        table_text,
        transform=ax3.transAxes,
        fontfamily="monospace",
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )

    plt.suptitle("Threshold Optimization Analysis", fontsize=16, fontweight="bold")
    ensure_dirs(EVALUATION_MODELS_PLOTS_DIR)
    plt.savefig(
        os.path.join(
            EVALUATION_MODELS_PLOTS_DIR, f"plot_optimize_threshold_{TIMESTAMP}.png"
        ),
        dpi=150,
    )
    print(
        f"📊 Plot: {os.path.join(EVALUATION_MODELS_PLOTS_DIR, f'plot_optimize_threshold_{TIMESTAMP}.png')}"
    )
    plt.close()


def recommend_threshold(optimal_thresholds):
    """Gibt eine Empfehlung basierend auf typischen GW-Anwendungen."""

    print("\n" + "=" * 70)
    print("🎯 THRESHOLD-EMPFEHLUNGEN FÜR VERSCHIEDENE SZENARIEN")
    print("=" * 70)

    scenarios = [
        {
            "name": "🔬 Wissenschaftliche Analyse (Paper)",
            "recommendation": "fixed_fpr_1",
            "reason": "Niedriger False Alarm Rate wichtig für Glaubwürdigkeit",
        },
        {
            "name": "🚨 Trigger für Follow-up Analysen",
            "recommendation": "youden",
            "reason": "Ausgewogenes Verhältnis zwischen TPR und FPR",
        },
        {
            "name": "📊 Statistik-Studien (Große Datenmengen)",
            "recommendation": "f1",
            "reason": "Maximiert Gesamtqualität bei unbalancierten Daten",
        },
        {
            "name": "⚡ Real-time Alert System",
            "recommendation": "fixed_tpr_90",
            "reason": "Garantiert hohe Detection Rate, FPR kann später gefiltert werden",
        },
        {
            "name": "🎓 Lehre / Demo",
            "recommendation": "youden",
            "reason": "Intuitiv verstehbarer Optimierungspunkt",
        },
    ]

    for scenario in scenarios:
        rec = optimal_thresholds[scenario["recommendation"]]
        print(f"\n{scenario['name']}")
        print(f"  → Empfohlen: {scenario['recommendation'].upper()}")
        print(f"  → Threshold: {rec['threshold']:.4f}")
        print(f"  → TPR: {rec['tpr']:.2%} | FPR: {rec['fpr']:.2%}")
        print(f"  → Grund: {scenario['reason']}")

    print("\n" + "=" * 70)


def run(model="latest", save=True):
    """Find the thresholds of one model and save them unless save is False. Returns the exit code."""
    print("=" * 70)
    print("🎯 THRESHOLD OPTIMIZATION TOOL")
    print("=" * 70)

    # 1. Modell laden
    print("\n📦 Lade Modell...")
    try:
        model_path = resolve(model)
    except FileNotFoundError as error:
        print(f"❌ {error}")
        return 1

    loaded = load_model(model_path)
    name = os.path.basename(model_path)
    print(f"✓ Geladen: {name}")

    # 2. Test-Daten generieren
    print(f"\n📊 Generating the test data ({EVAL_V1.name})...")
    X_test, y_true = make_test_set(EVAL_V1, 500)
    print(f"✓ {len(X_test)} Samples generiert")

    # 3. Predictions
    print("\n🧠 Berechne Predictions...")
    y_scores = loaded.predict(X_test, verbose=0).flatten()
    print("✓ Fertig")

    # 4. Optimale Thresholds finden
    print("\n🔍 Suche optimale Thresholds...")
    optimal_thresholds, roc_data = find_optimal_thresholds(y_true, y_scores)
    print("✓ Gefunden")

    # 5. Visualisierung
    print("\n📈 Erstelle Visualisierung...")
    visualize_thresholds(y_true, y_scores, optimal_thresholds, roc_data)

    # 6. Empfehlungen
    recommend_threshold(optimal_thresholds)

    # 7. Export (default; --no-save skips it)
    if not save:
        print("\nNot saved (--no-save).")
        return 0

    output = {
        "model": name,
        "optimal_thresholds": {
            k: {
                **v,
                "tpr": float(v["tpr"]),
                "fpr": float(v["fpr"]),
                "threshold": float(v["threshold"]),
            }
            for k, v in optimal_thresholds.items()
        },
    }

    ensure_dirs(os.path.dirname(OPTIMAL_THRESHOLDS_FILE))
    with open(OPTIMAL_THRESHOLDS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✓ Gespeichert in {OPTIMAL_THRESHOLDS_FILE}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit("The options are on the gwd command: gwd eval thresholds --help")
    sys.exit(run())
