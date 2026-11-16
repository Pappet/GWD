#!/usr/bin/env python3
"""
ROC Comparison Tool - Vergleicht mehrere Modelle in einer ROC-Kurve
Nützlich um zu sehen, welches Modell threshold-unabhängig besser ist.
"""

import argparse
import datetime
import os

import matplotlib
import numpy as np

from gwd.config import (
    EVALUATION_MODELS_PLOTS_DIR,
    ensure_dirs,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

from gwd.data.injection import make_test_set
from gwd.data.scenarios import EVAL_V1
from gwd.registry import MODEL_HELP, list_models, load_model, model_name, resolve

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def load_models(paths):
    """Load the given models, oldest first when they come from list_models()."""
    models = []
    for path in paths:
        try:
            model = load_model(path)
            name = model_name(path)
            models.append((name, model))
            print(f"✓ Geladen: {name}")
        except Exception as e:
            print(f"✗ Fehler bei {path}: {e}")

    return models


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
        ax1.plot(
            fpr, tpr, color=color, lw=2, label=f"{name[:30]}... (AUC={roc_auc:.3f})"
        )

    # Zufall-Linie
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Classifier")

    ax1.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax1.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax1.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- AUC Bar Chart ---
    auc_scores_sorted = sorted(auc_scores, key=lambda x: x[1], reverse=True)
    names = [x[0][:30] for x in auc_scores_sorted]
    aucs = [x[1] for x in auc_scores_sorted]

    bars = ax2.barh(range(len(names)), aucs, color=colors[: len(names)])
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("ROC AUC", fontsize=12, fontweight="bold")
    ax2.set_title("Model Ranking by AUC", fontsize=14, fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax2.grid(True, alpha=0.3, axis="x")

    # Werte in die Bars schreiben
    for i, (bar, auc_val) in enumerate(zip(bars, aucs)):
        ax2.text(
            auc_val - 0.05,
            i,
            f"{auc_val:.4f}",
            va="center",
            ha="right",
            fontweight="bold",
            color="white",
        )

    plt.tight_layout()
    ensure_dirs(EVALUATION_MODELS_PLOTS_DIR)
    plt.savefig(
        os.path.join(EVALUATION_MODELS_PLOTS_DIR, f"plot_compare_roc_{TIMESTAMP}.png"),
        dpi=150,
    )
    print(
        f"📊 Plot: {os.path.join(EVALUATION_MODELS_PLOTS_DIR, f'plot_compare_roc_{TIMESTAMP}.png')}"
    )
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
        ax.plot(
            fpr[mask],
            tpr[mask],
            color=color,
            lw=2,
            marker="o",
            markersize=3,
            label=f"{name[:30]}... (AUC={roc_auc:.3f})",
        )

    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title(
        "ROC Curve - Zoom auf relevante Region (FPR < 0.1)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    ensure_dirs(EVALUATION_MODELS_PLOTS_DIR)
    plt.savefig(
        os.path.join(EVALUATION_MODELS_PLOTS_DIR, f"plot_compare_roc_{TIMESTAMP}.png"),
        dpi=150,
    )
    print(
        f"📊 Plot: {os.path.join(EVALUATION_MODELS_PLOTS_DIR, f'plot_compare_roc_{TIMESTAMP}.png')}"
    )
    plt.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare models in one ROC plot (default: all models in models_registry/)."
    )
    parser.add_argument(
        "--model",
        action="append",
        help=f"One model to compare, repeatable: {MODEL_HELP}. Default: all models.",
    )
    args = parser.parse_args(argv)

    print("=" * 70)
    print("🏆 ROC CURVE COMPARISON TOOL")
    print("=" * 70)

    # 1. Modelle laden
    try:
        paths = [resolve(spec) for spec in args.model] if args.model else list_models()
    except FileNotFoundError as error:
        print(f"❌ {error}")
        return
    models = load_models(paths)

    if len(models) == 0:
        print("\n❌ Keine Modelle zum Vergleichen gefunden.")
        return

    if len(models) == 1:
        print("\n⚠️ Nur ein Modell gefunden. Trainiere mehr Modelle für Vergleiche!")
        return

    print(f"\n✅ {len(models)} Modelle geladen.")

    # 2. Test-Set generieren
    n_samples = 1000
    print(f"\n📊 Generating {n_samples} test samples ({EVAL_V1.name})...")
    X_test, y_true = make_test_set(EVAL_V1, n_samples)

    # 3. Vergleich durchführen
    rankings = compare_models_roc(models, X_test, y_true)

    # 4. Ranking ausgeben
    print("\n" + "=" * 70)
    print("📊 FINAL RANKING:")
    print("=" * 70)
    for i, (name, auc_val) in enumerate(rankings, 1):
        print(f"  {i}. {name[:50]:<50s} | AUC: {auc_val:.4f}")
    print("=" * 70)

    # 5. Bonus: Zoom
    if len(models) > 1:
        print("\n💡 Erstelle zusätzlichen Zoom-Plot für niedrige FPR...")
        zoom_to_relevant_region(models, X_test, y_true)


if __name__ == "__main__":
    main()
