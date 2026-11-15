#!/usr/bin/env python3
"""
ROC comparison: several models on one test set, independent of a threshold.

Run it with ``gwd eval compare``; it writes the comparison plot and a zoom on low FPR.
"""

import datetime
import os
import sys

import matplotlib

from gwd.config import (
    EVALUATION_MODELS_PLOTS_DIR,
    ensure_dirs,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

from gwd.data.injection import make_test_set
from gwd.data.scenarios import EVAL_V1
from gwd.registry import list_models, load_model, model_name, resolve

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def score_models(paths, X_test, y_true):
    """Load the models one after another and compute each ROC curve once.

    Only one model is in memory at a time. Returns (name, fpr, tpr, roc_auc) per model that
    loads, in the order of paths (oldest first when they come from list_models()).
    """
    curves = []
    for path in paths:
        try:
            model = load_model(path)
        except Exception as e:
            print(f"✗ Error for {path}: {e}")
            continue
        name = model_name(path)
        print(f"✓ Loaded: {name}")
        y_scores = model.predict(X_test, verbose=0).flatten()
        del model
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        curves.append((name, fpr, tpr, auc(fpr, tpr)))
    return curves


def model_colors(names):
    """One fixed color per model name, the same in every plot of a run.

    The names are sorted, so the same set of models gets the same colors in every run.
    """
    unique = sorted(set(names))
    cmap = plt.cm.tab10 if len(unique) <= 10 else plt.cm.tab20
    return {name: cmap(i % cmap.N) for i, name in enumerate(unique)}


def short_name(name, length=30):
    """The model name for a legend, cut to length characters."""
    return name if len(name) <= length else name[:length] + "..."


def save_plot(fig, filename):
    """Save fig in the evaluation plot folder and return the path."""
    ensure_dirs(EVALUATION_MODELS_PLOTS_DIR)
    path = os.path.join(EVALUATION_MODELS_PLOTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"📊 Plot: {path}")
    return path


def compare_models_roc(curves, colors):
    """Plot all ROC curves and the AUC ranking. Returns the ranking, best first."""

    print("\n🧠 Plotting the ROC curves of all models...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for name, fpr, tpr, roc_auc in curves:
        ax1.plot(
            fpr,
            tpr,
            color=colors[name],
            lw=2,
            label=f"{short_name(name)} (AUC={roc_auc:.3f})",
        )

    # Random classifier line
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Classifier")

    ax1.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax1.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax1.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- AUC Bar Chart ---
    auc_scores_sorted = sorted(
        ((name, roc_auc) for name, _, _, roc_auc in curves),
        key=lambda x: x[1],
        reverse=True,
    )
    names = [short_name(name) for name, _ in auc_scores_sorted]
    aucs = [x[1] for x in auc_scores_sorted]

    bars = ax2.barh(
        range(len(names)),
        aucs,
        color=[colors[name] for name, _ in auc_scores_sorted],
    )
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("ROC AUC", fontsize=12, fontweight="bold")
    ax2.set_title("Model Ranking by AUC", fontsize=14, fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax2.grid(True, alpha=0.3, axis="x")

    # Write the values into the bars
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

    fig.tight_layout()
    save_plot(fig, f"plot_compare_roc_{TIMESTAMP}.png")

    return auc_scores_sorted


def zoom_to_relevant_region(curves, colors):
    """
    Zoomed ROC plot: gravitational wave searches care mostly about very low FPR,
    and a zoom on FPR <= 0.1 shows the differences there better.
    """
    print("\n🔍 Zooming in on the relevant region (FPR <= 0.1)...")

    fig, ax = plt.subplots(figsize=(10, 8))

    for name, fpr, tpr, roc_auc in curves:
        # Plot only FPR <= 0.1
        mask = fpr <= 0.1
        ax.plot(
            fpr[mask],
            tpr[mask],
            color=colors[name],
            lw=2,
            marker="o",
            markersize=3,
            label=f"{short_name(name)} (AUC={roc_auc:.3f})",
        )

    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title(
        "ROC Curve - Zoom on the relevant region (FPR <= 0.1)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    save_plot(fig, f"plot_compare_roc_{TIMESTAMP}_zoom.png")


def run(models=None):
    """Compare the given models (default: all in models_registry/). Returns the exit code."""
    print("=" * 70)
    print("🏆 ROC CURVE COMPARISON TOOL")
    print("=" * 70)

    # 1. Find the models
    try:
        paths = [resolve(spec) for spec in models] if models else list_models()
    except FileNotFoundError as error:
        print(f"❌ {error}")
        return 1

    if len(paths) == 0:
        print("\n❌ No models to compare.")
        return 1

    if len(paths) == 1:
        print("\n⚠️ Only one model found. Train more models to compare them.")
        return 0

    # 2. Generate the test set
    n_samples = 1000
    print(f"\n📊 Generating {n_samples} test samples ({EVAL_V1.name})...")
    X_test, y_true = make_test_set(EVAL_V1, n_samples)

    # 3. Load each model in turn and score it once
    curves = score_models(paths, X_test, y_true)

    if len(curves) == 0:
        print("\n❌ No models to compare.")
        return 1

    if len(curves) == 1:
        print("\n⚠️ Only one model found. Train more models to compare them.")
        return 0

    print(f"\n✅ {len(curves)} models loaded.")
    colors = model_colors([name for name, *_ in curves])

    # 4. Compare
    rankings = compare_models_roc(curves, colors)

    # 5. Print the ranking
    print("\n" + "=" * 70)
    print("📊 FINAL RANKING:")
    print("=" * 70)
    for i, (name, auc_val) in enumerate(rankings, 1):
        print(f"  {i}. {name[:50]:<50s} | AUC: {auc_val:.4f}")
    print("=" * 70)

    # 6. Bonus: Zoom
    print("\n💡 Creating the zoom plot for low FPR...")
    zoom_to_relevant_region(curves, colors)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit("The options are on the gwd command: gwd eval compare --help")
    sys.exit(run())
