#!/usr/bin/env python3
"""
Model evaluation: ROC analysis on simulated data and an optional field test on real LIGO data.

Run it with ``gwd eval run``; it writes one plot with six panels and adds a row to the
leaderboard. The field test (gwd.evaluation.real_data) runs only with ``--real``.
"""

import datetime
import os
import sys

import matplotlib
import numpy as np

from gwd.config import (
    EVALUATION_MODELS_PLOTS_DIR,
    LEADERBOARD_FILE,
    MODELS_DIR,
    N_SAMPLES,
    ensure_dirs,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from gwd.data.injection import make_sample, make_test_set
from gwd.data.preprocessing import normalize
from gwd.data.provenance import announce
from gwd.data.scenarios import EVAL_V1
from gwd.evaluation.leaderboard import update_leaderboard_roc
from gwd.evaluation.metrics import (
    calculate_operating_point,
    confusion_counts,
    detection_rate,
    format_snr,
    predict_labels,
    snr_at_rate,
)
from gwd.evaluation.real_data import NO_FIELD_TEST, run_field_test
from gwd.registry import load_model, resolve

# ==========================================
# CONFIGURATION
# ==========================================
BASELINE_FILE = os.path.join(MODELS_DIR, "physics_baseline.json")
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Operating point of the tests; the ROC curve itself does not depend on it
THRESHOLD = 0.15


# ==========================================
# HELPERS
# ==========================================
def metrics_box(rows, width=27):
    """The metrics table of the plot: (label, formatted value) rows, None for a blank line."""
    lines = ["╔" + "═" * width + "╗", "║" + "PERFORMANCE METRICS".center(width) + "║"]
    lines.append("╠" + "═" * width + "╣")
    for row in rows:
        text = "" if row is None else f" {row[0]:<15s} {row[1]}"
        lines.append("║" + text.ljust(width) + "║")
    lines.append("╚" + "═" * width + "╝")
    return "\n".join(lines)


# ==========================================
# TEST 1: SIMULATION WITH ROC ANALYSIS
# ==========================================
def evaluate_simulation_with_roc(model):
    """ROC, precision-recall, sensitivity curve, and confusion matrix on EVAL_V1; one plot."""
    print("\n🧪 [LAB] Starting the ROC analysis on simulated data...")

    # --- A. ROC curve (independent of the threshold) ---
    print(f"   📊 Generating the ROC test set ({EVAL_V1.name})...")
    n_roc_samples = 1000  # A large sample for a stable ROC curve

    X_roc, y_true_roc = make_test_set(EVAL_V1, n_roc_samples)

    # Predictions (probabilities, not labels)
    print("   🧠 Computing the model predictions...")
    y_scores = model.predict(X_roc, verbose=0).flatten()

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_roc, y_scores)
    roc_auc = auc(fpr, tpr)

    # Precision-recall curve (another view)
    precision, recall, pr_thresholds = precision_recall_curve(y_true_roc, y_scores)
    avg_precision = average_precision_score(y_true_roc, y_scores)

    # Operating point at THRESHOLD
    op_tpr, op_fpr = calculate_operating_point(y_true_roc, y_scores, THRESHOLD)

    print(f"   ✅ ROC AUC: {roc_auc:.4f}")
    print(f"   ✅ Average Precision: {avg_precision:.4f}")
    print(
        f"   📍 Operating Point (Threshold={THRESHOLD}): TPR={op_tpr:.2%}, FPR={op_fpr:.2%}"
    )

    # --- B. Sensitivity curve (detection rate against SNR) ---
    print("   📈 Computing the sensitivity curve...")
    snr_levels = np.linspace(0.0, 2.5, 15)
    detection_rates = []

    for snr in snr_levels:
        n_batch = 50
        X_batch = []

        for _ in range(n_batch):
            sample = make_sample(EVAL_V1, has_signal=True, snr=snr)
            X_batch.append(normalize(sample.data))

        X = np.array(X_batch).reshape(n_batch, N_SAMPLES, 1)
        preds = model.predict(X, verbose=0).flatten()
        detection_rates.append(detection_rate(preds, THRESHOLD))

    # SNR50 / SNR90: the first SNR level whose detection rate reaches 50% / 90%
    snr_50 = snr_at_rate(snr_levels, detection_rates, 0.5)
    snr_90 = snr_at_rate(snr_levels, detection_rates, 0.9)

    # --- C. Confusion matrix at THRESHOLD ---
    y_pred = predict_labels(y_scores, THRESHOLD)
    tn, fp, fn, tp = confusion_counts(y_true_roc, y_scores, THRESHOLD)
    acc = (tp + tn) / n_roc_samples

    # --- PLOTTING ---
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. ROC curve (main plot)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(fpr, tpr, color="blue", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax1.plot(
        op_fpr, op_tpr, "ro", markersize=10, label=f"Operating Point (τ={THRESHOLD})"
    )
    ax1.set_xlabel("False Positive Rate", fontweight="bold")
    ax1.set_ylabel("True Positive Rate", fontweight="bold")
    ax1.set_title("ROC curve (threshold-independent)", fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # 2. Precision-recall curve
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        recall, precision, color="green", lw=2, label=f"PR (AP = {avg_precision:.3f})"
    )
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Recall (TPR)", fontweight="bold")
    ax2.set_ylabel("Precision", fontweight="bold")
    ax2.set_title("Precision-recall curve", fontweight="bold")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    # 3. Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 2])
    ConfusionMatrixDisplay.from_predictions(
        y_true_roc, y_pred, ax=ax3, colorbar=False, cmap="Blues"
    )
    ax3.set_title(f"Confusion Matrix (τ={THRESHOLD})", fontweight="bold")

    # 4. Sensitivity Curve
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(snr_levels, detection_rates, "o-", color="purple", lw=2)
    ax4.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50%")
    ax4.axhline(0.9, color="blue", linestyle="--", alpha=0.5, label="90%")
    if snr_50 is not None:
        ax4.axvline(snr_50, color="orange", linestyle=":", label=f"SNR50={snr_50:.2f}")
    if snr_90 is not None:
        ax4.axvline(snr_90, color="red", linestyle=":", label=f"SNR90={snr_90:.2f}")
    ax4.set_xlabel("Injected SNR", fontweight="bold")
    ax4.set_ylabel("Detection Probability", fontweight="bold")
    ax4.set_title("Sensitivity Curve", fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Score Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    scores_signal = y_scores[y_true_roc == 1]
    scores_noise = y_scores[y_true_roc == 0]
    ax5.hist(
        scores_noise, bins=30, alpha=0.5, label="Noise", color="blue", density=True
    )
    ax5.hist(
        scores_signal, bins=30, alpha=0.5, label="Signal", color="red", density=True
    )
    ax5.axvline(
        THRESHOLD, color="black", linestyle="--", lw=2, label=f"Threshold={THRESHOLD}"
    )
    ax5.set_xlabel("Model Score", fontweight="bold")
    ax5.set_ylabel("Density", fontweight="bold")
    ax5.set_title("Score Distributions", fontweight="bold")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Metrics table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    metrics_text = metrics_box(
        [
            ("ROC AUC:", f"{roc_auc:.4f}"),
            ("Avg Precision:", f"{avg_precision:.4f}"),
            None,
            ("Accuracy:", f"{acc:.2%}"),
            ("TPR (Recall):", f"{op_tpr:.2%}"),
            ("FPR:", f"{op_fpr:.2%}"),
            None,
            ("SNR50:", format_snr(snr_50)),
            ("SNR90:", format_snr(snr_90)),
        ]
    )

    ax6.text(
        0.1,
        0.5,
        metrics_text,
        fontfamily="monospace",
        fontsize=10,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.suptitle("Comprehensive Model Evaluation", fontsize=16, fontweight="bold")
    ensure_dirs(EVALUATION_MODELS_PLOTS_DIR)
    plt.savefig(
        os.path.join(
            EVALUATION_MODELS_PLOTS_DIR, f"plot_evaluate_model_with_roc_{TIMESTAMP}.png"
        ),
        dpi=150,
    )
    print(
        f"📊 Plot: {os.path.join(EVALUATION_MODELS_PLOTS_DIR, f'plot_evaluate_model_with_roc_{TIMESTAMP}.png')}"
    )
    plt.close()

    return {
        "acc": acc,
        "snr50": snr_50,
        "snr90": snr_90,
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "tpr": op_tpr,
        "fpr": op_fpr,
    }


# ==========================================
# MAIN
# ==========================================
def noise_description(real):
    """The noise part of the start line: simulated test sets, and the field test."""
    field_test = "field test on real H1 data" if real else "no field test (--no-real)"
    return f"gaussian ({EVAL_V1.name} test sets); {field_test}"


def run(model="latest", real=False):
    """Evaluate one model and add a leaderboard row. Returns the exit code.

    real=True adds the real data field test, which needs network access (D10: optional,
    off by default); without it the row reports 'n/a' for the events and the false
    alarms.
    """
    try:
        model_path = resolve(model)
    except FileNotFoundError as error:
        print(f"❌ {error}")
        return 1

    print("=" * 70)
    print(f"🤖 COMPREHENSIVE EVALUATION: {os.path.basename(model_path)}")
    announce(noise_description(real))

    loaded = load_model(model_path)

    # 1. Simulation with ROC
    sim_res = evaluate_simulation_with_roc(loaded)

    # 2. Real Data
    if real:
        real_res = run_field_test(loaded, THRESHOLD)
    else:
        print("\nField test skipped; gwd eval run --real runs it (network).")
        real_res = NO_FIELD_TEST

    # 3. Final Report
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY:")
    print("=" * 70)
    print(f"  ROC AUC:              {sim_res['roc_auc']:.4f}")
    print(f"  Average Precision:    {sim_res['avg_precision']:.4f}")
    print(f"  Accuracy:             {sim_res['acc']:.1%}")
    print(f"  TPR (Sensitivity):    {sim_res['tpr']:.2%}")
    print(f"  FPR:                  {sim_res['fpr']:.2%}")
    print(f"  SNR90:                {format_snr(sim_res['snr90'])}")
    print(f"  Real Events Found:    {real_res['events_text']}")
    print(f"  Real Noise:           {real_res['far_text']}")
    print("=" * 70)

    update_leaderboard_roc(
        os.path.basename(model_path), sim_res, real_res, LEADERBOARD_FILE, BASELINE_FILE
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit("The options are on the gwd command: gwd eval run --help")
    sys.exit(run())
