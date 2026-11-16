#!/usr/bin/env python3
"""
Model evaluation: ROC analysis on simulated data and an optional field test on real LIGO data.

Run it with ``gwd eval run``; it writes one plot with six panels and, for the test set
stress_v1, adds a row to the leaderboard. The field test (gwd.evaluation.real_data)
runs only with ``--real``.

Since P5.7 the simulated data is a fixed, hashed test set of gwd.evaluation.testsets
(--testset, --seed, --n; default stress_v1, which is EVAL_V1), drawn once and then
loaded. Its part 'test' gives the ROC curve, the operating point, and the efficiency
curve: the detection rate of its signal samples in each peak SNR bin, with the mean
optimal (matched filter) SNR of the bin's samples (P5.5); SNR50 and SNR90 are reported
both ways, and the leaderboard keeps the peak values. The metrics are also reported per
peak SNR bin, optimal SNR bin, total mass bin, and noise source; a bin whose signal
samples all lie outside the training ranges is marked 'Extrapolation'.
"""

import datetime
import os
import sys

import matplotlib
import numpy as np

from gwd.config import (
    DEFAULT_TESTSET,
    EVALUATION_MODELS_PLOTS_DIR,
    LEADERBOARD_FILE,
    MODELS_DIR,
    TESTSET_SEED,
    TESTSET_SIZE,
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

from gwd.data.provenance import announce
from gwd.evaluation import testsets
from gwd.evaluation.leaderboard import update_leaderboard_roc
from gwd.evaluation.metrics import (
    bin_index,
    calculate_operating_point,
    confusion_counts,
    detection_rate,
    format_snr,
    group_metrics,
    predict_labels,
    signal_bin_metrics,
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
# The leaderboard (12 columns, no test set column) only takes rows of this set, the
# distribution of all its older rows (EVAL_V1). Leaderboard v2 (P5.8) records the set.
LEADERBOARD_TESTSET = "stress_v1"


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


def efficiency_curve(built, y_scores, threshold=THRESHOLD):
    """(bin centers, detection rates, mean optimal SNRs) per peak SNR bin of a test set.

    The signal samples of the set's part 'test' fill its peak SNR bins
    (EvalSet.peak_snr_edges()); each bin gives the share of them that reach threshold
    and the mean of their optimal SNR (Sample.optimal_snr, measured in
    gwd.data.injection). An empty bin gives NaN for both.
    """
    edges = np.asarray(built.spec.peak_snr_edges())
    signal = built.y == 1
    scores = np.asarray(y_scores)[signal]
    optimal = built.optimal_snr[signal]
    index = bin_index(built.peak_snr[signal], edges)
    rates, means = [], []
    for b in range(len(edges) - 1):
        mask = index == b
        found = mask.any()
        rates.append(detection_rate(scores[mask], threshold) if found else np.nan)
        means.append(float(np.mean(optimal[mask])) if found else np.nan)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, rates, means


def binned_metrics(built, y_scores, threshold=THRESHOLD):
    """The metrics per peak SNR, optimal SNR, and total mass bin and per noise source.

    The SNR and mass rows describe the signal samples of a bin against all noise-only
    samples of the set (see gwd.evaluation.metrics.signal_bin_metrics); the noise
    source rows the signal and noise samples of that source.
    """
    y_scores = np.asarray(y_scores)
    signal = built.y == 1
    flags = testsets.outside_training(built)
    edges = testsets.bins(built)
    values = {
        "peak_snr": built.peak_snr,
        "optimal_snr": built.optimal_snr,
        "total_mass": built.total_mass,
    }
    report = {
        key: signal_bin_metrics(
            values[key][signal],
            edges[key],
            y_scores[signal],
            y_scores[~signal],
            threshold,
            flags[signal],
        )
        for key in values
    }
    report["noise_source"] = group_metrics(
        built.noise_source, built.y, y_scores, threshold, flags
    )
    return report


def extrapolation_label(outside):
    """'Extrapolation' if all signal samples lie outside the training ranges."""
    if not outside:
        return ""
    if outside >= 1.0:
        return "Extrapolation"
    return f"partly extrapolation ({outside:.0%})"


def _number(value, spec):
    return "-" if value is None else format(value, spec)


def bin_table(title, rows, value_format=".2f"):
    """A text table of signal_bin_metrics or group_metrics rows."""
    if rows and "group" in rows[0]:
        lines = [
            f"   {title:<16s} {'signal':>6s} {'noise':>6s} {'TPR':>7s} "
            f"{'FPR':>7s} {'AUC':>6s}"
        ]
        for row in rows:
            lines.append(
                f"   {row['group']:<16s} {row['n_signal']:>6d} {row['n_noise']:>6d} "
                f"{_number(row['tpr'], '>7.1%')} {_number(row['fpr'], '>7.1%')} "
                f"{_number(row['auc'], '>6.3f')}  {extrapolation_label(row['outside'])}"
            )
        return "\n".join(lines)
    lines = [f"   {title:<16s} {'n':>6s} {'TPR':>7s} {'AUC':>6s}"]
    for row in rows:
        span = f"{row['low']:{value_format}}–{row['high']:{value_format}}"
        lines.append(
            f"   {span:<16s} {row['n']:>6d} {_number(row['tpr'], '>7.1%')} "
            f"{_number(row['auc'], '>6.3f')}  {extrapolation_label(row['outside'])}"
        )
    return "\n".join(lines)


def print_binned_metrics(report, threshold=THRESHOLD):
    print(
        f"\n   📋 Per bin (TPR at threshold {threshold}; the AUC of an SNR or mass "
        "bin is its signal samples against all noise samples):"
    )
    print(bin_table("Peak SNR", report["peak_snr"]))
    print(bin_table("Optimal SNR", report["optimal_snr"], ".1f"))
    print(bin_table("Total mass (M☉)", report["total_mass"], ".1f"))
    print(bin_table("Noise source", report["noise_source"]))
    print(
        "   'Extrapolation': all signal samples of the bin lie outside the training "
        "ranges (TRAIN_V1)."
    )


def optimal_per_peak(snr_levels, optimal_snrs):
    """The mean optimal SNR per unit of peak SNR over the curve; None without signal."""
    total = float(np.sum(snr_levels))
    return float(np.sum(optimal_snrs)) / total if total > 0 else None


# ==========================================
# TEST 1: SIMULATION WITH ROC ANALYSIS
# ==========================================
def evaluate_simulation_with_roc(model, built):
    """ROC, precision-recall, efficiency curve, confusion matrix, and the binned
    metrics on the part 'test' of a test set (gwd.evaluation.testsets); one plot."""
    print(f"\n🧪 [LAB] Starting the ROC analysis on the test set {built.label()}...")

    # --- A. ROC curve (independent of the threshold) ---
    X_roc, y_true_roc = built.X, built.y
    n_roc_samples = len(y_true_roc)

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

    # --- B. Efficiency curve: detection rate per peak SNR bin of the signal samples ---
    snr_levels, detection_rates, optimal_snrs = efficiency_curve(built, y_scores)

    # SNR50 / SNR90: the first peak SNR bin (its center) whose detection rate reaches
    # 50% / 90%, and the mean optimal SNR of that bin's samples (P5.5).
    snr_50 = snr_at_rate(snr_levels, detection_rates, 0.5)
    snr_90 = snr_at_rate(snr_levels, detection_rates, 0.9)
    optimal_50 = snr_at_rate(np.array(optimal_snrs), detection_rates, 0.5)
    optimal_90 = snr_at_rate(np.array(optimal_snrs), detection_rates, 0.9)
    signal = y_true_roc == 1
    ratio = optimal_per_peak(built.peak_snr[signal], built.optimal_snr[signal])
    if ratio is not None:
        print(
            f"   ✅ Optimal SNR per peak SNR ({built.spec.name}, "
            f"{built.spec.scenario.name}): {ratio:.1f}"
        )

    report = binned_metrics(built, y_scores)
    print_binned_metrics(report)

    # --- C. Confusion matrix at THRESHOLD ---
    y_pred = predict_labels(y_scores, THRESHOLD)
    tn, fp, fn, tp = confusion_counts(y_true_roc, y_scores, THRESHOLD)
    acc = (tp + tn) / n_roc_samples

    # --- PLOTTING ---
    fig = plt.figure(figsize=(15, 10))
    # hspace leaves room for the optimal SNR axis on top of the efficiency curve.
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.3)

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

    # 4. Efficiency curve
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(snr_levels, detection_rates, "o-", color="purple", lw=2)
    ax4.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50%")
    ax4.axhline(0.9, color="blue", linestyle="--", alpha=0.5, label="90%")
    if snr_50 is not None:
        ax4.axvline(snr_50, color="orange", linestyle=":", label=f"SNR50={snr_50:.2f}")
    if snr_90 is not None:
        ax4.axvline(snr_90, color="red", linestyle=":", label=f"SNR90={snr_90:.2f}")
    ax4.set_xlabel("Injected peak SNR (bin center)", fontweight="bold")
    if ratio:
        # The optimal SNR is about proportional to the peak SNR for fixed masses.
        top = ax4.secondary_xaxis(
            "top", functions=(lambda x: x * ratio, lambda x: x / ratio)
        )
        top.set_xlabel(f"Optimal SNR (≈ {ratio:.1f} × peak)")
    ax4.set_ylabel("Detection Probability", fontweight="bold")
    ax4.set_title("Efficiency curve (signal samples per bin)", fontweight="bold")
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
            ("Peak SNR50:", format_snr(snr_50)),
            ("Peak SNR90:", format_snr(snr_90)),
            ("Optimal SNR50:", format_snr(optimal_50)),
            ("Optimal SNR90:", format_snr(optimal_90)),
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

    plt.suptitle(
        f"Comprehensive Model Evaluation: {built.label()}, SHA-256 {built.short_hash()}",
        fontsize=14,
        fontweight="bold",
    )
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
        "optimal_snr50": optimal_50,
        "optimal_snr90": optimal_90,
        "optimal_per_peak": ratio,
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "tpr": op_tpr,
        "fpr": op_fpr,
        "bins": report,
    }


# ==========================================
# MAIN
# ==========================================
def noise_description(spec, real):
    """The noise part of the start line: the test set's noise, and the field test."""
    field_test = "field test on real H1 data" if real else "no field test (--no-real)"
    scenario = spec.scenario
    if scenario.real_noise_fraction == 0:
        noise = "gaussian"
    else:
        noise = (
            f"real for {scenario.real_noise_fraction:.0%} of samples if the test set "
            "was built with noise chunks, else gaussian"
        )
    return f"{noise} (test set {spec.name} = {scenario.name}); {field_test}"


def run(
    model="latest",
    real=False,
    testset=DEFAULT_TESTSET,
    seed=TESTSET_SEED,
    n=TESTSET_SIZE,
):
    """Evaluate one model on a fixed test set and report it. Returns the exit code.

    testset, seed, and n name the set (gwd.evaluation.testsets.build: built once,
    then loaded). real=True adds the real data field test, which needs network access
    (D10: optional, off by default); without it the row reports 'n/a' for the events
    and the false alarms. Only the set LEADERBOARD_TESTSET adds a leaderboard row.
    """
    try:
        model_path = resolve(model)
    except FileNotFoundError as error:
        print(f"❌ {error}")
        return 1
    spec = testsets.get(testset)

    print("=" * 70)
    print(f"🤖 COMPREHENSIVE EVALUATION: {os.path.basename(model_path)}")
    announce(noise_description(spec, real))

    # The set first: drawing it needs PyCBC, so a missing backend stops before the model.
    try:
        built = testsets.build(testset, n, seed)
    except testsets.InvalidTestsetError as error:
        print(f"❌ {error}", file=sys.stderr)
        return 1
    print("\n".join(testsets.describe(built)))

    loaded = load_model(model_path)

    # 1. Simulation with ROC
    sim_res = evaluate_simulation_with_roc(loaded, built)

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
    print(f"  Test set:             {built.label()}")
    print(f"  Test set SHA-256:     {built.sha256}")
    print(f"  ROC AUC:              {sim_res['roc_auc']:.4f}")
    print(f"  Average Precision:    {sim_res['avg_precision']:.4f}")
    print(f"  Accuracy:             {sim_res['acc']:.1%}")
    print(f"  TPR (Sensitivity):    {sim_res['tpr']:.2%}")
    print(f"  FPR:                  {sim_res['fpr']:.2%}")
    print(f"  Peak SNR90:           {format_snr(sim_res['snr90'])}")
    print(f"  Optimal SNR90:        {format_snr(sim_res['optimal_snr90'])}")
    print(f"  Real Events Found:    {real_res['events_text']}")
    print(f"  Real Noise:           {real_res['far_text']}")
    print("=" * 70)

    if spec.name != LEADERBOARD_TESTSET:
        print(
            f"\nLeaderboard not updated: its rows all come from {LEADERBOARD_TESTSET} "
            "(EVAL_V1), and it has no test set column until leaderboard v2 (P5.8)."
        )
        return 0
    update_leaderboard_roc(
        os.path.basename(model_path), sim_res, real_res, LEADERBOARD_FILE, BASELINE_FILE
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit("The options are on the gwd command: gwd eval run --help")
    sys.exit(run())
