#!/usr/bin/env python3
"""
Threshold calibration: one detection threshold per model at a fixed false alarm rate.

Since P5.8 (D10 = A) ``gwd eval thresholds`` calibrates the threshold on the
noise-only part calib_bg of a fixed, hashed test set (gwd.evaluation.testsets;
--testset, --seed, --n, default stress_v1), never on the part test: the lowest
threshold whose false alarm rate on calib_bg is at most THRESHOLD_TARGET_FPR (1 %).
With --save (the default) it stores the threshold, the target and the reached FPR, the
strategy, and the test set with its SHA-256 in the model's metadata
(gwd.registry.set_threshold), where gwd eval run, gwd eval compare, and the Glitch Hunter
read it, and it writes a report to results/thresholds/<model name>.json, one per model
(results/optimal_thresholds.json, which every run overwrote, is no longer written).

It then reports the TPR and FPR of that threshold on the part test, with n. The six
strategies of find_optimal_thresholds are still shown for reference, but they are
chosen on the part test, so their test numbers are optimistic, and nothing uses them:

1. Youden's J statistic: maximizes (TPR - FPR)
2. F1 score: harmonic mean of precision and recall
3. Cost-based: weights the two error types differently
4. Fixed FPR: maximizes the TPR at a given FPR limit (5% and 1%)
5. Fixed TPR: minimizes the FPR at a TPR of 90%
"""

import datetime
import json
import os
import sys

import matplotlib
import numpy as np

from gwd.config import (
    DEFAULT_TESTSET,
    EVALUATION_MODELS_PLOTS_DIR,
    TESTSET_SEED,
    TESTSET_SIZE,
    THRESHOLD_REPORTS_DIR,
    THRESHOLD_TARGET_FPR,
    ensure_dirs,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gwd import registry
from gwd.evaluation import testsets
from gwd.evaluation.metrics import find_optimal_thresholds, rate_at, threshold_at_fpr
from gwd.registry import load_model, model_name, resolve

# The strategy that gwd eval thresholds saves in the registry.
STRATEGY = "fixed_fpr on calib_bg"

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def report_path(model_path):
    """The report of one model: results/thresholds/<model name>.json."""
    return os.path.join(THRESHOLD_REPORTS_DIR, model_name(model_path) + ".json")


def plot_path(model_path):
    """The plot of one run: plot_optimize_threshold_<model name>_<timestamp>.png."""
    return os.path.join(
        EVALUATION_MODELS_PLOTS_DIR,
        f"plot_optimize_threshold_{model_name(model_path)}_{TIMESTAMP}.png",
    )


def visualize_thresholds(
    y_true, y_scores, optimal_thresholds, roc_data, calibrated, model_path
):
    """Plot the calibrated threshold and the reference strategies on the ROC curve.

    calibrated is the result of calibrate(): its point is the TPR and FPR of the
    calib_bg threshold on the part test. model_path names the plot file (plot_path).
    """

    fpr, tpr, thresholds = roc_data

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. ROC with all chosen points
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(fpr, tpr, "b-", lw=2, label="ROC Curve")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax1.plot(
        calibrated["test_fpr"],
        calibrated["test_tpr"],
        marker="*",
        markersize=18,
        color="black",
        linestyle="none",
        label=f"saved: FPR ≤ {calibrated['target_fpr']:.0%} on calib_bg, "
        f"τ={calibrated['threshold']:.4f}",
    )

    colors = ["red", "orange", "green", "cyan", "purple", "magenta"]
    markers = ["o", "s", "^", "v", "D", "P"]

    for (name, opt), color, marker in zip(optimal_thresholds.items(), colors, markers):
        ax1.plot(
            opt["fpr"],
            opt["tpr"],
            marker=marker,
            markersize=12,
            color=color,
            label=f"{name} (on test, reference): τ={opt['threshold']:.3f}",
        )

    ax1.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax1.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax1.set_title(
        "ROC curve with the chosen thresholds", fontsize=14, fontweight="bold"
    )
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Threshold vs Metrics
    ax2 = fig.add_subplot(gs[1, 0])

    # TPR and FPR against the threshold
    ax2.plot(thresholds, tpr, "g-", lw=2, label="TPR (Sensitivity)")
    ax2.plot(thresholds, fpr, "r-", lw=2, label="FPR (False Alarms)")
    ax2.plot(thresholds, tpr - fpr, "b--", lw=2, label="Youden's J (TPR-FPR)")

    # Mark the chosen points
    for name, opt in optimal_thresholds.items():
        ax2.axvline(opt["threshold"], color="gray", linestyle=":", alpha=0.5)
    ax2.axvline(calibrated["threshold"], color="black", linestyle="--", lw=2)

    ax2.set_xlabel("Threshold", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Rate", fontsize=12, fontweight="bold")
    ax2.set_title("Metrics vs threshold", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # 3. Comparison table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")

    table_text = "╔════════════════════════════════════════════════╗\n"
    table_text += "║  SAVED THRESHOLD (calib_bg)                    ║\n"
    table_text += "╠════════════════════════════════════════════════╣\n"
    table_text += (
        f"║   Threshold:  {calibrated['threshold']:.4f}                       ║\n"
    )
    table_text += (
        f"║   FPR calib:  {calibrated['calib_fpr']:.2%} (n {calibrated['n_background']})".ljust(
            49
        )
        + "║\n"
    )
    table_text += (
        f"║   TPR test:   {calibrated['test_tpr']:.2%} (n {calibrated['n_signal']})".ljust(
            49
        )
        + "║\n"
    )
    table_text += (
        f"║   FPR test:   {calibrated['test_fpr']:.2%} (n {calibrated['n_noise']})".ljust(
            49
        )
        + "║\n"
    )
    table_text += "╠════════════════════════════════════════════════╣\n"
    table_text += "║  REFERENCE STRATEGIES (chosen on test)         ║\n"
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

    plt.suptitle(
        f"Threshold Optimization Analysis: {model_name(model_path)}",
        fontsize=16,
        fontweight="bold",
    )
    path = plot_path(model_path)
    ensure_dirs(os.path.dirname(path))
    plt.savefig(path, dpi=150)
    print(f"📊 Plot: {path}")
    plt.close()


def recommend_threshold(optimal_thresholds):
    """Print which strategy suits which typical GW use."""

    print("\n" + "=" * 70)
    print("🎯 REFERENCE: STRATEGIES FOR TYPICAL USES")
    print("=" * 70)
    print(
        "These thresholds are chosen on the part 'test', so their TPR and FPR there are\n"
        "optimistic. They are not saved; the registry keeps the calib_bg threshold."
    )

    scenarios = [
        {
            "name": "🔬 Scientific analysis (paper)",
            "recommendation": "fixed_fpr_1",
            "reason": "A low false alarm rate matters for credibility",
        },
        {
            "name": "🚨 Trigger for follow-up analyses",
            "recommendation": "youden",
            "reason": "Balance between TPR and FPR",
        },
        {
            "name": "📊 Statistical studies (large data sets)",
            "recommendation": "f1",
            "reason": "Maximizes overall quality on unbalanced data",
        },
        {
            "name": "⚡ Real-time alert system",
            "recommendation": "fixed_tpr_90",
            "reason": "Guarantees a high detection rate; false alarms can be filtered later",
        },
        {
            "name": "🎓 Teaching / demo",
            "recommendation": "youden",
            "reason": "An operating point that is easy to understand",
        },
    ]

    for scenario in scenarios:
        rec = optimal_thresholds[scenario["recommendation"]]
        print(f"\n{scenario['name']}")
        print(f"  → Recommended: {scenario['recommendation'].upper()}")
        print(f"  → Threshold: {rec['threshold']:.4f}")
        print(f"  → TPR: {rec['tpr']:.2%} | FPR: {rec['fpr']:.2%}")
        print(f"  → Reason: {scenario['reason']}")

    print("\n" + "=" * 70)


def calibrate(background_scores, y_true, y_scores, target_fpr=THRESHOLD_TARGET_FPR):
    """The threshold at target_fpr on the noise-only scores, and its rates on test.

    background_scores come from the part calib_bg, y_true and y_scores from the part
    test, which only reports: TPR on its signal samples and FPR on its noise samples,
    each with its n.
    """
    chosen = threshold_at_fpr(background_scores, target_fpr)
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    threshold = chosen["threshold"]
    test_tpr, n_signal = rate_at(y_scores[y_true == 1], threshold)
    test_fpr, n_noise = rate_at(y_scores[y_true == 0], threshold)
    return {
        "threshold": threshold,
        "target_fpr": float(target_fpr),
        "calib_fpr": chosen["fpr"],
        "n_background": chosen["n"],
        "test_tpr": test_tpr,
        "n_signal": n_signal,
        "test_fpr": test_fpr,
        "n_noise": n_noise,
    }


def describe_calibration(calibrated):
    """The lines that report a calibrated threshold."""
    return [
        f"   Threshold: {calibrated['threshold']:.6g} (FPR ≤ "
        f"{calibrated['target_fpr']:.0%} on calib_bg)",
        f"   calib_bg: FPR {calibrated['calib_fpr']:.2%} reached "
        f"(n {calibrated['n_background']} noise samples)",
        f"   test:     TPR {calibrated['test_tpr']:.2%} (n {calibrated['n_signal']} "
        f"signal) at FPR {calibrated['test_fpr']:.2%} (n {calibrated['n_noise']} noise)",
    ]


def testset_record(built, part):
    """The test set and part a threshold comes from, as the registry stores it."""
    return {
        "name": built.spec.name,
        "seed": built.seed,
        "n": built.n,
        "part": part,
        "sha256": built.sha256,
    }


def run(
    model="latest",
    save=True,
    testset=DEFAULT_TESTSET,
    seed=TESTSET_SEED,
    n=TESTSET_SIZE,
):
    """Calibrate one model's threshold and save it unless save is False.

    The threshold comes from the part calib_bg of the test set testset, seed, n
    (gwd.evaluation.testsets.build) at THRESHOLD_TARGET_FPR; the part test only
    reports. Returns the exit code.
    """
    print("=" * 70)
    print("🎯 THRESHOLD CALIBRATION")
    print("=" * 70)

    # 1. Find the model
    print("\n📦 Loading the model...")
    try:
        model_path = resolve(model)
    except FileNotFoundError as error:
        print(f"❌ {error}")
        return 1

    # 2. The test set first: drawing it needs PyCBC, so a missing backend stops
    # before the model takes its time to load.
    try:
        built = testsets.build(testset, n, seed)
    except testsets.InvalidTestsetError as error:
        print(f"❌ {error}", file=sys.stderr)
        return 1
    print("\n".join(testsets.describe(built)))
    X_test, y_true = built.X, built.y

    loaded = load_model(model_path)
    name = os.path.basename(model_path)
    print(f"✓ Loaded: {name}")

    # 3. Predictions on both parts
    print("\n🧠 Computing predictions on calib_bg and test...")
    background_scores = loaded.predict(built.calib_X, verbose=0).flatten()
    y_scores = loaded.predict(X_test, verbose=0).flatten()
    print("✓ Done")

    # 4. The threshold on calib_bg, never on test
    print(
        f"\n🔍 Calibrating at FPR ≤ {THRESHOLD_TARGET_FPR:.0%} on calib_bg of "
        f"{built.label()}..."
    )
    calibrated = calibrate(background_scores, y_true, y_scores)
    print("\n".join(describe_calibration(calibrated)))

    # 5. Reference strategies on test, and the plot
    optimal_thresholds, roc_data = find_optimal_thresholds(y_true, y_scores)
    print("\n📈 Creating the plot...")
    visualize_thresholds(
        y_true, y_scores, optimal_thresholds, roc_data, calibrated, model_path
    )
    recommend_threshold(optimal_thresholds)

    # 6. Save (default; --no-save skips it)
    if not save:
        print("\nNot saved (--no-save).")
        return 0

    calibration_set = testset_record(built, "calib_bg")
    written = registry.set_threshold(
        model_path,
        calibrated["threshold"],
        calibrated["target_fpr"],
        calibrated["calib_fpr"],
        calibrated["n_background"],
        STRATEGY,
        calibration_set,
    )
    print(f"✓ Threshold saved in {written}")

    output = {
        "model": name,
        "testset": {**calibration_set, "part": "calib_bg and test"},
        "calibrated": {"strategy": STRATEGY, **calibrated},
        "reference_strategies_on_test": {
            k: {
                **v,
                "tpr": float(v["tpr"]),
                "fpr": float(v["fpr"]),
                "threshold": float(v["threshold"]),
            }
            for k, v in optimal_thresholds.items()
        },
    }

    report = report_path(model_path)
    ensure_dirs(os.path.dirname(report))
    with open(report, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✓ Report saved to {report}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit("The options are on the gwd command: gwd eval thresholds --help")
    sys.exit(run())
