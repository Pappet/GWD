#!/usr/bin/env python3
"""
Model evaluation: ROC analysis on simulated data and an optional field test on real LIGO data.

Run it with ``gwd eval run``; it writes one plot with six panels and adds a row to
leaderboard v2 (results/leaderboard_v2.csv, gwd.evaluation.leaderboard). The field test
(gwd.evaluation.real_data) runs only with ``--real``.

Since P5.8 the threshold is the model's own, from gwd.registry.get_threshold:
calibrated by gwd eval thresholds on the noise-only part calib_bg at FPR 1 % (D10 = A),
or 0.5 with a warning for a model without one. TPR, FPR, and SNR50/90 are reported at
that threshold, always with the FPR it reached on the part test and n.

Since P5.7 the simulated data is a fixed, hashed test set of gwd.evaluation.testsets
(--testset, --seed, --n; default stress_v1, which is EVAL_V1), drawn once and then
loaded. Its part 'test' gives the ROC curve, the operating point, and the efficiency
curve: the detection rate of its signal samples in each bin of the SNR the set was
drawn in (peak in V1; optimal in the V2 sets of P6.2), with the mean of the other SNR
of the bin's samples (P5.5); SNR50 and SNR90 are reported both ways, interpolated
between the bins (P5.8). The metrics are also reported per peak SNR bin, optimal SNR
bin, total mass bin, and noise source; a bin whose signal samples all lie outside the
training ranges is marked 'Extrapolation'.

Since V3 the data get the preprocessing the model was trained with (its metadata,
gwd.evaluation.testsets.set_for_model): a whitening model asked for indist_v2 or
stress_v2 is scored on indist_v3 or stress_v3, the same events whitened, and the field
test whitens the real data for it; the run says so.
"""

import datetime
import json
import os
import sys
from collections import Counter

import matplotlib
import numpy as np

from gwd.config import (
    DEFAULT_TESTSET,
    EVALUATION_MODELS_PLOTS_DIR,
    LEADERBOARD_FILE,
    TESTSET_SEED,
    TESTSET_SIZE,
    WAVEFORM_BACKEND,
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

from gwd import registry
from gwd.data.provenance import announce
from gwd.evaluation import testsets
from gwd.evaluation.leaderboard import add_row
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
from gwd.registry import load_model, model_name, resolve

# ==========================================
# CONFIGURATION
# ==========================================
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


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


def other_kind(kind):
    """The SNR kind a set is not drawn in: 'optimal' for 'peak' and the reverse."""
    return "optimal" if kind == "peak" else "peak"


def efficiency_curve(built, y_scores, threshold):
    """(bin centers, detection rates, mean other SNRs) per SNR bin of a test set.

    The signal samples of the set's part 'test' fill the bins of the SNR the set is
    drawn in (EvalSet.snr_edges(): peak SNR in V1, optimal SNR in V2); each bin gives
    the share of them that reach threshold and the mean of their other SNR (in V1 the
    optimal SNR, Sample.optimal_snr, measured in gwd.data.injection; in V2 the peak
    SNR). An empty bin gives NaN for both.
    """
    kind = built.spec.snr_kind
    edges = np.asarray(built.spec.snr_edges())
    signal = built.y == 1
    scores = np.asarray(y_scores)[signal]
    optimal = testsets.snr_values(built, other_kind(kind))[signal]
    index = bin_index(testsets.snr_values(built, kind)[signal], edges)
    rates, means = [], []
    for b in range(len(edges) - 1):
        mask = index == b
        found = mask.any()
        rates.append(detection_rate(scores[mask], threshold) if found else np.nan)
        means.append(float(np.mean(optimal[mask])) if found else np.nan)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, rates, means


def binned_metrics(built, y_scores, threshold):
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


def print_binned_metrics(report, threshold, training="TRAIN_V1"):
    print(
        f"\n   📋 Per bin (TPR at threshold {threshold:.4g}; the AUC of an SNR or mass "
        "bin is its signal samples against all noise samples):"
    )
    print(bin_table("Peak SNR", report["peak_snr"]))
    print(bin_table("Optimal SNR", report["optimal_snr"], ".1f"))
    print(bin_table("Total mass (M☉)", report["total_mass"], ".1f"))
    print(bin_table("Noise source", report["noise_source"]))
    print(
        "   'Extrapolation': all signal samples of the bin lie outside the training "
        f"ranges ({training})."
    )


def optimal_per_peak(snr_levels, optimal_snrs):
    """The mean optimal SNR per unit of peak SNR over the curve; None without signal."""
    total = float(np.sum(snr_levels))
    return float(np.sum(optimal_snrs)) / total if total > 0 else None


# ==========================================
# TEST 1: SIMULATION WITH ROC ANALYSIS
# ==========================================
def plot_path(model_path):
    """The plot of one run: plot_evaluate_model_with_roc_<model name>_<timestamp>.png."""
    return os.path.join(
        EVALUATION_MODELS_PLOTS_DIR,
        f"plot_evaluate_model_with_roc_{model_name(model_path)}_{TIMESTAMP}.png",
    )


def evaluate_simulation_with_roc(model, built, threshold, model_path):
    """ROC, precision-recall, efficiency curve, confusion matrix, and the binned
    metrics on the part 'test' of a test set (gwd.evaluation.testsets); one plot.

    threshold is the model's threshold (gwd.registry.get_threshold); the operating
    point, the efficiency curve, and SNR50/90 use it. model_path names the plot file
    (plot_path)."""
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

    # Operating point at the model's threshold, with n of each class
    op_tpr, op_fpr = calculate_operating_point(y_true_roc, y_scores, threshold)
    n_signal = int(np.sum(y_true_roc == 1))
    n_noise = int(np.sum(y_true_roc == 0))

    print(f"   ✅ ROC AUC: {roc_auc:.4f}")
    print(f"   ✅ Average Precision: {avg_precision:.4f}")
    print(
        f"   📍 Operating point (threshold {threshold:.4g}): TPR {op_tpr:.2%} "
        f"(n {n_signal}) at FPR {op_fpr:.2%} (n {n_noise})"
    )

    # --- B. Efficiency curve: detection rate per SNR bin of the signal samples ---
    kind = built.spec.snr_kind
    snr_levels, detection_rates, other_snrs = efficiency_curve(
        built, y_scores, threshold
    )
    if kind == "peak":
        peak_levels, optimal_levels = snr_levels, np.array(other_snrs)
    else:
        peak_levels, optimal_levels = np.array(other_snrs), snr_levels

    # SNR50 / SNR90: where the detection rate over the bin centers reaches 50% / 90%,
    # interpolated between the bins (P5.8), and the same over the mean other SNR of
    # the bins' samples (P5.5): peak from the centers in V1, optimal in V2.
    snr_50 = snr_at_rate(peak_levels, detection_rates, 0.5)
    snr_90 = snr_at_rate(peak_levels, detection_rates, 0.9)
    optimal_50 = snr_at_rate(optimal_levels, detection_rates, 0.5)
    optimal_90 = snr_at_rate(optimal_levels, detection_rates, 0.9)
    drawn_50, drawn_90 = (
        (snr_50, snr_90) if kind == "peak" else (optimal_50, optimal_90)
    )
    signal = y_true_roc == 1
    ratio = optimal_per_peak(built.peak_snr[signal], built.optimal_snr[signal])
    if ratio is not None:
        print(
            f"   ✅ Optimal SNR per peak SNR ({built.spec.name}, "
            f"{built.spec.scenario.name}): {ratio:.1f}"
        )

    print(
        f"   ✅ Peak SNR50 {format_snr(snr_50)}, SNR90 {format_snr(snr_90)} at FPR "
        f"{op_fpr:.2%} (threshold {threshold:.4g})"
    )

    report = binned_metrics(built, y_scores, threshold)
    print_binned_metrics(report, threshold, built.spec.training.name)

    # --- C. Confusion matrix at the threshold ---
    y_pred = predict_labels(y_scores, threshold)
    tn, fp, fn, tp = confusion_counts(y_true_roc, y_scores, threshold)
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
        op_fpr,
        op_tpr,
        "ro",
        markersize=10,
        label=f"Operating Point (τ={threshold:.4g})",
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
    ax3.set_title(f"Confusion Matrix (τ={threshold:.4g})", fontweight="bold")

    # 4. Efficiency curve
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(snr_levels, detection_rates, "o-", color="purple", lw=2)
    ax4.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50%")
    ax4.axhline(0.9, color="blue", linestyle="--", alpha=0.5, label="90%")
    if drawn_50 is not None:
        ax4.axvline(
            drawn_50, color="orange", linestyle=":", label=f"SNR50={drawn_50:.2f}"
        )
    if drawn_90 is not None:
        ax4.axvline(drawn_90, color="red", linestyle=":", label=f"SNR90={drawn_90:.2f}")
    ax4.text(
        0.98,
        0.04,
        f"at FPR {op_fpr:.2%}",
        transform=ax4.transAxes,
        ha="right",
        fontsize=9,
    )
    ax4.set_xlabel(f"Injected {kind} SNR (bin center)", fontweight="bold")
    if ratio and kind == "peak":
        # The optimal SNR is about proportional to the peak SNR for fixed masses.
        top = ax4.secondary_xaxis(
            "top", functions=(lambda x: x * ratio, lambda x: x / ratio)
        )
        top.set_xlabel(f"Optimal SNR (≈ {ratio:.1f} × peak)")
    elif ratio:
        top = ax4.secondary_xaxis(
            "top", functions=(lambda x: x / ratio, lambda x: x * ratio)
        )
        top.set_xlabel(f"Peak SNR (≈ optimal / {ratio:.1f})")
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
        threshold,
        color="black",
        linestyle="--",
        lw=2,
        label=f"Threshold={threshold:.4g}",
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
            ("Threshold:", f"{threshold:.4g}"),
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
        f"Comprehensive Model Evaluation: {model_name(model_path)} on "
        f"{built.label()}, SHA-256 {built.short_hash()}",
        fontsize=14,
        fontweight="bold",
    )
    path = plot_path(model_path)
    ensure_dirs(os.path.dirname(path))
    plt.savefig(path, dpi=150)
    print(f"📊 Plot: {path}")
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
        "n_signal": n_signal,
        "n_noise": n_noise,
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
        noise = scenario.noise_text()
    else:
        noise = (
            f"real for {scenario.real_noise_fraction:.0%} of samples if the test set "
            "was built with noise chunks, else gaussian"
        )
    return f"{noise} (test set {spec.name} = {scenario.name}); {field_test}"


def threshold_info(model_path):
    """The model's threshold for the run and the leaderboard.

    {"threshold", "fpr" (the target of its calibration), "testset_hash" (of the set it
    was calibrated on), "testset" (name, seed, n, part, sha256)}; the last three are
    None for the fallback, for which get_threshold warns.
    """
    threshold = registry.get_threshold(model_path)
    record = registry.threshold_record(model_path) or {}
    calibration = record.get("threshold_testset") or {}
    return {
        "threshold": threshold,
        "fpr": record.get("threshold_fpr"),
        "testset_hash": calibration.get("sha256"),
        "testset": calibration or None,
    }


def describe_threshold(info, built):
    """The line that names the threshold and where it comes from."""
    if info["testset"] is None:
        return (
            f"Threshold: {info['threshold']} (fallback: the model has no calibrated "
            f"threshold; gwd eval thresholds --model <m> --save)"
        )
    calibration = info["testset"]
    line = (
        f"Threshold: {info['threshold']:.6g} (FPR ≤ {info['fpr']:.0%} on calib_bg of "
        f"{calibration.get('name')} (seed {calibration.get('seed')}, "
        f"n {calibration.get('n')}))"
    )
    if calibration.get("sha256") != built.sha256:
        line += (
            f"; this run uses another test set, so the FPR on {built.spec.name} can "
            "differ from the target"
        )
    return line


def testset_info(built):
    """The test set columns of the leaderboard: name, seed, n, hash, scenario, ..."""
    try:
        with open(built.manifest_path, encoding="utf-8") as f:
            backend = json.load(f).get("waveform_backend")
    except (OSError, ValueError):
        backend = None
    sources = sorted(Counter(map(str, built.noise_source)))
    return {
        "name": built.spec.name,
        "seed": built.seed,
        "n": built.n,
        "sha256": built.sha256,
        "scenario": built.spec.scenario.name,
        "backend": backend or WAVEFORM_BACKEND,
        "noise_source": "+".join(sources),
    }


def run(
    model="latest",
    real=False,
    testset=DEFAULT_TESTSET,
    seed=TESTSET_SEED,
    n=TESTSET_SIZE,
):
    """Evaluate one model on a fixed test set and report it. Returns the exit code.

    testset, seed, and n name the set (gwd.evaluation.testsets.build: built once,
    then loaded). The threshold is the model's own (gwd.registry.get_threshold).
    real=True adds the real data field test, which needs network access (D10:
    optional, off by default); without it the field test cells of the row stay empty.
    Every run adds a row to leaderboard v2, which records the test set and its hash.
    """
    try:
        model_path = resolve(model)
    except FileNotFoundError as error:
        print(f"❌ {error}")
        return 1
    testset, preprocessing, note = testsets.set_for_model(testset, model_path)
    spec = testsets.get(testset)

    print("=" * 70)
    print(f"🤖 COMPREHENSIVE EVALUATION: {os.path.basename(model_path)}")
    announce(noise_description(spec, real))
    print(f"Preprocessing of the model: {preprocessing}")
    if note:
        print(note)

    # The set first: drawing it needs PyCBC, so a missing backend stops before the model.
    try:
        built = testsets.build(testset, n, seed)
    except testsets.InvalidTestsetError as error:
        print(f"❌ {error}", file=sys.stderr)
        return 1
    print("\n".join(testsets.describe(built)))

    info = threshold_info(model_path)
    threshold = info["threshold"]
    print(describe_threshold(info, built))

    loaded = load_model(model_path)

    # 1. Simulation with ROC
    sim_res = evaluate_simulation_with_roc(loaded, built, threshold, model_path)

    # 2. Real Data
    if real:
        real_res = run_field_test(loaded, threshold, preprocessing=preprocessing)
    else:
        print("\nField test skipped; gwd eval run --real runs it (network).")
        real_res = NO_FIELD_TEST

    # 3. Final Report
    fpr = f"at FPR {sim_res['fpr']:.2%}"
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY:")
    print("=" * 70)
    print(f"  Test set:             {built.label()}")
    print(f"  Test set SHA-256:     {built.sha256}")
    print(f"  ROC AUC:              {sim_res['roc_auc']:.4f}")
    print(f"  Average Precision:    {sim_res['avg_precision']:.4f}")
    print(
        f"  Threshold:            {threshold:.6g}"
        + ("" if info["testset"] else " (fallback)")
    )
    print(
        f"  TPR at threshold:     {sim_res['tpr']:.2%} (n {sim_res['n_signal']}) "
        f"at FPR {sim_res['fpr']:.2%} (n {sim_res['n_noise']})"
    )
    print(f"  Accuracy:             {sim_res['acc']:.1%}")
    print(f"  Peak SNR50:           {format_snr(sim_res['snr50'])} {fpr}")
    print(f"  Peak SNR90:           {format_snr(sim_res['snr90'])} {fpr}")
    print(f"  Optimal SNR90:        {format_snr(sim_res['optimal_snr90'])} {fpr}")
    print(f"  Real Events Found:    {real_res['events_text']}")
    print(f"  Real Noise:           {real_res['far_text']}")
    print("=" * 70)

    add_row(
        os.path.basename(model_path),
        testset_info(built),
        sim_res,
        real_res,
        info,
        LEADERBOARD_FILE,
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit("The options are on the gwd command: gwd eval run --help")
    sys.exit(run())
