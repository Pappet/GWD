#!/usr/bin/env python3
"""
Threshold optimizer: finds detection thresholds for one model from its ROC curve.

Strategies:
1. Youden's J statistic: maximizes (TPR - FPR)
2. F1 score: harmonic mean of precision and recall
3. Cost-based: weights the two error types differently
4. Fixed FPR: maximizes the TPR at a given FPR limit (5% and 1%)
5. Fixed TPR: minimizes the FPR at a TPR of 90%

Run it with ``gwd eval thresholds``; the results go to results/optimal_thresholds.json
unless --no-save is given.
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
    Compute thresholds by several criteria.

    Returns:
        (dict of threshold proposals by strategy name, (fpr, tpr, thresholds))
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)

    results = {}

    # 1. Youden's J statistic (maximizes TPR - FPR)
    j_scores = tpr - fpr
    j_max_idx = np.argmax(j_scores)
    results["youden"] = {
        "threshold": thresholds[j_max_idx],
        "tpr": tpr[j_max_idx],
        "fpr": fpr[j_max_idx],
        "j_stat": j_scores[j_max_idx],
        "description": "Maximizes (TPR - FPR) - balanced approach",
    }

    # 2. Maximum F1 score
    # F1 = 2 * (precision * recall) / (precision + recall)
    f1_scores = []
    for t in pr_thresholds:
        y_pred = (y_scores >= t).astype(int)
        f1_scores.append(f1_score(y_true, y_pred))

    f1_scores = np.array(f1_scores)
    f1_max_idx = np.argmax(f1_scores)

    # Find the TPR/FPR that belong to this threshold
    opt_threshold = pr_thresholds[f1_max_idx]
    opt_idx = np.argmin(np.abs(thresholds - opt_threshold))

    results["f1"] = {
        "threshold": opt_threshold,
        "tpr": tpr[opt_idx],
        "fpr": fpr[opt_idx],
        "f1_score": f1_scores[f1_max_idx],
        "description": "Maximizes the F1 score - good for unbalanced data",
    }

    # 3. Fixed FPR = 0.05 (5% false alarm rate)
    target_fpr = 0.05
    idx_fpr = np.argmin(np.abs(fpr - target_fpr))
    results["fixed_fpr_5"] = {
        "threshold": thresholds[idx_fpr],
        "tpr": tpr[idx_fpr],
        "fpr": fpr[idx_fpr],
        "description": f"Maximizes TPR at FPR ≈ {target_fpr:.0%}",
    }

    # 4. Fixed FPR = 0.01 (1% false alarm rate - conservative)
    target_fpr = 0.01
    idx_fpr = np.argmin(np.abs(fpr - target_fpr))
    results["fixed_fpr_1"] = {
        "threshold": thresholds[idx_fpr],
        "tpr": tpr[idx_fpr],
        "fpr": fpr[idx_fpr],
        "description": f"Maximizes TPR at FPR ≈ {target_fpr:.0%} - conservative",
    }

    # 5. Cost-based (example: false negatives are 5x worse than false positives)
    # Cost = FN_cost * FNR + FP_cost * FPR
    fn_cost = 5.0
    fp_cost = 1.0

    fnr = 1 - tpr  # false negative rate
    costs = fn_cost * fnr + fp_cost * fpr
    cost_min_idx = np.argmin(costs)

    results["cost_based"] = {
        "threshold": thresholds[cost_min_idx],
        "tpr": tpr[cost_min_idx],
        "fpr": fpr[cost_min_idx],
        "cost": costs[cost_min_idx],
        "description": f"Minimizes the cost (FN:{fn_cost}x, FP:{fp_cost}x)",
    }

    # 6. TPR = 0.9 (90% detection rate guaranteed)
    target_tpr = 0.90
    idx_tpr = np.argmin(np.abs(tpr - target_tpr))
    results["fixed_tpr_90"] = {
        "threshold": thresholds[idx_tpr],
        "tpr": tpr[idx_tpr],
        "fpr": fpr[idx_tpr],
        "description": f"Minimizes FPR at TPR ≈ {target_tpr:.0%}",
    }

    return results, (fpr, tpr, thresholds)


def visualize_thresholds(y_true, y_scores, optimal_thresholds, roc_data):
    """Plot the threshold options on the ROC curve and against the threshold."""

    fpr, tpr, thresholds = roc_data

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. ROC with all chosen points
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
    table_text += "║  CHOSEN THRESHOLDS - COMPARISON               ║\n"
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
    """Print which strategy suits which typical GW use."""

    print("\n" + "=" * 70)
    print("🎯 THRESHOLD RECOMMENDATIONS FOR TYPICAL USES")
    print("=" * 70)

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


def run(model="latest", save=True):
    """Find the thresholds of one model and save them unless save is False. Returns the exit code."""
    print("=" * 70)
    print("🎯 THRESHOLD OPTIMIZATION TOOL")
    print("=" * 70)

    # 1. Load the model
    print("\n📦 Loading the model...")
    try:
        model_path = resolve(model)
    except FileNotFoundError as error:
        print(f"❌ {error}")
        return 1

    loaded = load_model(model_path)
    name = os.path.basename(model_path)
    print(f"✓ Loaded: {name}")

    # 2. Generate the test data
    print(f"\n📊 Generating the test data ({EVAL_V1.name})...")
    X_test, y_true = make_test_set(EVAL_V1, 500)
    print(f"✓ {len(X_test)} samples generated")

    # 3. Predictions
    print("\n🧠 Computing predictions...")
    y_scores = loaded.predict(X_test, verbose=0).flatten()
    print("✓ Done")

    # 4. Find the thresholds
    print("\n🔍 Searching for thresholds...")
    optimal_thresholds, roc_data = find_optimal_thresholds(y_true, y_scores)
    print("✓ Found")

    # 5. Plot
    print("\n📈 Creating the plot...")
    visualize_thresholds(y_true, y_scores, optimal_thresholds, roc_data)

    # 6. Recommendations
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
    print(f"✓ Saved to {OPTIMAL_THRESHOLDS_FILE}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit("The options are on the gwd command: gwd eval thresholds --help")
    sys.exit(run())
