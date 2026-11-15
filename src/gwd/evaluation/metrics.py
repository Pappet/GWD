"""Metrics and threshold strategies as pure functions.

Nothing here loads a model, draws data, plots, or writes files: every function takes
labels and scores and returns numbers.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)


def calculate_operating_point(y_true, y_scores, threshold):
    """Return (TPR, FPR) at one threshold, to mark the operating point on the ROC curve."""
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return tpr, fpr


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
