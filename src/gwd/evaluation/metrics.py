"""Metrics and threshold strategies as pure functions.

Nothing here loads a model, draws data, plots, or writes files: every function takes
labels and scores and returns numbers.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
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


def finite_roc(fpr, tpr, thresholds):
    """Drop the points of a ROC curve whose threshold is not finite."""
    finite = np.isfinite(thresholds)
    return fpr[finite], tpr[finite], thresholds[finite]


def point_at_fpr(fpr, tpr, thresholds, target_fpr, y_scores):
    """The ROC point with the highest TPR among those with FPR <= target_fpr.

    Ties go to the lower FPR. If no point meets the target, the threshold lies just above the
    highest score: nothing is detected, so TPR and FPR are 0.
    """
    allowed = np.flatnonzero(fpr <= target_fpr)
    if len(allowed) == 0:
        return {
            "threshold": float(np.nextafter(np.max(y_scores), np.inf)),
            "tpr": 0.0,
            "fpr": 0.0,
        }
    best = allowed[tpr[allowed] == tpr[allowed].max()]
    i = best[np.argmin(fpr[best])]
    return {"threshold": thresholds[i], "tpr": tpr[i], "fpr": fpr[i]}


def point_at_tpr(fpr, tpr, thresholds, target_tpr):
    """The ROC point with the lowest FPR among those with TPR >= target_tpr.

    Ties go to the higher TPR. The lowest threshold detects every signal, so a point exists
    for any target up to 1.
    """
    allowed = np.flatnonzero(tpr >= target_tpr)
    best = allowed[fpr[allowed] == fpr[allowed].min()]
    i = best[np.argmax(tpr[best])]
    return {"threshold": thresholds[i], "tpr": tpr[i], "fpr": fpr[i]}


def f1_curve(y_true, y_scores):
    """F1 = 2PR / (P + R) at every threshold of precision_recall_curve, as (f1, thresholds).

    One pass over the sorted scores instead of one f1_score call per threshold. F1 is 0
    where precision and recall are both 0.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # The last point (precision 1, recall 0) has no threshold.
    precision, recall = precision[:-1], recall[:-1]
    total = precision + recall
    f1 = np.divide(
        2 * precision * recall, total, out=np.zeros_like(total), where=total > 0
    )
    return f1, thresholds


def find_optimal_thresholds(y_true, y_scores):
    """
    Compute thresholds by several criteria.

    Returns:
        (dict of threshold proposals by strategy name, (fpr, tpr, thresholds)). The ROC
        curve is the full one from roc_curve, for plotting; the proposals never use its
        infinite threshold.
    """
    roc = roc_curve(y_true, y_scores)
    # roc_curve starts with an infinite threshold (nothing detected); no strategy may choose
    # it, because it is no usable threshold and json writes it as the invalid 'Infinity'.
    fpr, tpr, thresholds = finite_roc(*roc)

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
    f1_scores, pr_thresholds = f1_curve(y_true, y_scores)
    f1_max_idx = np.argmax(f1_scores)

    # The TPR/FPR at exactly this threshold
    opt_threshold = pr_thresholds[f1_max_idx]
    opt_tpr, opt_fpr = calculate_operating_point(y_true, y_scores, opt_threshold)

    results["f1"] = {
        "threshold": opt_threshold,
        "tpr": opt_tpr,
        "fpr": opt_fpr,
        "f1_score": f1_scores[f1_max_idx],
        "description": "Maximizes the F1 score - good for unbalanced data",
    }

    # 3. Fixed FPR <= 0.05 (5% false alarm rate)
    target_fpr = 0.05
    results["fixed_fpr_5"] = {
        **point_at_fpr(fpr, tpr, thresholds, target_fpr, y_scores),
        "description": f"Maximizes TPR at FPR ≤ {target_fpr:.0%}",
    }

    # 4. Fixed FPR <= 0.01 (1% false alarm rate - conservative)
    target_fpr = 0.01
    results["fixed_fpr_1"] = {
        **point_at_fpr(fpr, tpr, thresholds, target_fpr, y_scores),
        "description": f"Maximizes TPR at FPR ≤ {target_fpr:.0%} - conservative",
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

    # 6. TPR >= 0.9 (90% detection rate guaranteed)
    target_tpr = 0.90
    results["fixed_tpr_90"] = {
        **point_at_tpr(fpr, tpr, thresholds, target_tpr),
        "description": f"Minimizes FPR at TPR ≥ {target_tpr:.0%}",
    }

    return results, roc
