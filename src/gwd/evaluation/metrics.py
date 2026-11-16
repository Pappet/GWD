"""Metrics and threshold strategies as pure functions.

Nothing here loads a model, draws data, plots, or writes files: every function takes
labels and scores and returns numbers.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def predict_labels(y_scores, threshold):
    """1 where a score reaches the threshold, else 0.

    The one detection rule of the evaluation: a score counts if it is >= the threshold, as
    in roc_curve and precision_recall_curve.
    """
    return (np.asarray(y_scores) >= threshold).astype(int)


def detection_rate(y_scores, threshold):
    """The share of scores that reach the threshold."""
    return float(np.mean(predict_labels(y_scores, threshold)))


def confusion_counts(y_true, y_scores, threshold):
    """(tn, fp, fn, tp) at one threshold, also when a class is missing."""
    y_pred = predict_labels(y_scores, threshold)
    return confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()


def snr_at_rate(snr_levels, rates, level):
    """The first SNR whose detection rate reaches level, or None if none does.

    0.0 is a valid answer (a model that fires on pure noise), so test the result with
    'is None', not for truth.
    """
    reached = np.flatnonzero(np.asarray(rates) >= level)
    return float(snr_levels[reached[0]]) if len(reached) else None


def format_snr(snr):
    """An SNR from snr_at_rate for display: two decimals, or N/A for None."""
    return "N/A" if snr is None else f"{snr:.2f}"


def auc_or_none(y_true, y_scores):
    """The ROC AUC, or None if y_true lacks a class."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_scores))


def bin_index(values, edges):
    """The bin of each value: i for edges[i] <= v < edges[i+1], -1 outside.

    The last bin includes its upper edge. Two equal edges make one bin that holds
    exactly that value (a fixed mass).
    """
    values = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if len(edges) == 2 and edges[0] == edges[1]:
        return np.where(values == edges[0], 0, -1)
    index = np.searchsorted(edges, values, side="right") - 1
    index[values == edges[-1]] = len(edges) - 2
    index[(values < edges[0]) | (values > edges[-1])] = -1
    return index


def signal_bin_metrics(values, edges, signal_scores, noise_scores, threshold, flags):
    """One row per bin of the signal samples' values.

    values, signal_scores, and flags (True: outside the training ranges) belong to the
    signal samples; noise_scores to all noise-only samples. Each row has the bin edges
    (low, high), n, tpr (the detection rate at threshold), auc (the bin's signal
    samples against all noise samples), and outside (the share of flagged samples);
    tpr, auc, and outside are None for an empty bin.
    """
    signal_scores = np.asarray(signal_scores)
    noise_scores = np.asarray(noise_scores)
    flags = np.asarray(flags, dtype=bool)
    index = bin_index(values, edges)
    rows = []
    for b in range(len(edges) - 1):
        mask = index == b
        n = int(mask.sum())
        scores = signal_scores[mask]
        labels = np.r_[np.ones(n), np.zeros(len(noise_scores))]
        rows.append(
            {
                "low": float(edges[b]),
                "high": float(edges[b + 1]),
                "n": n,
                "tpr": detection_rate(scores, threshold) if n else None,
                "auc": auc_or_none(labels, np.r_[scores, noise_scores]) if n else None,
                "outside": float(flags[mask].mean()) if n else None,
            }
        )
    return rows


def group_metrics(groups, y_true, y_scores, threshold, flags):
    """One row per group (for example the noise source) of a labelled set.

    Each row has the group, n_signal, n_noise, tpr and fpr at threshold, the auc of
    the group alone, and outside (the share of its signal samples that flags marks).
    A rate or AUC whose class is missing in the group is None.
    """
    groups = np.asarray(groups)
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    flags = np.asarray(flags, dtype=bool)
    rows = []
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        signal = mask & (y_true == 1)
        noise = mask & (y_true == 0)
        rows.append(
            {
                "group": str(group),
                "n_signal": int(signal.sum()),
                "n_noise": int(noise.sum()),
                "tpr": detection_rate(y_scores[signal], threshold)
                if signal.any()
                else None,
                "fpr": detection_rate(y_scores[noise], threshold)
                if noise.any()
                else None,
                "auc": auc_or_none(y_true[mask], y_scores[mask]),
                "outside": float(flags[signal].mean()) if signal.any() else None,
            }
        )
    return rows


def calculate_operating_point(y_true, y_scores, threshold):
    """Return (TPR, FPR) at one threshold, to mark the operating point on the ROC curve."""
    tn, fp, fn, tp = confusion_counts(y_true, y_scores, threshold)

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
