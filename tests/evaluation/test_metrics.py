"""The pure metrics of gwd.evaluation.metrics (P4.2)."""

import json
import subprocess
import sys

import numpy as np
import pytest
from sklearn.metrics import f1_score

from gwd.evaluation import metrics


def scores(seed=0, n=500, shift=1.5):
    """n noise and n signal scores; the signal scores are shifted up."""
    rng = np.random.RandomState(seed)
    y_true = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    y_scores = np.r_[rng.normal(0, 1, n), rng.normal(shift, 1, n)]
    return y_true, y_scores


def all_points(y_true, y_scores):
    """(threshold, tpr, fpr) at every distinct score, by brute force with '>='."""
    points = []
    for t in np.unique(y_scores):
        tpr, fpr = metrics.calculate_operating_point(y_true, y_scores, t)
        points.append((t, tpr, fpr))
    return points


@pytest.mark.parametrize("seed", range(5))
def test_fixed_fpr_1_stays_within_the_limit_with_the_best_tpr(seed):
    y_true, y_scores = scores(seed)
    best = max(tpr for _, tpr, fpr in all_points(y_true, y_scores) if fpr <= 0.01)

    chosen = metrics.find_optimal_thresholds(y_true, y_scores)[0]["fixed_fpr_1"]

    assert chosen["fpr"] <= 0.01
    assert np.isfinite(chosen["threshold"])
    assert chosen["tpr"] == best
    # The reported rates are the ones the threshold gives.
    assert metrics.calculate_operating_point(
        y_true, y_scores, chosen["threshold"]
    ) == pytest.approx((chosen["tpr"], chosen["fpr"]))


@pytest.mark.parametrize("seed", range(5))
def test_fixed_fpr_5_stays_within_the_limit(seed):
    y_true, y_scores = scores(seed)
    chosen = metrics.find_optimal_thresholds(y_true, y_scores)[0]["fixed_fpr_5"]
    assert chosen["fpr"] <= 0.05


@pytest.mark.parametrize("seed", range(5))
def test_fixed_tpr_90_reaches_the_target_with_the_lowest_fpr(seed):
    y_true, y_scores = scores(seed)
    best = min(fpr for _, tpr, fpr in all_points(y_true, y_scores) if tpr >= 0.9)

    chosen = metrics.find_optimal_thresholds(y_true, y_scores)[0]["fixed_tpr_90"]

    assert chosen["tpr"] >= 0.9
    assert chosen["fpr"] == best


def test_no_strategy_chooses_an_infinite_threshold():
    # Scores that rank noise above signal: Youden's J is largest at the infinite
    # threshold of roc_curve, where TPR = FPR = 0.
    y_true, y_scores = scores(shift=-3.0)

    results, roc = metrics.find_optimal_thresholds(y_true, y_scores)

    assert not np.isfinite(roc[2][0])  # The full curve keeps it for plotting.
    for name, result in results.items():
        assert np.isfinite(result["threshold"]), name
    json.dumps(
        {k: {**v, "threshold": float(v["threshold"])} for k, v in results.items()},
        allow_nan=False,
    )


def test_an_unreachable_fpr_limit_detects_nothing():
    # The highest score is noise, so every finite threshold has FPR >= 1/3.
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.9, 0.2, 0.1, 0.8, 0.7, 0.6])

    chosen = metrics.find_optimal_thresholds(y_true, y_scores)[0]["fixed_fpr_1"]

    assert (chosen["tpr"], chosen["fpr"]) == (0.0, 0.0)
    assert chosen["threshold"] > y_scores.max()
    assert metrics.predict_labels(y_scores, chosen["threshold"]).sum() == 0


@pytest.mark.parametrize("seed", range(3))
def test_vectorized_f1_matches_sklearn(seed):
    y_true, y_scores = scores(seed, shift=1.0)
    f1, thresholds = metrics.f1_curve(y_true, y_scores)
    assert len(f1) == len(thresholds)

    rng = np.random.RandomState(seed)
    for i in rng.choice(len(thresholds), 25, replace=False):
        expected = f1_score(y_true, metrics.predict_labels(y_scores, thresholds[i]))
        assert f1[i] == pytest.approx(expected, abs=1e-12)


def test_the_f1_strategy_reports_the_rates_of_its_threshold():
    y_true, y_scores = scores(1, shift=1.0)
    chosen = metrics.find_optimal_thresholds(y_true, y_scores)[0]["f1"]
    f1, _ = metrics.f1_curve(y_true, y_scores)

    assert chosen["f1_score"] == f1.max()
    assert metrics.calculate_operating_point(y_true, y_scores, chosen["threshold"]) == (
        chosen["tpr"],
        chosen["fpr"],
    )


def test_f1_is_zero_where_precision_and_recall_are_zero():
    # The highest score is noise: at that threshold precision and recall are both 0.
    f1, thresholds = metrics.f1_curve(np.array([0, 1]), np.array([0.9, 0.1]))
    assert f1[thresholds == 0.9] == 0.0
    assert np.all(np.isfinite(f1))


def test_a_score_equal_to_the_threshold_counts_as_detection():
    y_scores = np.array([0.1, 0.15, 0.2])
    assert list(metrics.predict_labels(y_scores, 0.15)) == [0, 1, 1]
    assert metrics.detection_rate(y_scores, 0.15) == pytest.approx(2 / 3)


def test_operating_point_and_confusion_counts_agree():
    y_true, y_scores = scores(2)
    tn, fp, fn, tp = metrics.confusion_counts(y_true, y_scores, 0.5)
    assert metrics.calculate_operating_point(y_true, y_scores, 0.5) == (
        tp / (tp + fn),
        fp / (fp + tn),
    )


def test_confusion_counts_with_one_predicted_class():
    # Nothing reaches the threshold: still four counts.
    counts = metrics.confusion_counts(np.array([0, 1]), np.array([0.1, 0.2]), 0.5)
    assert list(counts) == [1, 0, 1, 0]


def test_snr_at_rate():
    levels = np.linspace(0.0, 2.5, 6)
    assert metrics.snr_at_rate(levels, [0.1, 0.4, 0.5, 0.95, 1, 1], 0.5) == 1.0
    assert metrics.snr_at_rate(levels, [0.1, 0.4, 0.5, 0.95, 1, 1], 0.9) == 1.5
    # 0.0 is a valid result, and None means that no level reaches the rate.
    assert metrics.snr_at_rate(levels, [0.6] * 6, 0.5) == 0.0
    assert metrics.snr_at_rate(levels, [0.1] * 6, 0.5) is None


def test_format_snr():
    assert metrics.format_snr(0.0) == "0.00"
    assert metrics.format_snr(1.2345) == "1.23"
    assert metrics.format_snr(None) == "N/A"


def test_import_loads_no_plotting_and_no_tensorflow():
    code = (
        "import sys, gwd.evaluation.metrics; "
        "print('matplotlib' in sys.modules, 'tensorflow' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False False"


# --- per-bin metrics (P5.7) ---


def test_bin_index_closes_the_last_bin_and_marks_values_outside():
    edges = [1.0, 2.0, 3.0]
    index = metrics.bin_index([0.5, 1.0, 1.99, 2.0, 3.0, 3.5], edges)
    assert list(index) == [-1, 0, 0, 1, 1, -1]


def test_two_equal_edges_are_one_bin_of_that_value():
    assert list(metrics.bin_index([60.0, 60.0, 59.0], [60.0, 60.0])) == [0, 0, -1]


def test_signal_bin_metrics_counts_rates_and_auc_per_bin():
    values = np.array([1.5, 1.6, 2.5, 2.6])
    signal_scores = np.array([0.1, 0.9, 0.9, 0.95])
    noise_scores = np.array([0.0, 0.2])
    flags = np.array([True, True, False, True])
    rows = metrics.signal_bin_metrics(
        values, [1.0, 2.0, 3.0, 4.0], signal_scores, noise_scores, 0.5, flags
    )
    assert [row["n"] for row in rows] == [2, 2, 0]
    assert [row["tpr"] for row in rows] == [0.5, 1.0, None]
    # Bin 0: 0.1 and 0.9 against 0.0 and 0.2 -> 3 of 4 pairs ordered right.
    assert rows[0]["auc"] == pytest.approx(0.75)
    assert rows[1]["auc"] == 1.0
    assert [row["outside"] for row in rows] == [1.0, 0.5, None]
    assert rows[2]["auc"] is None
    assert (rows[0]["low"], rows[0]["high"]) == (1.0, 2.0)


def test_group_metrics_per_noise_source():
    groups = np.array(["real", "real", "gaussian", "gaussian", "gaussian"])
    y_true = np.array([1, 0, 1, 0, 0])
    y_scores = np.array([0.9, 0.6, 0.4, 0.1, 0.7])
    rows = metrics.group_metrics(groups, y_true, y_scores, 0.5, np.zeros(5, bool))
    by_group = {row["group"]: row for row in rows}
    assert [row["group"] for row in rows] == ["gaussian", "real"]
    assert by_group["real"]["tpr"] == 1.0 and by_group["real"]["fpr"] == 1.0
    assert by_group["real"]["auc"] == 1.0
    assert by_group["gaussian"]["tpr"] == 0.0
    assert by_group["gaussian"]["fpr"] == 0.5
    assert by_group["gaussian"]["auc"] == 0.5
    assert by_group["gaussian"]["outside"] == 0.0


def test_a_group_without_noise_has_no_fpr_or_auc():
    rows = metrics.group_metrics(
        np.array(["real"]), np.array([1]), np.array([0.9]), 0.5, np.array([False])
    )
    assert (rows[0]["fpr"], rows[0]["auc"], rows[0]["tpr"]) == (None, None, 1.0)
