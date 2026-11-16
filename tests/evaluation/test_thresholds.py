"""The per-model threshold at a fixed false alarm rate (P5.8, D10 = A).

threshold_at_fpr picks the threshold on noise-only scores (the part calib_bg of a test
set), never on the part test. Everything runs on synthetic scores or stub models in a
tmp folder; nothing reads or writes the real models_registry/ or results/.
"""

import numpy as np
import pytest

from gwd.evaluation import metrics


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n", [100, 1000, 1234])
def test_the_strategy_meets_fpr_1_percent_on_synthetic_scores(seed, n):
    rng = np.random.default_rng(seed)
    background = rng.normal(0.0, 1.0, n)
    chosen = metrics.threshold_at_fpr(background, 0.01)
    alarms = np.sum(background >= chosen["threshold"])
    assert alarms / n <= 0.01
    assert chosen["fpr"] == alarms / n
    assert chosen["n"] == n
    # As low as the target allows: the next lower score would exceed it.
    lower = background[background < chosen["threshold"]].max()
    assert np.sum(background >= lower) / n > 0.01


def test_fpr_1_percent_of_1000_allows_exactly_10_alarms():
    background = np.arange(1000) / 1000
    chosen = metrics.threshold_at_fpr(background, 0.01)
    assert chosen["threshold"] == 0.99
    assert chosen["fpr"] == 0.01


def test_ties_at_the_top_detect_nothing_instead_of_exceeding_the_target():
    """An untrained model that scores everything 1.0 cannot reach FPR <= 1 %."""
    chosen = metrics.threshold_at_fpr(np.ones(500), 0.01)
    assert chosen["threshold"] > 1.0
    assert chosen["fpr"] == 0.0
    assert metrics.detection_rate(np.ones(500), chosen["threshold"]) == 0.0


def test_too_few_background_samples_detect_nothing():
    chosen = metrics.threshold_at_fpr([0.1, 0.2, 0.3], 0.01)
    assert chosen["threshold"] > 0.3
    assert chosen["fpr"] == 0.0


def test_threshold_at_fpr_rejects_empty_scores_and_bad_targets():
    with pytest.raises(ValueError, match="No background"):
        metrics.threshold_at_fpr([], 0.01)
    with pytest.raises(ValueError, match="target_fpr"):
        metrics.threshold_at_fpr([0.5], 1.5)


def test_rate_at_counts_and_handles_no_scores():
    assert metrics.rate_at([0.1, 0.5, 0.9], 0.5) == (pytest.approx(2 / 3), 3)
    assert metrics.rate_at([], 0.5) == (None, 0)
