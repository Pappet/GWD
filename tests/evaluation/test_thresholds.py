"""The per-model threshold at a fixed false alarm rate (P5.8, D10 = A).

threshold_at_fpr picks the threshold on noise-only scores (the part calib_bg of a test
set), never on the part test. Everything runs on synthetic scores or stub models in a
tmp folder; nothing reads or writes the real models_registry/ or results/.
"""

import json
import os

import numpy as np
import pytest

from gwd import registry
from gwd.evaluation import metrics, testsets, thresholds


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


# --- gwd eval thresholds ---


class PeakModel:
    """A deterministic score from the data: the peak of |x|, squashed to (0, 1)."""

    def predict(self, X, verbose=0):
        peak = np.abs(np.asarray(X)).max(axis=(1, 2))
        return (1 - np.exp(-peak / 5)).reshape(-1, 1)


class ConstantModel:
    """An untrained model that fires on everything: TPR = FPR = 1 at any threshold <= 1."""

    def predict(self, X, verbose=0):
        return np.ones((len(X), 1))


@pytest.fixture(scope="module")
def small_set(tmp_path_factory):
    """stress_v1 with 64 samples per part, PyCBC chirps, in a tmp folder."""
    root = tmp_path_factory.mktemp("sets")
    return testsets.build(
        "stress_v1",
        64,
        5,
        data_dir=str(root / "eval"),
        manifest_dir=str(root / "manifests"),
        noise_dir=str(root / "noise"),
    )


@pytest.fixture
def stubbed(monkeypatch, tmp_path, small_set):
    """thresholds.run on the small set, with a model file in tmp_path."""
    model_path = tmp_path / "models_registry" / "gwd_model_20260101-000000.keras"
    model_path.parent.mkdir()
    model_path.write_bytes(b"")
    state = {"model": PeakModel(), "path": str(model_path)}
    monkeypatch.setattr(thresholds, "resolve", lambda model: state["path"])
    monkeypatch.setattr(thresholds, "load_model", lambda path: state["model"])
    monkeypatch.setattr(thresholds.testsets, "build", lambda name, n, seed: small_set)
    monkeypatch.setattr(
        thresholds, "EVALUATION_MODELS_PLOTS_DIR", str(tmp_path / "plots")
    )
    monkeypatch.setattr(
        thresholds, "THRESHOLD_REPORTS_DIR", str(tmp_path / "results" / "thresholds")
    )
    return state


def test_run_calibrates_on_calib_bg_and_saves_in_the_registry(stubbed, small_set):
    assert thresholds.run() == 0
    record = registry.threshold_record(stubbed["path"])
    background = PeakModel().predict(small_set.calib_X).flatten()
    expected = metrics.threshold_at_fpr(background, 0.01)
    assert record["threshold"] == expected["threshold"]
    assert record["threshold_fpr"] == 0.01
    assert record["threshold_fpr_achieved"] == expected["fpr"] <= 0.01
    assert record["threshold_n_background"] == 64
    assert record["threshold_strategy"] == thresholds.STRATEGY
    assert record["threshold_testset"] == {
        "name": "stress_v1",
        "seed": 5,
        "n": 64,
        "part": "calib_bg",
        "sha256": small_set.sha256,
    }
    assert registry.get_threshold(stubbed["path"]) == expected["threshold"]


def test_the_threshold_never_comes_from_test(stubbed, small_set, monkeypatch):
    """Scores on test change nothing; scores on calib_bg decide."""

    class TestOnlyChanges(PeakModel):
        def predict(self, X, verbose=0):
            scores = super().predict(X)
            if X is not small_set.calib_X and len(X) == small_set.n:
                return np.full_like(scores, 0.123)
            return scores

    stubbed["model"] = TestOnlyChanges()
    monkeypatch.setattr(type(small_set), "calib_X", small_set.calib_X)
    assert thresholds.run() == 0
    first = registry.get_threshold(stubbed["path"])
    stubbed["model"] = PeakModel()
    assert thresholds.run() == 0
    assert registry.get_threshold(stubbed["path"]) == first


def test_each_model_gets_its_own_report(stubbed, tmp_path, capsys):
    """A loop over models keeps every report; before, each run overwrote the last."""
    first = stubbed["path"]
    second = str(tmp_path / "models_registry" / "gwd_model_20260202-000000.keras")
    assert thresholds.run() == 0
    stubbed["path"] = second
    assert thresholds.run() == 0
    reports = tmp_path / "results" / "thresholds"
    assert sorted(p.name for p in reports.iterdir()) == [
        "gwd_model_20260101-000000.json",
        "gwd_model_20260202-000000.json",
    ]
    for path in (first, second):
        report = json.loads((reports / f"{registry.model_name(path)}.json").read_text())
        assert report["model"] == os.path.basename(path)
        assert report["calibrated"]["threshold"] == registry.get_threshold(path)
    assert f"Report saved to {reports / 'gwd_model_20260202-000000.json'}" in (
        capsys.readouterr().out
    )
    assert not (tmp_path / "results" / "optimal_thresholds.json").exists()


def test_no_save_writes_nothing(stubbed, tmp_path):
    assert thresholds.run(save=False) == 0
    assert registry.read_metadata(stubbed["path"]) is None
    assert not (tmp_path / "results").exists()


def test_an_untrained_model_gets_a_threshold_that_detects_nothing(stubbed, capsys):
    """evaluation:MISS-1: a model that always fires must not look sensitive."""
    stubbed["model"] = ConstantModel()
    assert thresholds.run() == 0
    threshold = registry.get_threshold(stubbed["path"])
    assert threshold > 1.0
    assert (
        "TPR 0.00% (n 32 signal) at FPR 0.00% (n 32 noise)" in capsys.readouterr().out
    )


def test_calibrate_reports_tpr_and_fpr_on_test_with_n():
    background = np.arange(100) / 100
    y_true = np.array([1, 1, 1, 0, 0])
    y_scores = np.array([0.995, 0.999, 0.5, 0.999, 0.1])
    result = thresholds.calibrate(background, y_true, y_scores)
    assert result["threshold"] == pytest.approx(0.99)
    assert result["calib_fpr"] == 0.01
    assert (result["test_tpr"], result["n_signal"]) == (pytest.approx(2 / 3), 3)
    assert (result["test_fpr"], result["n_noise"]) == (0.5, 2)
