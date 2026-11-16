"""Leaderboard v2 (P5.8): raw numbers, the test set hash, ranking by ROC AUC.

The moratorium tests of P0.4 (Physik_Gap always '-') end here: v2 has no Physik_Gap
and no baseline at all (D5 = B). Every test writes into tmp_path or a tmp GWD_HOME;
the real results/ is never read or written.
"""

import csv
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from gwd.evaluation import leaderboard
from gwd.evaluation.real_data import NO_FIELD_TEST, EventCount

TESTSET = {
    "name": "stress_v1",
    "seed": 0,
    "n": 1000,
    "sha256": "0123456789abcdef" * 4,
    "scenario": "EVAL_V1",
    "backend": "pycbc 2.8.3",
    "noise_source": "gaussian",
}
CALIBRATED = {"threshold": 0.83, "fpr": 0.01, "testset_hash": "f" * 64}
FALLBACK = {"threshold": 0.5, "fpr": None, "testset_hash": None}
FIELD_TEST = {
    "events_text": "3/4; OOD n/a (0 of 1 loaded)",
    "far_text": "2 false alarms in 0.0556 h",
    "in_range": EventCount(3, 4, 4),
    "ood": EventCount(0, 0, 1),
    "false_alarms": 2,
    "seconds": 200.0,
}


def sim(**overrides):
    metrics = {
        "roc_auc": np.float64(0.9),
        "avg_precision": 0.8,
        "acc": 0.75,
        "tpr": 0.6,
        "fpr": 0.008,
        "n_signal": 500,
        "n_noise": 500,
        "snr50": 1.25,
        "snr90": None,
        "optimal_snr50": 17.0,
        "optimal_snr90": None,
    }
    metrics.update(overrides)
    return metrics


def add(path, model="m.keras", real=NO_FIELD_TEST, threshold=CALIBRATED, **metrics):
    return leaderboard.add_row(model, TESTSET, sim(**metrics), real, threshold, path)


def run_python(home, *args):
    """Run Python with GWD_HOME in home, so no real file is ever read or written."""
    return subprocess.run(
        [sys.executable, *args],
        env={**os.environ, "GWD_HOME": str(home), "MPLBACKEND": "Agg"},
        cwd=home,
        capture_output=True,
        text=True,
    )


def test_the_csv_holds_raw_numbers(tmp_path):
    path = tmp_path / "board.csv"
    add(path, real=FIELD_TEST)
    df = leaderboard.read_leaderboard(path)
    assert tuple(df.columns) == leaderboard.COLUMNS
    row = df.iloc[0]
    assert row["schema_version"] == 2
    for column in ("roc_auc", "ap", "threshold", "tpr_at_fpr", "fpr_achieved", "snr50"):
        assert df[column].dtype.kind == "f", column
    assert (row["roc_auc"], row["tpr_at_fpr"], row["fpr_achieved"]) == (0.9, 0.6, 0.008)
    assert (row["n_signal"], row["n_noise"]) == (500, 500)
    assert (row["real_found"], row["real_loaded"], row["real_total"]) == (3, 4, 4)
    assert (row["real_ood_loaded"], row["real_ood_total"]) == (0, 1)
    assert (row["real_false_alarms"], row["real_noise_seconds"]) == (2, 200.0)
    assert row["testset_hash"] == TESTSET["sha256"]
    assert row["threshold_testset_hash"] == "f" * 64
    # No formatted strings: no %, no '/', no '-' placeholders.
    text = path.read_text()
    assert "%" not in text and "/y" not in text and ",-," not in text


def test_missing_values_are_empty_cells(tmp_path):
    path = tmp_path / "board.csv"
    add(path, threshold=FALLBACK)
    with path.open(newline="") as f:
        row = next(csv.DictReader(f))
    assert row["snr90"] == "" and row["optimal_snr90"] == ""
    assert row["threshold_fpr"] == "" and row["threshold_testset_hash"] == ""
    assert {row[c] for c in leaderboard.COLUMNS if c.startswith("real_")} == {""}
    assert leaderboard.read_leaderboard(path)["snr90"].isna().all()


def test_zero_snr_is_kept_as_zero(tmp_path):
    path = tmp_path / "board.csv"
    add(path, snr50=0.0)
    assert leaderboard.read_leaderboard(path)["snr50"].iloc[0] == 0.0


def test_rows_are_appended_and_old_rows_stay_byte_identical(tmp_path):
    path = tmp_path / "board.csv"
    add(path, model="first.keras")
    before = path.read_bytes()
    add(path, model="second.keras", roc_auc=0.7)
    assert path.read_bytes().startswith(before)
    assert list(leaderboard.read_leaderboard(path)["model"]) == [
        "first.keras",
        "second.keras",
    ]


def test_a_file_with_another_header_is_refused(tmp_path):
    path = tmp_path / "board.csv"
    original = "Model,Date,ROC_AUC,Physik_Gap\nm0,x,0.9,+1.39\n"
    path.write_text(original)
    with pytest.raises(ValueError, match="another header"):
        add(path)
    assert path.read_text() == original


def test_no_physik_gap_or_baseline():
    """D5 = B: the column and the baseline are gone (moratorium of P0.4 ends here)."""
    assert not any("physik" in c.lower() or "gap" in c for c in leaderboard.COLUMNS)
    assert "baseline" not in leaderboard.add_row.__code__.co_varnames


def test_an_untrained_model_is_not_ranked_well():
    """evaluation:MISS-1: TPR = FPR = 1 and the lowest SNR50 must not win."""
    df = pd.DataFrame(
        {
            "model": ["fires_always", "good", "fallback_good"],
            "roc_auc": [0.5, 0.95, 0.95],
            "tpr_at_fpr": [1.0, 0.6, 0.4],
            "fpr_achieved": [1.0, 0.01, 0.02],
            "snr50": [0.5, 1.5, 1.7],
        }
    )
    assert list(leaderboard.rank(df)["model"]) == [
        "good",
        "fallback_good",
        "fires_always",
    ]


def test_rank_puts_missing_auc_last():
    df = pd.DataFrame({"model": list("abc"), "roc_auc": [0.8, np.nan, 0.9]})
    df["tpr_at_fpr"] = 0.5
    assert list(leaderboard.rank(df)["model"]) == ["c", "a", "b"]


def write_legacy(home):
    path = home / "results" / "model_leaderboard.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    original = (
        "Model,Date,ROC_AUC,Physik_Gap,Real_Noise_FAR\n"
        "old.keras,2025-12-05 23:36,0.9092,+0.67,n/a\n"
    ).encode()
    path.write_bytes(original)
    return path, original


EVALUATE_IN_HOME = """
import numpy as np
from gwd import registry
from gwd.evaluation import evaluate

class PeakModel:
    def predict(self, X, verbose=0):
        peak = np.abs(np.asarray(X)).max(axis=(1, 2))
        return (1 - np.exp(-peak / 5)).reshape(-1, 1)

evaluate.resolve = lambda model: "gwd_model_20260101-000000.keras"
evaluate.load_model = lambda path: PeakModel()
registry.get_threshold = lambda path: 0.123
raise SystemExit(evaluate.run(testset="stress_v1", seed=2, n=16))
"""


def test_evaluate_writes_v2_in_a_tmp_home_and_keeps_the_old_csv(tmp_path):
    """The checks of the plan: roc_auc is a float column, the old CSV byte-identical."""
    legacy, original = write_legacy(tmp_path)
    result = run_python(tmp_path, "-W", "ignore", "-c", EVALUATE_IN_HOME)
    assert result.returncode == 0, result.stderr[-2000:]
    df = pd.read_csv(tmp_path / "results" / "leaderboard_v2.csv")
    assert df["roc_auc"].dtype.kind == "f"
    assert df["threshold"].iloc[0] == 0.123
    assert df["testset"].iloc[0] == "stress_v1"
    assert (df["testset_seed"].iloc[0], df["testset_n"].iloc[0]) == (2, 16)
    assert len(df["testset_hash"].iloc[0]) == 64
    assert legacy.read_bytes() == original


def show(home):
    result = run_python(home, "-m", "gwd.evaluation.leaderboard")
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    return result.stdout


def test_show_without_file_explains_and_exits_zero(tmp_path):
    out = show(tmp_path)
    assert "No leaderboard yet" in out
    assert not (tmp_path / "results").exists()


def test_show_ranks_per_test_set_and_formats_only_there(tmp_path):
    path = tmp_path / "results" / "leaderboard_v2.csv"
    add(path, model="weak.keras", roc_auc=0.8)
    add(path, model="fallback.keras", roc_auc=0.95, threshold=FALLBACK)
    add(path, model="strong.keras", roc_auc=0.9, real=FIELD_TEST)
    out = show(tmp_path)
    assert (
        f"Test set stress_v1 (seed 0, n 1000), SHA-256 {TESTSET['sha256'][:12]}" in out
    )
    positions = [out.index(m) for m in ("fallback.keras", "strong.keras", "weak.keras")]
    assert positions == sorted(positions)
    assert "0.5 (fallback)" in out
    assert "60.0%" in out and "0.80%" in out and "500/500" in out
    assert "3/4" in out and "n/a (0 of 1 loaded)" in out and "2 in 0.0556 h" in out
    assert "%" not in path.read_text()


def test_show_names_the_archives_and_never_changes_them(tmp_path):
    legacy, original = write_legacy(tmp_path)
    analysis = tmp_path / "analysis" / "model_leaderboard.csv"
    analysis.parent.mkdir()
    analysis.write_bytes(original)
    out = show(tmp_path)
    assert "No leaderboard yet" in out
    assert f"Archive, not shown: {legacy} (1 rows" in out
    assert f"Archive, not shown: {analysis} (1 rows" in out
    assert "old.keras" not in out and "+0.67" not in out
    assert legacy.read_bytes() == original and analysis.read_bytes() == original
    assert not (tmp_path / "results" / "leaderboard_v2.csv").exists()


def test_the_old_leaderboard_stays_readable(tmp_path):
    legacy, _ = write_legacy(tmp_path)
    df = leaderboard.read_legacy(legacy)
    assert df.loc[0, "Physik_Gap"] == "+0.67"
    assert df.loc[0, "Real_Noise_FAR"] == "n/a"


def test_show_has_main_guard(tmp_path):
    code = "import runpy; runpy.run_module('gwd.evaluation.leaderboard', run_name='x')"
    result = run_python(tmp_path, "-c", code)
    assert result.returncode == 0
    assert result.stdout == ""
