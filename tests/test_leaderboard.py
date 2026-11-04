import csv
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from src.utils.evaluation import update_leaderboard_roc

pytestmark = pytest.mark.filterwarnings("ignore:Physics baseline currently invalid")

SHOW_LEADERBOARD = Path(__file__).resolve().parent.parent / "src" / "show_leaderboard.py"
COLUMNS = [
    "Model", "Date", "ROC_AUC", "Avg_Precision", "Sim_Accuracy", "TPR", "FPR",
    "Sim_SNR50", "Sim_SNR90", "Physik_Gap", "Real_Events_Found", "Real_Noise_FAR",
]


def sim_metrics(**overrides):
    metrics = {"roc_auc": 0.9, "avg_precision": 0.8, "acc": 0.75, "tpr": 0.7, "fpr": 0.01,
               "snr50": 2.0, "snr90": 3.0}
    metrics.update(overrides)
    return metrics


REAL_METRICS = {"found": 3, "total": 5, "far_text": "< 1.0/y"}


def write_row(tmp_path, **overrides):
    """Write one leaderboard row in tmp_path and return it as raw strings."""
    leaderboard = tmp_path / "leaderboard.csv"
    baseline = tmp_path / "physics_baseline.json"
    baseline.write_text(json.dumps({"SNR90": 1.0}))
    update_leaderboard_roc("model.keras", sim_metrics(**overrides), REAL_METRICS,
                           str(leaderboard), str(baseline))
    with leaderboard.open(newline="") as f:
        return list(csv.DictReader(f))[-1]


def run_show(tmp_path, code=None):
    """Run show_leaderboard with GWD_HOME in tmp_path, so the real leaderboard is never read."""
    args = ["-c", code] if code else [str(SHOW_LEADERBOARD)]
    return subprocess.run([sys.executable, *args], env={**os.environ, "GWD_HOME": str(tmp_path)},
                          capture_output=True, text=True)


def write_show_csv(tmp_path, roc_values, folder="results"):
    target = tmp_path / folder
    target.mkdir()
    rows = [f"m{i},{value}" for i, value in enumerate(roc_values)]
    (target / "model_leaderboard.csv").write_text("\n".join(["Model,ROC_AUC", *rows]) + "\n")


def test_physik_gap_ignores_baseline(tmp_path):
    assert write_row(tmp_path, snr90=2.0)["Physik_Gap"] == "-"


def test_baseline_file_is_not_read(tmp_path):
    leaderboard = tmp_path / "leaderboard.csv"
    update_leaderboard_roc("model.keras", sim_metrics(), REAL_METRICS,
                           str(leaderboard), str(tmp_path / "missing.json"))
    assert leaderboard.exists()


def test_invalid_baseline_warning_is_shown_once(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        write_row(tmp_path)
        write_row(tmp_path)
    messages = [str(w.message) for w in caught]
    assert messages == ["Physics baseline currently invalid (EVAL-6, see D5)"]


def test_zero_snr_is_valid(tmp_path):
    row = write_row(tmp_path, snr50=0.0, snr90=0.0)
    assert (row["Sim_SNR50"], row["Sim_SNR90"]) == ("0.00", "0.00")


def test_missing_snr_is_dash(tmp_path):
    row = write_row(tmp_path, snr50=None, snr90=None)
    assert (row["Sim_SNR50"], row["Sim_SNR90"]) == ("-", "-")


def test_schema_stays_twelve_columns(tmp_path):
    assert list(write_row(tmp_path)) == COLUMNS


def test_rows_are_appended(tmp_path):
    write_row(tmp_path)
    write_row(tmp_path)
    with (tmp_path / "leaderboard.csv").open(newline="") as f:
        assert len(list(csv.DictReader(f))) == 2


def test_console_output_has_real_newline(tmp_path, capsys):
    write_row(tmp_path)
    out = capsys.readouterr().out
    assert "\\n" not in out
    assert out.startswith("\n🏆")


def test_show_without_file_explains_and_exits_zero(tmp_path):
    result = run_show(tmp_path)
    assert result.returncode == 0
    assert "No leaderboard yet" in result.stdout
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "results" / "model_leaderboard.csv").exists()


def test_show_sorts_by_roc_auc_descending_with_failures_last(tmp_path):
    write_show_csv(tmp_path, ["0.80", "-", "0.95", "0.90"])
    result = run_show(tmp_path)
    assert result.returncode == 0
    positions = [result.stdout.index(name) for name in ("m2", "m3", "m0", "m1")]
    assert positions == sorted(positions)


def test_show_reads_a_leaderboard_migrated_from_analysis(tmp_path):
    write_show_csv(tmp_path, ["0.80", "0.95"], folder="analysis")
    result = run_show(tmp_path)
    assert result.returncode == 0
    assert "Copied leaderboard" in result.stdout
    assert result.stdout.index("m1") < result.stdout.rindex("m0")


def test_show_has_main_guard(tmp_path):
    write_show_csv(tmp_path, ["0.80"])
    code = f"import runpy; runpy.run_path({str(SHOW_LEADERBOARD)!r}, run_name='imported')"
    result = run_show(tmp_path, code)
    assert result.returncode == 0
    assert result.stdout == ""


def test_show_hides_old_physik_gap_and_keeps_csv(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    leaderboard = results / "model_leaderboard.csv"
    original = "Model,ROC_AUC,Physik_Gap\nm0,0.9,+1.39\n"
    leaderboard.write_text(original)

    result = run_show(tmp_path)

    assert result.returncode == 0
    assert "+1.39" not in result.stdout
    *table, note = result.stdout.strip().splitlines()
    assert [cell.strip() for cell in table[-1].split("|")][1:-1] == ["m0", "0.9", "-"]
    assert note == "Physik_Gap hidden: values were based on an invalid baseline (EVAL-6)."
    assert leaderboard.read_text() == original
