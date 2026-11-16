"""Where the results go (P1.2), and since P5.8 that the old leaderboards stay archives.

Before P5.8, show() and the writer copied analysis/model_leaderboard.csv to
results/model_leaderboard.csv once. Leaderboard v2 writes results/leaderboard_v2.csv
only; no old file is copied, created, or changed any more.
"""

import os
import subprocess
import sys

HEADER = "Model,Date,ROC_AUC,Physik_Gap\n"
ANALYSIS_CSV = HEADER + "from_analysis.keras,2025-12-05 23:36,0.9092,+0.67\n"
ROOT_CSV = HEADER + "from_root.keras,2025-11-21 18:00,0.5,-\n"


def run_python(home, *args):
    """Run Python with GWD_HOME in home, so no real file is ever read or written."""
    return subprocess.run(
        [sys.executable, *args],
        env={**os.environ, "GWD_HOME": str(home)},
        cwd=home,
        capture_output=True,
        text=True,
    )


def write_legacy(home):
    analysis = home / "analysis" / "model_leaderboard.csv"
    analysis.parent.mkdir()
    analysis.write_bytes(ANALYSIS_CSV.encode())
    root = home / "model_leaderboard.csv"
    root.write_bytes(ROOT_CSV.encode())
    return analysis, root


def old_target(home):
    return home / "results" / "model_leaderboard.csv"


def test_import_config_does_not_create_results(tmp_path):
    result = run_python(tmp_path, "-c", "import gwd.config")
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "results").exists()


def test_show_copies_nothing_and_names_the_archives(tmp_path):
    analysis, root = write_legacy(tmp_path)
    result = run_python(tmp_path, "-m", "gwd.evaluation.leaderboard")
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "results").exists()
    assert analysis.read_bytes() == ANALYSIS_CSV.encode()
    assert root.read_bytes() == ROOT_CSV.encode()
    assert "from_analysis.keras" not in result.stdout
    assert f"Archive, not shown: {analysis}" in result.stdout
    assert f"Archive, not shown: {root}" in result.stdout


WRITE_ROW = """
from gwd import config
from gwd.evaluation.leaderboard import add_row
from gwd.evaluation.real_data import NO_FIELD_TEST
testset = dict(name="stress_v1", seed=0, n=10, sha256="a" * 64, scenario="EVAL_V1",
               backend="pycbc", noise_source="gaussian")
sim = dict(roc_auc=0.9, avg_precision=0.8, tpr=0.6, fpr=0.01, n_signal=5, n_noise=5)
threshold = dict(threshold=0.5, fpr=None, testset_hash=None)
add_row("new.keras", testset, sim, NO_FIELD_TEST, threshold, config.LEADERBOARD_FILE)
"""


def test_the_writer_creates_only_leaderboard_v2(tmp_path):
    analysis, root = write_legacy(tmp_path)
    result = run_python(tmp_path, "-c", WRITE_ROW)
    assert result.returncode == 0, result.stderr
    assert "new.keras" in (tmp_path / "results" / "leaderboard_v2.csv").read_text()
    assert not old_target(tmp_path).exists()
    assert analysis.read_bytes() == ANALYSIS_CSV.encode()
    assert root.read_bytes() == ROOT_CSV.encode()
