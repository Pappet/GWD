import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOW_LEADERBOARD = ROOT / "src" / "show_leaderboard.py"

HEADER = "Model,Date,ROC_AUC,Physik_Gap\n"
ANALYSIS_CSV = HEADER + "from_analysis.keras,2025-12-05 23:36,0.9092,+0.67\n"
ROOT_CSV = HEADER + "from_root.keras,2025-11-21 18:00,0.5,-\n"


def run_python(home, *args):
    """Run Python with GWD_HOME in home, so no real file is ever read or written."""
    return subprocess.run([sys.executable, *args], env={**os.environ, "GWD_HOME": str(home)}, cwd=home,
                          capture_output=True, text=True)


def show(home):
    result = run_python(home, str(SHOW_LEADERBOARD))
    assert result.returncode == 0, result.stderr
    return result.stdout


def write_analysis_csv(home):
    path = home / "analysis" / "model_leaderboard.csv"
    path.parent.mkdir()
    path.write_bytes(ANALYSIS_CSV.encode())
    return path


def write_root_csv(home):
    path = home / "model_leaderboard.csv"
    path.write_bytes(ROOT_CSV.encode())
    return path


def target(home):
    return home / "results" / "model_leaderboard.csv"


def test_import_config_does_not_create_results(tmp_path):
    result = run_python(tmp_path, "-c", f"import sys; sys.path.insert(0, {str(ROOT)!r}); import config")
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "results").exists()


def test_show_copies_analysis_csv_and_keeps_the_original(tmp_path):
    original = write_analysis_csv(tmp_path)

    output = show(tmp_path)

    assert target(tmp_path).read_bytes() == ANALYSIS_CSV.encode()
    assert original.read_bytes() == ANALYSIS_CSV.encode()
    assert "Copied leaderboard" in output
    assert "from_analysis.keras" in output


def test_copy_happens_only_once(tmp_path):
    write_analysis_csv(tmp_path)
    show(tmp_path)
    changed = ANALYSIS_CSV + "newer.keras,2026-01-01 00:00,0.95,-\n"
    target(tmp_path).write_text(changed)

    output = show(tmp_path)

    assert target(tmp_path).read_text() == changed
    assert "Copied leaderboard" not in output


def test_root_leaderboard_only_gives_a_note(tmp_path):
    original = write_root_csv(tmp_path)

    output = show(tmp_path)

    assert "was not copied" in output
    assert not target(tmp_path).exists()
    assert original.read_bytes() == ROOT_CSV.encode()
    assert "from_root.keras" not in output


def test_analysis_csv_wins_over_root_leaderboard(tmp_path):
    write_analysis_csv(tmp_path)
    write_root_csv(tmp_path)

    output = show(tmp_path)

    assert target(tmp_path).read_bytes() == ANALYSIS_CSV.encode()
    assert "was not copied" in output
    assert "from_root.keras" not in output


def write_row(home, destination):
    code = f"""
import sys
sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'src')!r}]
import config
from utils.evaluation import update_leaderboard_roc
destination = {destination}
sim = dict(roc_auc=0.9, avg_precision=0.8, acc=0.7, tpr=0.6, fpr=0.01, snr50=1.0, snr90=2.0)
real = dict(found=0, total=0, far_text="n/a")
update_leaderboard_roc("new.keras", sim, real, destination, "unused.json")
"""
    result = run_python(home, "-W", "ignore", "-c", code)
    assert result.returncode == 0, result.stderr


def test_writer_migrates_first_and_leaves_the_original(tmp_path):
    original = write_analysis_csv(tmp_path)

    write_row(tmp_path, "config.LEADERBOARD_FILE")

    written = target(tmp_path).read_text()
    assert "from_analysis.keras" in written
    assert "new.keras" in written
    assert original.read_bytes() == ANALYSIS_CSV.encode()


def test_writer_creates_results_without_a_legacy_file(tmp_path):
    write_row(tmp_path, "config.LEADERBOARD_FILE")
    assert "new.keras" in target(tmp_path).read_text()


def test_writing_elsewhere_does_not_migrate(tmp_path):
    write_analysis_csv(tmp_path)
    other = tmp_path / "other" / "board.csv"

    write_row(tmp_path, repr(str(other)))

    written = other.read_text()
    assert "new.keras" in written
    assert "from_analysis.keras" not in written
    assert not (tmp_path / "results").exists()
