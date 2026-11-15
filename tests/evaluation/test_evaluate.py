"""gwd eval run: the start line that names backend and noise (P4.1), the metrics box (P4.2)."""

import pytest

from gwd.core import waveforms
from gwd.evaluation import evaluate

SIM_RESULT = {
    "acc": 0.5,
    "snr50": None,
    "snr90": None,
    "roc_auc": 0.5,
    "avg_precision": 0.5,
    "tpr": 0.5,
    "fpr": 0.5,
}


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """run() without model, test set, plots, or leaderboard."""
    monkeypatch.setattr(evaluate, "resolve", lambda model: str(tmp_path / "m.keras"))
    monkeypatch.setattr(evaluate, "load_model", lambda path: object())
    monkeypatch.setattr(evaluate, "evaluate_simulation_with_roc", lambda m: SIM_RESULT)
    monkeypatch.setattr(evaluate, "update_leaderboard_roc", lambda *args: None)


def test_the_start_line_names_backend_and_noise(stubbed, capsys):
    assert evaluate.run(real=False) == 0
    out, err = capsys.readouterr()
    line = (
        f"Backend: {waveforms.active_backend()} | Noise: gaussian (EVAL_V1 test sets); "
        "no field test (--no-real)"
    )
    assert line in out
    assert out.index(line) < out.index("FINAL SUMMARY")


def test_the_field_test_is_named(stubbed, capsys, monkeypatch):
    monkeypatch.setattr(
        evaluate, "evaluate_real_world", lambda m: evaluate.NO_FIELD_TEST
    )
    assert evaluate.run(real=True) == 0
    assert "Noise: gaussian (EVAL_V1 test sets); field test on real H1 data" in (
        capsys.readouterr().out
    )


def test_newton_gets_a_loud_warning(stubbed, capsys, monkeypatch):
    monkeypatch.setattr(waveforms, "HAS_PYCBC", False)
    assert evaluate.run(real=False) == 0
    out, err = capsys.readouterr()
    assert "Backend: newton | Noise:" in out
    assert "the waveform backend is the Newtonian approximation" in err


def test_the_metrics_box_has_rows_of_one_width():
    box = evaluate.metrics_box(
        [("Accuracy:", "100.00%"), None, ("SNR50:", "0.00"), ("SNR90:", "N/A")]
    )
    lines = box.splitlines()
    assert len({len(line) for line in lines}) == 1
    assert "║ SNR50:          0.00" in box
    assert "N/A" in box
