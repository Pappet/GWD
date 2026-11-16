"""gwd eval run: the start line that names backend and noise (P4.1), the metrics box (P4.2)."""

import sys

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
    assert evaluate.run() == 0
    out, err = capsys.readouterr()
    line = (
        f"Backend: {waveforms.backend_label('pycbc')} | Noise: gaussian (EVAL_V1 test sets); "
        "no field test (--no-real)"
    )
    assert line in out
    assert out.index(line) < out.index("FINAL SUMMARY")


def test_the_field_test_is_named(stubbed, capsys, monkeypatch):
    calls = []

    def field_test(model, threshold):
        calls.append(threshold)
        return {"events_text": "4/4; OOD 0/1", "far_text": "0 false alarms in 0.0556 h"}

    monkeypatch.setattr(evaluate, "run_field_test", field_test)
    assert evaluate.run(real=True) == 0
    out = capsys.readouterr().out
    assert "Noise: gaussian (EVAL_V1 test sets); field test on real H1 data" in out
    assert calls == [evaluate.THRESHOLD]
    assert "Real Events Found:    4/4; OOD 0/1" in out
    assert "Real Noise:           0 false alarms in 0.0556 h" in out


def test_without_real_the_row_says_n_a(stubbed, capsys, monkeypatch):
    rows = []
    monkeypatch.setattr(evaluate, "update_leaderboard_roc", lambda *a: rows.append(a))
    assert evaluate.run() == 0
    real_res = rows[0][2]
    assert (real_res["events_text"], real_res["far_text"]) == ("n/a", "n/a")
    assert "gwd eval run --real" in capsys.readouterr().out


def test_without_pycbc_the_evaluation_stops_before_loading_the_model(
    stubbed, monkeypatch
):
    """P5.2: no Newton test sets; the error names the install command."""
    monkeypatch.setitem(sys.modules, "pycbc", None)
    monkeypatch.setattr(
        evaluate, "load_model", lambda path: pytest.fail("model loaded")
    )
    with pytest.raises(waveforms.WaveformBackendError, match="physics"):
        evaluate.run(real=False)


def test_the_metrics_box_has_rows_of_one_width():
    box = evaluate.metrics_box(
        [("Accuracy:", "100.00%"), None, ("SNR50:", "0.00"), ("SNR90:", "N/A")]
    )
    lines = box.splitlines()
    assert len({len(line) for line in lines}) == 1
    assert "║ SNR50:          0.00" in box
    assert "N/A" in box
