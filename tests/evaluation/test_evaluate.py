"""gwd eval run: the start line that names backend and noise (P4.1), the metrics box
(P4.2), and since P5.7 the fixed test set, the efficiency curve, and the binned metrics.
Every test set is built in a tmp folder; nothing reads or writes the real data/ or
results/."""

import sys

import numpy as np
import pytest

from gwd.core import waveforms
from gwd.evaluation import evaluate, testsets

SIM_RESULT = {
    "acc": 0.5,
    "snr50": None,
    "snr90": None,
    "optimal_snr50": None,
    "optimal_snr90": None,
    "optimal_per_peak": None,
    "roc_auc": 0.5,
    "avg_precision": 0.5,
    "tpr": 0.5,
    "fpr": 0.5,
}


@pytest.fixture(scope="module")
def small_sets(tmp_path_factory):
    """Both standard sets with 16 samples, Newton chirps, in a tmp folder."""
    root = tmp_path_factory.mktemp("sets")
    return {
        name: testsets.build(
            name,
            16,
            3,
            backend="newton",
            data_dir=str(root / "eval"),
            manifest_dir=str(root / "manifests"),
            noise_dir=str(root / "noise"),
        )
        for name in ("indist_v1", "stress_v1")
    }


@pytest.fixture
def tmp_set_folders(monkeypatch, tmp_path):
    """Point the default test set and plot folders of the modules into tmp_path."""
    monkeypatch.setattr(testsets, "TESTSET_DIR", str(tmp_path / "data" / "eval"))
    monkeypatch.setattr(testsets, "TESTSET_MANIFEST_DIR", str(tmp_path / "testsets"))
    monkeypatch.setattr(testsets, "DATA_DIR_NOISE", str(tmp_path / "noise"))
    monkeypatch.setattr(
        evaluate, "EVALUATION_MODELS_PLOTS_DIR", str(tmp_path / "plots")
    )
    return tmp_path


@pytest.fixture
def stubbed(monkeypatch, tmp_path, small_sets):
    """run() without model, plots, or leaderboard, on the small sets."""
    monkeypatch.setattr(evaluate, "resolve", lambda model: str(tmp_path / "m.keras"))
    monkeypatch.setattr(evaluate, "load_model", lambda path: object())
    monkeypatch.setattr(
        evaluate, "evaluate_simulation_with_roc", lambda m, built: SIM_RESULT
    )
    monkeypatch.setattr(evaluate, "update_leaderboard_roc", lambda *args: None)
    monkeypatch.setattr(
        evaluate.testsets, "build", lambda name, n, seed: small_sets[name]
    )


def test_the_start_line_names_backend_and_noise(stubbed, capsys):
    assert evaluate.run() == 0
    out, err = capsys.readouterr()
    line = (
        f"Backend: {waveforms.backend_label('pycbc')} | Noise: gaussian "
        "(test set stress_v1 = EVAL_V1); no field test (--no-real)"
    )
    assert line in out
    assert out.index(line) < out.index("FINAL SUMMARY")
    assert "Test set:             stress_v1 (seed 3, n 16)" in out


def test_indist_names_its_real_noise_share(stubbed, capsys):
    assert evaluate.run(testset="indist_v1") == 0
    out = capsys.readouterr().out
    assert "Noise: real for 80% of samples if the test set was built with" in out
    assert "(test set indist_v1 = TRAIN_V1)" in out


def test_the_field_test_is_named(stubbed, capsys, monkeypatch):
    calls = []

    def field_test(model, threshold):
        calls.append(threshold)
        return {"events_text": "4/4; OOD 0/1", "far_text": "0 false alarms in 0.0556 h"}

    monkeypatch.setattr(evaluate, "run_field_test", field_test)
    assert evaluate.run(real=True) == 0
    out = capsys.readouterr().out
    assert "(test set stress_v1 = EVAL_V1); field test on real H1 data" in out
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


def test_only_stress_v1_adds_a_leaderboard_row(stubbed, capsys, monkeypatch):
    rows = []
    monkeypatch.setattr(evaluate, "update_leaderboard_roc", lambda *a: rows.append(a))
    assert evaluate.run(testset="indist_v1") == 0
    assert rows == []
    assert "Leaderboard not updated" in capsys.readouterr().out
    assert evaluate.run(testset="stress_v1") == 0
    assert len(rows) == 1


def test_a_changed_test_set_ends_with_exit_code_1(stubbed, monkeypatch, capsys):
    def broken(name, n, seed):
        raise testsets.InvalidTestsetError("its arrays were changed after the build")

    monkeypatch.setattr(evaluate.testsets, "build", broken)
    monkeypatch.setattr(
        evaluate, "load_model", lambda path: pytest.fail("model loaded")
    )
    assert evaluate.run() == 1
    assert "changed after the build" in capsys.readouterr().err


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


class FiringModel:
    """Scores every sample 1.0, so every bin is detected."""

    def predict(self, X, verbose=0):
        return np.ones((len(X), 1))


class PeakModel:
    """A deterministic score from the data: the peak of |x|, squashed to (0, 1)."""

    def predict(self, X, verbose=0):
        peak = np.abs(np.asarray(X)).max(axis=(1, 2))
        return (1 - np.exp(-peak / 5)).reshape(-1, 1)


def test_the_efficiency_curve_records_the_optimal_snr_per_bin(tmp_path):
    """P5.5/P5.7: each peak SNR bin also gets the mean optimal SNR of its samples."""
    if not waveforms.pycbc_available():
        pytest.skip("pycbc missing")
    built = testsets.build(
        "stress_v1",
        32,
        0,
        data_dir=str(tmp_path / "eval"),
        manifest_dir=str(tmp_path / "manifests"),
        noise_dir=str(tmp_path / "noise"),
    )
    scores = FiringModel().predict(built.X).flatten()
    centers, rates, optimal = evaluate.efficiency_curve(built, scores)
    np.testing.assert_allclose(centers, np.arange(0.625, 2.5, 0.25))
    assert rates == [1.0] * 8
    signal = built.y == 1
    # EVAL_V1 is 30/30 Msun in white noise: rho_opt is about 13.5 times the peak SNR.
    ratio = evaluate.optimal_per_peak(built.peak_snr[signal], built.optimal_snr[signal])
    assert ratio == pytest.approx(13.5, rel=0.05)
    np.testing.assert_allclose(np.array(optimal) / centers, 13.5, rtol=0.25)
    assert evaluate.optimal_per_peak(np.zeros(3), [0.0] * 3) is None


def test_binned_metrics_cover_snr_mass_and_noise_source(small_sets):
    built = small_sets["stress_v1"]
    report = evaluate.binned_metrics(built, FiringModel().predict(built.X).flatten())
    assert set(report) == {"peak_snr", "optimal_snr", "total_mass", "noise_source"}
    assert len(report["peak_snr"]) == 8
    assert sum(row["n"] for row in report["peak_snr"]) == 8
    assert [row["n"] for row in report["total_mass"]] == [8]
    assert report["noise_source"][0]["group"] == "gaussian"
    assert report["noise_source"][0]["fpr"] == 1.0
    # stress_v1 starts at peak SNR 0.5; TRAIN_V1 at 1.0.
    labels = [evaluate.extrapolation_label(r["outside"]) for r in report["peak_snr"]]
    assert labels[:2] == ["Extrapolation"] * 2
    assert set(labels[2:]) == {""}
    table = evaluate.bin_table("Peak SNR", report["peak_snr"])
    assert "0.50–0.75" in table and "Extrapolation" in table


def test_extrapolation_labels():
    assert evaluate.extrapolation_label(None) == ""
    assert evaluate.extrapolation_label(0.0) == ""
    assert evaluate.extrapolation_label(1.0) == "Extrapolation"
    assert evaluate.extrapolation_label(0.25) == "partly extrapolation (25%)"


def test_run_twice_gives_the_same_roc_auc(tmp_set_folders, monkeypatch, capsys):
    """The check of the plan, in process: the second run loads the same set."""
    if not waveforms.pycbc_available():
        pytest.skip("pycbc missing")
    results = []
    monkeypatch.setattr(evaluate, "resolve", lambda model: "m.keras")
    monkeypatch.setattr(evaluate, "load_model", lambda path: PeakModel())
    monkeypatch.setattr(
        evaluate, "update_leaderboard_roc", lambda name, sim, *a: results.append(sim)
    )
    for _ in range(2):
        assert evaluate.run(testset="stress_v1", real=False, seed=1, n=24) == 0
    out = capsys.readouterr().out
    assert out.count("Built: test set stress_v1 (seed 1, n 24)") == 1
    assert out.count("Cache hit: test set stress_v1 (seed 1, n 24)") == 1
    assert results[0]["roc_auc"] == results[1]["roc_auc"]
    assert results[0]["bins"] == results[1]["bins"]
    assert (tmp_set_folders / "testsets" / "stress_v1_1_24.json").exists()
