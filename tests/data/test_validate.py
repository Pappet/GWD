"""gwd.data.validate checks every sample, the manifest, and orphans (P4.6)."""

import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from gwd.config import N_SAMPLES
from gwd.data import validate
from gwd.data.generate import generate_dataset
from gwd.data.scenarios import EVAL_V1, TRAIN_V1


@pytest.fixture(scope="module")
def pristine(tmp_path_factory):
    """A correct TRAIN_V1 dataset of 16 samples; tests work on a copy."""
    root = tmp_path_factory.mktemp("pristine")
    # An empty noise folder, so the test never reads data/noise_background/.
    generate_dataset(16, str(root / "data"), TRAIN_V1, seed=4, noise_dir=str(root))
    return root / "data"


@pytest.fixture
def dataset(pristine, tmp_path):
    folder = tmp_path / "data"
    shutil.copytree(pristine, folder)
    return folder


def checks(findings):
    return [finding.check for finding in findings]


def test_a_correct_dataset_has_no_findings(dataset):
    assert validate.check_integrity(str(dataset)) == []


def test_a_deleted_file_is_one_finding(dataset):
    (dataset / "sample_00003.npy").unlink()
    findings = validate.check_integrity(str(dataset))
    assert checks(findings) == ["missing"]
    assert findings[0].file == "sample_00003.npy"


def test_a_nan_sample_is_one_finding(dataset):
    data = np.load(dataset / "sample_00005.npy")
    data[100] = np.nan
    np.save(dataset / "sample_00005.npy", data)
    findings = validate.check_integrity(str(dataset))
    assert checks(findings) == ["nan"]
    assert findings[0].file == "sample_00005.npy"


def test_a_wrong_length_is_one_finding(dataset):
    np.save(dataset / "sample_00002.npy", np.ones(N_SAMPLES - 1))
    findings = validate.check_integrity(str(dataset))
    assert checks(findings) == ["shape"]
    assert str(N_SAMPLES - 1) in findings[0].message


@pytest.mark.parametrize(
    "data, check",
    [
        (np.ones(N_SAMPLES, dtype=np.int64), "dtype"),
        (np.zeros(N_SAMPLES), "zero"),
        (np.full(N_SAMPLES, np.inf), "nan"),
        (np.ones((N_SAMPLES, 1)), "shape"),
    ],
    ids=["int", "zeros", "inf", "2d"],
)
def test_other_broken_samples(dataset, data, check):
    np.save(dataset / "sample_00000.npy", data)
    assert checks(validate.check_integrity(str(dataset))) == [check]


def test_a_float32_sample_is_intact(dataset):
    data = np.load(dataset / "sample_00001.npy").astype(np.float32)
    np.save(dataset / "sample_00001.npy", data)
    assert validate.check_integrity(str(dataset)) == []


def test_an_unreadable_file_is_a_finding(dataset):
    (dataset / "sample_00004.npy").write_bytes(b"not a numpy file")
    assert checks(validate.check_integrity(str(dataset))) == ["unreadable"]


def test_an_orphan_is_a_finding(dataset):
    np.save(dataset / "sample_00099.npy", np.ones(N_SAMPLES))
    # Other files are not samples and are left alone.
    (dataset / "notes.txt").write_text("mine")
    findings = validate.check_integrity(str(dataset))
    assert checks(findings) == ["orphan"]
    assert findings[0].file == "sample_00099.npy"


def test_missing_labels_is_one_finding(tmp_path):
    findings = validate.check_integrity(str(tmp_path))
    assert checks(findings) == ["labels"]
    assert "labels.csv is missing" in findings[0].message


def test_broken_labels_are_findings(dataset):
    labels = pd.read_csv(dataset / "labels.csv", keep_default_na=False)
    labels.loc[1, "has_signal"] = 2
    labels["snr"] = labels["snr"].astype(object)
    labels.loc[2, "snr"] = "high"
    labels.loc[3, "filename"] = labels.loc[4, "filename"]
    labels.to_csv(dataset / "labels.csv", index=False)
    findings = validate.check_integrity(str(dataset))
    # The duplicate leaves sample_00003.npy unlisted, and labels.csv no longer
    # matches its hash in the manifest.
    assert sorted(checks(findings)) == [
        "labels",
        "labels",
        "labels",
        "manifest",
        "orphan",
    ]


def test_a_labels_file_without_a_column_is_one_finding(dataset):
    labels = pd.read_csv(dataset / "labels.csv")
    labels.drop(columns="snr").to_csv(dataset / "labels.csv", index=False)
    findings = validate.check_integrity(str(dataset))
    assert [f.check for f in findings if f.check == "labels"] == ["labels"]
    assert "snr" in findings[0].message


def test_a_dropped_row_disagrees_with_the_manifest(dataset):
    labels = pd.read_csv(dataset / "labels.csv", keep_default_na=False)
    labels.iloc[:-1].to_csv(dataset / "labels.csv", index=False)
    findings = validate.check_integrity(str(dataset))
    # n and the hash differ, and the last sample is now an orphan.
    assert sorted(checks(findings)) == ["manifest", "manifest", "orphan"]


def test_a_changed_scenario_is_a_finding(dataset):
    manifest = (dataset / "manifest.json").read_text()
    (dataset / "manifest.json").write_text(
        manifest.replace(TRAIN_V1.hash(), "000000000000")
    )
    findings = validate.check_integrity(str(dataset))
    assert checks(findings) == ["manifest"]
    assert "TRAIN_V1" in findings[0].message


def test_without_manifest_only_the_files_are_checked(dataset):
    (dataset / "manifest.json").unlink()
    assert validate.check_integrity(str(dataset)) == []


def test_on_sample_gets_every_intact_sample_in_order(dataset):
    (dataset / "sample_00003.npy").unlink()
    seen = []
    validate.check_integrity(str(dataset), on_sample=lambda i, data: seen.append(i))
    assert seen == [i for i in range(16) if i != 3]


def test_the_error_names_the_files(dataset):
    (dataset / "sample_00003.npy").unlink()
    error = validate.InvalidDatasetError(
        str(dataset), validate.check_integrity(str(dataset))
    )
    assert "sample_00003.npy" in str(error)
    assert "gwd data validate --data" in str(error)


def test_format_findings_caps_each_check():
    findings = [validate.Finding("orphan", "x", f"s{i}") for i in range(25)]
    text = validate.format_findings(findings, limit=20)
    assert "[orphan] 25 finding(s)" in text
    assert "s19: x" in text and "s20: x" not in text
    assert "... and 5 more" in text


# --- distribution and exit code ---


@pytest.fixture
def labels(pristine):
    return validate.read_labels(str(pristine))


def test_the_ranges_come_from_the_scenario():
    assert validate.signal_ranges(TRAIN_V1) == {
        "snr": TRAIN_V1.snr_range,
        "mass1": TRAIN_V1.masses,
        "mass2": TRAIN_V1.masses,
        "merger_time": TRAIN_V1.t_merger_range,
    }
    assert validate.signal_ranges(EVAL_V1)["mass1"] == (30.0, 30.0)


def test_a_generated_dataset_matches_its_scenario(labels):
    assert validate.check_distribution(labels, TRAIN_V1) == []


def test_another_scenario_reports_its_ranges(labels):
    findings = validate.check_distribution(labels, EVAL_V1)
    messages = " ".join(f.message for f in findings)
    assert set(checks(findings)) == {"range"}
    # The merger times of this seed happen to lie inside 2.0-3.5 s as well.
    for column in ("snr", "mass1", "mass2"):
        assert f"have {column} outside" in messages


@pytest.mark.parametrize(
    "column, value",
    [("snr", 12.5), ("snr", 0.0), ("mass1", 101.0), ("merger_time", 1.9)],
)
def test_a_value_outside_the_range_is_one_finding(labels, column, value):
    labels.loc[labels.index[0], column] = value  # sample_00000 has a signal
    findings = validate.check_distribution(labels, TRAIN_V1)
    assert checks(findings) == ["range"]
    assert f"1 signal sample(s) have {column} outside" in findings[0].message


def test_mass2_above_mass1_is_a_finding(labels):
    first = labels.index[0]
    labels.loc[first, ["mass1", "mass2"]] = [10.0, 20.0]
    findings = validate.check_distribution(labels, TRAIN_V1)
    assert checks(findings) == ["range"]
    assert "mass2 above mass1" in findings[0].message


def test_noise_samples_are_not_range_checked(labels):
    # Noise-only samples carry zeros, which lie outside every TRAIN_V1 range.
    assert (labels.loc[labels["has_signal"] == 0, "snr"] == 0).all()
    assert validate.check_distribution(labels, TRAIN_V1) == []


def test_the_balance_allows_five_percent(labels):
    # 16 samples, 8 with a signal: 9 is 56 %, 7 of 15 is 46.7 %.
    tipped = labels.copy()
    tipped.loc[tipped.index[1], "has_signal"] = 1
    tipped.loc[tipped.index[1], ["snr", "mass1", "mass2", "merger_time"]] = [
        5.0,
        20.0,
        10.0,
        3.0,
    ]
    assert checks(validate.check_distribution(tipped, TRAIN_V1)) == ["balance"]
    assert validate.check_distribution(labels.iloc[1:], TRAIN_V1) == []


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_a_correct_dataset_of_64_gives_rc_0_for_any_seed(tmp_path, seed, capsys):
    """The old validator failed on a low median SNR by chance (DATA-4)."""
    folder = tmp_path / "data"
    generate_dataset(64, str(folder), TRAIN_V1, seed=seed, noise_dir=str(tmp_path))
    assert validate.run(str(folder)) == 0
    out = capsys.readouterr().out
    assert "Scenario: TRAIN_V1" in out
    assert "The dataset is fit for training." in out


def test_each_broken_case_gives_rc_1(pristine, tmp_path, capsys):
    cases = {
        "missing": lambda d: (d / "sample_00003.npy").unlink(),
        "nan": lambda d: np.save(d / "sample_00005.npy", np.full(N_SAMPLES, np.nan)),
        "shape": lambda d: np.save(d / "sample_00002.npy", np.ones(N_SAMPLES + 1)),
    }
    for check, damage in cases.items():
        folder = tmp_path / check
        shutil.copytree(pristine, folder)
        damage(folder)
        integrity, distribution, _ = validate.check_dataset(str(folder))
        assert (checks(integrity), distribution) == ([check], [])
        assert validate.run(str(folder)) == 1
        assert "1 finding(s): the dataset is not fit" in capsys.readouterr().out


def test_without_manifest_only_the_integrity_is_checked(dataset, capsys):
    (dataset / "manifest.json").unlink()
    labels = validate.read_labels(str(dataset))
    # Outside TRAIN_V1, but not checked; peak_snr is the alias of snr (P5.5).
    labels.loc[labels.index[0], ["snr", "peak_snr"]] = 99.0
    labels.to_csv(dataset / "labels.csv", index=False)
    assert validate.check_dataset(str(dataset)) == ([], [], "unknown (v1)")
    assert validate.run(str(dataset)) == 0
    out = capsys.readouterr().out
    assert "Scenario: unknown (v1)" in out
    assert "DISTRIBUTION" not in out


# --- the SNR columns of P5.5 ---


def rewrite_labels(dataset, change):
    labels = validate.read_labels(str(dataset))
    change(labels)
    labels.to_csv(dataset / "labels.csv", index=False)
    (dataset / "manifest.json").unlink()  # its labels hash would be a second finding


def test_the_snr_columns_of_a_generated_dataset_are_consistent(labels):
    assert validate.check_snr_columns(labels) == []
    signal = labels["has_signal"] == 1
    assert (labels.loc[signal, "optimal_snr"] > labels.loc[signal, "peak_snr"]).all()


@pytest.mark.parametrize(
    "column, row, value, message",
    [
        ("peak_snr", 0, 1.5, "peak_snr different from snr"),
        ("optimal_snr", 0, 0.0, "optimal_snr <= 0"),
        ("optimal_snr", 1, 3.0, "noise sample(s) have a peak or optimal SNR"),
        ("optimal_snr", 0, "high", "non-numeric"),
    ],
)
def test_a_broken_snr_column_is_a_finding(dataset, column, row, value, message):
    def change(labels):
        labels[column] = labels[column].astype(object)
        labels.loc[labels.index[row], column] = value

    rewrite_labels(dataset, change)
    findings = validate.check_integrity(str(dataset))
    assert checks(findings) == ["snr"]
    assert message in findings[0].message


def test_one_snr_column_alone_is_a_finding(dataset):
    rewrite_labels(
        dataset, lambda labels: labels.drop(columns="optimal_snr", inplace=True)
    )
    findings = validate.check_integrity(str(dataset))
    assert checks(findings) == ["snr"]
    assert "lacks ['optimal_snr']" in findings[0].message


def test_a_dataset_before_p5_5_reads_snr_as_the_peak_snr(dataset, capsys):
    """Old labels.csv files have snr only: no finding, and the report says so."""
    rewrite_labels(
        dataset,
        lambda labels: labels.drop(columns=["peak_snr", "optimal_snr"], inplace=True),
    )
    assert validate.check_integrity(str(dataset)) == []
    assert validate.run(str(dataset)) == 0
    out = capsys.readouterr().out
    assert "peak_snr" in out
    assert "optimal_snr: not recorded (dataset written before P5.5)." in out


def test_the_report_names_both_snrs_and_their_ratio(dataset, capsys):
    assert validate.run(str(dataset)) == 0
    out = capsys.readouterr().out
    assert "   optimal_snr " in out and "   peak_snr " in out
    assert "Median optimal_snr / peak_snr by noise: gaussian " in out


def test_missing_labels_gives_rc_1(tmp_path, capsys):
    assert validate.run(str(tmp_path / "nothing")) == 1
    assert "labels.csv is missing" in capsys.readouterr().out


# --- the gwd data validate command ---


def run_gwd(args, home):
    env = {**os.environ, "GWD_HOME": str(home), "MPLBACKEND": "Agg"}
    return subprocess.run(
        [sys.executable, "-m", "gwd.cli", *args],
        cwd=home,
        env=env,
        capture_output=True,
        text=True,
    )


PLOTS = [
    "01_snr_distribution.png",
    "02_physical_parameters.png",
    "03_sample_timeseries.png",
    "04_spectrograms.png",
    "05_correlation_matrix.png",
]


def test_plots_are_written_only_with_plots(dataset, tmp_path):
    result = run_gwd(["data", "validate", "--data", str(dataset)], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "plots").exists()

    result = run_gwd(["data", "validate", "--data", str(dataset), "--plots"], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    written = sorted(
        p.name for p in (tmp_path / "plots" / "validation_dataset").iterdir()
    )
    assert written == PLOTS

    other = tmp_path / "elsewhere"
    args = ["data", "validate", "--data", str(dataset), "--plots", str(other)]
    assert run_gwd(args, tmp_path).returncode == 0
    assert sorted(p.name for p in other.iterdir()) == PLOTS


def test_the_command_fails_for_a_broken_dataset(dataset, tmp_path):
    (dataset / "sample_00003.npy").unlink()
    args = ["data", "validate", "--data", str(dataset), "--plots"]
    result = run_gwd(args, tmp_path)
    assert result.returncode == 1
    assert "sample_00003.npy: listed in labels.csv, but missing" in result.stdout
    assert "Plots skipped" in result.stdout
    assert "Traceback" not in result.stderr
