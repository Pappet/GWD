"""gwd.data.validate checks every sample, the manifest, and orphans (P4.6)."""

import shutil

import numpy as np
import pandas as pd
import pytest

from gwd.config import N_SAMPLES
from gwd.data import validate
from gwd.data.generate import generate_dataset
from gwd.data.scenarios import TRAIN_V1


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
