"""The three-way split of gwd train: stratified, grouped by noise chunk (P5.6)."""

import numpy as np
import pandas as pd
import pytest

from gwd.config import TEST_SPLIT, VAL_SPLIT
from gwd.training.split import PARTS, UNGROUPED_WARNING, split_dataset


def make_labels(n, real_share=0.8, chunks=50, seed=0):
    """labels.csv as gwd.data.generate writes it: 50/50 classes, chunks reused."""
    rng = np.random.default_rng(seed)
    real = rng.random(n) < real_share
    chunk = rng.integers(0, chunks, n)
    return pd.DataFrame(
        {
            "filename": [f"sample_{i:05d}.npy" for i in range(n)],
            "has_signal": [1 - i % 2 for i in range(n)],
            "noise_source": np.where(real, "real", "gaussian"),
            "noise_file": [
                f"real_noise_{c:04d}.npy" if r else "" for c, r in zip(chunk, real)
            ],
        }
    )


def parts(labels, split):
    return {part: labels.iloc[getattr(split, part)] for part in PARTS}


@pytest.mark.parametrize("n, seed", [(10000, 1), (2000, 2), (1000, 3), (500, 4)])
def test_each_part_is_70_15_15_with_balanced_classes(n, seed):
    labels = make_labels(n, seed=seed)
    split = split_dataset(labels, seed)
    sizes = split.sizes()
    assert sum(sizes.values()) == n
    expected = {
        "train": 1 - VAL_SPLIT - TEST_SPLIT,
        "val": VAL_SPLIT,
        "test": TEST_SPLIT,
    }
    for name, part in parts(labels, split).items():
        assert abs(len(part) / n - expected[name]) <= 0.02, name
        # Class balance within ±2 % of the dataset's 50 %.
        assert abs(part["has_signal"].mean() - 0.5) <= 0.02, name
        real = (part["noise_source"] == "real").mean()
        assert abs(real - (labels["noise_source"] == "real").mean()) <= 0.05, name


@pytest.mark.parametrize("seed", range(5))
def test_no_noise_chunk_is_in_two_parts(seed):
    labels = make_labels(2000, seed=seed)
    split = split_dataset(labels, seed)
    assert split.grouped
    chunks = {
        name: set(part["noise_file"]) - {""}
        for name, part in parts(labels, split).items()
    }
    assert not chunks["train"] & chunks["val"]
    assert not chunks["train"] & chunks["test"]
    assert not chunks["val"] & chunks["test"]
    # Every part gets real noise chunks of its own.
    assert all(chunks.values())


def test_the_parts_cover_every_row_once():
    labels = make_labels(1000)
    split = split_dataset(labels, 7)
    rows = np.concatenate([split.train, split.val, split.test])
    assert sorted(rows) == list(range(1000))


def test_the_same_seed_gives_the_same_split_and_another_seed_another():
    labels = make_labels(1000)
    first, again, other = (split_dataset(labels, s) for s in (5, 5, 6))
    for part in PARTS:
        np.testing.assert_array_equal(getattr(first, part), getattr(again, part))
    assert not np.array_equal(first.test, other.test)


def test_gaussian_samples_are_groups_of_their_own():
    labels = make_labels(400, real_share=0.0)
    split = split_dataset(labels, 1)
    assert split.grouped and split.n_groups == 400


def test_without_the_noise_file_column_it_warns_and_splits_ungrouped(capsys):
    """A dataset from before P4.1: stratified only, and recorded as such."""
    labels = make_labels(1000)[["filename", "has_signal"]]
    split = split_dataset(labels, 1)
    assert not split.grouped
    assert UNGROUPED_WARNING in capsys.readouterr().err
    assert "Regenerating the dataset" in UNGROUPED_WARNING
    for part in parts(labels, split).values():
        assert abs(part["has_signal"].mean() - 0.5) <= 0.02


def test_the_noise_file_column_without_a_warning(capsys):
    split_dataset(make_labels(100), 1)
    assert capsys.readouterr().err == ""


def test_a_dataset_too_small_to_split_is_refused():
    with pytest.raises(ValueError, match="too few to split"):
        split_dataset(make_labels(2), 1)


@pytest.mark.parametrize("val, test", [(0.5, 0.5), (0.0, 0.15), (0.15, -0.1)])
def test_fractions_without_a_training_share_are_refused(val, test):
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(make_labels(100), 1, val, test)
