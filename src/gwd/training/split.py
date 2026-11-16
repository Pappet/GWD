"""The train/val/test split of a dataset, grouped by noise chunk (P5.6).

A real noise chunk is used for several samples (about 7.8 times in TRAIN_V1). A
random split puts the same noise realization into training and test, so the test
accuracy is optimistic (evaluation:MISS-4). split_dataset() therefore keeps every
sample of one chunk in the same part: a group is the noise_file of labels.csv, and a
sample with Gaussian noise (empty noise_file) is a group of its own.

The groups are assigned greedily, the largest first and ties in random order, each to
the part that is furthest behind its share in the group's strata. The strata are
has_signal and noise_source, so every part gets about 70/15/15 of each class and of
each noise source. This replaces sklearn's GroupShuffleSplit, which does not stratify.

A dataset without the noise_file column (written before P4.1) is split by sample,
stratified by has_signal only, with a warning; the model's metadata then records
split_grouped = false.

Only numpy and pandas are used, so the split can be tested without TensorFlow.
"""

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gwd.config import TEST_SPLIT, VAL_SPLIT

UNGROUPED_WARNING = (
    "⚠️ labels.csv has no noise_file column (dataset written before P4.1): the split "
    "is ungrouped, so one real noise chunk can land in train and test. Regenerating "
    "the dataset with 'gwd data generate' is recommended."
)
PARTS = ("train", "val", "test")


@dataclass(frozen=True)
class Split:
    """Row positions in labels.csv of each part, and how they were drawn."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    # True if the samples were grouped by noise_file.
    grouped: bool
    n_groups: int

    def sizes(self):
        return {part: len(getattr(self, part)) for part in PARTS}


def split_dataset(labels, seed, val_fraction=VAL_SPLIT, test_fraction=TEST_SPLIT):
    """Split the rows of labels into train, val, and test; see the module docstring.

    labels is labels.csv as a DataFrame (gwd.data.validate.read_labels), seed an int
    for np.random.default_rng. The same labels and seed give the same split. Raises
    ValueError if the fractions leave no training share or a part stays empty.
    """
    train_fraction = 1.0 - val_fraction - test_fraction
    if min(train_fraction, val_fraction, test_fraction) <= 0:
        raise ValueError(
            f"val ({val_fraction}) and test ({test_fraction}) fractions must be "
            "positive and leave a training share"
        )
    has_signal = labels["has_signal"].astype(int).astype(str)
    names = labels["filename"].astype(str)
    grouped = "noise_file" in labels.columns
    if grouped:
        noise_files = labels["noise_file"].fillna("").astype(str)
        real = noise_files != ""
        keys = ("sample:" + names).where(~real, "chunk:" + noise_files)
        if "noise_source" in labels.columns:
            source = labels["noise_source"].fillna("").astype(str)
        else:
            source = real.map({True: "real", False: "gaussian"})
    else:
        print(UNGROUPED_WARNING, file=sys.stderr)
        keys = names
        source = ""

    strata, _ = pd.factorize(has_signal + "|" + source, sort=True)
    group_of, groups = pd.factorize(keys, sort=True)
    counts = np.zeros((len(groups), strata.max() + 1))
    np.add.at(counts, (group_of, strata), 1)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    # Largest groups first; the stable sort keeps the random order among equal sizes.
    order = order[np.argsort(-counts[order].sum(axis=1), kind="stable")]

    shares = np.array([train_fraction, val_fraction, test_fraction])[:, None]
    targets = shares * counts.sum(axis=0)[None, :]
    scale = np.where(targets > 0, targets, 1.0)
    filled = np.zeros_like(targets)
    part_of_group = np.empty(len(groups), dtype=int)
    for group in order:
        # Each part's squared shortfall against its targets, relative to them,
        # before and after taking the group; the group goes where it helps most.
        before = (((filled - targets) / scale) ** 2).sum(axis=1)
        after = (((filled + counts[group] - targets) / scale) ** 2).sum(axis=1)
        part = int(np.argmin(after - before))
        part_of_group[group] = part
        filled[part] += counts[group]

    part_of_row = part_of_group[group_of]
    rows = [np.flatnonzero(part_of_row == part) for part in range(len(PARTS))]
    empty = [name for name, positions in zip(PARTS, rows) if len(positions) == 0]
    if empty:
        raise ValueError(
            f"{len(labels)} samples in {len(groups)} groups are too few to split: "
            f"{', '.join(empty)} would be empty"
        )
    return Split(*rows, grouped=grouped, n_groups=len(groups))
