"""Leaderboard v2 (P5.8): one row per ``gwd eval run``, with raw numbers.

The CSV results/leaderboard_v2.csv holds floats, integers, and empty cells for missing
values, so pandas reads it as numbers and it can be sorted and aggregated
(evaluation:MISS-5). Every row names its test set with seed, n, and SHA-256, so rows of
different test sets are never ranked against each other; show() prints one table per
test set, ranked by ROC AUC, then TPR at the model's threshold, never by SNR50
(D10 = A, evaluation:MISS-1). Only show() formats.

The 12-column leaderboards from before P5.8 (results/model_leaderboard.csv and the
legacy paths in LEGACY_LEADERBOARD_FILES) are never written again; they stay as
archives and read_legacy() still reads them. Their baseline gap column is not carried
over: there is no valid Matched Filter baseline (D5 = B).
"""

import csv
import datetime
import math
import os
import sys

import pandas as pd  # type: ignore

from gwd.config import LEADERBOARD_FILE, LEGACY_LEADERBOARD_FILES, ensure_dirs

SCHEMA_VERSION = 2

COLUMNS = (
    "schema_version",
    "model",
    "date",
    # the test set: name, seed, n, SHA-256, its scenario, backend, and noise
    "testset",
    "testset_seed",
    "testset_n",
    "testset_hash",
    "eval_scenario",
    "backend",
    "noise_source",
    # threshold-independent metrics on the part test
    "roc_auc",
    "ap",
    # the model's threshold (gwd.registry); threshold_fpr and threshold_testset_hash
    # are empty for the fallback
    "threshold",
    "threshold_fpr",
    "threshold_testset_hash",
    # at that threshold on the part test: TPR on n_signal, FPR on n_noise samples
    "tpr_at_fpr",
    "fpr_achieved",
    "n_signal",
    "n_noise",
    # peak and optimal SNR50/90 at that threshold, interpolated; empty if not reached
    "snr50",
    "snr90",
    "optimal_snr50",
    "optimal_snr90",
    # the field test (--real); all empty without it
    "real_loaded",
    "real_found",
    "real_total",
    "real_ood_loaded",
    "real_ood_found",
    "real_ood_total",
    "real_false_alarms",
    "real_noise_seconds",
)


def _number(value):
    """A cell: None and NaN become None (empty), numpy scalars plain Python numbers."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def field_test_cells(real):
    """The field test columns from the result of run_field_test; empty without it."""
    cells = dict.fromkeys(c for c in COLUMNS if c.startswith("real_"))
    in_range, ood = real.get("in_range"), real.get("ood")
    if in_range is not None and ood is not None:
        cells.update(
            real_loaded=in_range.loaded,
            real_found=in_range.found,
            real_total=in_range.total,
            real_ood_loaded=ood.loaded,
            real_ood_found=ood.found,
            real_ood_total=ood.total,
        )
    if real.get("false_alarms") is not None:
        cells.update(
            real_false_alarms=real["false_alarms"],
            real_noise_seconds=real["seconds"],
        )
    return cells


def make_row(model_name, testset, sim, real, threshold, date=None):
    """One leaderboard row as a dict with the keys COLUMNS.

    testset has name, seed, n, sha256, scenario, backend, and noise_source; sim is the
    result of evaluate_simulation_with_roc; real the field test result (or
    NO_FIELD_TEST); threshold has threshold, fpr (the target, None for the fallback),
    and testset_hash (the calibration set, None for the fallback).
    """
    row = {
        "schema_version": SCHEMA_VERSION,
        "model": model_name,
        "date": date or datetime.datetime.now().isoformat(timespec="seconds"),
        "testset": testset["name"],
        "testset_seed": testset["seed"],
        "testset_n": testset["n"],
        "testset_hash": testset["sha256"],
        "eval_scenario": testset["scenario"],
        "backend": testset["backend"],
        "noise_source": testset["noise_source"],
        "roc_auc": sim["roc_auc"],
        "ap": sim["avg_precision"],
        "threshold": threshold["threshold"],
        "threshold_fpr": threshold.get("fpr"),
        "threshold_testset_hash": threshold.get("testset_hash"),
        "tpr_at_fpr": sim["tpr"],
        "fpr_achieved": sim["fpr"],
        "n_signal": sim["n_signal"],
        "n_noise": sim["n_noise"],
        "snr50": sim.get("snr50"),
        "snr90": sim.get("snr90"),
        "optimal_snr50": sim.get("optimal_snr50"),
        "optimal_snr90": sim.get("optimal_snr90"),
        **field_test_cells(real),
    }
    return {column: _number(row[column]) for column in COLUMNS}


def append_row(row, path=LEADERBOARD_FILE):
    """Append one row to the v2 CSV; write the header if the file is new.

    Raises ValueError if the file has another header, so rows of another schema are
    never mixed in. Older rows are not rewritten.
    """
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    if exists:
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if tuple(header) != COLUMNS:
            raise ValueError(
                f"{path} has another header than leaderboard v{SCHEMA_VERSION}; "
                "move it away to start a new one."
            )
    ensure_dirs(os.path.dirname(os.path.abspath(path)))
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(COLUMNS)
        writer.writerow(["" if row[c] is None else row[c] for c in COLUMNS])


def add_row(model_name, testset, sim, real, threshold, path=LEADERBOARD_FILE):
    """make_row, then append_row; prints where the row went. Returns the row."""
    row = make_row(model_name, testset, sim, real, threshold)
    append_row(row, path)
    print(f"\n🏆 Leaderboard v{SCHEMA_VERSION} updated: {path}")
    print(f"   -> ROC AUC: {row['roc_auc']:.4f} on {testset['name']}")
    return row


# The text columns; all others are numbers.
TEXT_COLUMNS = (
    "model",
    "date",
    "testset",
    "testset_hash",
    "eval_scenario",
    "backend",
    "noise_source",
    "threshold_testset_hash",
)


def read_leaderboard(path=LEADERBOARD_FILE):
    """The v2 leaderboard: numbers as numbers, empty cells as NaN, hashes as text.

    round_trip reads each float exactly as written; pandas' default parser can be off
    in the last digit.
    """
    return pd.read_csv(
        path, dtype=dict.fromkeys(TEXT_COLUMNS, str), float_precision="round_trip"
    )


def read_legacy(path):
    """An old 12-column leaderboard as strings, exactly as written.

    'n/a' and 'N/A' stay text; pandas would read them as NaN.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def rank(df):
    """Best first: by ROC AUC, then by TPR at the model's threshold; NaN last.

    Never by SNR50: a model that fires on everything reaches every SNR level
    (evaluation:MISS-1).
    """
    return df.sort_values(
        by=["roc_auc", "tpr_at_fpr"],
        ascending=False,
        na_position="last",
        kind="stable",
    )


def _fmt(value, spec):
    return "-" if pd.isna(value) else format(value, spec)


def _count(found, loaded, total):
    if pd.isna(total):
        return "-"
    if loaded < total:
        return f"n/a ({int(loaded)} of {int(total)} loaded)"
    return f"{int(found)}/{int(total)}"


def display_table(df):
    """The formatted columns that show() prints, in rank order."""
    out = pd.DataFrame(
        {
            "model": df["model"],
            "date": df["date"],
            "ROC AUC": df["roc_auc"].map(lambda v: _fmt(v, ".4f")),
            "AP": df["ap"].map(lambda v: _fmt(v, ".4f")),
            "threshold": [
                _fmt(t, ".4g") + ("" if isinstance(h, str) else " (fallback)")
                for t, h in zip(df["threshold"], df["threshold_testset_hash"])
            ],
            "TPR": df["tpr_at_fpr"].map(lambda v: _fmt(v, ".1%")),
            "at FPR": df["fpr_achieved"].map(lambda v: _fmt(v, ".2%")),
            "n sig/noise": [
                f"{int(s)}/{int(n)}" for s, n in zip(df["n_signal"], df["n_noise"])
            ],
            "SNR50": df["snr50"].map(lambda v: _fmt(v, ".2f")),
            "SNR90": df["snr90"].map(lambda v: _fmt(v, ".2f")),
            "real events": [
                _count(f, lo, t)
                for f, lo, t in zip(
                    df["real_found"], df["real_loaded"], df["real_total"]
                )
            ],
            "OOD": [
                _count(f, lo, t)
                for f, lo, t in zip(
                    df["real_ood_found"], df["real_ood_loaded"], df["real_ood_total"]
                )
            ],
            "false alarms": [
                "-" if pd.isna(k) else f"{int(k)} in {s / 3600:.3g} h"
                for k, s in zip(df["real_false_alarms"], df["real_noise_seconds"])
            ],
        }
    )
    return out


def legacy_note():
    """One line per old leaderboard that exists: it is an archive and not shown."""
    lines = []
    for path in LEGACY_LEADERBOARD_FILES:
        if os.path.exists(path):
            rows = len(read_legacy(path))
            lines.append(
                f"Archive, not shown: {path} ({rows} rows, before P5.8; another "
                "schema, test set, and threshold, so not comparable)."
            )
    return lines


def show():
    """Print one table per test set, best ROC AUC first. Returns the exit code."""
    if not os.path.exists(LEADERBOARD_FILE):
        print(
            f"No leaderboard yet: {LEADERBOARD_FILE} does not exist. "
            "Run 'gwd eval run' first."
        )
    else:
        df = read_leaderboard(LEADERBOARD_FILE)
        keys = ["testset", "testset_seed", "testset_n", "testset_hash"]
        for (name, seed, n, sha), group in df.groupby(keys, sort=True):
            print(f"\nTest set {name} (seed {seed}, n {n}), SHA-256 {sha[:12]}:")
            print(display_table(rank(group)).to_markdown(index=False))
        print(
            "\nRanked by ROC AUC, then TPR. TPR and SNR50/90 are at each model's "
            "threshold, at the FPR next to them; '(fallback)' marks a model without "
            "a calibrated threshold (gwd eval thresholds --model <m> --save)."
        )
    for line in legacy_note():
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(show())
