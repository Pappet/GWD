"""Validation of a training dataset (P4.6).

check_integrity(folder) checks labels.csv, every sample it lists, the sample_*.npy
files it does not list (orphans), and manifest.json, and returns a list of Finding.
check_distribution(labels, scenario) checks the signal samples against the ranges of
the scenario the manifest names, and the class balance. A dataset without a manifest
was written before P4.1; its scenario is unknown, so only its integrity is checked.
run() prints both with the statistics and returns 1 if there is any finding.

Usage:
    gwd data validate [--data FOLDER] [--plots [FOLDER]]
"""

import glob
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal

from gwd.config import (
    DATA_DIR_TRAIN,
    LABELS_FILE,
    N_SAMPLES,
    SAMPLE_RATE,
    VALIDATION_DATASET_PLOTS_DIR,
    ensure_dirs,
)
from gwd.core.timebase import time_axis
from gwd.data.provenance import UNKNOWN_V1, read_manifest, sha256_file
from gwd.data.scenarios import SCENARIOS

# --- integrity: every listed file, no orphans, consistent with the manifest ---

# The seven columns every labels.csv has had; P4.1 added three after them.
REQUIRED_COLUMNS = (
    "filename",
    "has_signal",
    "merger_time",
    "snr",
    "mass1",
    "mass2",
    "distance",
)
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:]
# The files gwd.data.generate writes; any other one in the folder is not checked.
SAMPLE_PATTERN = "sample_*.npy"
# float64 as written today; float32 is accepted for P5.6.
SAMPLE_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


@dataclass(frozen=True)
class Finding:
    """One problem of a dataset. file is the sample it concerns, None for the dataset."""

    check: str
    message: str
    file: Optional[str] = None

    def __str__(self):
        return f"{self.file}: {self.message}" if self.file else self.message


class InvalidDatasetError(ValueError):
    """A dataset with integrity findings; findings holds them all."""

    def __init__(self, folder, findings):
        self.folder = folder
        self.findings = list(findings)
        super().__init__(
            f"The dataset in '{folder}' has {len(self.findings)} integrity "
            f"finding(s):\n{format_findings(self.findings)}\n"
            f"Run 'gwd data validate --data {folder}' for the full report."
        )


def format_findings(findings, limit=20):
    """The findings as indented lines, at most limit per check, with the rest counted."""
    lines = []
    by_check = {}
    for finding in findings:
        by_check.setdefault(finding.check, []).append(finding)
    for check, group in by_check.items():
        lines.append(f"  [{check}] {len(group)} finding(s)")
        lines.extend(f"    - {finding}" for finding in group[:limit])
        if len(group) > limit:
            lines.append(f"    ... and {len(group) - limit} more")
    return "\n".join(lines)


# What pandas raises for a labels.csv it cannot read.
LABELS_READ_ERRORS = (
    pd.errors.ParserError,
    pd.errors.EmptyDataError,
    UnicodeDecodeError,
)


def read_labels(folder):
    """labels.csv of a dataset. Empty cells stay '' (no backend, no chunk)."""
    return pd.read_csv(os.path.join(folder, LABELS_FILE), keep_default_na=False)


def check_labels(labels):
    """Findings in labels.csv itself: columns, values, duplicate file names."""
    missing = [column for column in REQUIRED_COLUMNS if column not in labels.columns]
    if missing:
        return [Finding("labels", f"{LABELS_FILE} lacks the column(s) {missing}")]
    if labels.empty:
        return [Finding("labels", f"{LABELS_FILE} lists no samples")]

    findings = []
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(labels[column], errors="coerce")
        bad = int((~np.isfinite(values)).sum())
        if bad:
            findings.append(
                Finding("labels", f"column {column} has {bad} non-numeric value(s)")
            )
    has_signal = pd.to_numeric(labels["has_signal"], errors="coerce")
    bad = int((has_signal.notna() & ~has_signal.isin([0, 1])).sum())
    if bad:
        findings.append(
            Finding("labels", f"column has_signal has {bad} value(s) other than 0 or 1")
        )
    duplicates = labels["filename"][labels["filename"].duplicated()]
    for name in sorted(set(duplicates)):
        findings.append(Finding("labels", "listed more than once", name))
    return findings


def check_sample(path):
    """(data, findings) of one sample file; data is None if a finding was made.

    A sample must exist, load without pickle, have the shape (N_SAMPLES,), a float
    dtype, only finite values, and not be all zeros.
    """
    name = os.path.basename(path)
    if not os.path.isfile(path):
        return None, [Finding("missing", "listed in labels.csv, but missing", name)]
    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as error:
        return None, [Finding("unreadable", f"cannot be loaded ({error})", name)]
    if data.shape != (N_SAMPLES,):
        message = f"shape {data.shape} instead of ({N_SAMPLES},)"
        return None, [Finding("shape", message, name)]
    if data.dtype not in SAMPLE_DTYPES:
        return None, [Finding("dtype", f"dtype {data.dtype} is not a float", name)]
    if not np.isfinite(data).all():
        return None, [Finding("nan", "contains NaN or Inf", name)]
    if not data.any():
        return None, [Finding("zero", "is all zeros", name)]
    return data, []


def find_orphans(folder, labels):
    """Findings for sample_*.npy files in folder that labels.csv does not list."""
    listed = set(labels["filename"])
    present = {
        os.path.basename(path)
        for path in glob.glob(os.path.join(folder, SAMPLE_PATTERN))
    }
    return [
        Finding("orphan", "not listed in labels.csv", name)
        for name in sorted(present - listed)
    ]


def check_manifest(folder, labels):
    """Findings where manifest.json does not match labels.csv or its scenario.

    A dataset without a manifest (written before P4.1) has none.
    """
    manifest, _ = read_manifest(folder)
    if manifest is None:
        return []
    findings = []
    if manifest.get("n") != len(labels):
        findings.append(
            Finding(
                "manifest",
                f"manifest.json says n={manifest.get('n')}, "
                f"labels.csv lists {len(labels)} samples",
            )
        )
    digest = sha256_file(os.path.join(folder, LABELS_FILE))
    if manifest.get("labels_sha256") != digest:
        findings.append(
            Finding(
                "manifest", "labels.csv has changed since manifest.json was written"
            )
        )
    name = manifest.get("scenario")
    scenario = SCENARIOS.get(name)
    if scenario is None:
        findings.append(
            Finding("manifest", f"manifest.json names an unknown scenario {name!r}")
        )
    elif manifest.get("scenario_hash") != scenario.hash():
        findings.append(
            Finding(
                "manifest",
                f"the scenario hash {manifest.get('scenario_hash')!r} in manifest.json "
                f"is not that of {name} today ({scenario.hash()!r}); the scenario was "
                "changed after the data was written",
            )
        )
    return findings


def check_integrity(folder, on_sample=None):
    """Check a dataset folder and return a list of Finding; empty means it is intact.

    Checked: labels.csv (columns, values, duplicates); every file it lists (exists,
    loads, shape, dtype, NaN/Inf, not all zeros); sample_*.npy files it does not list
    (orphans); and manifest.json against labels.csv and its scenario, if there is one.
    on_sample(position, data) is called for every sample without a finding, with its
    row position in labels.csv, so a caller can load the data in the same pass.
    """
    if not os.path.isfile(os.path.join(folder, LABELS_FILE)):
        return [Finding("labels", f"{LABELS_FILE} is missing in '{folder}'")]
    try:
        labels = read_labels(folder)
    except LABELS_READ_ERRORS as e:
        return [Finding("labels", f"{LABELS_FILE} cannot be read ({e})")]

    findings = check_labels(labels)
    if "filename" not in labels.columns or labels.empty:
        return findings
    for position, name in enumerate(labels["filename"]):
        data, sample_findings = check_sample(os.path.join(folder, str(name)))
        findings.extend(sample_findings)
        if data is not None and on_sample is not None:
            on_sample(position, data)
    findings.extend(find_orphans(folder, labels))
    findings.extend(check_manifest(folder, labels))
    return findings


# --- distribution: the ranges of the scenario in manifest.json ---

# Largest distance of the signal share from one half (50 % ± 5 %).
BALANCE_TOLERANCE = 0.05
# Relative slack of the range checks: labels.csv stores the floats as text, and
# 10 ** log10(5) is not exactly 5.
RANGE_RTOL = 1e-9


def dataset_scenario(folder):
    """(name, Scenario) of the dataset's manifest.json.

    Without a manifest the name is UNKNOWN_V1, and for a name gwd.data.scenarios does
    not know, the Scenario is None.
    """
    manifest, _ = read_manifest(folder)
    if manifest is None:
        return UNKNOWN_V1, None
    name = manifest.get("scenario")
    return name, SCENARIOS.get(name)


def signal_ranges(scenario):
    """{labels.csv column: (low, high)} that every signal sample of scenario lies in.

    The bounds are the scenario's own fields. With random masses, mass2 also lies at
    or below mass1, which check_distribution checks on its own.
    """
    if scenario.mass_mode == "random":
        mass1 = mass2 = tuple(scenario.masses)
    else:
        mass1 = (scenario.masses[0],) * 2
        mass2 = (scenario.masses[1],) * 2
    return {
        "snr": tuple(scenario.snr_range),
        "mass1": mass1,
        "mass2": mass2,
        "merger_time": tuple(scenario.t_merger_range),
    }


def _slack(value):
    return RANGE_RTOL * max(1.0, abs(value))


def check_distribution(labels, scenario):
    """Findings where labels does not match scenario: class balance and ranges."""
    findings = []
    has_signal = pd.to_numeric(labels["has_signal"], errors="coerce")
    signal_rows = labels[has_signal == 1]
    share = len(signal_rows) / len(labels)
    if abs(share - 0.5) > BALANCE_TOLERANCE + 1e-12:
        findings.append(
            Finding(
                "balance",
                f"{share:.1%} of the samples have a signal, not "
                f"{0.5:.0%} ± {BALANCE_TOLERANCE:.0%}",
            )
        )
    if signal_rows.empty:
        return findings

    columns = {
        column: pd.to_numeric(signal_rows[column], errors="coerce")
        for column in signal_ranges(scenario)
    }
    for column, (low, high) in signal_ranges(scenario).items():
        values = columns[column]
        inside = values.between(low - _slack(low), high + _slack(high))
        outside = int((~inside).sum())
        if outside:
            findings.append(
                Finding(
                    "range",
                    f"{outside} signal sample(s) have {column} outside "
                    f"[{low:g}, {high:g}] of {scenario.name} (min {values.min():.6g}, "
                    f"max {values.max():.6g})",
                )
            )
    if scenario.mass_mode == "random":
        above = columns["mass2"] > columns["mass1"] + RANGE_RTOL * columns["mass1"]
        if above.any():
            findings.append(
                Finding(
                    "range",
                    f"{int(above.sum())} signal sample(s) have mass2 above mass1, "
                    f"which {scenario.name} does not draw",
                )
            )
    return findings


def check_dataset(folder):
    """(integrity findings, distribution findings, scenario name) of a dataset.

    The distribution is checked only if labels.csv has every required column and the
    manifest names a known scenario.
    """
    integrity = check_integrity(folder)
    name, scenario = dataset_scenario(folder)
    distribution = []
    if scenario is not None and not any(f.check == "labels" for f in integrity):
        distribution = check_distribution(read_labels(folder), scenario)
    return integrity, distribution, name


# --- report ---

STATISTIC_COLUMNS = ("snr", "mass1", "mass2", "merger_time", "distance")


def print_statistics(labels, scenario):
    """Print the class counts and the range of each signal parameter."""
    total = len(labels)
    has_signal = pd.to_numeric(labels["has_signal"], errors="coerce")
    signal_rows = labels[has_signal == 1]
    print("\n1. OVERVIEW:")
    print(f"   {'Samples:':<24} {total:>8,}")
    print(
        f"   {'With signal:':<24} {len(signal_rows):>8,} "
        f"({len(signal_rows) / total:.1%})"
    )
    print(
        f"   {'Noise only:':<24} {total - len(signal_rows):>8,} "
        f"({(total - len(signal_rows)) / total:.1%})"
    )
    if "noise_source" in labels.columns:
        for source, count in labels["noise_source"].value_counts().sort_index().items():
            print(f"   {'Noise ' + str(source) + ':':<24} {count:>8,}")
    if "waveform_backend" in labels.columns and not signal_rows.empty:
        backends = signal_rows["waveform_backend"].value_counts().sort_index()
        for backend, count in backends.items():
            print(f"   {'Chirps ' + str(backend) + ':':<24} {count:>8,}")

    if signal_rows.empty:
        return
    ranges = signal_ranges(scenario) if scenario is not None else {}
    print("\n2. SIGNAL PARAMETERS (signal samples):")
    print(f"   {'column':<12} {'min':>10} {'median':>10} {'max':>10}   scenario range")
    for column in STATISTIC_COLUMNS:
        values = pd.to_numeric(signal_rows[column], errors="coerce")
        low_high = ranges.get(column)
        bound = f"{low_high[0]:g} – {low_high[1]:g}" if low_high else "–"
        print(
            f"   {column:<12} {values.min():>10.4g} {values.median():>10.4g} "
            f"{values.max():>10.4g}   {bound}"
        )
    if scenario is None:
        return
    print(
        f"   snr is a {scenario.snr_kind} SNR (peak amplitude over noise std), not a "
        "matched filter SNR."
    )
    print(
        "   distance does not change the sample: the chirp is normalized before "
        "it is scaled to the SNR."
    )


def _print_findings(title, findings, ok_text):
    print(f"\n{title}")
    if findings:
        print(f"   ❌ {len(findings)} finding(s):")
        print(format_findings(findings))
    else:
        print(f"   ✅ {ok_text}")


def snr_examples(signal_samples):
    """(label, row) of the signal samples with the lowest, median, and highest SNR."""
    if signal_samples.empty:
        return []
    ordered = signal_samples.sort_values("snr", kind="stable")
    picks = {"Lowest SNR": 0, "Median SNR": len(ordered) // 2, "Highest SNR": -1}
    examples, seen = [], set()
    for label, position in picks.items():
        row = ordered.iloc[position]
        if row["filename"] not in seen:
            seen.add(row["filename"])
            examples.append((label, row))
    return examples


def create_visualizations(
    df, data_folder, output_folder=VALIDATION_DATASET_PLOTS_DIR, scenario=None
):
    """Write the report plots of the dataset to output_folder.

    The SNR histogram marks the SNR range of scenario, if it is known. The samples
    shown are the first noise sample and the signal samples with the lowest, the
    median, and the highest SNR; the fixed SNR classes up to '>20' are gone (DATA-4).
    """
    # Lazy, so importing gwd.data does not load matplotlib (docs/architecture.md).
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    ensure_dirs(output_folder)

    signal_samples = df[df["has_signal"] == 1]
    noise_samples = df[df["has_signal"] == 0]

    print("\n5. PLOTS:")
    print(f"   Writing plots to '{output_folder}/'...")

    # ========== PLOT 1: SNR distribution ==========
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes1[0].hist(
        signal_samples["snr"], bins=50, edgecolor="black", alpha=0.7, color="steelblue"
    )
    axes1[0].axvline(
        signal_samples["snr"].median(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median = {signal_samples['snr'].median():.2f}",
    )
    axes1[0].axvline(
        signal_samples["snr"].mean(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {signal_samples['snr'].mean():.2f}",
    )
    if scenario is not None:
        for bound in scenario.snr_range:
            axes1[0].axvline(bound, color="orange", linestyle=":", linewidth=2)
        axes1[0].axvspan(
            *scenario.snr_range,
            color="orange",
            alpha=0.08,
            label=f"{scenario.name} range",
        )
    axes1[0].set_xlabel("Peak SNR (peak amplitude / noise std)", fontsize=12)
    axes1[0].set_ylabel("Number of samples", fontsize=12)
    axes1[0].set_title("SNR distribution", fontsize=14, fontweight="bold")
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.3)

    # Box Plot
    axes1[1].boxplot([signal_samples["snr"]], tick_labels=["Signal SNR"])
    axes1[1].set_ylabel("SNR", fontsize=12)
    axes1[1].set_title("SNR Box Plot", fontsize=14, fontweight="bold")
    axes1[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "01_snr_distribution.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print("   ✅ 01_snr_distribution.png")
    plt.close()

    # ========== PLOT 2: Physical parameters ==========
    fig2 = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.3, wspace=0.3)

    # Mass 1 vs mass 2
    ax1 = fig2.add_subplot(gs[0, 0])
    scatter = ax1.scatter(
        signal_samples["mass1"],
        signal_samples["mass2"],
        c=signal_samples["snr"],
        cmap="viridis",
        alpha=0.6,
        s=10,
    )
    ax1.plot([0, 100], [0, 100], "r--", alpha=0.3, label="m1 = m2")
    ax1.set_xlabel("Mass 1 [M☉]", fontsize=11)
    ax1.set_ylabel("Mass 2 [M☉]", fontsize=11)
    ax1.set_title("Mass distribution", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label("SNR", fontsize=10)

    # Total mass distribution
    ax2 = fig2.add_subplot(gs[0, 1])
    total_mass = signal_samples["mass1"] + signal_samples["mass2"]
    ax2.hist(total_mass, bins=40, edgecolor="black", alpha=0.7, color="coral")
    ax2.set_xlabel("Total mass [M☉]", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Total mass distribution", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Mass ratio (q = m1/m2)
    ax3 = fig2.add_subplot(gs[0, 2])
    mass_ratio = signal_samples["mass1"] / signal_samples["mass2"]
    ax3.hist(mass_ratio, bins=40, edgecolor="black", alpha=0.7, color="lightgreen")
    ax3.set_xlabel("Mass ratio q = m1/m2", fontsize=11)
    ax3.set_ylabel("Count", fontsize=11)
    ax3.set_title("Mass ratio distribution", fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # Distance distribution
    # The distance has no effect on the sample (normalized chirp), so there is no
    # 'SNR vs distance' plot; this histogram shows only the drawn metadata.
    ax5 = fig2.add_subplot(gs[1, 0])
    ax5.hist(
        signal_samples["distance"],
        bins=40,
        edgecolor="black",
        alpha=0.7,
        color="mediumpurple",
    )
    ax5.set_xlabel("Distance [Mpc]", fontsize=11)
    ax5.set_ylabel("Count", fontsize=11)
    ax5.set_title("Distance distribution (metadata only)", fontweight="bold")
    ax5.grid(True, alpha=0.3)

    # Merger time distribution
    ax6 = fig2.add_subplot(gs[1, 1])
    ax6.hist(
        signal_samples["merger_time"],
        bins=40,
        edgecolor="black",
        alpha=0.7,
        color="gold",
    )
    ax6.set_xlabel("Merger time [s]", fontsize=11)
    ax6.set_ylabel("Count", fontsize=11)
    ax6.set_title("Merger time distribution", fontweight="bold")
    ax6.grid(True, alpha=0.3)

    plt.savefig(
        os.path.join(output_folder, "02_physical_parameters.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print("   ✅ 02_physical_parameters.png")
    plt.close()

    # ========== PLOT 3: Example samples (different SNR) ==========
    # Pick representative samples
    samples_to_plot = []

    # Noise only
    if len(noise_samples) > 0:
        samples_to_plot.append(("Noise only", noise_samples.iloc[0]))

    for label, row in snr_examples(signal_samples):
        samples_to_plot.append((label, row))

    n_plots = len(samples_to_plot)
    fig3, axes3 = plt.subplots(n_plots, 1, figsize=(16, 3 * n_plots))
    if n_plots == 1:
        axes3 = [axes3]

    for ax, (label, row) in zip(axes3, samples_to_plot):
        filepath = os.path.join(data_folder, row["filename"])
        data = np.load(filepath)
        time = time_axis(SAMPLE_RATE, len(data) / SAMPLE_RATE)

        ax.plot(time, data, linewidth=0.5, color="steelblue")

        # Title with the parameters
        if row["has_signal"] == 1:
            title = f"{label}: SNR={row['snr']:.2f} | M1={row['mass1']:.1f}M☉, M2={row['mass2']:.1f}M☉ | D={row['distance']:.0f}Mpc"
        else:
            title = f"{label}"

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("Time [s]", fontsize=10)
        ax.set_ylabel("Strain", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Mark the merger time (signal samples only)
        if row["has_signal"] == 1 and row["merger_time"] > 0:
            ax.axvline(
                row["merger_time"],
                color="red",
                linestyle="--",
                alpha=0.7,
                linewidth=2,
                label=f"Merger @ {row['merger_time']:.2f}s",
            )
            ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "03_sample_timeseries.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print("   ✅ 03_sample_timeseries.png")
    plt.close()

    # ========== PLOT 4: Spectrograms ==========
    # Noise only, the weakest and the strongest signal
    spec_samples = []

    if len(noise_samples) > 0:
        spec_samples.append(("Noise only", noise_samples.iloc[0]))

    examples = snr_examples(signal_samples)
    if examples:
        spec_samples.append(examples[0])
    if len(examples) > 1:
        spec_samples.append(examples[-1])

    fig4, axes4 = plt.subplots(
        len(spec_samples), 2, figsize=(16, 5 * len(spec_samples))
    )
    if len(spec_samples) == 1:
        axes4 = axes4.reshape(1, -1)

    for i, (label, row) in enumerate(spec_samples):
        filepath = os.path.join(data_folder, row["filename"])
        data = np.load(filepath)
        fs = SAMPLE_RATE

        # Time series
        time = time_axis(SAMPLE_RATE, len(data) / SAMPLE_RATE)
        axes4[i, 0].plot(time, data, linewidth=0.5, color="steelblue")
        axes4[i, 0].set_xlabel("Time [s]", fontsize=10)
        axes4[i, 0].set_ylabel("Strain", fontsize=10)

        if row["has_signal"] == 1:
            title_left = f"{label} | SNR={row['snr']:.2f}"
            if row["merger_time"] > 0:
                axes4[i, 0].axvline(
                    row["merger_time"],
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=2,
                )
        else:
            title_left = label
        axes4[i, 0].set_title(title_left, fontweight="bold", fontsize=11)
        axes4[i, 0].grid(True, alpha=0.3)

        # Spectrogram
        f, t, Sxx = signal.spectrogram(data, fs, nperseg=256)
        im = axes4[i, 1].pcolormesh(
            t, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="viridis"
        )
        axes4[i, 1].set_ylabel("Frequency [Hz]", fontsize=10)
        axes4[i, 1].set_xlabel("Time [s]", fontsize=10)
        axes4[i, 1].set_title("Spectrogram", fontweight="bold", fontsize=11)
        axes4[i, 1].set_ylim([0, 500])  # the band where the chirps are
        plt.colorbar(im, ax=axes4[i, 1], label="Power [dB]")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "04_spectrograms.png"), dpi=150, bbox_inches="tight"
    )
    print("   ✅ 04_spectrograms.png")
    plt.close()

    # ========== PLOT 5: Correlation heatmap ==========
    if len(signal_samples) > 0:
        fig5, ax5 = plt.subplots(figsize=(10, 8))

        # Correlation matrix
        corr_cols = ["snr", "mass1", "mass2", "distance", "merger_time"]
        corr_matrix = signal_samples[corr_cols].corr()

        im = ax5.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

        # Axis labels
        ax5.set_xticks(range(len(corr_cols)))
        ax5.set_yticks(range(len(corr_cols)))
        ax5.set_xticklabels(corr_cols, rotation=45, ha="right")
        ax5.set_yticklabels(corr_cols)

        # Values in the cells
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                ax5.text(
                    j,
                    i,
                    f"{corr_matrix.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=11,
                )

        ax5.set_title("Correlation between parameters", fontweight="bold", fontsize=14)
        plt.colorbar(im, ax=ax5, label="Correlation coefficient")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_folder, "05_correlation_matrix.png"),
            dpi=150,
            bbox_inches="tight",
        )
        print("   ✅ 05_correlation_matrix.png")
        plt.close()

    print(f"\n   ✅ All plots written to '{output_folder}/'")


def run(data_folder=DATA_DIR_TRAIN, output_folder=None):
    """Check the dataset in data_folder and print the report. Returns the exit code.

    0 without findings, 1 with any integrity or distribution finding. The plots are
    optional: they are written to output_folder if it is given and the samples are
    intact. gwd data validate --plots passes VALIDATION_DATASET_PLOTS_DIR.
    """
    print("\n" + "=" * 80)
    print("🔍 DATASET VALIDATION")
    print("=" * 80)
    print(f"\nDataset folder: {data_folder}")

    integrity, distribution, name = check_dataset(data_folder)
    _, scenario = dataset_scenario(data_folder)
    if name == UNKNOWN_V1:
        print(
            f"Scenario: {UNKNOWN_V1}. There is no manifest.json (written before "
            "P4.1), so only the integrity is checked."
        )
    elif scenario is None:
        print(f"Scenario: {name!r} is not in gwd.data.scenarios.")
    else:
        print(f"Scenario: {name} (hash {scenario.hash()})")

    labels_usable = not any(f.check == "labels" for f in integrity)
    if labels_usable:
        labels = read_labels(data_folder)
        print_statistics(labels, scenario)

    _print_findings(
        "3. INTEGRITY (every listed sample, orphans, manifest):",
        integrity,
        "All listed samples are intact, no orphans.",
    )
    if scenario is not None and labels_usable:
        _print_findings(
            f"4. DISTRIBUTION (ranges of {scenario.name}, balance):",
            distribution,
            f"Every signal sample lies in the ranges of {scenario.name}; "
            "the classes are balanced.",
        )

    if output_folder and integrity:
        print("\nPlots skipped: they need intact samples.")
    elif output_folder:
        create_visualizations(labels, data_folder, output_folder, scenario)

    findings = len(integrity) + len(distribution)
    print("\n" + "=" * 80)
    if findings:
        print(f"❌ {findings} finding(s): the dataset is not fit for training.")
        print("=" * 80)
        return 1
    print("✅ The dataset is fit for training.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit("The options are on the gwd command: gwd data validate --help")
    sys.exit(run())
