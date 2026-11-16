"""
Datensatz-Validierung für Gravitationswellen Training

Dieses Script überprüft die Qualität des generierten Datensatzes und erstellt
umfassende Visualisierungen zur Analyse.

Usage:
    gwd data validate [--data FOLDER] [--plots FOLDER] [--no-plots]
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
    DURATION,
    LABELS_FILE,
    N_SAMPLES,
    SAMPLE_RATE,
    VALIDATION_DATASET_PLOTS_DIR,
    ensure_dirs,
)
from gwd.data.provenance import read_manifest, sha256_file
from gwd.data.scenarios import SCENARIOS


def load_dataset_info(data_folder):
    """Lädt die Labels-CSV und gibt Statistiken zurück."""
    csv_path = os.path.join(data_folder, LABELS_FILE)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Konnte {csv_path} nicht finden!")

    df = pd.read_csv(csv_path)
    return df


def print_statistics(df):
    """Gibt detaillierte Statistiken über den Datensatz aus."""
    print("\n" + "=" * 80)
    print("📊 DATENSATZ STATISTIK")
    print("=" * 80)

    # Grundlegende Statistik
    total_samples = len(df)
    signal_samples = df[df["has_signal"] == 1]
    noise_samples = df[df["has_signal"] == 0]

    print("\n1. ALLGEMEINE ÜBERSICHT:")
    print(f"   {'Gesamt Samples:':<30} {total_samples:>10,}")
    print(
        f"   {'Samples mit Signal:':<30} {len(signal_samples):>10,} ({len(signal_samples) / total_samples * 100:>5.1f}%)"
    )
    print(
        f"   {'Samples nur Rauschen:':<30} {len(noise_samples):>10,} ({len(noise_samples) / total_samples * 100:>5.1f}%)"
    )

    # SNR Statistik (nur Signal-Samples)
    if len(signal_samples) > 0:
        print("\n2. SNR VERTEILUNG (Signal-Samples):")
        print(f"   {'Minimum SNR:':<30} {signal_samples['snr'].min():>10.4f}")
        print(f"   {'Maximum SNR:':<30} {signal_samples['snr'].max():>10.4f}")
        print(f"   {'Durchschnitt SNR:':<30} {signal_samples['snr'].mean():>10.4f}")
        print(f"   {'Median SNR:':<30} {signal_samples['snr'].median():>10.4f}")
        print(f"   {'Standardabweichung:':<30} {signal_samples['snr'].std():>10.4f}")

        # SNR Kategorien
        print("\n   SNR Kategorien:")
        print(
            f"   {'SNR < 5 (sehr schwach):':<30} {(signal_samples['snr'] < 5).sum():>10,} ({(signal_samples['snr'] < 5).sum() / len(signal_samples) * 100:>5.1f}%)"
        )
        print(
            f"   {'SNR 5-10 (schwach):':<30} {((signal_samples['snr'] >= 5) & (signal_samples['snr'] < 10)).sum():>10,} ({((signal_samples['snr'] >= 5) & (signal_samples['snr'] < 10)).sum() / len(signal_samples) * 100:>5.1f}%)"
        )
        print(
            f"   {'SNR 10-15 (mittel):':<30} {((signal_samples['snr'] >= 10) & (signal_samples['snr'] < 15)).sum():>10,} ({((signal_samples['snr'] >= 10) & (signal_samples['snr'] < 15)).sum() / len(signal_samples) * 100:>5.1f}%)"
        )
        print(
            f"   {'SNR 15-20 (stark):':<30} {((signal_samples['snr'] >= 15) & (signal_samples['snr'] < 20)).sum():>10,} ({((signal_samples['snr'] >= 15) & (signal_samples['snr'] < 20)).sum() / len(signal_samples) * 100:>5.1f}%)"
        )
        print(
            f"   {'SNR > 20 (sehr stark):':<30} {(signal_samples['snr'] >= 20).sum():>10,} ({(signal_samples['snr'] >= 20).sum() / len(signal_samples) * 100:>5.1f}%)"
        )

        # Qualitäts-Check
        print("\n   💡 QUALITÄTS-CHECK:")
        if signal_samples["snr"].median() < 5:
            print("   ❌ WARNUNG: Median SNR zu niedrig! Signale schwer detektierbar.")
        elif signal_samples["snr"].median() < 8:
            print("   ⚠️  ACHTUNG: Median SNR niedrig. Training könnte schwierig sein.")
        elif signal_samples["snr"].median() < 12:
            print("   ✅ OK: Median SNR im akzeptablen Bereich.")
        else:
            print("   ✅ EXCELLENT: Median SNR optimal für Training!")

    # Physikalische Parameter (nur Signal-Samples)
    if len(signal_samples) > 0:
        print("\n3. PHYSIKALISCHE PARAMETER:")

        # Massen
        print("\n   Massen (Sonnenmassen):")
        print(
            f"   {'Mass 1 - Bereich:':<30} {signal_samples['mass1'].min():>8.1f} - {signal_samples['mass1'].max():.1f}"
        )
        print(
            f"   {'Mass 1 - Durchschnitt:':<30} {signal_samples['mass1'].mean():>10.1f}"
        )
        print(
            f"   {'Mass 2 - Bereich:':<30} {signal_samples['mass2'].min():>8.1f} - {signal_samples['mass2'].max():.1f}"
        )
        print(
            f"   {'Mass 2 - Durchschnitt:':<30} {signal_samples['mass2'].mean():>10.1f}"
        )

        # Gesamtmasse und Massenverhältnis
        total_mass = signal_samples["mass1"] + signal_samples["mass2"]
        mass_ratio = signal_samples["mass1"] / signal_samples["mass2"]
        print(
            f"   {'Gesamtmasse - Bereich:':<30} {total_mass.min():>8.1f} - {total_mass.max():.1f}"
        )
        print(
            f"   {'Massenverhältnis (q):':<30} {mass_ratio.min():>8.2f} - {mass_ratio.max():.2f}"
        )

        # Distanz
        print("\n   Distanz (Megaparsec):")
        print(
            f"   {'Bereich:':<30} {signal_samples['distance'].min():>8.0f} - {signal_samples['distance'].max():.0f}"
        )
        print(f"   {'Durchschnitt:':<30} {signal_samples['distance'].mean():>10.0f}")
        print(f"   {'Median:':<30} {signal_samples['distance'].median():>10.0f}")

        # Merger Time
        print("\n   Merger Zeitpunkt (Sekunden):")
        print(
            f"   {'Bereich:':<30} {signal_samples['merger_time'].min():>8.2f} - {signal_samples['merger_time'].max():.2f}"
        )
        print(f"   {'Durchschnitt:':<30} {signal_samples['merger_time'].mean():>10.2f}")


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
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
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


def create_visualizations(df, data_folder, output_folder=VALIDATION_DATASET_PLOTS_DIR):
    """Erstellt umfassende Visualisierungen des Datensatzes."""
    # Lazy, so importing gwd.data does not load matplotlib (docs/architecture.md).
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    ensure_dirs(output_folder)

    signal_samples = df[df["has_signal"] == 1]
    noise_samples = df[df["has_signal"] == 0]

    print("\n5. VISUALISIERUNGEN:")
    print(f"   Erstelle Plots in '{output_folder}/'...")

    # ========== PLOT 1: SNR Verteilung ==========
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
    axes1[0].axvline(
        5, color="orange", linestyle=":", linewidth=2, alpha=0.5, label="SNR = 5"
    )
    axes1[0].axvline(
        10, color="purple", linestyle=":", linewidth=2, alpha=0.5, label="SNR = 10"
    )
    axes1[0].set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=12)
    axes1[0].set_ylabel("Anzahl Samples", fontsize=12)
    axes1[0].set_title("SNR Verteilung", fontsize=14, fontweight="bold")
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

    # ========== PLOT 2: Physikalische Parameter ==========
    fig2 = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.3, wspace=0.3)

    # Masse 1 vs Masse 2
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
    ax1.set_title("Massenverteilung", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label("SNR", fontsize=10)

    # Gesamtmasse Distribution
    ax2 = fig2.add_subplot(gs[0, 1])
    total_mass = signal_samples["mass1"] + signal_samples["mass2"]
    ax2.hist(total_mass, bins=40, edgecolor="black", alpha=0.7, color="coral")
    ax2.set_xlabel("Gesamtmasse [M☉]", fontsize=11)
    ax2.set_ylabel("Anzahl", fontsize=11)
    ax2.set_title("Gesamtmassen-Verteilung", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Massenverhältnis (q = m1/m2)
    ax3 = fig2.add_subplot(gs[0, 2])
    mass_ratio = signal_samples["mass1"] / signal_samples["mass2"]
    ax3.hist(mass_ratio, bins=40, edgecolor="black", alpha=0.7, color="lightgreen")
    ax3.set_xlabel("Massenverhältnis q = m1/m2", fontsize=11)
    ax3.set_ylabel("Anzahl", fontsize=11)
    ax3.set_title("Massenverhältnis-Verteilung", fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # Distanz vs SNR
    ax4 = fig2.add_subplot(gs[1, 0])
    ax4.scatter(
        signal_samples["distance"],
        signal_samples["snr"],
        alpha=0.5,
        s=10,
        color="steelblue",
    )
    ax4.set_xlabel("Distanz [Mpc]", fontsize=11)
    ax4.set_ylabel("SNR", fontsize=11)
    ax4.set_title("SNR vs Distanz", fontweight="bold")
    ax4.grid(True, alpha=0.3)

    # Distanz Distribution
    ax5 = fig2.add_subplot(gs[1, 1])
    ax5.hist(
        signal_samples["distance"],
        bins=40,
        edgecolor="black",
        alpha=0.7,
        color="mediumpurple",
    )
    ax5.set_xlabel("Distanz [Mpc]", fontsize=11)
    ax5.set_ylabel("Anzahl", fontsize=11)
    ax5.set_title("Distanz-Verteilung", fontweight="bold")
    ax5.grid(True, alpha=0.3)

    # Merger Time Distribution
    ax6 = fig2.add_subplot(gs[1, 2])
    ax6.hist(
        signal_samples["merger_time"],
        bins=40,
        edgecolor="black",
        alpha=0.7,
        color="gold",
    )
    ax6.set_xlabel("Merger Zeitpunkt [s]", fontsize=11)
    ax6.set_ylabel("Anzahl", fontsize=11)
    ax6.set_title("Merger Zeitpunkt-Verteilung", fontweight="bold")
    ax6.grid(True, alpha=0.3)

    plt.savefig(
        os.path.join(output_folder, "02_physical_parameters.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print("   ✅ 02_physical_parameters.png")
    plt.close()

    # ========== PLOT 3: Sample Visualisierung (verschiedene SNR) ==========
    # Wähle repräsentative Samples aus
    samples_to_plot = []

    # Nur Rauschen
    if len(noise_samples) > 0:
        samples_to_plot.append(("Nur Rauschen", noise_samples.iloc[0]))

    # Verschiedene SNR-Bereiche
    snr_ranges = [
        ("Sehr Schwach (SNR < 5)", signal_samples[signal_samples["snr"] < 5]),
        (
            "Schwach (SNR 5-10)",
            signal_samples[(signal_samples["snr"] >= 5) & (signal_samples["snr"] < 10)],
        ),
        (
            "Mittel (SNR 10-15)",
            signal_samples[
                (signal_samples["snr"] >= 10) & (signal_samples["snr"] < 15)
            ],
        ),
        (
            "Stark (SNR 15-20)",
            signal_samples[
                (signal_samples["snr"] >= 15) & (signal_samples["snr"] < 20)
            ],
        ),
        ("Sehr Stark (SNR > 20)", signal_samples[signal_samples["snr"] >= 20]),
    ]

    for label, subset in snr_ranges:
        if len(subset) > 0:
            samples_to_plot.append((label, subset.iloc[0]))

    n_plots = len(samples_to_plot)
    fig3, axes3 = plt.subplots(n_plots, 1, figsize=(16, 3 * n_plots))
    if n_plots == 1:
        axes3 = [axes3]

    for ax, (label, row) in zip(axes3, samples_to_plot):
        filepath = os.path.join(data_folder, row["filename"])
        data = np.load(filepath)
        time = np.linspace(0, DURATION, len(data))

        ax.plot(time, data, linewidth=0.5, color="steelblue")

        # Title mit Infos
        if row["has_signal"] == 1:
            title = f"{label}: SNR={row['snr']:.2f} | M1={row['mass1']:.1f}M☉, M2={row['mass2']:.1f}M☉ | D={row['distance']:.0f}Mpc"
        else:
            title = f"{label}"

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("Zeit [s]", fontsize=10)
        ax.set_ylabel("Strain", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Merger-Zeitpunkt markieren (falls Signal)
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

    # ========== PLOT 4: Spektrogramme ==========
    # Wähle 3 interessante Samples für Spektrogramm
    spec_samples = []

    if len(noise_samples) > 0:
        spec_samples.append(("Nur Rauschen", noise_samples.iloc[0]))

    if len(signal_samples[signal_samples["snr"] < 10]) > 0:
        spec_samples.append(
            ("Schwaches Signal", signal_samples[signal_samples["snr"] < 10].iloc[0])
        )

    if len(signal_samples[signal_samples["snr"] >= 15]) > 0:
        spec_samples.append(
            ("Starkes Signal", signal_samples[signal_samples["snr"] >= 15].iloc[0])
        )

    fig4, axes4 = plt.subplots(
        len(spec_samples), 2, figsize=(16, 5 * len(spec_samples))
    )
    if len(spec_samples) == 1:
        axes4 = axes4.reshape(1, -1)

    for i, (label, row) in enumerate(spec_samples):
        filepath = os.path.join(data_folder, row["filename"])
        data = np.load(filepath)
        fs = SAMPLE_RATE

        # Zeitreihe
        time = np.linspace(0, DURATION, len(data))
        axes4[i, 0].plot(time, data, linewidth=0.5, color="steelblue")
        axes4[i, 0].set_xlabel("Zeit [s]", fontsize=10)
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

        # Spektrogramm
        f, t, Sxx = signal.spectrogram(data, fs, nperseg=256)
        im = axes4[i, 1].pcolormesh(
            t, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="viridis"
        )
        axes4[i, 1].set_ylabel("Frequenz [Hz]", fontsize=10)
        axes4[i, 1].set_xlabel("Zeit [s]", fontsize=10)
        axes4[i, 1].set_title("Spektrogramm", fontweight="bold", fontsize=11)
        axes4[i, 1].set_ylim([0, 500])  # Fokus auf GW-relevanten Bereich
        plt.colorbar(im, ax=axes4[i, 1], label="Power [dB]")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "04_spectrograms.png"), dpi=150, bbox_inches="tight"
    )
    print("   ✅ 04_spectrograms.png")
    plt.close()

    # ========== PLOT 5: Korrelations-Heatmap ==========
    if len(signal_samples) > 0:
        fig5, ax5 = plt.subplots(figsize=(10, 8))

        # Korrelationsmatrix berechnen
        corr_cols = ["snr", "mass1", "mass2", "distance", "merger_time"]
        corr_matrix = signal_samples[corr_cols].corr()

        im = ax5.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

        # Achsen beschriften
        ax5.set_xticks(range(len(corr_cols)))
        ax5.set_yticks(range(len(corr_cols)))
        ax5.set_xticklabels(corr_cols, rotation=45, ha="right")
        ax5.set_yticklabels(corr_cols)

        # Werte in Zellen schreiben
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

        ax5.set_title("Korrelation zwischen Parametern", fontweight="bold", fontsize=14)
        plt.colorbar(im, ax=ax5, label="Korrelationskoeffizient")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_folder, "05_correlation_matrix.png"),
            dpi=150,
            bbox_inches="tight",
        )
        print("   ✅ 05_correlation_matrix.png")
        plt.close()

    print(f"\n   ✅ Alle Visualisierungen erstellt in '{output_folder}/'")


def run(
    data_folder=DATA_DIR_TRAIN, output_folder=VALIDATION_DATASET_PLOTS_DIR, plots=True
):
    """Check the dataset in data_folder, with plots in output_folder. Returns the exit code."""
    print("\n" + "=" * 80)
    print("🔍 DATENSATZ VALIDIERUNG - GRAVITATIONSWELLEN TRAINING")
    print("=" * 80)
    print(f"\nDataset folder: {data_folder}")
    print(f"Plot folder: {output_folder}")

    try:
        # Labels laden
        df = load_dataset_info(data_folder)

        # Statistiken ausgeben
        print_statistics(df)

        # Every listed sample, orphans, and the manifest
        print("\n4. INTEGRITY:")
        integrity = check_integrity(data_folder)
        if integrity:
            print(f"   {len(integrity)} finding(s):")
            print(format_findings(integrity))
        else:
            print(f"   ✅ All {len(df)} samples are intact, no orphans.")

        # Visualisierungen erstellen
        if plots and integrity:
            print("\n   Plots skipped: they need intact samples.")
        elif plots:
            create_visualizations(df, data_folder, output_folder)

        # Finale Bewertung
        signal_samples = df[df["has_signal"] == 1]

        print("\n" + "=" * 80)
        print("📋 FINALE BEWERTUNG")
        print("=" * 80)

        issues = (
            [f"❌ {len(integrity)} integrity finding(s), see above"]
            if integrity
            else []
        )
        warnings_list = []

        # Check 1: SNR
        if len(signal_samples) > 0:
            median_snr = signal_samples["snr"].median()
            if median_snr < 5:
                issues.append("❌ KRITISCH: Median SNR < 5 - Signale zu schwach!")
            elif median_snr < 8:
                warnings_list.append(
                    "⚠️  WARNUNG: Median SNR < 8 - Training könnte schwierig sein"
                )
            else:
                print("✅ SNR: Optimal für Training")

        # Check 2: Balance
        balance = len(signal_samples) / len(df) * 100
        if balance < 40 or balance > 60:
            warnings_list.append(f"⚠️  WARNUNG: Unbalanciert ({balance:.1f}% Signale)")
        else:
            print(f"✅ Balance: Gut balanciert ({balance:.1f}% Signale)")

        # Check 3: Diversität
        if len(signal_samples) > 0:
            mass_range = signal_samples["mass1"].max() - signal_samples["mass1"].min()
            if mass_range < 20:
                warnings_list.append("⚠️  WARNUNG: Geringe Massen-Diversität")
            else:
                print("✅ Diversität: Gute Parameter-Abdeckung")

        # Ausgabe der Warnings und Errors
        if warnings_list:
            print("\nWarnungen:")
            for w in warnings_list:
                print(f"  {w}")

        if issues:
            print("\nKritische Probleme:")
            for issue in issues:
                print(f"  {issue}")
            print("\n❌ DATENSATZ NICHT GEEIGNET FÜR TRAINING!")
            return 1
        else:
            print("\n" + "=" * 80)
            print("✅ DATENSATZ IST BEREIT FÜR TRAINING!")
            print("=" * 80)
            return 0

    except Exception as e:
        print(f"\n❌ FEHLER: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit("The options are on the gwd command: gwd data validate --help")
    sys.exit(run())
