"""gwd env: what this installation runs with (P3.6).

Prints versions, the installed extras, the waveform backend, the resolved paths, and
the matplotlib backend. Availability is checked with
importlib.util.find_spec and the installed package metadata, so TensorFlow and gwpy are
never imported. The one exception is PyCBC: only importing it shows whether it works.
"""

import importlib.metadata
import importlib.util
import os
import platform
import sys

# (import name, distribution name) of the base dependencies in pyproject.toml
BASE = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("sklearn", "scikit-learn"),
    ("tabulate", "tabulate"),
)
# PyCBC is a base dependency too since P6.1 (D1 = A), but not on Windows, so it is
# reported with the waveform backend instead of counting as missing.

# The extras in pyproject.toml and what each one installs
EXTRAS = {
    "gwdata": (("gwpy", "gwpy"),),
    "ml": (("tensorflow", "tensorflow"),),
    "dev": (("pytest", "pytest"), ("ruff", "ruff")),
}

# (label, name in gwd.config)
PATHS = (
    ("Project root", "ROOT_DIR"),
    ("Training data", "DATA_DIR_TRAIN"),
    ("Noise chunks", "DATA_DIR_NOISE"),
    ("Models", "MODELS_DIR"),
    ("Results", "RESULTS_DIR"),
    ("Plots", "PLOTS_DIR"),
    ("Leaderboard", "LEADERBOARD_FILE"),
)


def package_version(module, distribution):
    """The installed version, or None if the module cannot be found. Imports nothing."""
    if importlib.util.find_spec(module) is None:
        return None
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown version"


def pycbc_works():
    """True if PyCBC can be imported, which the pipeline needs (P5.2)."""
    from gwd.core import waveforms

    return waveforms.pycbc_available()


def gui_backend():
    """(name, interactive) of the matplotlib backend, as the apps check it (P4.5)."""
    from gwd.apps._backend import backend_status

    return backend_status()


def _line(label, value):
    print(f"  {label:<16s} {value}")


def main():
    """Print the report. Returns 1 if a base dependency is missing, else 0."""
    print("=" * 60)
    print("GWD ENVIRONMENT")
    print("=" * 60)
    _line("Python", f"{platform.python_version()} ({sys.executable})")
    _line("Platform", platform.platform())
    _line("gwd", importlib.metadata.version("gwd"))

    print("\nBase dependencies:")
    missing = []
    for module, distribution in BASE:
        version = package_version(module, distribution)
        if version is None:
            missing.append(distribution)
        _line(distribution, version or "MISSING")

    print("\nExtras:")
    for extra, packages in EXTRAS.items():
        versions = [
            (distribution, package_version(module, distribution))
            for module, distribution in packages
        ]
        state = "installed" if all(v for _, v in versions) else "not installed"
        details = ", ".join(f"{d} {v or 'missing'}" for d, v in versions)
        _line(extra, f"{state} ({details})")

    print("\nWaveform backend:")
    from gwd.config import WAVEFORM_BACKEND
    from gwd.core.waveforms import backend_label

    if pycbc_works():
        _line("Pipeline", backend_label(WAVEFORM_BACKEND))
        _line("Demo apps", backend_label(WAVEFORM_BACKEND))
    else:
        _line(
            "Pipeline",
            f"{WAVEFORM_BACKEND} MISSING: gwd data generate and gwd eval run, "
            "thresholds, and compare stop (pip install -e . on Linux or WSL)",
        )
        _line(
            "Demo apps",
            "the waveform explorer stops; the Glitch Hunter has real data only",
        )

    print("\nPaths:")
    from gwd import config

    home = os.environ.get("GWD_HOME", "").strip()
    _line("GWD_HOME", home or "not set")
    for label, name in PATHS:
        path = getattr(config, name)
        _line(label, f"{path}{'' if os.path.exists(path) else ' (does not exist)'}")

    print("\nGUI:")
    name, interactive = gui_backend()
    _line(
        "matplotlib", f"{name} ({'interactive' if interactive else 'not interactive'})"
    )

    print("=" * 60)
    if missing:
        print(f"Missing base dependencies: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
