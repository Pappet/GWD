# File: src/gwd/config.py
import os
import warnings


def _find_root():
    """Return GWD_HOME if set, else the nearest parent folder with a pyproject.toml.

    The root is not derived from this module's own folder, so data never lands
    silently in src/gwd/ or in site-packages.
    """
    home = os.environ.get("GWD_HOME", "").strip()
    if home:
        root = os.path.abspath(home)
        # Warn so that a forgotten variable does not redirect all data unnoticed.
        warnings.warn(f"Using GWD_HOME={root}", stacklevel=2)
        return root
    folder = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isfile(os.path.join(folder, "pyproject.toml")):
            return folder
        parent = os.path.dirname(folder)
        if parent == folder:
            raise RuntimeError(
                "Cannot find the GWD project root: no pyproject.toml above "
                f"{os.path.abspath(__file__)}. Set GWD_HOME to the project folder."
            )
        folder = parent


def ensure_dirs(*paths):
    """Create the given folders if missing. Call it right before writing there."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


# --- PATHS ---
# Nothing is created on import. Code that writes calls ensure_dirs() first.
ROOT_DIR = _find_root()

# Daten-Ordner
DATA_DIR_TRAIN = os.path.join(ROOT_DIR, "data/training")
DATA_DIR_NOISE = os.path.join(ROOT_DIR, "data/noise_background")
MODELS_DIR = os.path.join(ROOT_DIR, "models_registry")

RESULTS_DIR = os.path.join(ROOT_DIR, "results")

PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
VALIDATION_DATASET_PLOTS_DIR = os.path.join(ROOT_DIR, "plots/validation_dataset")
TRAINING_PLOTS_DIR = os.path.join(ROOT_DIR, "plots/training")
EVALUATION_MODELS_PLOTS_DIR = os.path.join(ROOT_DIR, "plots/evaluation_models")

# Dateinamen
LEADERBOARD_FILE = os.path.join(RESULTS_DIR, "model_leaderboard.csv")
LEGACY_LEADERBOARD_FILES = (
    os.path.join(ROOT_DIR, "analysis/model_leaderboard.csv"),
    os.path.join(ROOT_DIR, "model_leaderboard.csv"),
)
LABELS_FILE = "labels.csv"  # Relativ zu DATA_DIR_TRAIN
OPTIMAL_THRESHOLDS_FILE = os.path.join(RESULTS_DIR, "optimal_thresholds.json")

# --- KONSTANTEN ---
SAMPLE_RATE = 4096  # Hz
DURATION = 4.0  # seconds per sample
N_SAMPLES = int(SAMPLE_RATE * DURATION)  # 16384 values per sample
INPUT_SHAPE = (N_SAMPLES, 1)  # (time steps, channels)
SAMPLES_FOR_TRAINING = 10000

# Real-data conditioning: notch the power line and its harmonics, then band-pass.
NOTCH_FREQS = (60, 120, 180, 240)  # Hz
BANDPASS = (20, 300)  # Hz, (low, high)

# Length of the real-noise segment that fetch_noise downloads (about 68 minutes).
NOISE_FETCH_SECONDS = 4096
