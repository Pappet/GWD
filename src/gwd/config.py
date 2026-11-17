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

# Data folders
DATA_DIR_TRAIN = os.path.join(ROOT_DIR, "data/training")
DATA_DIR_NOISE = os.path.join(ROOT_DIR, "data/noise_background")
MODELS_DIR = os.path.join(ROOT_DIR, "models_registry")

RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Fixed, hashed test sets (P5.7, gwd.evaluation.testsets): the arrays are a local cache
# under data/eval/, the manifests with the hash under results/testsets/ (D9 = A: both
# stay local; under D9 = B the manifests could be versioned, which data/ cannot).
TESTSET_DIR = os.path.join(ROOT_DIR, "data/eval")
TESTSET_MANIFEST_DIR = os.path.join(RESULTS_DIR, "testsets")

PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
VALIDATION_DATASET_PLOTS_DIR = os.path.join(ROOT_DIR, "plots/validation_dataset")
TRAINING_PLOTS_DIR = os.path.join(ROOT_DIR, "plots/training")
EVALUATION_MODELS_PLOTS_DIR = os.path.join(ROOT_DIR, "plots/evaluation_models")

# File names
# Leaderboard v2 (P5.8): raw numbers, one row per gwd eval run, with the test set hash.
LEADERBOARD_FILE = os.path.join(RESULTS_DIR, "leaderboard_v2.csv")
# The 12-column leaderboards from before P5.8. Nothing writes them any more; they stay
# as archives and can still be read (gwd.evaluation.leaderboard.read_legacy).
LEGACY_LEADERBOARD_FILES = (
    os.path.join(RESULTS_DIR, "model_leaderboard.csv"),
    os.path.join(ROOT_DIR, "analysis/model_leaderboard.csv"),
    os.path.join(ROOT_DIR, "model_leaderboard.csv"),
)
LABELS_FILE = "labels.csv"  # relative to DATA_DIR_TRAIN
# gwd eval thresholds writes one report per model, <model name>.json, into this folder.
THRESHOLD_REPORTS_DIR = os.path.join(RESULTS_DIR, "thresholds")
# The single report that every gwd eval thresholds overwrote before. Nothing writes or
# reads it any more; an existing file stays as it is.
OPTIMAL_THRESHOLDS_FILE = os.path.join(RESULTS_DIR, "optimal_thresholds.json")

# --- CONSTANTS ---
SAMPLE_RATE = 4096  # Hz
DURATION = 4.0  # seconds per sample
N_SAMPLES = int(SAMPLE_RATE * DURATION)  # 16384 values per sample
INPUT_SHAPE = (N_SAMPLES, 1)  # (time steps, channels)
SAMPLES_FOR_TRAINING = 10000
# dtype of the sample files gwd data generate writes since P5.6 (float64 before).
# 16384 float32 values are 65 536 bytes of data per file; float64 files stay readable.
SAMPLE_DTYPE = "float32"

# The waveform backend of the pipeline (datasets, test sets, evaluation). gwd.core reads
# no config, so the callers pass it on. Without PyCBC the pipeline stops with
# gwd.core.waveforms.WaveformBackendError (P5.2); since P6.1 it is the only backend (D1).
WAVEFORM_BACKEND = "pycbc"

# Real-data conditioning: notch the power line and its harmonics, then band-pass.
NOTCH_FREQS = (60, 120, 180, 240)  # Hz
BANDPASS = (20, 300)  # Hz, (low, high)

# Whitening (V3, preprocessing "whiten_bandpass" in gwd.data.scenarios): simulated
# samples whiten with the known PSD of their noise, real data with the Welch PSD of
# their padded segment (gwd.data.preprocessing). That Welch PSD takes Hann segments of
# this length (s), half overlap, and the median average (robust against a loud event).
WHITEN_PSD_SEGMENT_S = 4.0
# Real data are tapered with a Tukey window over this many seconds at each end before
# they are whitened, so the jump between their two ends does not ring into the window.
# The taper lies inside the pads, which are cut off afterwards.
WHITEN_TAPER_S = 1.0

# Length of the real-noise segment that fetch_noise downloads (about 68 minutes).
# The segment start is START_GPS in gwd.data.fetch_noise.
NOISE_FETCH_SECONDS = 4096

# --- TEST SET DEFAULTS ---
# gwd eval run, thresholds, and compare override them with --testset, --seed, and --n.
# Without --testset they take the set of the model's training scenario version
# (gwd.evaluation.testsets.default_for_model): the in-distribution set of V2 and V3, and
# for V1, and for models that do not name their scenario, DEFAULT_TESTSET.
DEFAULT_TESTSET = "stress_v1"  # = EVAL_V1, comparable with the numbers before P5.7
DEFAULT_TESTSET_BY_VERSION = {1: DEFAULT_TESTSET, 2: "indist_v2", 3: "indist_v3"}
TESTSET_SEED = 0
TESTSET_SIZE = 1000  # samples of the balanced part 'test'; calib_bg has as many

# --- DETECTION THRESHOLD (P5.8, D10 = A) ---
# gwd eval thresholds calibrates one threshold per model on the noise-only part calib_bg
# of a test set, at this false alarm rate, and stores it in the model's metadata
# (gwd.registry.set_threshold). evaluate, compare, and the Glitch Hunter read it with
# gwd.registry.get_threshold; a model without one gets THRESHOLD_FALLBACK, with a warning.
THRESHOLD_TARGET_FPR = 0.01
THRESHOLD_FALLBACK = 0.5

# --- TRAINING DEFAULTS ---
# gwd train overrides them with --epochs, --batch-size, --lr, and --patience.
EPOCHS = 40  # upper limit; early stopping usually ends sooner
BATCH_SIZE = 128
LEARNING_RATE = 5e-4  # Adam
EARLY_STOPPING_PATIENCE = 5  # epochs without a better val_loss before stopping
REDUCE_LR_PATIENCE = 3  # epochs without a better val_loss before halving the rate
# The three-way split of gwd train (P5.6): 70 % train, 15 % val, 15 % test, grouped by
# noise chunk (gwd.training.split). Until P5.6, one 20 % share was val and test at once.
VAL_SPLIT = 0.15  # steers early stopping, the checkpoint, and the learning rate
TEST_SPLIT = 0.15  # evaluated once, with the best epoch, after training
# Before every validation, gwd train recomputes the moving BatchNorm statistics from
# this many fixed training samples with the current weights (gwd.training.callbacks).
BN_RECALIBRATION_SAMPLES = 1024

# --- ONLINE TRAINING (gwd train --online, gwd.data.online) ---
# Fresh samples from the scenario for every epoch; val and test are fixed, seeded sets.
# gwd train overrides them with --samples-per-epoch, --val-samples, --test-samples, and
# --workers.
ONLINE_SAMPLES_PER_EPOCH = 8192  # rounded up to whole batches
ONLINE_VAL_SAMPLES = 2048
ONLINE_TEST_SAMPLES = 2048
# Worker processes that draw the samples; at most the CPU count minus 2 at run time.
ONLINE_WORKERS = 12
ONLINE_CHUNK = 16  # samples per task of a worker
