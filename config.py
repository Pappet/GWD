# Datei: GWD/config.py
import os

# --- PFADE ---
# Das Root-Verzeichnis ist dort, wo diese Datei liegt
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Daten-Ordner
DATA_DIR_TRAIN = os.path.join(ROOT_DIR, "data/training")
DATA_DIR_NOISE = os.path.join(ROOT_DIR, "data/noise_background")
MODELS_DIR = os.path.join(ROOT_DIR, "models_registry")

PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
VALIDATION_DATASET_PLOTS_DIR = os.path.join(ROOT_DIR, "plots/validation_dataset")
TRAINING_PLOTS_DIR = os.path.join(ROOT_DIR, "plots/training")
EVALUATION_MODELS_PLOTS_DIR = os.path.join(ROOT_DIR, "plots/evaluation_models")

# Dateinamen
LEADERBOARD_FILE = os.path.join(ROOT_DIR, "analysis/model_leaderboard.csv")
LABELS_FILE = "labels.csv" # Relativ zu DATA_DIR_TRAIN
OPTIMAL_THRESHOLDS_FILE = os.path.join(EVALUATION_MODELS_PLOTS_DIR, "optimal_thresholds.json")

# --- KONSTANTEN ---
SAMPLE_RATE = 4096
DURATION = 4.0
INPUT_SHAPE = (16384, 1) # (SampleRate * Duration, Channels)
SAMPLES_FOR_TRAINING = 10000

# Sicherstellen, dass Ordner existieren
os.makedirs(DATA_DIR_TRAIN, exist_ok=True)
os.makedirs(DATA_DIR_NOISE, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(VALIDATION_DATASET_PLOTS_DIR, exist_ok=True)
os.makedirs(TRAINING_PLOTS_DIR, exist_ok=True)
os.makedirs(EVALUATION_MODELS_PLOTS_DIR, exist_ok=True)