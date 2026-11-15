"""The one place that finds, selects, and loads models (P3.3).

Models are files ``gwd_model_YYYYMMDD-HHMMSS.keras`` in ``MODELS_DIR``, each with an
optional ``.json`` metadata file next to it. "Latest" is the newest timestamp in the
file name, not the file's ctime: on Linux and WSL, ctime changes on chmod, copy, or
restore (apps:APPX-4). Files without a timestamp in the name fall back to their mtime,
with a warning.

TensorFlow is imported only inside load_model, so importing this module stays light.
"""

import datetime
import glob
import json
import os
import re
import warnings
from typing import Optional

from gwd.config import MODELS_DIR

MODEL_EXTENSION = ".keras"
TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
_NAME_PATTERN = re.compile(r"^gwd_model_(\d{8}-\d{6})$")


def model_name(path: str) -> str:
    """The model's name: the file name without folder and extension."""
    return os.path.splitext(os.path.basename(path))[0]


def model_time(path: str) -> datetime.datetime:
    """When the model was trained: the timestamp in its name, or its mtime with a warning."""
    match = _NAME_PATTERN.match(model_name(path))
    if match:
        try:
            return datetime.datetime.strptime(match.group(1), TIMESTAMP_FORMAT)
        except ValueError:
            pass  # e.g. month 13: treat it like a name without a timestamp
    warnings.warn(
        f"{os.path.basename(path)} has no gwd_model_YYYYMMDD-HHMMSS name; "
        "ordering it by its modification time.",
        stacklevel=2,
    )
    return datetime.datetime.fromtimestamp(os.path.getmtime(path))


def list_models(models_dir: str = MODELS_DIR) -> list:
    """All models in models_dir, oldest first. Empty if the folder does not exist."""
    files = glob.glob(os.path.join(models_dir, "*" + MODEL_EXTENSION))
    return sorted(files, key=lambda path: (model_time(path), model_name(path)))


def latest_model(models_dir: str = MODELS_DIR) -> Optional[str]:
    """The newest model in models_dir, or None if there is none."""
    models = list_models(models_dir)
    return models[-1] if models else None


def resolve(model: str = "latest", models_dir: str = MODELS_DIR) -> str:
    """Turn 'latest', a model name, or a path into the path of an existing model file.

    Raises FileNotFoundError if there is no such model.
    """
    if model == "latest":
        path = latest_model(models_dir)
        if path is None:
            raise FileNotFoundError(
                f"No models ({MODEL_EXTENSION}) in '{models_dir}'. "
                "Run 'python -m gwd.training.train' first to train a model."
            )
        return path
    if os.path.isfile(model):
        return model
    for candidate in (model, model + MODEL_EXTENSION):
        path = os.path.join(models_dir, candidate)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Model '{model}' is neither a file nor a model in '{models_dir}'."
    )


def load_model(path: str, compile: bool = False):
    """Load a Keras model. Without compile, Keras skips the optimizer state and its warning."""
    import tensorflow as tf

    return tf.keras.models.load_model(path, compile=compile)


def metadata_path(model_path: str) -> str:
    """The metadata file that belongs to a model: same name, .json instead of .keras."""
    return os.path.splitext(model_path)[0] + ".json"


def read_metadata(model_path: str) -> Optional[dict]:
    """The model's metadata, or None if it has no metadata file."""
    path = metadata_path(model_path)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_metadata(model_path: str, metadata: dict) -> str:
    """Write the model's metadata file and return its path."""
    path = metadata_path(model_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    return path


def get_latest_model(models_dir: str) -> Optional[str]:
    """Sucht das neueste Modell im Registry-Ordner."""
    if not os.path.exists(models_dir):
        print(f"❌ Ordner '{models_dir}' nicht gefunden.")
        return None

    files = glob.glob(os.path.join(models_dir, "*.keras"))
    if not files:
        print("❌ Keine Modelle (.keras) gefunden.")
        return None

    latest_file = max(files, key=os.path.getctime)
    return latest_file
