"""Training of the CNN: train(cfg) loads a dataset, trains, and saves the model (P3.7).

Importing this module changes nothing: the timestamp, the model name, the paths, and
the precision policy come into being when train() runs. The plot uses a matplotlib
Figure without pyplot, so no backend is chosen. The hyperparameter defaults are in
gwd.config, and gwd train overrides them.
"""

import datetime
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, models

from gwd.config import (
    BATCH_SIZE,
    DATA_DIR_TRAIN,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LABELS_FILE,
    LEARNING_RATE,
    MODELS_DIR,
    N_SAMPLES,
    REDUCE_LR_PATIENCE,
    TEST_SPLIT,
    TRAINING_PLOTS_DIR,
    ensure_dirs,
)
from gwd.data.preprocessing import normalize
from gwd.data.provenance import UNKNOWN_V1, read_manifest
from gwd.registry import (
    MODEL_EXTENSION,
    TIMESTAMP_FORMAT,
    metadata_path,
    write_metadata,
)
from gwd.training.model import ARCHITECTURE, build_model


@dataclass(frozen=True)
class TrainConfig:
    """Settings of one training run. The defaults come from gwd.config."""

    data_folder: str = DATA_DIR_TRAIN
    models_dir: str = MODELS_DIR
    plots_dir: str = TRAINING_PLOTS_DIR
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    patience: int = EARLY_STOPPING_PATIENCE
    # True forces mixed_float16; otherwise only a GPU turns it on (D7).
    mixed_precision: bool = False
    # Seeds Python, numpy, and TensorFlow when train() starts; None leaves them alone.
    seed: Optional[int] = None


@dataclass(frozen=True)
class Run:
    """Name and output files of one training run."""

    timestamp: str
    name: str
    model_path: str
    config_path: str


def new_run(models_dir, now=None):
    """Name a run after the current time, as gwd.registry expects: gwd_model_YYYYMMDD-HHMMSS."""
    timestamp = (now or datetime.datetime.now()).strftime(TIMESTAMP_FORMAT)
    name = f"gwd_model_{timestamp}"
    model_path = os.path.join(models_dir, name + MODEL_EXTENSION)
    return Run(
        timestamp=timestamp,
        name=name,
        model_path=model_path,
        config_path=metadata_path(model_path),
    )


def choose_precision(mixed_precision=False, gpus=None):
    """Return the Keras policy for a run and why: mixed_float16 on request or with a GPU.

    On the CPU, mixed_float16 is slow and only emulated, so float32 is used there (D7).
    gpus=None asks TensorFlow for the visible GPUs.
    """
    if mixed_precision:
        return "mixed_float16", "--mixed-precision"
    if gpus is None:
        gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return "mixed_float16", f"{len(gpus)} GPU(s) found"
    return "float32", "no GPU found"


def missing_dataset_message(data_folder):
    """The error for a dataset folder without labels.csv."""
    return (
        f"No dataset in '{data_folder}': {LABELS_FILE} is missing. "
        "Run 'gwd data generate' first."
    )


def dataset_provenance(data_folder):
    """What the model's metadata records about its dataset, from manifest.json.

    A dataset without a manifest was written before P4.1; every field is then
    UNKNOWN_V1.
    """
    manifest, digest = read_manifest(data_folder)
    if manifest is None:
        return dict.fromkeys(
            (
                "dataset_manifest_sha256",
                "dataset_scenario",
                "waveform_backend",
                "real_noise_fraction",
            ),
            UNKNOWN_V1,
        )
    return {
        "dataset_manifest_sha256": digest,
        "dataset_scenario": manifest["scenario"],
        "waveform_backend": manifest["waveform_backend"],
        "real_noise_fraction": manifest["real_noise_fraction"],
    }


def load_data(data_folder=DATA_DIR_TRAIN):
    """
    Load the training dataset from the .npy files listed in labels.csv.

    Args:
        data_folder: Dataset folder with labels.csv.

    Returns:
        X: array of shape (n_samples, n_timesteps, 1)
        y: array of labels (0 or 1)
    """
    print("📂 Loading dataset...")
    csv_path = os.path.join(data_folder, LABELS_FILE)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(missing_dataset_message(data_folder))

    df = pd.read_csv(csv_path)
    signals = []
    labels = []

    for index, row in df.iterrows():
        try:
            file_path = os.path.join(data_folder, row["filename"])
            data = np.load(file_path)

            # Z-score with the default eps, as before
            data = normalize(data)

            # Clip or pad to the expected length
            TARGET_LEN = N_SAMPLES
            if len(data) > TARGET_LEN:
                data = data[:TARGET_LEN]
            elif len(data) < TARGET_LEN:
                data = np.pad(data, (0, TARGET_LEN - len(data)))

            signals.append(data)
            labels.append(row["has_signal"])
        except Exception as e:
            print(f"⚠️ Error loading {row['filename']}: {e}")

    X = np.array(signals)
    y = np.array(labels)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    print(f"✅ Data loaded: {len(X)} samples. Shape: {X.shape}")
    print(f"   - Samples with signal: {np.sum(y)} ({np.sum(y) / len(y) * 100:.1f}%)")
    print(
        f"   - Samples without signal: {len(y) - np.sum(y)} ({(len(y) - np.sum(y)) / len(y) * 100:.1f}%)"
    )

    return X, y


def save_experiment_config(history_obj, final_acc, cfg, run, policy):
    """
    Save the metadata of a run as .json next to the model, so it can be traced later.

    Args:
        history_obj: Keras History object of the training
        final_acc: test set accuracy
        cfg: TrainConfig of the run
        run: Run with the name and the .json path
        policy: Name of the Keras precision policy the model was trained with
    """
    history = history_obj.history
    # The values of the last epoch
    final_train_acc = history["accuracy"][-1]
    final_val_acc = history["val_accuracy"][-1]
    final_train_loss = history["loss"][-1]
    final_val_loss = history["val_loss"][-1]

    config = {
        "model_name": run.name,
        "timestamp": run.timestamp,
        "dataset_folder": cfg.data_folder,
        **dataset_provenance(cfg.data_folder),
        "seed": cfg.seed,
        # epochs is the maximum; early stopping can end the run before it.
        "epochs": cfg.epochs,
        "epochs_trained": len(history["loss"]),
        # The saved weights are those of this epoch (1-based, lowest val_loss).
        "best_epoch": int(np.argmin(history["val_loss"])) + 1,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "early_stopping_patience": cfg.patience,
        "precision_policy": policy,
        "test_split": TEST_SPLIT,
        "architecture": ARCHITECTURE,
        "metrics": {
            "final_training_acc": float(final_train_acc),
            "final_val_acc": float(final_val_acc),
            "final_training_loss": float(final_train_loss),
            "final_val_loss": float(final_val_loss),
            "test_set_accuracy": float(final_acc),
        },
        "note": "Training with automatic versioning",
    }

    path = write_metadata(run.model_path, config)
    print(f"📋 Experiment config saved: {path}")


def plot_history(history, plot_path):
    """
    Plot accuracy and loss per epoch, with the best epoch marked.

    Args:
        history: Keras History object
        plot_path: PNG file to write; its folder is created if missing.
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    best_epoch = np.argmin(val_loss)
    epochs_range = range(len(acc))

    # A Figure without pyplot: no backend is chosen or changed, no window can open.
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 5))
    ax_acc, ax_loss = fig.subplots(1, 2)

    # Accuracy
    ax_acc.plot(epochs_range, acc, label="Training Accuracy", linewidth=2)
    ax_acc.plot(epochs_range, val_acc, label="Validation Accuracy", linewidth=2)
    ax_acc.axvline(
        best_epoch, color="green", linestyle="--", label="Best Model", alpha=0.7
    )
    ax_acc.legend(loc="lower right")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.grid(True, alpha=0.3)

    # Loss
    ax_loss.plot(epochs_range, loss, label="Training Loss", linewidth=2)
    ax_loss.plot(epochs_range, val_loss, label="Validation Loss", linewidth=2)
    ax_loss.axvline(
        best_epoch, color="green", linestyle="--", label="Best Model", alpha=0.7
    )
    ax_loss.legend(loc="upper right")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.grid(True, alpha=0.3)

    fig.tight_layout()
    ensure_dirs(os.path.dirname(plot_path))
    fig.savefig(plot_path, dpi=150)
    print(f"📊 Training plot saved: {plot_path}")


def train(cfg=None):
    """
    Train one model with the precision from choose_precision() and save it.

    The precision policy is set only for the run and restored afterwards.

    Args:
        cfg: TrainConfig; None means all defaults.

    Returns:
        The Run with the name and the paths of the saved model and its .json.
    """
    cfg = cfg or TrainConfig()
    if cfg.seed is not None:
        # Seeds Python's random, numpy, and TensorFlow.
        tf.keras.utils.set_random_seed(cfg.seed)
    policy, reason = choose_precision(cfg.mixed_precision)
    print(f"Precision: {policy} ({reason})")
    previous = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy(policy)
    try:
        return _train_and_save(cfg, policy)
    finally:
        tf.keras.mixed_precision.set_global_policy(previous)


def _train_and_save(cfg, policy):
    """Load the data, train, evaluate the best epoch, and save model, plot, and .json."""
    run = new_run(cfg.models_dir)

    # Load and split the data
    X, y = load_data(cfg.data_folder)
    provenance = dataset_provenance(cfg.data_folder)
    if provenance["dataset_scenario"] == UNKNOWN_V1:
        print(
            "⚠️ The dataset has no manifest.json (written before P4.1); the model's "
            f"metadata records its backend and noise as '{UNKNOWN_V1}'."
        )
    else:
        print(
            f"Dataset: {provenance['dataset_scenario']}, backend "
            f"{provenance['waveform_backend']}, real noise "
            f"{provenance['real_noise_fraction']:.0%}"
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
    )

    print("\n📊 Data split:")
    print(f"   - Training: {len(X_train)} Samples")
    print(f"   - Test: {len(X_test)} Samples")

    # Build the model
    input_shape = (X_train.shape[1], 1)
    model = build_model(input_shape, cfg.learning_rate)

    print("\n🏗️ Model architecture:")
    model.summary()

    # Callbacks
    my_callbacks = [
        callbacks.EarlyStopping(
            patience=cfg.patience,
            monitor="val_loss",
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=run.model_path, save_best_only=True, monitor="val_loss", verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"\n🚀 Starting training: {run.name}")
    print(f"💾 Saving to: {cfg.models_dir}")

    # Training (the checkpoint callback writes into models_dir)
    ensure_dirs(cfg.models_dir)
    history = model.fit(
        X_train,
        y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_data=(X_test, y_test),
        callbacks=my_callbacks,
        verbose=1,
    )

    # Evaluate the BEST model (the checkpoint), not the one from the last epoch
    best_model = models.load_model(run.model_path)

    # Plot the history
    plot_history(
        history, os.path.join(cfg.plots_dir, f"training_history_{run.timestamp}.png")
    )

    # Final evaluation
    print("\n📈 Final evaluation:")
    loss, acc = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"   - Test Loss: {loss:.4f}")
    print(f"   - Test Accuracy: {acc * 100:.2f}%")

    # Save the metadata
    save_experiment_config(history, acc, cfg, run, policy)

    print("\n✅ Training finished!")
    print(f"   - Model: {run.model_path}")
    print(f"   - Config: {run.config_path}")
    return run


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit("The options are on the gwd command: gwd train --help")
    train()
