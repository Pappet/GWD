"""Training of the CNN: train(cfg) loads a dataset, trains, and saves the model (P3.7).

The timestamp, the model name, and the paths come into being when train() runs, not on
import. The hyperparameter defaults are in gwd.config, and gwd train overrides them.
"""

import datetime
import json
import os
import sys
from dataclasses import dataclass

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
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
from gwd.training.model import ARCHITECTURE, build_model

# Mixed precision für schnelleres Training auf RTX GPUs
tf.keras.mixed_precision.set_global_policy("mixed_float16")


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


@dataclass(frozen=True)
class Run:
    """Name and output files of one training run."""

    timestamp: str
    name: str
    model_path: str
    config_path: str


def new_run(models_dir, now=None):
    """Name a run after the current time, as gwd.registry expects: gwd_model_YYYYMMDD-HHMMSS."""
    timestamp = (now or datetime.datetime.now()).strftime("%Y%m%d-%H%M%S")
    name = f"gwd_model_{timestamp}"
    return Run(
        timestamp=timestamp,
        name=name,
        model_path=os.path.join(models_dir, f"{name}.keras"),
        config_path=os.path.join(models_dir, f"{name}.json"),
    )


def load_data(data_folder=DATA_DIR_TRAIN):
    """
    Lädt den Trainingsdatensatz aus Numpy-Dateien und CSV-Labels.

    Args:
        data_folder: Dataset folder with labels.csv.

    Returns:
        X: Array der Form (n_samples, n_timesteps, 1)
        y: Array der Labels (0 oder 1)
    """
    print("📂 Lade Datensatz...")
    csv_path = os.path.join(data_folder, LABELS_FILE)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Konnte {csv_path} nicht finden.")

    df = pd.read_csv(csv_path)
    signals = []
    labels = []

    for index, row in df.iterrows():
        try:
            file_path = os.path.join(data_folder, row["filename"])
            data = np.load(file_path)

            # Z-score with the default eps, as before
            data = normalize(data)

            # Sicherstellen, dass die Länge stimmt (Clip/Pad)
            TARGET_LEN = N_SAMPLES
            if len(data) > TARGET_LEN:
                data = data[:TARGET_LEN]
            elif len(data) < TARGET_LEN:
                data = np.pad(data, (0, TARGET_LEN - len(data)))

            signals.append(data)
            labels.append(row["has_signal"])
        except Exception as e:
            print(f"⚠️ Fehler beim Laden von {row['filename']}: {e}")

    X = np.array(signals)
    y = np.array(labels)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    print(f"✅ Daten geladen: {len(X)} Samples. Shape: {X.shape}")
    print(f"   - Samples mit Signal: {np.sum(y)} ({np.sum(y) / len(y) * 100:.1f}%)")
    print(
        f"   - Samples ohne Signal: {len(y) - np.sum(y)} ({(len(y) - np.sum(y)) / len(y) * 100:.1f}%)"
    )

    return X, y


def save_experiment_config(history_obj, final_acc, cfg, run):
    """
    Speichert Metadaten zum Training, damit wir später wissen, was wir getan haben.

    Args:
        history_obj: Keras History-Objekt vom Training
        final_acc: Test-Set Accuracy
        cfg: TrainConfig of the run
        run: Run with the name and the .json path
    """
    # Wir holen uns die letzten Werte aus der History
    final_train_acc = history_obj.history["accuracy"][-1]
    final_val_acc = history_obj.history["val_accuracy"][-1]
    final_train_loss = history_obj.history["loss"][-1]
    final_val_loss = history_obj.history["val_loss"][-1]

    config = {
        "model_name": run.name,
        "timestamp": run.timestamp,
        "dataset_folder": cfg.data_folder,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "early_stopping_patience": cfg.patience,
        "test_split": TEST_SPLIT,
        "architecture": ARCHITECTURE,
        "metrics": {
            "final_training_acc": float(final_train_acc),
            "final_val_acc": float(final_val_acc),
            "final_training_loss": float(final_train_loss),
            "final_val_loss": float(final_val_loss),
            "test_set_accuracy": float(final_acc),
        },
        "note": "Training mit automatischer Versionierung",
    }

    with open(run.config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"📋 Experiment-Konfiguration gespeichert: {run.config_path}")


def plot_history(history, plot_path):
    """
    Erstellt Visualisierung des Trainingsverlaufs.

    Args:
        history: Keras History-Objekt
        plot_path: PNG file to write; its folder is created if missing.
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    best_epoch = np.argmin(val_loss)
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", linewidth=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", linewidth=2)
    plt.axvline(
        best_epoch, color="green", linestyle="--", label="Best Model", alpha=0.7
    )
    plt.legend(loc="lower right")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Genauigkeit")
    plt.grid(True, alpha=0.3)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", linewidth=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", linewidth=2)
    plt.axvline(
        best_epoch, color="green", linestyle="--", label="Best Model", alpha=0.7
    )
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    ensure_dirs(os.path.dirname(plot_path))
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"📊 Training-Plot gespeichert: {plot_path}")


def train(cfg=None):
    """
    Lädt Daten, trainiert Modell, evaluiert und speichert alles.

    Args:
        cfg: TrainConfig; None means all defaults.

    Returns:
        The Run with the name and the paths of the saved model and its .json.
    """
    cfg = cfg or TrainConfig()
    run = new_run(cfg.models_dir)

    # Daten laden und aufteilen
    X, y = load_data(cfg.data_folder)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
    )

    print("\n📊 Datensplit:")
    print(f"   - Training: {len(X_train)} Samples")
    print(f"   - Test: {len(X_test)} Samples")

    # Modell bauen
    input_shape = (X_train.shape[1], 1)
    model = build_model(input_shape, cfg.learning_rate)

    print("\n🏗️ Modell-Architektur:")
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

    print(f"\n🚀 Starte Training: {run.name}")
    print(f"💾 Speicherort: {cfg.models_dir}")

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

    # Wir laden das BESTE Modell (nicht das vom letzten Epoch) für die Evaluation
    best_model = models.load_model(run.model_path)

    # Visualisierung erstellen
    plot_history(
        history, os.path.join(cfg.plots_dir, f"training_history_{run.timestamp}.png")
    )

    # Finale Evaluation
    print("\n📈 Finale Evaluation:")
    loss, acc = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"   - Test Loss: {loss:.4f}")
    print(f"   - Test Accuracy: {acc * 100:.2f}%")

    # Metadaten speichern
    save_experiment_config(history, acc, cfg, run)

    print("\n✅ Training abgeschlossen!")
    print(f"   - Modell: {run.model_path}")
    print(f"   - Config: {run.config_path}")
    return run


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit("The options are on the gwd command: gwd train --help")
    train()
