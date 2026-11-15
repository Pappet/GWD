"""The CNN (P3.7). Building it changes no global state.

The architecture is fixed by tests/training/test_model.py (1 074 609 parameters);
changing it is a deliberate step of its own (DATA-26).
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from gwd.config import LEARNING_RATE

# Stored in each model's .json.
ARCHITECTURE = "CNN_V2_BatchNorm"


def build_model(input_shape, learning_rate=LEARNING_RATE):
    """
    Baut ein modernes CNN-Modell mit BatchNormalization.

    Architektur:
    - 3 Conv1D Blöcke mit steigenden Filtern (16 -> 32 -> 64)
    - BatchNormalization nach jeder Conv-Schicht
    - MaxPooling für Dimensionsreduktion
    - Dense Layer mit Dropout zur Regularisierung
    - Sigmoid Output für binäre Klassifikation

    Args:
        input_shape: Tuple (n_timesteps, n_channels)
        learning_rate: Adam learning rate. Default: LEARNING_RATE in gwd.config.

    Returns:
        Kompiliertes Keras Model
    """
    model = models.Sequential(
        [
            # --- INPUT LAYER MODERNISIEREN ---
            # Statt input_shape=... im Conv1D, nutzen wir einen expliziten Input-Layer
            layers.Input(shape=input_shape),
            # Block 1
            layers.Conv1D(filters=16, kernel_size=32, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling1D(pool_size=4),
            # Block 2
            layers.Conv1D(filters=32, kernel_size=16, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling1D(pool_size=4),
            # Block 3
            layers.Conv1D(filters=64, kernel_size=8, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling1D(pool_size=4),
            # Classifier
            layers.Flatten(),
            layers.Dense(64),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(0.3),
            layers.Dense(
                1, activation="sigmoid", dtype="float32"
            ),  # Output immer in float32
        ]
    )

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    return model
