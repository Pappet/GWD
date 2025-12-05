import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

# Mixed precision für schnelleres Training auf RTX GPUs
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ==========================================
# KONFIGURATION & VERSIONIERUNG
# ==========================================
DATA_FOLDER = "gw_training_data"
LABEL_FILE = "labels.csv"

# Wir erstellen einen Zeitstempel für dieses Experiment
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_NAME = f"gwd_model_{TIMESTAMP}"

# Alle Modelle kommen in einen eigenen Ordner, damit es ordentlich bleibt
MODELS_DIR = "models_registry"
PLOTS_DIR = "plots"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Pfade für Modell und Config
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, f"{MODEL_NAME}.keras")
CONFIG_SAVE_PATH = os.path.join(MODELS_DIR, f"{MODEL_NAME}.json")

BATCH_SIZE = 128
EPOCHS = 40 
TEST_SPLIT = 0.2


def load_data():
    """
    Lädt den Trainingsdatensatz aus Numpy-Dateien und CSV-Labels.
    
    Returns:
        X: Array der Form (n_samples, n_timesteps, 1)
        y: Array der Labels (0 oder 1)
    """
    print("📂 Lade Datensatz...")
    csv_path = os.path.join(DATA_FOLDER, LABEL_FILE)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Konnte {csv_path} nicht finden.")
        
    df = pd.read_csv(csv_path)
    signals = []
    labels = []
    
    for index, row in df.iterrows():
        try:
            file_path = os.path.join(DATA_FOLDER, row['filename'])
            data = np.load(file_path)
            
            # Robustere Normalisierung
            # WICHTIG: Addiere kleines Epsilon auch im Zähler, falls std sehr klein ist (selten)
            data = (data - np.mean(data)) / (np.std(data) + 1e-10)
            
            # Sicherstellen, dass die Länge stimmt (Clip/Pad)
            TARGET_LEN = 16384 # 4s * 4096Hz
            if len(data) > TARGET_LEN:
                data = data[:TARGET_LEN]
            elif len(data) < TARGET_LEN:
                data = np.pad(data, (0, TARGET_LEN - len(data)))
                
            signals.append(data)
            labels.append(row['has_signal'])
        except Exception as e:
            print(f"⚠️ Fehler beim Laden von {row['filename']}: {e}")

    X = np.array(signals)
    y = np.array(labels)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    print(f"✅ Daten geladen: {len(X)} Samples. Shape: {X.shape}")
    print(f"   - Samples mit Signal: {np.sum(y)} ({np.sum(y)/len(y)*100:.1f}%)")
    print(f"   - Samples ohne Signal: {len(y) - np.sum(y)} ({(len(y)-np.sum(y))/len(y)*100:.1f}%)")
    
    return X, y


def build_model(input_shape):
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
    
    Returns:
        Kompiliertes Keras Model
    """
    model = models.Sequential([
        # Block 1
        layers.Conv1D(filters=16, kernel_size=32, padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=4),
        
        # Block 2
        layers.Conv1D(filters=32, kernel_size=16, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=4),
        
        # Block 3
        layers.Conv1D(filters=64, kernel_size=8, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=4),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        layers.Dense(1, activation='sigmoid', dtype='float32')  # Output immer in float32
    ])
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def save_experiment_config(history_obj, final_acc):
    """
    Speichert Metadaten zum Training, damit wir später wissen, was wir getan haben.
    
    Args:
        history_obj: Keras History-Objekt vom Training
        final_acc: Test-Set Accuracy
    """
    # Wir holen uns die letzten Werte aus der History
    final_train_acc = history_obj.history['accuracy'][-1]
    final_val_acc = history_obj.history['val_accuracy'][-1]
    final_train_loss = history_obj.history['loss'][-1]
    final_val_loss = history_obj.history['val_loss'][-1]

    config = {
        "model_name": MODEL_NAME,
        "timestamp": TIMESTAMP,
        "dataset_folder": DATA_FOLDER,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "test_split": TEST_SPLIT,
        "architecture": "CNN_V2_BatchNorm",
        "metrics": {
            "final_training_acc": float(final_train_acc),
            "final_val_acc": float(final_val_acc),
            "final_training_loss": float(final_train_loss),
            "final_val_loss": float(final_val_loss),
            "test_set_accuracy": float(final_acc)
        },
        "note": "Training mit automatischer Versionierung"
    }
    
    with open(CONFIG_SAVE_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print(f"📋 Experiment-Konfiguration gespeichert: {CONFIG_SAVE_PATH}")


def plot_history(history):
    """
    Erstellt Visualisierung des Trainingsverlaufs.
    
    Args:
        history: Keras History-Objekt
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    best_epoch = np.argmin(val_loss)
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2)
    plt.axvline(best_epoch, color='green', linestyle='--', label='Best Model', alpha=0.7)
    plt.legend(loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Genauigkeit')
    plt.grid(True, alpha=0.3)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', linewidth=2)
    plt.axvline(best_epoch, color='green', linestyle='--', label='Best Model', alpha=0.7)
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f'training_history_{TIMESTAMP}.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"📊 Training-Plot gespeichert: {plot_path}")


def main():
    """
    Hauptfunktion: Lädt Daten, trainiert Modell, evaluiert und speichert alles.
    """
    # Daten laden und aufteilen
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
    )
    
    print(f"\n📊 Datensplit:")
    print(f"   - Training: {len(X_train)} Samples")
    print(f"   - Test: {len(X_test)} Samples")
    
    # Modell bauen
    input_shape = (X_train.shape[1], 1)
    model = build_model(input_shape)
    
    print(f"\n🏗️ Modell-Architektur:")
    model.summary()
    
    # Callbacks
    my_callbacks = [
        callbacks.EarlyStopping(
            patience=5, 
            monitor='val_loss', 
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH, 
            save_best_only=True, 
            monitor='val_loss',
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print(f"\n🚀 Starte Training: {MODEL_NAME}")
    print(f"💾 Speicherort: {MODELS_DIR}")
    
    # Training
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=my_callbacks,
        verbose=1
    )
    
    # Wir laden das BESTE Modell (nicht das vom letzten Epoch) für die Evaluation
    best_model = models.load_model(MODEL_SAVE_PATH)
    
    # Visualisierung erstellen
    plot_history(history)
    
    # Finale Evaluation
    print("\n📈 Finale Evaluation:")
    loss, acc = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"   - Test Loss: {loss:.4f}")
    print(f"   - Test Accuracy: {acc*100:.2f}%")
    
    # Metadaten speichern
    save_experiment_config(history, acc)
    
    print(f"\n✅ Training abgeschlossen!")
    print(f"   - Modell: {MODEL_SAVE_PATH}")
    print(f"   - Config: {CONFIG_SAVE_PATH}")


if __name__ == "__main__":
    main()