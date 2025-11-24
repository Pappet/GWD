import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

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
os.makedirs(MODELS_DIR, exist_ok=True)

# Pfade für Modell und Config
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, f"{MODEL_NAME}.keras")
CONFIG_SAVE_PATH = os.path.join(MODELS_DIR, f"{MODEL_NAME}.json")

BATCH_SIZE = 32
EPOCHS = 20 
TEST_SPLIT = 0.2

def load_data():
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
            # Robustere Normalisierung: Standardisierung (Mean=0, Std=1)
            data = (data - np.mean(data)) / np.std(data)
            signals.append(data)
            labels.append(row['has_signal'])
        except Exception as e:
            pass

    X = np.array(signals)
    y = np.array(labels)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    print(f"✅ Daten geladen: {len(X)} Samples. Shape: {X.shape}")
    return X, y

def build_model(input_shape):
    # Ein modernes CNN Design mit BatchNormalization
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
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def save_experiment_config(history_obj, final_acc):
    """Speichert Metadaten zum Training, damit wir später wissen, was wir getan haben."""
    
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
        "architecture": "CNN_V2_BatchNorm", # Manuelle Info zur Architektur
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
    print(f"📝 Experiment-Konfiguration gespeichert: {CONFIG_SAVE_PATH}")

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    best_epoch = np.argmin(val_loss)
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.axvline(best_epoch, color='green', linestyle='--', label='Best Model')
    plt.legend(loc='lower right')
    plt.title('Genauigkeit')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.axvline(best_epoch, color='green', linestyle='--', label='Best Model')
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.show()

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42)
    
    input_shape = (X_train.shape[1], 1)
    model = build_model(input_shape)
    
    # Callbacks
    my_callbacks = [
        callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
        # Speichert jetzt in den neuen, versionierten Pfad
        callbacks.ModelCheckpoint(filepath=MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss')
    ]
    
    print(f"\n🚀 Starte Training: {MODEL_NAME}")
    print(f"💾 Speicherort: {MODELS_DIR}")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=my_callbacks
    )
    
    # Wir laden das BESTE Modell (nicht das vom letzten Epoch) für die Evaluation
    best_model = models.load_model(MODEL_SAVE_PATH)
    plot_history(history)
    
    loss, acc = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"Finale Test-Genauigkeit: {acc*100:.2f}%")
    
    # Speichern der Metadaten
    save_experiment_config(history, acc)

if __name__ == "__main__":
    main()