import pandas as pd
import os
import numpy as np
import sys
import glob
import random

# Import-Pfad anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gwd_core.waveforms import (
    generate_astrophysical_chirp, 
    generate_realistic_chirp, 
    generate_random_bbh_parameters
)
from gwd_core.noise import generate_gaussian_noise

REAL_NOISE_DIR = "gw_noise_background"

def scale_signal_to_snr(signal, noise, target_snr):
    """
    Skaliert das Signal basierend auf PEAK-Amplitude relativ zur Noise-Std.
    Das entspricht der Logik in evaluate_model_with_roc.py.
    """
    # 1. Noise Level messen (Standardabweichung)
    noise_level = np.std(noise)
    if noise_level == 0:
        noise_level = 1.0
        
    # 2. Signal normalisieren (auf Peak = 1.0)
    sig_max = np.max(np.abs(signal))
    if sig_max == 0:
        return signal, 0.0
        
    signal_normalized = signal / sig_max
    
    # 3. Auf Ziel-SNR skalieren
    # Peak Amplitude = target_snr * noise_level
    scaled_signal = signal_normalized * target_snr * noise_level
    
    return scaled_signal, target_snr

def load_random_real_noise(length):
    """Lädt echtes Rauschen und normalisiert es auf Std=1."""
    files = glob.glob(os.path.join(REAL_NOISE_DIR, "*.npy"))
    if not files:
        return generate_gaussian_noise(length, 1.0)
    
    f = random.choice(files)
    try:
        data = np.load(f)
        if len(data) != length:
            # Einfaches Resizing falls nötig
            if len(data) > length:
                start = np.random.randint(0, len(data) - length)
                data = data[start:start+length]
            else:
                return generate_gaussian_noise(length, 1.0)
        
        # WICHTIG: Normalisierung, damit SNR-Berechnung konsistent ist
        data = (data - np.mean(data)) / (np.std(data) + 1e-10)
        return data
    except:
        return generate_gaussian_noise(length, 1.0)

def generate_dataset(num_samples=10000, output_folder="gw_training_data"):
    print(f"🚀 Starte Daten-Generierung (Peak-SNR Methode). Ziel: {num_samples}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    fs = 4096 
    duration = 4.0
    time = np.linspace(0, duration, int(fs * duration))
    labels = []
    
    has_real_noise = len(glob.glob(os.path.join(REAL_NOISE_DIR, "*.npy"))) > 0
    
    for i in range(num_samples):
        # Default Werte
        has_signal = 0
        t_merger = 0.0
        snr = 0.0
        m1 = 0.0
        m2 = 0.0
        dist = 0.0
        data = None
        
        dice = np.random.rand()
        
        # Rauschen laden (Echt oder Simuliert)
        if has_real_noise and dice < 0.8:
            noise = load_random_real_noise(len(time))
        else:
            noise = generate_gaussian_noise(len(time), 1.0)
            
        # Soll ein Signal rein? (50% Chance in den Signal-Blöcken)
        # Wir vereinfachen die Logik hier etwas für klarere Verteilung
        # 50% Rauschen pur, 50% Signal+Rauschen
        should_have_signal = (i % 2 == 0) # Strikt 50/50 Verteilung
        
        if should_have_signal:
            # Parameter würfeln
            params = generate_random_bbh_parameters()
            m1 = params['mass1']
            m2 = params['mass2']
            dist = params['distance']
            t_merger = np.random.uniform(duration * 0.5, duration * 0.9) # Merger eher hinten
            
            # Signal generieren
            # Wir nutzen hier generate_astrophysical_chirp als Wrapper, 
            # da es normalize=True unterstützt, was wir für Peak-Scaling brauchen
            signal_raw, _ = generate_astrophysical_chirp(
                time, t_merger=t_merger, mass1=m1, mass2=m2, distance_mpc=dist, normalize=True
            )
            
            # SNR wählen: WICHTIG! Jetzt kleinere Werte nehmen!
            # Vorher: 5-25 (bei falscher Berechnung)
            # Jetzt: 1.0 - 15.0 (realistische Peak SNRs für Training)
            # Wir wollen auch schwache Signale (SNR 1-3) lernen!
            target_snr = np.random.uniform(1.0, 12.0)
            
            # Skalieren
            signal_scaled, actual_snr = scale_signal_to_snr(signal_raw, noise, target_snr)
            
            data = noise + signal_scaled
            has_signal = 1
            snr = actual_snr
            
        else:
            data = noise
            has_signal = 0
            
        # Speichern
        filename = f"sample_{i:05d}.npy"
        filepath = os.path.join(output_folder, filename)
        np.save(filepath, data)
        
        labels.append({
            "filename": filename,
            "has_signal": has_signal,
            "merger_time": float(t_merger),
            "snr": float(snr),
            "mass1": float(m1),
            "mass2": float(m2),
            "distance": float(dist)
        })
        
        if i % 1000 == 0:
            print(f"   ... {i}/{num_samples}")

    df = pd.DataFrame(labels)
    df.to_csv(os.path.join(output_folder, "labels.csv"), index=False)
    print("✅ Fertig.")

if __name__ == "__main__":
    generate_dataset()