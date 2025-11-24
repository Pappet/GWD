import pandas as pd
import os
import numpy as np
import sys
import glob
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Wir nutzen jetzt das verbesserte Super-Hirn (waveforms.py)
from gwd_core.waveforms import generate_astrophysical_chirp
from gwd_core.noise import generate_gaussian_noise

REAL_NOISE_DIR = "gw_noise_background"

def load_random_real_noise(length):
    """Lädt echtes Rauschen und NORMALISIERT es sofort."""
    files = glob.glob(os.path.join(REAL_NOISE_DIR, "*.npy"))
    if not files: return generate_gaussian_noise(length, 1.0)
    
    f = random.choice(files)
    try:
        data = np.load(f)
        if len(data) != length: return generate_gaussian_noise(length, 1.0)
        
        # Rauschen auf Standardabweichung 1.0 zwingen für korrekte SNR-Berechnung
        if np.std(data) > 0:
            data = (data - np.mean(data)) / np.std(data)
            
        return data
    except:
        return generate_gaussian_noise(length, 1.0)

def generate_dataset(num_samples=5000, output_folder="gw_training_data"):
    print(f"🚀 Starte INTELLIGENTE Daten-Fabrik (Diverse Physik). Ziel: {num_samples}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    fs = 4096 
    duration = 4.0
    time = np.linspace(0, duration, int(fs * duration))
    labels = []
    
    has_real_noise = len(glob.glob(os.path.join(REAL_NOISE_DIR, "*.npy"))) > 0
    
    for i in range(num_samples):
        # Default Werte für leere Samples (nur Rauschen)
        has_signal = 0
        t_merger = 0
        snr = 0
        m1 = 0
        m2 = 0
        dist = 0
        data = None
        
        dice = np.random.rand()
        
        # --- SZENARIO 1: Nur Rauschen (40%) ---
        if dice < 0.4 and has_real_noise:
            data = load_random_real_noise(len(time))
            has_signal = 0
            
        # --- SZENARIO 2: Signal in Echtem Rauschen (40%) ---
        elif dice < 0.8 and has_real_noise:
            noise = load_random_real_noise(len(time))
            
            # Zufällige Physik-Parameter würfeln
            t_merger = np.random.uniform(duration * 0.4, duration * 0.95)
            m1 = np.random.uniform(10, 80)   # Kleine bis supermassive Löcher
            m2 = np.random.uniform(10, 80)
            dist = np.random.uniform(200, 800) # Nah bis fern
            
            # Signal generieren (nutzt jetzt deine neue waveforms.py)
            # normalize=True ist wichtig für ML, damit Amplitude ~1.0 ist
            signal_raw, _ = generate_astrophysical_chirp(
                time, 
                t_merger=t_merger,
                mass1=m1, 
                mass2=m2, 
                distance_mpc=dist,
                normalize=True 
            )
            
            snr = np.random.uniform(1.0, 3.0)
            data = noise + (signal_raw * snr)
            has_signal = 1
            
        # --- SZENARIO 3: Simulation/Fallback (20%) ---
        else:
            noise = generate_gaussian_noise(len(time), 1.0)
            if np.random.rand() > 0.5:
                # Auch hier: Zufällige Physik statt Hardcoding!
                t_merger = np.random.uniform(duration * 0.4, duration * 0.95)
                m1 = np.random.uniform(10, 80)
                m2 = np.random.uniform(10, 80)
                dist = np.random.uniform(200, 800)
                
                signal_raw, _ = generate_astrophysical_chirp(
                    time, 
                    t_merger=t_merger,
                    mass1=m1, 
                    mass2=m2, 
                    distance_mpc=dist,
                    normalize=True
                )
                
                snr = np.random.uniform(0.8, 2.5)
                data = noise + (signal_raw * snr)
                has_signal = 1
            else:
                data = noise
                has_signal = 0
        
        # Speichern
        filename = f"sample_{i:05d}.npy"
        filepath = os.path.join(output_folder, filename)
        np.save(filepath, data)
        
        # Metadaten erfassen (jetzt viel detaillierter!)
        labels.append({
            "filename": filename,
            "has_signal": has_signal,
            "merger_time": t_merger,
            "snr": snr,
            "mass1": m1,
            "mass2": m2,
            "distance": dist
        })
        
        if i % 500 == 0: print(f"   ... {i}/{num_samples} (Letztes Event: M1={m1:.1f}, Dist={dist:.0f})")

    df = pd.DataFrame(labels)
    df.to_csv(os.path.join(output_folder, "labels.csv"), index=False)
    print("✅ Fertig! Der Datensatz ist jetzt physikalisch divers und bereit fürs Training.")

if __name__ == "__main__":
    generate_dataset()