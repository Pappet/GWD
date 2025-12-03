import pandas as pd
import os
import numpy as np
import sys
import glob
import random

# Import-Pfad anpassen (falls nötig)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importiere die Wellenform- und Rausch-Funktionen
from gwd_core.waveforms import (
    generate_astrophysical_chirp, 
    generate_realistic_chirp, 
    generate_random_bbh_parameters
)
from gwd_core.noise import generate_gaussian_noise

REAL_NOISE_DIR = "gw_noise_background"


def calculate_snr(signal, noise):
    """
    Berechnet das Signal-to-Noise Ratio (SNR) eines Signals in Rauschen.
    
    SNR = sqrt(sum(signal^2) / sum(noise^2))
    
    Args:
        signal: Das reine Signal (ohne Rauschen)
        noise: Das Rauschen (ohne Signal)
    
    Returns:
        float: Der SNR-Wert
    """
    signal_power = np.sum(signal**2)
    noise_power = np.sum(noise**2)
    
    if noise_power == 0:
        return 0.0
    
    snr = np.sqrt(signal_power / noise_power)
    return snr


def calculate_snr_whitened(signal, noise, psd=None):
    """
    Berechnet SNR mit Frequency-Domain Whitening (wie LIGO es macht).
    
    Args:
        signal: Das reine Signal
        noise: Das Rauschen
        psd: Power Spectral Density (optional, sonst aus noise geschätzt)
    
    Returns:
        float: Der optimal gefilterte SNR
    """
    # Fourier Transform
    signal_fft = np.fft.rfft(signal)
    noise_fft = np.fft.rfft(noise)
    
    # PSD schätzen (oder vorgeben)
    if psd is None:
        psd = np.abs(noise_fft)**2
        psd[psd == 0] = 1e-40  # Verhindere Division durch 0
    
    # Matched Filter SNR (optimal)
    snr_squared = np.sum(np.abs(signal_fft)**2 / psd)
    snr = np.sqrt(snr_squared.real)
    
    return snr


def load_random_real_noise(length):
    """
    Lädt echtes Rauschen und NORMALISIERT es sofort.
    
    Args:
        length: Erwartete Länge des Rauschens
    
    Returns:
        np.array: Normalisiertes Rauschen (mean=0, std=1)
    """
    files = glob.glob(os.path.join(REAL_NOISE_DIR, "*.npy"))
    if not files:
        return generate_gaussian_noise(length, 1.0)
    
    f = random.choice(files)
    try:
        data = np.load(f)
        if len(data) != length:
            return generate_gaussian_noise(length, 1.0)
        
        # Rauschen auf Standardabweichung 1.0 zwingen für korrekte SNR-Berechnung
        if np.std(data) > 0:
            data = (data - np.mean(data)) / np.std(data)
            
        return data
    except:
        return generate_gaussian_noise(length, 1.0)


def generate_dataset(num_samples=20000, output_folder="gw_training_data"):
    """
    Generiert einen diversen Trainingsdatensatz mit verschiedenen Szenarien:
    - 40% nur Rauschen
    - 40% Signal in echtem Rauschen (physikalisch diverse Parameter)
    - 20% Simuliertes Rauschen mit/ohne Signal
    
    WICHTIG: SNR-Bereiche sind jetzt realistisch (5-25) für besseres Training!
    
    Args:
        num_samples: Anzahl zu generierender Samples
        output_folder: Zielordner für die Daten
    """
    print(f"🚀 Starte INTELLIGENTE Daten-Fabrik (Diverse Physik + Realistischer SNR). Ziel: {num_samples}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    fs = 4096 
    duration = 4.0
    time = np.linspace(0, duration, int(fs * duration))
    labels = []
    
    has_real_noise = len(glob.glob(os.path.join(REAL_NOISE_DIR, "*.npy"))) > 0
    
    for i in range(num_samples):
        # Default-Werte für Metadaten
        has_signal = 0
        t_merger = 0.0
        snr = 0.0
        m1 = 0.0
        m2 = 0.0
        dist = 0.0
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
            params = generate_random_bbh_parameters()
            m1 = params['mass1']
            m2 = params['mass2']
            dist = params['distance']
            t_merger = np.random.uniform(duration * 0.4, duration * 0.95)
            
            # Signal generieren (nutzt PyCBC wenn verfügbar)
            signal_raw, _ = generate_realistic_chirp(
                time, 
                mass1=m1,
                mass2=m2,
                distance=dist,
                spin1z=params['spin1z'],
                spin2z=params['spin2z'],
                inclination=params['inclination']
            )
            
            # ⭐ NEU: Signal auf realistischen SNR skalieren (5-25)
            # LIGO detektiert typischerweise Events mit SNR 8-50
            # Für Training: 5-25 ist ein guter Bereich (nicht zu leicht, nicht unmöglich)
            target_snr = np.random.uniform(5, 25)
            
            # Normalisiere Signal auf Amplitude 1.0
            if np.max(np.abs(signal_raw)) > 0:
                signal_normalized = signal_raw / np.max(np.abs(signal_raw))
            else:
                signal_normalized = signal_raw
            
            # Skaliere auf gewünschten SNR
            signal_scaled = signal_normalized * target_snr
            
            # Beobachtete Daten = Signal + Rauschen
            data = noise + signal_scaled
            
            # SNR neu berechnen zur Verifikation
            snr = calculate_snr(signal_scaled, noise)
            has_signal = 1
            
        # --- SZENARIO 3: Simulation/Fallback (20%) ---
        else:
            noise = generate_gaussian_noise(len(time), 1.0)
            
            if np.random.rand() > 0.5:
                # Signal mit zufälliger Physik
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
                
                # ⭐ NEU: Realistischer SNR-Bereich (5-25)
                target_snr = np.random.uniform(5, 25)
                
                # Signal skalieren, um gewünschten SNR zu erreichen
                signal_scaled = signal_raw * target_snr
                snr = calculate_snr(signal_scaled, noise)
                
                data = noise + signal_scaled
                has_signal = 1
            else:
                # Nur Rauschen
                data = noise
                has_signal = 0
        
        # Speichern
        filename = f"sample_{i:05d}.npy"
        filepath = os.path.join(output_folder, filename)
        np.save(filepath, data)
        
        # Metadaten erfassen
        labels.append({
            "filename": filename,
            "has_signal": has_signal,
            "merger_time": float(t_merger),
            "snr": float(snr),
            "mass1": float(m1),
            "mass2": float(m2),
            "distance": float(dist)
        })
        
        if i % 500 == 0:
            print(f"   ... {i}/{num_samples} (Letztes Event: M1={m1:.1f}, M2={m2:.1f}, Dist={dist:.0f}, SNR={snr:.2f})")

    # Labels als CSV speichern
    df = pd.DataFrame(labels)
    df.to_csv(os.path.join(output_folder, "labels.csv"), index=False)
    
    # Statistik ausgeben
    signal_samples = df[df['has_signal'] == 1]
    print(f"\n✅ Fertig! Datensatz-Statistik:")
    print(f"   - Gesamt: {len(df)} Samples")
    print(f"   - Mit Signal: {len(signal_samples)} ({len(signal_samples)/len(df)*100:.1f}%)")
    print(f"   - Nur Rauschen: {len(df) - len(signal_samples)} ({(len(df)-len(signal_samples))/len(df)*100:.1f}%)")
    if len(signal_samples) > 0:
        print(f"\n   📊 SNR Statistik (Signal-Samples):")
        print(f"      - Bereich: {signal_samples['snr'].min():.2f} - {signal_samples['snr'].max():.2f}")
        print(f"      - Durchschnitt: {signal_samples['snr'].mean():.2f}")
        print(f"      - Median: {signal_samples['snr'].median():.2f}")
        print(f"      - Samples mit SNR > 8: {(signal_samples['snr'] > 8).sum()} ({(signal_samples['snr'] > 8).sum()/len(signal_samples)*100:.1f}%)")

if __name__ == "__main__":
    generate_dataset()