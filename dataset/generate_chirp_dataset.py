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


def scale_signal_to_snr(signal, noise, target_snr):
    """
    Skaliert ein Signal, um einen bestimmten SNR zu erreichen.
    
    Diese Funktion ist KRITISCH für die korrekte SNR-Kontrolle!
    
    Args:
        signal: Das reine Signal (normalisiert oder nicht)
        noise: Das Rauschen
        target_snr: Gewünschter SNR-Wert
    
    Returns:
        tuple: (skaliertes_signal, tatsächlicher_snr)
    """
    # Berechne aktuellen SNR
    current_snr = calculate_snr(signal, noise)
    
    if current_snr == 0:
        return signal, 0.0
    
    # Berechne Skalierungsfaktor
    scale_factor = target_snr / current_snr
    
    # Skaliere Signal
    scaled_signal = signal * scale_factor
    
    # Verifiziere finalen SNR
    final_snr = calculate_snr(scaled_signal, noise)
    
    return scaled_signal, final_snr


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


def generate_dataset(num_samples=10000, output_folder="gw_training_data"):
    """
    Generiert einen diversen Trainingsdatensatz mit verschiedenen Szenarien:
    - 40% nur Rauschen
    - 40% Signal in echtem Rauschen (physikalisch diverse Parameter)
    - 20% Simuliertes Rauschen mit/ohne Signal
    
    WICHTIG: SNR-Bereiche sind jetzt realistisch (5-25) für besseres Training!
    Die SNR-Skalierung ist jetzt KORREKT implementiert.
    
    Args:
        num_samples: Anzahl zu generierender Samples
        output_folder: Zielordner für die Daten
    """
    print(f"🚀 Starte INTELLIGENTE Daten-Fabrik (Diverse Physik + KORREKTER SNR). Ziel: {num_samples}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    fs = 4096 
    duration = 4.0
    time = np.linspace(0, duration, int(fs * duration))
    labels = []
    
    has_real_noise = len(glob.glob(os.path.join(REAL_NOISE_DIR, "*.npy"))) > 0
    
    # Zähler für Debugging
    snr_too_low_count = 0
    
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
            
            # Signal generieren (nutzt PyCBC wenn verfügbar, sonst Newton)
            signal_raw, _ = generate_realistic_chirp(
                time, 
                mass1=m1,
                mass2=m2,
                distance=dist,
                spin1z=params['spin1z'],
                spin2z=params['spin2z'],
                inclination=params['inclination']
            )
            
            # ⭐ KORRIGIERTE SNR-SKALIERUNG
            # Ziel-SNR zwischen 5 und 25 (realistisch für LIGO)
            target_snr = np.random.uniform(5, 25)
            
            # Nutze die neue scale_signal_to_snr Funktion
            signal_scaled, actual_snr = scale_signal_to_snr(signal_raw, noise, target_snr)
            
            # Tracking für zu niedrige SNRs
            if actual_snr < 5:
                snr_too_low_count += 1
            
            # Beobachtete Daten = Signal + Rauschen
            data = noise + signal_scaled
            snr = actual_snr
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
                
                # ⭐ KORRIGIERTE SNR-SKALIERUNG
                target_snr = np.random.uniform(5, 25)
                signal_scaled, actual_snr = scale_signal_to_snr(signal_raw, noise, target_snr)
                
                if actual_snr < 5:
                    snr_too_low_count += 1
                
                data = noise + signal_scaled
                snr = actual_snr
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
        
        if i % 500 == 0 and has_signal == 1:
            print(f"   ... {i}/{num_samples} | Signal: M1={m1:.1f}☉, M2={m2:.1f}☉, D={dist:.0f}Mpc, SNR={snr:.2f}")
        elif i % 500 == 0:
            print(f"   ... {i}/{num_samples} | Nur Rauschen")

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
        print(f"      - Minimum: {signal_samples['snr'].min():.2f}")
        print(f"      - Maximum: {signal_samples['snr'].max():.2f}")
        print(f"      - Durchschnitt: {signal_samples['snr'].mean():.2f}")
        print(f"      - Median: {signal_samples['snr'].median():.2f}")
        print(f"      - Standardabweichung: {signal_samples['snr'].std():.2f}")
        print(f"\n      SNR Kategorien:")
        print(f"      - SNR < 5:    {(signal_samples['snr'] < 5).sum():>5} ({(signal_samples['snr'] < 5).sum()/len(signal_samples)*100:>5.1f}%)")
        print(f"      - SNR 5-10:   {((signal_samples['snr'] >= 5) & (signal_samples['snr'] < 10)).sum():>5} ({((signal_samples['snr'] >= 5) & (signal_samples['snr'] < 10)).sum()/len(signal_samples)*100:>5.1f}%)")
        print(f"      - SNR 10-15:  {((signal_samples['snr'] >= 10) & (signal_samples['snr'] < 15)).sum():>5} ({((signal_samples['snr'] >= 10) & (signal_samples['snr'] < 15)).sum()/len(signal_samples)*100:>5.1f}%)")
        print(f"      - SNR 15-20:  {((signal_samples['snr'] >= 15) & (signal_samples['snr'] < 20)).sum():>5} ({((signal_samples['snr'] >= 15) & (signal_samples['snr'] < 20)).sum()/len(signal_samples)*100:>5.1f}%)")
        print(f"      - SNR > 20:   {(signal_samples['snr'] >= 20).sum():>5} ({(signal_samples['snr'] >= 20).sum()/len(signal_samples)*100:>5.1f}%)")
        print(f"      - SNR > 8:    {(signal_samples['snr'] > 8).sum():>5} ({(signal_samples['snr'] > 8).sum()/len(signal_samples)*100:>5.1f}%)")
        
        # Qualitäts-Check
        if signal_samples['snr'].median() >= 12:
            print(f"\n      ✅ EXCELLENT: Median SNR = {signal_samples['snr'].median():.2f} - Optimal für Training!")
        elif signal_samples['snr'].median() >= 8:
            print(f"\n      ✅ GUT: Median SNR = {signal_samples['snr'].median():.2f} - Gut für Training")
        elif signal_samples['snr'].median() >= 5:
            print(f"\n      ⚠️  OK: Median SNR = {signal_samples['snr'].median():.2f} - Akzeptabel")
        else:
            print(f"\n      ❌ PROBLEM: Median SNR = {signal_samples['snr'].median():.2f} - Zu niedrig!")
        
        if snr_too_low_count > 0:
            print(f"\n      ⚠️  Warnung: {snr_too_low_count} Signale hatten SNR < 5 nach Skalierung")
            print(f"         (Das kann bei sehr schwachen Wellenformen passieren)")

if __name__ == "__main__":
    generate_dataset()