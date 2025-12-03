import os
import numpy as np
from gwpy.timeseries import TimeSeries

# Output Ordner für das echte Rauschen
OUTPUT_DIR = "gw_noise_background"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ein Zeitraum im O3b Run, wo definitiv keine Events waren
# GPS 1262304018 = Jan 2020
START_GPS = 1262304018
DURATION = 4096  # Wir laden ca. 68 Minuten Daten

def fetch_and_slice():
    """
    Lädt echte LIGO-Daten aus einem ruhigen Zeitraum und schneidet sie
    in 4-Sekunden-Schnipsel für das Training.
    """
    print(f"🌊 Lade {DURATION} Sekunden echtes LIGO-Rauschen (H1)...")
    
    try:
        # 1. Download am Stück
        strain = TimeSeries.fetch_open_data('H1', START_GPS, START_GPS + DURATION, verbose=True)
        
        # 2. Preprocessing (EXAKT wie im Training/App!)
        # Resample auf 4096 Hz
        if strain.sample_rate.value != 4096:
            strain = strain.resample(4096)
            
        # Notch Filter (Stromnetz entfernen)
        strain = strain.notch(60).notch(120).notch(180).notch(240)
        
        # Bandpass
        strain = strain.bandpass(20, 300)
        
        # Als Numpy Array holen
        data = strain.value
        
        # 3. Zerschneiden in 4s Stücke
        fs = 4096
        chunk_len = 4 * fs
        num_chunks = len(data) // chunk_len
        
        print(f"✂️ Schneide in {num_chunks} Trainings-Schnipsel...")
        
        for i in range(num_chunks):
            chunk = data[i*chunk_len : (i+1)*chunk_len]
            
            # Speichern
            filename = os.path.join(OUTPUT_DIR, f"real_noise_{i:04d}.npy")
            np.save(filename, chunk)
            
        print(f"✅ Fertig! {num_chunks} Dateien in '{OUTPUT_DIR}' gespeichert.")
        
    except Exception as e:
        print(f"❌ Fehler: {e}")

if __name__ == "__main__":
    fetch_and_slice()