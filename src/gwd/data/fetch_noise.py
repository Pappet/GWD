import os

import numpy as np

from gwd.config import (
    DATA_DIR_NOISE,
    N_SAMPLES,
    NOISE_FETCH_SECONDS,
    SAMPLE_RATE,
    ensure_dirs,
)
from gwd.data.gwosc import fetch_strain
from gwd.data.preprocessing import condition_strain

# Output Ordner für das echte Rauschen
OUTPUT_DIR = DATA_DIR_NOISE

# Ein Zeitraum im O3b Run, wo definitiv keine Events waren
# GPS 1262304018 = Jan 2020
START_GPS = 1262304018


def fetch_and_slice():
    """
    Lädt echte LIGO-Daten aus einem ruhigen Zeitraum und schneidet sie
    in 4-Sekunden-Schnipsel für das Training.
    """
    print(f"🌊 Loading {NOISE_FETCH_SECONDS} seconds of real LIGO noise (H1)...")

    try:
        # 1. One download, checked for length and NaN
        strain = fetch_strain(
            "H1", START_GPS, START_GPS + NOISE_FETCH_SECONDS, verbose=True
        )

        # 2. Conditioning, the same chain as the field test and the Glitch Hunter
        data = condition_strain(strain, SAMPLE_RATE)

        # 3. Zerschneiden in 4s Stücke
        chunk_len = N_SAMPLES
        num_chunks = len(data) // chunk_len

        print(f"✂️ Schneide in {num_chunks} Trainings-Schnipsel...")

        ensure_dirs(OUTPUT_DIR)
        for i in range(num_chunks):
            chunk = data[i * chunk_len : (i + 1) * chunk_len]

            # Speichern
            filename = os.path.join(OUTPUT_DIR, f"real_noise_{i:04d}.npy")
            np.save(filename, chunk)

        print(f"✅ Fertig! {num_chunks} Dateien in '{OUTPUT_DIR}' gespeichert.")

    except Exception as e:
        print(f"❌ Fehler: {e}")


if __name__ == "__main__":
    fetch_and_slice()
