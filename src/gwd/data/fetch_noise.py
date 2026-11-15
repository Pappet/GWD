"""Download real detector noise from GWOSC and cut it into training chunks.

gwd data fetch-noise runs fetch_and_slice(). Any error ends it with exit code 1 and a
message on stderr (DATA-6), and the menu reports that exit code.
"""

import os
import sys

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

# Output folder for the real noise chunks
OUTPUT_DIR = DATA_DIR_NOISE

# A stretch of the O3b run assumed to be free of events (not verified, DATA-25).
# GPS 1262304018 = January 2020
START_GPS = 1262304018


def fetch_and_slice(detector="H1", start_gps=START_GPS, duration=NOISE_FETCH_SECONDS):
    """Download duration seconds of detector data from start_gps and cut them into chunks.

    The chunks of N_SAMPLES samples go to OUTPUT_DIR. Returns the exit code, 1 on any
    error; a rejected download writes nothing.
    """
    print(f"🌊 Loading {duration} seconds of real LIGO noise ({detector})...")

    try:
        # 1. One download, checked for length and NaN
        strain = fetch_strain(detector, start_gps, start_gps + duration, verbose=True)

        # 2. Conditioning, the same chain as the field test and the Glitch Hunter
        data = condition_strain(strain, SAMPLE_RATE)

        # 3. Cut into chunks of N_SAMPLES (4 s)
        chunk_len = N_SAMPLES
        num_chunks = len(data) // chunk_len

        print(f"✂️ Cutting into {num_chunks} training chunks...")

        ensure_dirs(OUTPUT_DIR)
        for i in range(num_chunks):
            chunk = data[i * chunk_len : (i + 1) * chunk_len]

            filename = os.path.join(OUTPUT_DIR, f"real_noise_{i:04d}.npy")
            np.save(filename, chunk)

        print(f"✅ Done: {num_chunks} files saved in '{OUTPUT_DIR}'.")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit("The options are on the gwd command: gwd data fetch-noise --help")
    sys.exit(fetch_and_slice())
