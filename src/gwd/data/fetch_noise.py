"""Download real detector noise from GWOSC and cut it into training chunks.

gwd data fetch-noise runs fetch_and_slice(). Any error ends it with exit code 1 and a
message on stderr (DATA-6), and the menu reports that exit code.

The segment is conditioned in one piece, so the filters ring only at its two ends. Since
P4.3 the first and the last chunk, which hold that ringing, are not written
(evaluation:MISS-4); every file keeps the position of its chunk in the segment in its
name. Files already in the folder are never deleted.
"""

import glob
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


def edge_free_chunks(num_chunks):
    """The chunk positions that are written: all but the first and the last."""
    return range(1, num_chunks - 1)


def fetch_and_slice(detector="H1", start_gps=START_GPS, duration=NOISE_FETCH_SECONDS):
    """Download duration seconds of detector data from start_gps and cut them into chunks.

    The chunks of N_SAMPLES samples go to OUTPUT_DIR, without the first and the last
    one. Returns the exit code, 1 on any error; a rejected download writes nothing.
    """
    print(f"🌊 Loading {duration} seconds of real LIGO noise ({detector})...")

    try:
        # 1. One download, checked for length and NaN
        strain = fetch_strain(detector, start_gps, start_gps + duration, verbose=True)

        # 2. Conditioning, the same chain as the field test and the Glitch Hunter
        data = condition_strain(strain, SAMPLE_RATE)

        # 3. Cut into chunks of N_SAMPLES (4 s); drop the two edge chunks
        chunk_len = N_SAMPLES
        num_chunks = len(data) // chunk_len
        kept = edge_free_chunks(num_chunks)
        if not kept:
            raise ValueError(
                f"{num_chunks} chunks leave none after dropping the first and the last"
            )

        print(
            f"✂️ Cutting into {num_chunks} chunks, writing {len(kept)} "
            "(the first and the last hold filter edge transients)..."
        )

        ensure_dirs(OUTPUT_DIR)
        written = set()
        for i in kept:
            chunk = data[i * chunk_len : (i + 1) * chunk_len]

            filename = os.path.join(OUTPUT_DIR, f"real_noise_{i:04d}.npy")
            np.save(filename, chunk)
            written.add(filename)

        print(f"✅ Done: {len(kept)} files saved in '{OUTPUT_DIR}'.")
        others = sorted(set(glob.glob(os.path.join(OUTPUT_DIR, "*.npy"))) - written)
        if others:
            names = ", ".join(os.path.basename(f) for f in others[:5])
            more = f" and {len(others) - 5} more" if len(others) > 5 else ""
            print(
                f"⚠️ {len(others)} older files in '{OUTPUT_DIR}' are not from this "
                f"download and were kept: {names}{more}. gwd data generate still "
                "uses them; remove them yourself if they should not be used.",
                file=sys.stderr,
            )

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit("The options are on the gwd command: gwd data fetch-noise --help")
    sys.exit(fetch_and_slice())
