import os

import numpy as np
import pandas as pd

from gwd.config import (
    DATA_DIR_TRAIN,
    LABELS_FILE,
    SAMPLES_FOR_TRAINING,
    ensure_dirs,
)
from gwd.data.injection import make_sample, real_noise_provider
from gwd.data.scenarios import TRAIN_V1


def generate_dataset(
    num_samples=SAMPLES_FOR_TRAINING,
    output_folder=DATA_DIR_TRAIN,
    scenario=TRAIN_V1,
    rng=None,
):
    """Write num_samples samples of scenario and their labels.csv to output_folder.

    Even indices are signal samples, odd ones noise only (strictly 50/50). The noise
    is real where scenario and the chunks in DATA_DIR_NOISE allow it. rng defaults to
    the global numpy RNG (see gwd.data.injection).
    """
    print(
        f"🚀 Generating data ({scenario.name}, {scenario.snr_kind} SNR). Target: {num_samples}"
    )

    ensure_dirs(output_folder)

    noise_provider = real_noise_provider()
    labels = []

    for i in range(num_samples):
        sample = make_sample(
            scenario, rng, has_signal=(i % 2 == 0), noise_provider=noise_provider
        )
        params = sample.params

        # Speichern
        filename = f"sample_{i:05d}.npy"
        filepath = os.path.join(output_folder, filename)
        np.save(filepath, sample.data)

        labels.append(
            {
                "filename": filename,
                "has_signal": int(sample.has_signal),
                "merger_time": float(params.t_merger) if params else 0.0,
                "snr": float(sample.snr),
                "mass1": float(params.mass1) if params else 0.0,
                "mass2": float(params.mass2) if params else 0.0,
                "distance": float(params.distance) if params else 0.0,
            }
        )

        if i % 1000 == 0:
            print(f"   ... {i}/{num_samples}")

    df = pd.DataFrame(labels)
    df.to_csv(os.path.join(output_folder, LABELS_FILE), index=False)
    print("✅ Fertig.")


if __name__ == "__main__":
    generate_dataset()
