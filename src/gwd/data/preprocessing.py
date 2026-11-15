import numpy as np

from gwd.config import BANDPASS, SAMPLE_RATE


def preprocess_real_data(strain, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """WICHTIG: Die Signalverarbeitung für echte LIGO-Daten."""
    if strain.sample_rate.value != sample_rate:
        strain = strain.resample(sample_rate)

    # Cleaning: Notch & Bandpass
    strain = strain.notch(60).notch(120).notch(180)
    white_data = strain.bandpass(*BANDPASS)

    return white_data.value
