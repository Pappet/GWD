import numpy as np

def preprocess_real_data(strain, sample_rate: int = 4096) -> np.ndarray:
    """WICHTIG: Die Signalverarbeitung für echte LIGO-Daten."""
    if strain.sample_rate.value != sample_rate:
        strain = strain.resample(sample_rate)
    
    # Cleaning: Notch & Bandpass
    strain = strain.notch(60).notch(120).notch(180)
    white_data = strain.bandpass(20, 300)
    
    return white_data.value
