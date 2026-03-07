import numpy as np
import pytest
from src.gwd_core.waveforms import generate_astrophysical_chirp

def test_generate_astrophysical_chirp_normalized():
    time_array = np.linspace(0, 1, 1000)
    strain, freq = generate_astrophysical_chirp(time_array, t_merger=0.8, normalize=True)
    assert len(strain) == 1000
    assert len(freq) == 1000
    # Peak amplitude should be 1.0
    assert np.isclose(np.max(np.abs(strain)), 1.0)

def test_generate_astrophysical_chirp_unnormalized():
    time_array = np.linspace(0, 1, 1000)
    strain, freq = generate_astrophysical_chirp(time_array, t_merger=0.8, normalize=False)
    assert len(strain) == 1000
    assert len(freq) == 1000
    # Unnormalized strain is typically very small or depends on PYCBC vs Newton
    assert np.max(np.abs(strain)) > 0
