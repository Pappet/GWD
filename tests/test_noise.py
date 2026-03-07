import numpy as np
import pytest
from src.gwd_core.noise import generate_gaussian_noise, generate_colored_noise

def test_generate_gaussian_noise():
    noise = generate_gaussian_noise(1000, 1.0)
    assert len(noise) == 1000
    # Mean should be close to 0
    assert abs(np.mean(noise)) < 0.5
    # Std dev should be close to 1.0
    assert abs(np.std(noise) - 1.0) < 0.5

def test_generate_colored_noise():
    noise = generate_colored_noise(1000, 4096, 1.0)
    assert len(noise) == 1000
    assert abs(np.mean(noise)) < 0.5
