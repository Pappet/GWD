"""The time axis of every sample (P5.3, CORE-19, DATA-19)."""

import numpy as np


def time_axis(fs, duration, t0=0.0):
    """n = round(fs * duration) times t0 + k / fs, k = 0 .. n - 1.

    The spacing is exactly 1 / fs and the last time is t0 + duration - 1 / fs. Before
    P5.3 the code used np.linspace from 0 to duration with n points, whose spacing
    duration / (n - 1) is an effective 4095.75 Hz instead of 4096 Hz for 4 s.
    """
    n = int(round(fs * duration))
    if n < 2:
        raise ValueError(
            f"A time axis needs at least 2 samples, got {n} for fs={fs}, "
            f"duration={duration}"
        )
    return t0 + np.arange(n) / fs
