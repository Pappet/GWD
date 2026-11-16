"""gwd.core.timebase.time_axis: spacing exactly 1/fs (CORE-19, DATA-19)."""

import numpy as np
import pytest

from gwd.core.timebase import time_axis


@pytest.mark.parametrize("fs, duration", [(4096, 4.0), (4096, 0.5), (1024, 2.0)])
def test_the_spacing_is_exactly_one_over_fs(fs, duration):
    t = time_axis(fs, duration)
    n = int(fs * duration)
    assert t.shape == (n,)
    assert t[0] == 0.0
    np.testing.assert_array_equal(t, np.arange(n) / fs)
    np.testing.assert_allclose(np.diff(t), 1 / fs, rtol=1e-9, atol=0)
    assert t[-1] == pytest.approx(duration - 1 / fs, rel=1e-12)


def test_linspace_would_have_sampled_at_4095_75_hz():
    """The old axis: np.linspace(0, 4, 16384) has a spacing of 4 / 16383 s."""
    old = np.linspace(0, 4.0, 16384)
    assert 1 / (old[1] - old[0]) == pytest.approx(4095.75)
    assert 1 / (time_axis(4096, 4.0)[1]) == pytest.approx(4096)


def test_t0_shifts_the_axis():
    t = time_axis(4096, 1.0, t0=1e9)
    assert t[0] == 1e9
    assert t.shape == (4096,)


def test_a_too_short_axis_is_refused():
    with pytest.raises(ValueError, match="at least 2 samples"):
        time_axis(4096, 1e-4)
