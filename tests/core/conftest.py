import os
import warnings

import numpy as np
import pytest

from gwd.core import waveforms

SEED = 20251115
BACKENDS = ["pycbc", "newton"]
PYCBC_INSTALLED = waveforms.HAS_PYCBC  # Captured before any test patches it.


@pytest.fixture(autouse=True)
def fixed_seed():
    """Seed the global numpy RNG, which gwd.core still uses, and restore it afterwards."""
    state = np.random.get_state()
    np.random.seed(SEED)
    yield
    np.random.set_state(state)


def require_pycbc():
    if PYCBC_INSTALLED:
        return
    if os.environ.get("GWD_REQUIRE_PYCBC") == "1":
        pytest.fail("pycbc missing, but GWD_REQUIRE_PYCBC=1 requires it")
    pytest.skip("pycbc missing")


@pytest.fixture
def needs_pycbc():
    require_pycbc()


@pytest.fixture(params=BACKENDS)
def backend(request, monkeypatch):
    """Select the waveform backend through waveforms.HAS_PYCBC.

    On 'pycbc', a fallback to Newton (CORE-4) fails the test: its WaveformFallbackWarning
    becomes an error. The error is not an AssertionError, so xfail(raises=AssertionError)
    cannot hide it.
    """
    with warnings.catch_warnings():
        if request.param == "pycbc":
            require_pycbc()
            monkeypatch.setattr(waveforms, "HAS_PYCBC", True)
            warnings.simplefilter("error", waveforms.WaveformFallbackWarning)
        else:
            monkeypatch.setattr(waveforms, "HAS_PYCBC", False)
        yield request.param
