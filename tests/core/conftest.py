import os

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


def _no_silent_fallback(*args, **kwargs):
    message = " ".join(str(arg) for arg in args)
    # pytest.fail raises Failed, not AssertionError, so xfail(raises=AssertionError) cannot hide it.
    pytest.fail(f"PyCBC backend fell back to Newton: {message}")


@pytest.fixture
def needs_pycbc():
    require_pycbc()


@pytest.fixture(params=BACKENDS)
def backend(request, monkeypatch):
    """Select the waveform backend through waveforms.HAS_PYCBC.

    On 'pycbc', the silent fallback to Newton (CORE-4) fails the test: its only trace
    is a print, so the fixture replaces print in the waveforms module.
    """
    if request.param == "pycbc":
        require_pycbc()
        monkeypatch.setattr(waveforms, "HAS_PYCBC", True)
        monkeypatch.setattr(waveforms, "print", _no_silent_fallback, raising=False)
    else:
        monkeypatch.setattr(waveforms, "HAS_PYCBC", False)
    return request.param
