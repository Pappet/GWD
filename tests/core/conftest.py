import os

import numpy as np
import pytest

from gwd.core import waveforms

SEED = 20251115
BACKENDS = list(waveforms.BACKENDS)
PYCBC_INSTALLED = waveforms.pycbc_available()


@pytest.fixture
def rng():
    """A seeded numpy Generator; gwd.core draws only from the rng it is given (P5.3)."""
    return np.random.default_rng(SEED)


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
def backend(request):
    """The waveform backend a test passes explicitly; 'pycbc' skips without PyCBC.

    Since P5.2 the backend is an argument and nothing falls back, so a test on 'pycbc'
    runs PyCBC or fails with WaveformBackendError. Since P6.1 'pycbc' is the only one.
    """
    if request.param == "pycbc":
        require_pycbc()
    return request.param
