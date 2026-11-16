"""Physics checks for gwd.core.waveforms on both backends, and the backend itself.

Only physical quantities are checked, never the length of the raw waveform, because that
depends on the pycbc version. Known bugs are strict xfails with their finding ID from
docs/aufraeumen/BEFUNDE.md: once a fix makes one pass, strict forces removing the mark.
Every call names its backend (P5.2); none falls back to another.
"""

import subprocess
import sys

import numpy as np
import pytest

from gwd.core import waveforms
from gwd.core.waveforms import (
    WaveformBackendError,
    generate_chirp,
    generate_realistic_chirp,
)

FS = 4096
DURATION = 4.0
T_MERGER = 3.6

G = 6.674e-11
C = 2.998e8
M_SUN = 1.989e30


def time_axis():
    return np.arange(int(FS * DURATION)) / FS


def zero_crossing_frequency(t, strain, t_eval):
    """Frequency from the spacing of the zero crossings of the strain, at t_eval."""
    sign = np.signbit(strain)
    i = np.nonzero(sign[1:] != sign[:-1])[0]
    t_zero = t[i] - strain[i] * (t[i + 1] - t[i]) / (strain[i + 1] - strain[i])
    freq = 0.5 / np.diff(t_zero)
    return np.interp(t_eval, 0.5 * (t_zero[1:] + t_zero[:-1]), freq)


def test_normalized_peak_is_one(backend):
    t = time_axis()
    strain, freq = generate_chirp(t, t_merger=T_MERGER, backend=backend, normalize=True)
    assert strain.shape == t.shape
    assert freq.shape == t.shape
    assert np.isclose(np.max(np.abs(strain)), 1.0)


@pytest.mark.parametrize("t_merger", [2.0, T_MERGER])
def test_peak_is_at_t_merger(backend, t_merger):
    t = time_axis()
    strain, _ = generate_chirp(t, t_merger=t_merger, backend=backend, normalize=False)
    t_peak = t[np.argmax(np.abs(strain))]
    assert abs(t_peak - t_merger) <= 1e-3


def test_frequency_rises_monotonically_until_the_merger(backend):
    t = time_axis()
    strain, freq = generate_chirp(
        t, t_merger=T_MERGER, backend=backend, normalize=False
    )
    amplitude = np.abs(strain)
    i_peak = np.argmax(amplitude)
    # Start where the signal is visible: PyCBC's conditioning taper at the very start
    # has a tiny amplitude and an ill-defined phase.
    i_start = np.argmax(amplitude >= 0.1 * amplitude[i_peak])
    inspiral = freq[i_start : i_peak + 1]
    assert i_peak - i_start > FS // 4  # at least 0.25 s of inspiral
    assert np.all(np.diff(inspiral) >= 0)
    assert inspiral[-1] > inspiral[0]


def test_strain_scales_with_inverse_distance(backend):
    t = time_axis()
    near, _ = generate_chirp(
        t, t_merger=T_MERGER, backend=backend, distance_mpc=400, normalize=False
    )
    far, _ = generate_chirp(
        t, t_merger=T_MERGER, backend=backend, distance_mpc=800, normalize=False
    )
    assert np.max(np.abs(near)) > 0
    np.testing.assert_allclose(far, near / 2, rtol=1e-6, atol=0)


@pytest.mark.parametrize(
    "backend",
    [
        "pycbc",
        pytest.param(
            "newton",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="CORE-2: Newton frequency is pi^(5/8) ~ 2.045 times too high",
            ),
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("tau", [1.0, 0.5, 0.3, 0.2])
def test_frequency_matches_the_zero_crossings(backend, tau):
    t = time_axis()
    strain, freq = generate_chirp(
        t, t_merger=T_MERGER, backend=backend, normalize=False
    )
    t_eval = T_MERGER - tau
    expected = zero_crossing_frequency(t, strain, t_eval)
    assert np.interp(t_eval, t, freq) == pytest.approx(expected, rel=0.05)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="CORE-3: Newton strain amplitude is 4 times too large",
)
def test_newton_amplitude_matches_pycbc(needs_pycbc):
    t = time_axis()
    tau = 0.3
    near_tau = np.abs(t - (T_MERGER - tau)) <= 0.02  # about one cycle at ~30 Hz

    envelope = {}
    for name in ["pycbc", "newton"]:
        strain, _ = generate_chirp(
            t, t_merger=T_MERGER, backend=name, inclination=0, normalize=False
        )
        envelope[name] = np.max(np.abs(strain[near_tau]))

    assert 0.8 <= envelope["newton"] / envelope["pycbc"] <= 1.2


@pytest.mark.parametrize("backend", ["newton"], indirect=True)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="CORE-8: the Newton inspiral has no ISCO cut and runs to ~1 kHz",
)
def test_newton_inspiral_stops_at_isco(backend):
    t = time_axis()
    mass1 = mass2 = 30
    strain, freq = generate_chirp(
        t,
        t_merger=T_MERGER,
        backend=backend,
        mass1=mass1,
        mass2=mass2,
        normalize=False,
    )
    total_mass = (mass1 + mass2) * M_SUN
    f_isco = C**3 / (6**1.5 * np.pi * G * total_mass)  # GW frequency, ~73 Hz here
    assert np.max(freq[strain != 0]) <= 1.05 * f_isco


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            "pycbc",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="CORE-1: generate_realistic_chirp cuts off the merger",
            ),
        ),
        "newton",
    ],
    indirect=True,
)
def test_realistic_chirp_contains_the_merger(backend):
    t = time_axis()
    strain, _ = generate_realistic_chirp(t, backend=backend)
    reference, _ = generate_chirp(t, backend=backend, f_lower=20.0, normalize=False)
    i_peak = np.argmax(np.abs(strain))
    assert 0 < i_peak < len(t) - 1
    assert np.max(np.abs(strain)) == pytest.approx(np.max(np.abs(reference)), rel=0.01)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="CORE-15: t_merger outside the window is accepted silently",
)
def test_t_merger_outside_the_window_raises(backend):
    try:
        generate_chirp(time_axis(), t_merger=10.0, backend=backend)
    except ValueError:
        return
    raise AssertionError("t_merger=10 s in a 4 s window gave no ValueError")


def test_importing_the_module_prints_nothing_and_imports_no_pycbc():
    """CORE-17: no print on import, and PyCBC is imported only when a chirp needs it."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, gwd.core.waveforms; print('pycbc' in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "False\n"


def test_the_backend_is_required():
    with pytest.raises(TypeError, match="backend"):
        generate_chirp(time_axis(), t_merger=T_MERGER)


def test_an_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="newton"):
        generate_chirp(time_axis(), t_merger=T_MERGER, backend="lal")


def test_an_unknown_approximant_raises_instead_of_falling_back(needs_pycbc):
    """CORE-4: before P5.2 any PyCBC error gave a Newton chirp with a warning."""
    with pytest.raises(WaveformBackendError, match="GibtEsNicht") as raised:
        generate_chirp(
            time_axis(), t_merger=T_MERGER, backend="pycbc", approximant="GibtEsNicht"
        )
    assert raised.value.__cause__ is not None


def test_a_pycbc_error_raises_with_the_parameters(needs_pycbc, monkeypatch):
    def failing(**kwargs):
        raise ValueError("mass out of range")

    monkeypatch.setattr(waveforms, "_get_td_waveform", lambda: failing)
    with pytest.raises(WaveformBackendError, match="mass1=30.*mass out of range"):
        generate_chirp(time_axis(), t_merger=T_MERGER, backend="pycbc")


def test_without_pycbc_the_pycbc_backend_raises_with_the_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "pycbc", None)
    with pytest.raises(WaveformBackendError) as raised:
        generate_chirp(time_axis(), t_merger=T_MERGER, backend="pycbc")
    message = str(raised.value)
    assert "pip install -e '.[physics]'" in message
    assert "Linux or in WSL" in message
    assert not waveforms.pycbc_available()


def test_without_pycbc_newton_still_works(monkeypatch):
    monkeypatch.setitem(sys.modules, "pycbc", None)
    strain, _ = generate_chirp(time_axis(), t_merger=T_MERGER, backend="newton")
    assert np.isclose(np.max(np.abs(strain)), 1.0)


def test_the_backend_label_names_newton():
    assert waveforms.backend_label("newton") == "newton"


def test_the_backend_label_names_the_pycbc_version(needs_pycbc):
    import importlib.metadata

    version = importlib.metadata.version("pycbc")
    assert waveforms.backend_label("pycbc") == f"pycbc {version}"
