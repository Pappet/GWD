"""Gravitational waveforms of binary black hole mergers.

The chirps come from PyCBC's get_td_waveform, the one backend 'pycbc'. The caller still
names it (P5.2), and the pipeline passes gwd.config.WAVEFORM_BACKEND, so the backend is
recorded with every dataset. PyCBC is imported only when a chirp is drawn, and a
missing PyCBC or a PyCBC error raises WaveformBackendError; nothing falls back. Until
P6.1 a second, numpy-only backend drew a leading-order inspiral approximation with
known errors (CORE-2, CORE-3, CORE-8); D1 = A deleted it, and the Git history keeps it.

gwd.core does not read gwd.config, so there is no default backend here. Importing
this module imports no PyCBC and prints nothing.
"""

import importlib.metadata

import numpy as np

BACKENDS = ("pycbc",)

# Where the time axis cuts a PyCBC waveform, the cut edge is faded with the cosine half
# of a Tukey window of this duration, so the cut does not click (CORE-14).
TAPER_SECONDS = 0.05

PYCBC_INSTALL_HINT = (
    "PyCBC is a dependency of gwd except on Windows; install gwd with it:\n"
    "    pip install -e .\n"
    "PyCBC is not available on native Windows; use WSL there (see README.md)."
)


class WaveformBackendError(RuntimeError):
    """The waveform backend cannot draw the chirp: PyCBC is missing or failed."""


def _get_td_waveform():
    """PyCBC's get_td_waveform, imported on first use."""
    try:
        # pycbc first: sys.modules["pycbc"] = None then blocks it even when
        # pycbc.waveform is already imported.
        import pycbc  # noqa: F401
        from pycbc.waveform import get_td_waveform
    except ImportError as error:
        raise WaveformBackendError(
            f"The waveform backend 'pycbc' needs PyCBC, which cannot be imported "
            f"({error}).\n{PYCBC_INSTALL_HINT}"
        ) from error
    return get_td_waveform


def check_backend(backend):
    """Raise unless backend can draw chirps here.

    An unknown name raises ValueError; 'pycbc' without a working PyCBC raises
    WaveformBackendError with the install hint. Call it before work that would fail
    later, for example before an output folder is cleared.
    """
    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown waveform backend {backend!r}; choose one of {', '.join(BACKENDS)}"
        )
    _get_td_waveform()


def pycbc_available():
    """True if PyCBC can be imported; gwd env reports it."""
    try:
        check_backend("pycbc")
    except WaveformBackendError:
        return False
    return True


def backend_label(backend):
    """The backend with its version, as recorded: 'pycbc <version>'."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown waveform backend {backend!r}")
    try:
        version = importlib.metadata.version("pycbc")
    except importlib.metadata.PackageNotFoundError:
        version = "not installed"
    return f"pycbc {version}"


def generate_chirp(
    time_array,
    t_merger=None,
    *,
    backend,
    mass1=30,
    mass2=30,
    distance_mpc=400,
    f_lower=25.0,
    normalize=True,
    spin1z=0.0,
    spin2z=0.0,
    inclination=0,
    approximant="IMRPhenomD",
):
    """
    Generate the chirp of a binary black hole merger on time_array.

    Args:
        time_array: Time axis of the simulation, equally spaced, for example
                    gwd.core.timebase.time_axis(4096, 4.0).
        t_merger: Merger time in seconds, on the scale of time_array; it must lie
                  in [time_array[0], time_array[-1]]. If None, 90% of the span
                  after time_array[0].
        backend: 'pycbc', required. The pipeline passes gwd.config.WAVEFORM_BACKEND.
        mass1, mass2: Black hole masses in solar masses.
        distance_mpc: Distance in megaparsec (sets the physical amplitude).
        f_lower: Lower frequency limit of the waveform (Hz).
        normalize: If True (for ML), the peak amplitude is scaled to 1.0.
                   If False (for the physics simulator), the strain is returned.
        spin1z, spin2z: Spins along the line of sight (-1 to +1).
        inclination: Inclination (0 = face-on, π/2 = edge-on).
        approximant: Waveform approximant.

    Returns:
        tuple: (strain, frequency)
            - strain: np.array with the gravitational wave amplitude, 0 where the
              chirp has no signal
            - frequency: np.array with the instantaneous frequency in Hz, NaN where
              the chirp has no signal (before P5.3: 0, CORE-14). It is taken from
              the phase of h_plus - i h_cross, so it is only meaningful
              where h_cross is not zero, that is not for an edge-on inclination of
              pi/2. The pipeline draws face-on chirps (inclination 0).

    The waveform is shifted so its peak lies at t_merger. Where time_array
    cuts it off, at the start or the end, the cut edge is faded over TAPER_SECONDS; a
    merger within TAPER_SECONDS of such an edge is damped by it.

    Raises:
        ValueError: backend is not 'pycbc', or t_merger lies outside
            time_array. Before P5.3 it was moved to the edge silently (CORE-15).
        WaveformBackendError: PyCBC is missing or fails for these
            parameters (for example an unknown approximant).
    """
    check_backend(backend)

    t0, t_end = time_array[0], time_array[-1]
    if t_merger is None:
        t_merger = t0 + 0.9 * (t_end - t0)
    elif not t0 <= t_merger <= t_end:
        raise ValueError(
            f"t_merger={t_merger} s lies outside the time axis [{t0}, {t_end}] s"
        )

    return _pycbc_chirp(
        time_array,
        t_merger,
        mass1=mass1,
        mass2=mass2,
        distance_mpc=distance_mpc,
        f_lower=f_lower,
        normalize=normalize,
        spin1z=spin1z,
        spin2z=spin2z,
        inclination=inclination,
        approximant=approximant,
    )


def _pycbc_chirp(
    time_array,
    t_merger,
    mass1,
    mass2,
    distance_mpc,
    f_lower,
    normalize,
    spin1z,
    spin2z,
    inclination,
    approximant,
):
    get_td_waveform = _get_td_waveform()
    dt = time_array[1] - time_array[0]

    # Plus and cross polarization
    try:
        hp, hc = get_td_waveform(
            approximant=approximant,
            mass1=mass1,
            mass2=mass2,
            spin1z=spin1z,
            spin2z=spin2z,
            inclination=inclination,
            delta_t=dt,
            f_lower=f_lower,
            distance=distance_mpc,
        )
    except Exception as error:
        raise WaveformBackendError(
            f"PyCBC failed for approximant={approximant!r}, mass1={mass1}, "
            f"mass2={mass2}, f_lower={f_lower}: {error}"
        ) from error

    # Instantaneous frequency from the phase of the complex signal
    # h(t) = h_plus - i * h_cross
    complex_strain = hp.numpy() - 1j * hc.numpy()
    phase = np.unwrap(np.angle(complex_strain))

    # The frequency is the time derivative of the phase / 2π
    raw_freq = np.abs(np.gradient(phase, dt) / (2 * np.pi))

    waveform = hp.numpy()

    # Map onto time_array, shifted so the peak lies at t_merger
    # 1. Find the peak
    peak_idx = np.argmax(np.abs(waveform))

    # 2. Find the target index in time_array
    target_idx = np.argmin(np.abs(time_array - t_merger))

    # 3. Compute the shift
    shift = target_idx - peak_idx

    # Empty arrays for the result: no strain, and no frequency, outside the signal
    strain_out = np.zeros_like(time_array, dtype=float)
    freq_out = np.full_like(time_array, np.nan, dtype=float)

    # Copy the overlapping part
    start_src = max(0, -shift)
    end_src = min(len(waveform), len(time_array) - shift)

    start_dst = max(0, shift)
    end_dst = start_dst + (end_src - start_src)

    if end_dst > start_dst:
        strain_out[start_dst:end_dst] = waveform[start_src:end_src]
        freq_out[start_dst:end_dst] = raw_freq[start_src:end_src]

        # Fade the edges that time_array cuts off, over TAPER_SECONDS each. Where the
        # waveform starts inside the window, PyCBC's own taper is kept. PyCBC returns
        # about 5 s after the peak, so the end is nearly always cut, long after the
        # ringdown has decayed.
        ramp = _tukey_ramp(min(int(round(TAPER_SECONDS / dt)), end_dst - start_dst))
        if start_src > 0:
            strain_out[start_dst : start_dst + len(ramp)] *= ramp
        if end_src < len(waveform):
            strain_out[end_dst - len(ramp) : end_dst] *= ramp[::-1]

    # Normalization for machine learning
    if normalize:
        m = np.max(np.abs(strain_out))
        if m > 0:
            strain_out /= m

    return strain_out, freq_out


def _tukey_ramp(n):
    """The rising cosine half of a Tukey window, n samples from 0 towards 1."""
    if n <= 0:
        return np.ones(0)
    return 0.5 * (1 - np.cos(np.pi * np.arange(n) / n))
