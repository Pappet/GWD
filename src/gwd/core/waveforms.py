"""Gravitational waveforms of binary black hole mergers.

Two backends draw the chirps, and the caller always names one (P5.2):

- 'pycbc': PyCBC's get_td_waveform. The pipeline uses only this backend
  (gwd.config.WAVEFORM_BACKEND). PyCBC is imported only when a chirp is drawn, and a
  missing PyCBC or a PyCBC error raises WaveformBackendError; nothing falls back.
- 'newton': a Newtonian approximation of the inspiral with known errors (CORE-2,
  CORE-3, CORE-8). Only the demo apps choose it, and only when PyCBC is missing; they
  say so in their window. P6.1 deletes or corrects it (D1).

gwd.core does not read gwd.config, so there is no default backend here. Importing
this module imports no PyCBC and prints nothing.
"""

import importlib.metadata

import numpy as np

from .constants import M_SUN, MPC, C, G

BACKENDS = ("pycbc", "newton")

PYCBC_INSTALL_HINT = (
    "Install PyCBC on Linux or in WSL with the physics extra:\n"
    "    pip install -e '.[physics]'\n"
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
    if backend == "pycbc":
        _get_td_waveform()


def pycbc_available():
    """True if PyCBC can be imported. The demo apps use it to choose their backend."""
    try:
        check_backend("pycbc")
    except WaveformBackendError:
        return False
    return True


def backend_label(backend):
    """The backend with its version, as recorded: 'pycbc <version>' or 'newton'."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown waveform backend {backend!r}")
    if backend == "newton":
        return "newton"
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
        backend: 'pycbc' or 'newton', required. The pipeline passes
                 gwd.config.WAVEFORM_BACKEND.
        mass1, mass2: Black hole masses in solar masses.
        distance_mpc: Distance in megaparsec (sets the physical amplitude).
        f_lower: Lower frequency limit of the waveform (Hz).
        normalize: If True (for ML), the peak amplitude is scaled to 1.0.
                   If False (for the physics simulator), the strain is returned.
        spin1z, spin2z: Spins along the line of sight (-1 to +1). PyCBC only.
        inclination: Inclination (0 = face-on, π/2 = edge-on). PyCBC only.
        approximant: Waveform approximant. PyCBC only.

    Returns:
        tuple: (strain, frequency)
            - strain: np.array with the gravitational wave amplitude
            - frequency: np.array with the instantaneous frequency over time

    Raises:
        ValueError: backend is not 'pycbc' or 'newton', or t_merger lies outside
            time_array. Before P5.3 it was moved to the edge silently (CORE-15).
        WaveformBackendError: 'pycbc' and PyCBC is missing or fails for these
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

    if backend == "pycbc":
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
    return _newton_chirp(
        time_array,
        t_merger,
        mass1=mass1,
        mass2=mass2,
        distance_mpc=distance_mpc,
        normalize=normalize,
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

    # Empty arrays for the result
    strain_out = np.zeros_like(time_array)
    freq_out = np.zeros_like(time_array)

    # Copy the overlapping part
    start_src = max(0, -shift)
    end_src = min(len(waveform), len(time_array) - shift)

    start_dst = max(0, shift)
    end_dst = start_dst + (end_src - start_src)

    if end_dst > start_dst:
        strain_out[start_dst:end_dst] = waveform[start_src:end_src]
        freq_out[start_dst:end_dst] = raw_freq[start_src:end_src]

    # Taper the start of the window to avoid clicks
    taper_len = min(int(0.05 / dt), start_dst + 50)  # 50 ms taper
    if start_dst < taper_len and start_dst < len(strain_out):
        # Soft fade in
        window = np.linspace(0, 1, taper_len - start_dst)
        strain_out[start_dst:taper_len] *= window

    # Normalization for machine learning
    if normalize:
        m = np.max(np.abs(strain_out))
        if m > 0:
            strain_out /= m

    return strain_out, freq_out


def _newton_chirp(time_array, t_merger, mass1, mass2, distance_mpc, normalize):
    # Newtonian approximation: formulas for the inspiral (before the merger)

    # Chirp mass in kg, distance in m
    ch_mass = (mass1 * mass2) ** 0.6 / (mass1 + mass2) ** 0.2 * M_SUN
    dist_m = distance_mpc * MPC

    # Time to merger (tau)
    tau = np.clip(t_merger - time_array, 0.0001, None)

    # Frequency evolution (0th post-Newtonian order); CORE-2: too high by π^(5/8)
    # f(tau) ~ tau^(-3/8)
    freq_newton = (
        (5 / (256 * np.pi)) ** (3 / 8)
        * (G * ch_mass / C**3) ** (-5 / 8)
        * tau ** (-3 / 8)
    )

    # Amplitude (strain); CORE-3: 4 times too large
    # h(tau) ~ 1/D * tau^(-1/4)
    amp_prefactor = (
        (4 / dist_m) * (G * ch_mass / C**2) ** (5 / 4) * (5 / (C * tau)) ** (1 / 4)
    )

    # Phase
    # phi(tau) ~ tau^(5/8)
    phase = -2 * (5 * G * ch_mass / C**3) ** (-5 / 8) * tau ** (5 / 8)

    strain_out = amp_prefactor * np.cos(phase)

    # Cut everything after the merger (the approximation has no ringdown)
    mask_post_merger = time_array > t_merger
    strain_out[mask_post_merger] = 0
    freq_newton[mask_post_merger] = 0

    # Fade-in taper to avoid an abrupt start
    strain_out *= 1 - np.exp(-10 * (time_array - time_array[0]))

    if normalize:
        m = np.max(np.abs(strain_out))
        if m > 0:
            strain_out /= m

    return strain_out, freq_newton


def generate_realistic_chirp(
    time_array,
    *,
    backend,
    mass1=30,
    mass2=30,
    distance=400,
    spin1z=0.0,
    spin2z=0.0,
    inclination=0,
    approximant="IMRPhenomD",
):
    """
    Generate a strain signal with the given backend, unnormalized.

    Not used by the pipeline; P5.3 deletes it. With 'pycbc' it cuts off the merger
    (CORE-1); with 'newton' it returns the unnormalized chirp of generate_chirp.

    Args:
        time_array: np.array of time points (e.g. np.linspace(0, 4, 16384))
        backend: 'pycbc' or 'newton', required
        mass1, mass2: Masses in solar masses
        distance: Distance in Mpc
        spin1z, spin2z: Spins along the line of sight (-1 to +1)
        inclination: Inclination (0 = face-on, π/2 = edge-on)
        approximant: 'IMRPhenomD' (fast) or 'SEOBNRv4' (more accurate)

    Returns:
        tuple: (strain, freq)
            - strain: np.array (signal only, no noise)
            - freq: instantaneous frequency
    """
    check_backend(backend)
    if backend == "newton":
        return generate_chirp(
            time_array,
            backend="newton",
            mass1=mass1,
            mass2=mass2,
            distance_mpc=distance,
            normalize=False,
        )

    get_td_waveform = _get_td_waveform()
    sample_rate = 1.0 / (time_array[1] - time_array[0])

    # Generate the waveform
    hp, hc = get_td_waveform(
        approximant=approximant,
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        distance=distance,
        inclination=inclination,
        delta_t=1.0 / sample_rate,
        f_lower=20.0,
    )

    # Plus polarization only (simplified: optimal orientation)
    # In a detector: h(t) = F+ * hp + Fx * hc (antenna pattern)
    strain_raw = hp.numpy()

    # Instantaneous frequency
    complex_strain = hp.numpy() - 1j * hc.numpy()
    phase = np.unwrap(np.angle(complex_strain))
    dt = 1.0 / sample_rate
    freq = np.abs(np.gradient(phase, dt) / (2 * np.pi))

    # Time alignment: peak near the end
    peak_idx = np.argmax(np.abs(strain_raw))
    strain_raw = np.roll(strain_raw, len(strain_raw) - peak_idx - 100)
    freq = np.roll(freq, len(freq) - peak_idx - 100)

    # Crop or pad to the length of time_array
    if len(strain_raw) > len(time_array):
        strain_raw = strain_raw[: len(time_array)]
        freq = freq[: len(time_array)]
    elif len(strain_raw) < len(time_array):
        strain_raw = np.pad(strain_raw, (0, len(time_array) - len(strain_raw)))
        freq = np.pad(freq, (0, len(time_array) - len(freq)))

    return strain_raw, freq


def generate_random_bbh_parameters():
    """
    Draw random binary black hole parameters from the global numpy RNG.

    The distributions are simple choices, not fits to an observed population:
    - Masses: log-uniform (smaller masses more often)
    - Distance: uniform in volume (larger distances more often)
    - Spins: uniform between -0.9 and +0.9
    - Inclination: isotropic

    The pipeline does not call it since P3.5; gwd.data.injection draws the same
    values in the same order.

    Returns:
        dict with the parameters:
            - mass1, mass2: Masses in solar masses (mass1 >= mass2)
            - distance: Distance in Mpc
            - spin1z, spin2z: Spins
            - inclination: Inclination in radians
    """
    # Masses: log-uniform between 5 and 100 solar masses
    m1 = 10 ** np.random.uniform(np.log10(5), np.log10(100))
    m2 = 10 ** np.random.uniform(np.log10(5), np.log10(m1))  # m2 ≤ m1

    # Distance: uniform in volume, up to 2000 Mpc
    distance = (np.random.uniform(0, 1) ** (1 / 3)) * 2000

    # Spins: uniform between -0.9 and +0.9
    spin1z = np.random.uniform(-0.9, 0.9)
    spin2z = np.random.uniform(-0.9, 0.9)

    # Inclination: isotropic, so cos(i) is uniform
    cos_inclination = np.random.uniform(-1, 1)
    inclination = np.arccos(cos_inclination)

    return {
        "mass1": m1,
        "mass2": m2,
        "distance": distance,
        "spin1z": spin1z,
        "spin2z": spin2z,
        "inclination": inclination,
    }
