"""Gravitational waveforms of binary black hole mergers.

Two backends draw the chirps: PyCBC if it can be imported, otherwise a Newtonian
approximation of the inspiral with known errors (CORE-2, CORE-3, CORE-8). Which one
runs is a switch in the environment; active_backend() names it, and since P4.1 the
pipeline prints and records it (hygiene:HYG-M1). Importing this module prints nothing.
"""

import importlib.metadata
import logging
import warnings

import numpy as np

_log = logging.getLogger(__name__)

try:
    from pycbc.waveform import get_td_waveform

    HAS_PYCBC = True
except ImportError as error:
    HAS_PYCBC = False
    # Logged, not printed: importing this module writes nothing (CORE-17). The
    # pipeline states the backend itself, see active_backend().
    _log.info(
        "PyCBC not importable (%s); waveforms use the Newtonian approximation", error
    )


class WaveformFallbackWarning(RuntimeWarning):
    """PyCBC was asked for a waveform, but the Newtonian approximation produced it."""


def active_backend():
    """The waveform backend in use: 'pycbc <version>' or 'newton'.

    It is decided once, on import, by whether PyCBC can be imported. A single chirp can
    still fall back to Newton if PyCBC fails for its parameters; chirp_and_backend()
    reports that per chirp and warns with WaveformFallbackWarning.
    """
    if not HAS_PYCBC:
        return "newton"
    try:
        version = importlib.metadata.version("pycbc")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown version"
    return f"pycbc {version}"


def generate_astrophysical_chirp(time_array, **kwargs):
    """The chirp of chirp_and_backend() as (strain, frequency), without the backend name."""
    strain, frequency, _ = chirp_and_backend(time_array, **kwargs)
    return strain, frequency


def chirp_and_backend(
    time_array,
    t_merger=None,
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

    Uses PyCBC if it can be imported. Without PyCBC, or if PyCBC fails for these
    parameters, the Newtonian approximation produces the chirp; the second case
    warns with WaveformFallbackWarning (CORE-4, removed in P5.2).

    Args:
        time_array: Time axis of the simulation (e.g. np.linspace(0, 4, 16384))
        t_merger: Merger time in seconds. If None, 90% of the time span.
        mass1, mass2: Black hole masses in solar masses.
        distance_mpc: Distance in megaparsec (sets the physical amplitude).
        f_lower: Lower frequency limit of the waveform (Hz).
        normalize: If True (for ML), the peak amplitude is scaled to 1.0.
                   If False (for the physics simulator), the strain is returned.
        spin1z, spin2z: Spins along the line of sight (-1 to +1).
        inclination: Inclination (0 = face-on, π/2 = edge-on).
        approximant: Waveform approximant (PyCBC only).

    Returns:
        tuple: (strain, frequency, backend)
            - strain: np.array with the gravitational wave amplitude
            - frequency: np.array with the instantaneous frequency over time
            - backend: "pycbc" or "newton", the backend that produced this chirp
    """

    # Default merger time if none is given
    if t_merger is None:
        t_merger = time_array[-1] * 0.9  # 90% of the time span

    # --- Backend A: PyCBC ---
    if HAS_PYCBC:
        try:
            dt = time_array[1] - time_array[0]

            # Plus and cross polarization
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

            return strain_out, freq_out, "pycbc"

        except Exception as e:
            warnings.warn(
                f"PyCBC failed ({e}); this chirp uses the Newtonian approximation.",
                WaveformFallbackWarning,
                stacklevel=2,
            )

    # --- Backend B: Newtonian approximation (fallback) ---
    # Formulas for the inspiral (before the merger)

    # Constants
    G = 6.674e-11  # gravitational constant
    c = 2.998e8  # speed of light
    Msun = 1.989e30  # solar mass in kg

    # Chirp mass
    ch_mass = (mass1 * mass2) ** 0.6 / (mass1 + mass2) ** 0.2 * Msun
    dist_m = distance_mpc * 3.086e22  # megaparsec in meters

    # Time to merger (tau)
    tau = np.clip(t_merger - time_array, 0.0001, None)

    # Frequency evolution (0th post-Newtonian order); CORE-2: too high by π^(5/8)
    # f(tau) ~ tau^(-3/8)
    freq_newton = (
        (5 / (256 * np.pi)) ** (3 / 8)
        * (G * ch_mass / c**3) ** (-5 / 8)
        * tau ** (-3 / 8)
    )

    # Amplitude (strain); CORE-3: 4 times too large
    # h(tau) ~ 1/D * tau^(-1/4)
    amp_prefactor = (
        (4 / dist_m) * (G * ch_mass / c**2) ** (5 / 4) * (5 / (c * tau)) ** (1 / 4)
    )

    # Phase
    # phi(tau) ~ tau^(5/8)
    phase = -2 * (5 * G * ch_mass / c**3) ** (-5 / 8) * tau ** (5 / 8)

    strain_out = amp_prefactor * np.cos(phase)

    # Cut everything after the merger (the approximation has no ringdown)
    mask_post_merger = time_array > t_merger
    strain_out[mask_post_merger] = 0
    freq_newton[mask_post_merger] = 0

    # Fade-in taper to avoid an abrupt start
    strain_out *= 1 - np.exp(-10 * time_array)

    if normalize:
        m = np.max(np.abs(strain_out))
        if m > 0:
            strain_out /= m

    return strain_out, freq_newton, "newton"


def generate_realistic_chirp(
    time_array,
    mass1=30,
    mass2=30,
    distance=400,
    spin1z=0.0,
    spin2z=0.0,
    inclination=0,
    approximant="IMRPhenomD",
):
    """
    Generate a strain signal with PyCBC, or with the Newtonian approximation.

    Not used by the pipeline. With PyCBC it cuts off the merger (CORE-1, P5.3);
    without PyCBC it warns with WaveformFallbackWarning and returns the unnormalized
    chirp of generate_astrophysical_chirp.

    Args:
        time_array: np.array of time points (e.g. np.linspace(0, 4, 16384))
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

    if not HAS_PYCBC:
        warnings.warn(
            "PyCBC is not available; using the Newtonian approximation.",
            WaveformFallbackWarning,
            stacklevel=2,
        )
        return generate_astrophysical_chirp(
            time_array,
            mass1=mass1,
            mass2=mass2,
            distance_mpc=distance,
            spin1z=spin1z,
            spin2z=spin2z,
            inclination=inclination,
            normalize=False,
        )

    # PyCBC is available: use it directly
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
