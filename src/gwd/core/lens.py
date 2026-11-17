"""The chirp lens: mother curves of binary black hole chirps, stretched over the mass.

General relativity has no scale of its own. At a fixed mass ratio and without spins,
the waveform of total mass M is one curve in u = t / M,

    h(t) = A(t / M) * exp(i * phi(t / M)),

so another total mass is the same curve, stretched in time. A mother curve holds A and
phi of one mass ratio on a Grid in u (seconds per solar mass, 0 at the peak).
kernel_spectra() stretches it to a list of total masses and returns the spectrum of
each stretched curve in the band of a Frame: a bank of matched filter templates.
mass_ladder() spaces the total masses so that neighbouring templates overlap by a
fixed amount.

gwd.training.telescope makes these curves the weights of a model, the chirp telescope.
This module is its numpy reference and draws its start values: mother_curve() from
PyCBC, newtonian_curve() from the leading-order inspiral formula, pixel_evidence() and
channel_log_prior() for the pooling.

Like the rest of gwd.core it reads no gwd.config and prints nothing, and PyCBC is
imported only when mother_curve() runs.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ive, logsumexp

from gwd.core import waveforms

# G * solar mass / c^3 in seconds: the time scale of one solar mass.
SOLAR_MASS_S = 4.925490947e-6
# A mother curve is drawn at this total mass (solar masses), from this frequency (Hz),
# with this spacing (s). 20 solar masses from 10 Hz reach back more than 1 s per solar
# mass; the spacing resolves the ringdown of the heaviest system of the pipeline.
MOTHER_TOTAL_MASS = 20.0
MOTHER_F_LOWER = 10.0
MOTHER_DELTA_T = 1.0 / 16384
# The start of every kernel is faded in over this many seconds.
FADE_SECONDS = 0.05
# The leading-order inspiral reaches the innermost stable circular orbit of a
# Schwarzschild black hole of the total mass, f = 1 / (6^1.5 pi G M / c^3), this many
# G M / c^3, divided by the symmetric mass ratio, before its coalescence: 5 * 6^4 / 256.
ISCO_FACTOR = 5.0 * 6.0**4 / 256.0


@dataclass(frozen=True)
class Grid:
    """The nodes of a mother curve: values of u = t / M, uniform in asinh(u / scale).

    Within scale of the peak the nodes are uniform in u, further out uniform in
    log |u|, as the phase of an inspiral needs them: a curve between two nodes is
    interpolated linearly in the node index. u_min is negative (before the peak).
    """

    u_min: float
    u_max: float
    nodes: int = 512
    scale: float = 1e-4

    def _warp(self, u):
        return np.arcsinh(np.asarray(u, dtype=float) / self.scale)

    def u(self):
        """The nodes, from u_min to u_max."""
        warped = np.linspace(self._warp(self.u_min), self._warp(self.u_max), self.nodes)
        return self.scale * np.sinh(warped)

    def locate(self, u):
        """Where u lies between the nodes: (index, fraction, inside).

        A value at u is curve[index] + fraction * (curve[index + 1] - curve[index]).
        inside is False for u outside [u_min, u_max], where index and fraction point at
        the nearest end.
        """
        low, high = self._warp(self.u_min), self._warp(self.u_max)
        position = (self._warp(u) - low) / (high - low) * (self.nodes - 1)
        inside = (position >= 0) & (position <= self.nodes - 1)
        index = np.clip(np.floor(position), 0, self.nodes - 2).astype(np.int32)
        fraction = np.clip(position - index, 0.0, 1.0)
        return index, fraction, inside


@dataclass(frozen=True)
class Frame:
    """How a kernel sits in a sample.

    A sample has n_samples values at sample_rate Hz. A kernel covers pre_s seconds
    before its merger and post_s seconds after it, and the lens reads the rfft bins of
    band (Hz, both ends included). The focal plane, the output of the lens over the
    merger time, is sampled at sample_rate / decimation.
    """

    sample_rate: int
    n_samples: int
    pre_s: float
    post_s: float
    band: tuple
    decimation: int = 4

    @property
    def pre(self):
        """Kernel samples before the merger."""
        return round(self.pre_s * self.sample_rate)

    @property
    def length(self):
        """Samples of a kernel."""
        return self.pre + round(self.post_s * self.sample_rate)

    @property
    def times(self):
        """The sample times of a kernel in seconds, 0 at the merger."""
        return (np.arange(self.length) - self.pre) / self.sample_rate

    @property
    def bins(self):
        """(first, last + 1) rfft bin of the band."""
        spacing = self.sample_rate / self.n_samples
        low = int(np.ceil(self.band[0] / spacing - 1e-9))
        high = int(np.floor(self.band[1] / spacing + 1e-9)) + 1
        return low, high

    @property
    def frequencies(self):
        """The frequencies of the band's bins in Hz."""
        low, high = self.bins
        return np.arange(low, high) * self.sample_rate / self.n_samples

    @property
    def focal_samples(self):
        """Samples of the focal plane over the whole sample."""
        return self.n_samples // self.decimation

    @property
    def fade(self):
        """The factor on a kernel's samples: a cosine ramp over FADE_SECONDS at its start."""
        factor = np.ones(self.length)
        ramp = min(round(FADE_SECONDS * self.sample_rate), self.length)
        factor[:ramp] = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
        return factor

    @property
    def shift(self):
        """The factor on the band's bins that moves a kernel's merger to time 0."""
        return np.exp(2j * np.pi * self.frequencies * self.pre / self.sample_rate)

    def window(self, t_low, t_high):
        """(first, last + 1) focal plane sample of the merger times t_low ... t_high (s)."""
        rate = self.sample_rate / self.decimation
        first = max(int(np.floor(t_low * rate)), 0)
        last = min(int(np.ceil(t_high * rate)), self.focal_samples - 1)
        return first, last + 1


def mother_curve(
    mass_ratio,
    grid,
    *,
    backend,
    total_mass=MOTHER_TOTAL_MASS,
    f_lower=MOTHER_F_LOWER,
    approximant="IMRPhenomD",
):
    """Amplitude and phase of the chirp with mass_ratio = m1 / m2 on the nodes of grid.

    One PyCBC waveform, face-on and without spins, of total_mass solar masses from
    f_lower Hz, read as a curve in u = t / total_mass. The amplitude is 1 at its peak,
    where u is 0 and the phase is 0; the phase grows with u (positive frequencies).
    Before the start of the waveform the amplitude is 0.

    Raises ValueError for an unknown backend and WaveformBackendError if PyCBC is
    missing or fails.
    """
    waveforms.check_backend(backend)
    get_td_waveform = waveforms._get_td_waveform()
    mass2 = total_mass / (1.0 + mass_ratio)
    try:
        hp, hc = get_td_waveform(
            approximant=approximant,
            mass1=total_mass - mass2,
            mass2=mass2,
            delta_t=MOTHER_DELTA_T,
            f_lower=f_lower,
        )
    except Exception as error:
        raise waveforms.WaveformBackendError(
            f"PyCBC failed for approximant={approximant!r}, mass ratio {mass_ratio}, "
            f"total mass {total_mass}, f_lower={f_lower}: {error}"
        ) from error
    strain = hp.numpy() + 1j * hc.numpy()
    envelope = np.abs(strain)
    peak = int(np.argmax(envelope))
    u = (np.arange(len(strain)) - peak) * MOTHER_DELTA_T / total_mass
    phase = np.unwrap(np.angle(strain))
    if phase[peak] < phase[0]:
        phase = -phase
    nodes = grid.u()
    amplitude = np.interp(nodes, u, envelope / envelope[peak], left=0.0, right=0.0)
    return amplitude, np.interp(nodes, u, phase - phase[peak])


def newtonian_curve(mass_ratio, grid):
    """Amplitude and phase of the leading-order inspiral on the nodes of grid.

    The quadrupole formula for a circular orbit that shrinks by radiation: the phase is
    -2 (tau / (5 G Mc / c^3))^(5/8) for the time tau before the coalescence, with the
    chirp mass Mc, and the amplitude grows as tau^(-1/4). It has no merger and no
    ringdown: the amplitude is 1 at the innermost stable circular orbit (ISCO_FACTOR)
    and fades to 0 from there to the coalescence at u = 0, after which the phase stays
    0. No PyCBC is needed.
    """
    eta = mass_ratio / (1.0 + mass_ratio) ** 2
    u = grid.u()
    tau = np.maximum(-u, 0.0)
    phase = -2.0 * (tau / (5.0 * SOLAR_MASS_S * eta**0.6)) ** 0.625
    u_isco = -ISCO_FACTOR * SOLAR_MASS_S / eta
    amplitude = np.zeros_like(u)
    inspiral = u <= u_isco
    amplitude[inspiral] = (u_isco / u[inspiral]) ** 0.25
    plunge = (u > u_isco) & (u < 0)
    amplitude[plunge] = np.cos(0.5 * np.pi * (1.0 - u[plunge] / u_isco)) ** 2
    return amplitude, phase


def stretched(amplitude, phase, grid, masses, frame):
    """The mother curve at each total mass on frame.times: complex (masses, samples).

    Row j is A(t / masses[j]) * exp(i * phi(t / masses[j])), interpolated linearly
    between the nodes, 0 outside the grid, and faded in at its start (frame.fade).
    """
    u = frame.times[None, :] / np.asarray(masses, dtype=float)[:, None]
    index, fraction, inside = grid.locate(u)
    amplitude = np.asarray(amplitude, dtype=float)
    phase = np.asarray(phase, dtype=float)
    a = amplitude[index] + fraction * (amplitude[index + 1] - amplitude[index])
    p = phase[index] + fraction * (phase[index + 1] - phase[index])
    return np.where(inside, a, 0.0) * frame.fade * np.exp(1j * p)


def kernel_spectra(amplitude, phase, grid, masses, frame):
    """The spectrum of each stretched curve in the band: complex (masses, bins).

    The merger of every kernel lies at time 0 (frame.shift), so correlating a sample
    with a kernel peaks at the merger time of a matching signal.
    """
    kernels = stretched(amplitude, phase, grid, masses, frame)
    padded = np.zeros((len(kernels), frame.n_samples), dtype=complex)
    padded[:, : frame.length] = kernels
    low, high = frame.bins
    return np.fft.fft(padded, axis=1)[:, low:high] * frame.shift


def overlap(a, b, frame):
    """The normalized overlap of two band spectra, maximized over time and phase.

    1 for equal spectra (up to a factor), 0 for orthogonal ones. The time is searched
    on the focal plane grid of frame.
    """
    low, high = frame.bins
    spectrum = np.zeros(frame.focal_samples, dtype=complex)
    spectrum[low:high] = a * np.conj(b)
    peak = np.abs(np.fft.ifft(spectrum)).max() * frame.focal_samples
    return peak / np.sqrt((np.abs(a) ** 2).sum() * (np.abs(b) ** 2).sum())


def mass_ladder(amplitude, phase, grid, m_low, m_high, min_overlap, frame, weight=None):
    """Total masses from m_low towards m_high whose neighbours overlap by min_overlap.

    Starts at m_low; each next mass is the largest whose kernel still overlaps the
    previous one by min_overlap (found by bisection in log mass), so a signal between
    two rungs loses little against either. weight (real, per bin of the band)
    multiplies the kernels first, for example 1 / ASD of the noise; None means 1. The
    last rung lies below m_high by less than one step.
    """
    if not 0 < min_overlap < 1:
        raise ValueError(f"min_overlap must lie between 0 and 1, got {min_overlap}")
    if not 0 < m_low <= m_high:
        raise ValueError(f"need 0 < m_low <= m_high, got {m_low} and {m_high}")
    weight = 1.0 if weight is None else np.asarray(weight, dtype=float)

    def kernel(mass):
        return kernel_spectra(amplitude, phase, grid, [mass], frame)[0] * weight

    masses = [float(m_low)]
    last = kernel(m_low)
    while True:
        low, high = masses[-1], min(masses[-1] * 1.5, m_high)
        if high <= low:
            break
        if overlap(last, kernel(high), frame) < min_overlap:
            for _ in range(14):
                middle = np.sqrt(low * high)
                if overlap(last, kernel(middle), frame) >= min_overlap:
                    low = middle
                else:
                    high = middle
            high = low
        if high >= m_high or high <= masses[-1]:
            break
        masses.append(float(high))
        last = kernel(high)
    return np.array(masses)


def pixel_evidence(rho, snr_range, steps=321):
    """The log likelihood ratio of one focal plane pixel of amplitude rho.

    The pixel is the output of a matched filter, scaled so that noise alone gives
    rho^2 a chi-squared distribution with 2 degrees of freedom. A signal of SNR a with
    an unknown phase then has the likelihood ratio exp(-a^2 / 2) * I0(a * rho), and
    this is its logarithm averaged over a uniform in snr_range.
    """
    rho = np.asarray(rho, dtype=float)
    snr = np.linspace(snr_range[0], snr_range[1], steps)[:, None]
    x = snr * rho[None, :]
    terms = -(snr**2) / 2.0 + x + np.log(ive(0, x))
    return logsumexp(terms, axis=0) - np.log(steps)


def channel_log_prior(ladders):
    """The log prior weight of every channel: its share of the log mass axis.

    ladders is one array of total masses per mother curve; the result joins them in
    that order. Within a ladder a channel weighs as much as the log mass interval it
    covers, and all weights sum to 1.
    """
    weights = []
    for masses in ladders:
        log_mass = np.log(np.asarray(masses, dtype=float))
        weights.append(np.gradient(log_mass) if len(log_mass) > 1 else np.ones(1))
    log_weight = np.log(np.concatenate(weights))
    return log_weight - logsumexp(log_weight)
