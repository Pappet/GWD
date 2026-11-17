"""The chirp lens: grid, frame, mother curves, and the mass ladder (gwd.core.lens).

The physics is checked against textbook formulas and against PyCBC: a mother curve
stretched to another total mass is PyCBC's waveform of that mass.
"""

import numpy as np
import pytest
from scipy.special import i0

from gwd.core import lens, timebase, waveforms

FRAME = lens.Frame(
    sample_rate=4096, n_samples=16384, pre_s=2.0, post_s=0.4, band=(24.0, 300.0)
)
# u from 2 s before to 0.4 s after the peak of the lightest system, 10 solar masses.
GRID = lens.Grid(u_min=-0.2, u_max=0.04)


def test_the_grid_runs_from_u_min_to_u_max_and_is_dense_at_the_peak():
    u = GRID.u()
    assert len(u) == GRID.nodes == 512
    assert u[0] == pytest.approx(-0.2) and u[-1] == pytest.approx(0.04)
    assert np.all(np.diff(u) > 0)
    # Uniform in asinh(u / scale): the spacing grows with the distance from the peak.
    spacing = np.diff(u)
    assert spacing.min() < 5e-6 and spacing[0] > 1000 * spacing.min()


def test_locate_interpolates_between_the_nodes():
    u = GRID.u()
    curve = np.sin(np.linspace(0, 3, GRID.nodes))
    index, fraction, inside = GRID.locate(u[[0, 100, 511]])
    assert inside.all()
    values = curve[index] + fraction * (curve[index + 1] - curve[index])
    assert values == pytest.approx(curve[[0, 100, 511]], abs=1e-9)
    # Halfway between two nodes in asinh(u / scale), not in u.
    warped = np.arcsinh(u / GRID.scale)
    middle = GRID.scale * np.sinh(0.5 * (warped[200] + warped[201]))
    index, fraction, inside = GRID.locate(middle)
    assert (index, inside) == (200, True)
    assert fraction == pytest.approx(0.5, abs=1e-6)
    assert not GRID.locate(np.array([-0.21, 0.05]))[2].any()


def test_the_frame_fits_the_samples_of_the_pipeline():
    assert FRAME.bins == (96, 1201)  # 24 Hz and 300 Hz at 0.25 Hz per bin
    assert FRAME.frequencies[0] == 24.0 and FRAME.frequencies[-1] == 300.0
    assert (FRAME.pre, FRAME.length) == (8192, 8192 + 1638)
    assert FRAME.times[FRAME.pre] == 0.0
    assert FRAME.focal_samples == 4096  # the focal plane at 1024 Hz
    assert FRAME.window(2.0, 3.6) == (2048, 3688)
    assert FRAME.fade[0] == 0.0 and FRAME.fade[-1] == 1.0


def test_the_newtonian_curve_is_the_textbook_chirp():
    """f(tau) = (5 / (256 tau))^(3/8) (G Mc / c^3)^(-5/8) / pi, here for 15 + 15."""
    mass, tau = 30.0, 1.0
    amplitude, phase = lens.newtonian_curve(1.0, GRID)
    kernel = lens.stretched(amplitude, phase, GRID, [mass], FRAME)[0]
    unwrapped = np.unwrap(np.angle(kernel))
    frequency = np.gradient(unwrapped, 1 / FRAME.sample_rate) / (2 * np.pi)
    chirp_mass_s = lens.SOLAR_MASS_S * mass * 0.25**0.6
    expected = (5 / (256 * tau)) ** 0.375 * chirp_mass_s**-0.625 / np.pi
    at_tau = FRAME.pre - round(tau * FRAME.sample_rate)
    assert frequency[at_tau] == pytest.approx(expected, rel=0.01)
    # No merger and no ringdown: nothing after the coalescence.
    assert np.all(amplitude[GRID.u() >= 0] == 0)
    assert amplitude.max() == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("mass_ratio, total", [(1.0, 30.0), (1.0, 120.0), (4.0, 50.0)])
def test_a_stretched_mother_curve_is_the_pycbc_waveform_of_that_mass(
    needs_pycbc, mass_ratio, total
):
    """General relativity has no scale: one curve in t / M holds every total mass."""
    amplitude, phase = lens.mother_curve(mass_ratio, GRID, backend="pycbc")
    kernel = lens.kernel_spectra(amplitude, phase, GRID, [total], FRAME)[0]
    time = timebase.time_axis(FRAME.sample_rate, 4.0)
    mass2 = total / (1 + mass_ratio)
    strain, _ = waveforms.generate_chirp(
        time,
        t_merger=3.0,
        backend="pycbc",
        mass1=total - mass2,
        mass2=mass2,
        f_lower=15.0,
    )
    low, high = FRAME.bins
    spectrum = np.fft.rfft(strain)[low:high]
    assert lens.overlap(kernel, spectrum, FRAME) > 0.98


def test_the_mother_curve_peaks_at_zero_with_a_growing_phase(needs_pycbc):
    amplitude, phase = lens.mother_curve(1.0, GRID, backend="pycbc")
    u = GRID.u()
    assert amplitude.max() == pytest.approx(1.0, abs=0.01)
    assert abs(u[np.argmax(amplitude)]) < 1e-5
    inspiral = (u < -1e-3) & (amplitude > 0)
    assert np.all(np.diff(phase[inspiral]) > 0)
    assert np.interp(0.0, u, phase) == pytest.approx(0.0, abs=0.05)


def test_an_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="Unknown waveform backend"):
        lens.mother_curve(1.0, GRID, backend="numpy")


def test_the_overlap_is_one_for_a_shifted_copy_and_small_for_another_mass():
    amplitude, phase = lens.newtonian_curve(1.0, GRID)
    a, b = lens.kernel_spectra(amplitude, phase, GRID, [40.0, 60.0], FRAME)
    shifted = a * np.exp(-2j * np.pi * FRAME.frequencies * 0.25) * np.exp(1.3j)
    assert lens.overlap(a, a, FRAME) == pytest.approx(1.0)
    assert lens.overlap(a, shifted, FRAME) == pytest.approx(1.0, abs=1e-6)
    assert lens.overlap(a, b, FRAME) < 0.5


def test_the_ladder_keeps_the_overlap_between_neighbours():
    amplitude, phase = lens.newtonian_curve(1.0, GRID)
    masses = lens.mass_ladder(amplitude, phase, GRID, 40.0, 60.0, 0.9, FRAME)
    assert masses[0] == 40.0 and masses[-1] < 60.0
    assert np.all(np.diff(masses) > 0) and len(masses) > 3
    kernels = lens.kernel_spectra(amplitude, phase, GRID, masses, FRAME)
    neighbours = [lens.overlap(a, b, FRAME) for a, b in zip(kernels, kernels[1:])]
    assert min(neighbours) >= 0.9 and max(neighbours) < 0.92
    # One more step from the last rung would pass m_high.
    step = masses[-1] / masses[-2]
    assert masses[-1] * step > 60.0 * 0.98
    # A lower overlap needs fewer rungs.
    assert len(lens.mass_ladder(amplitude, phase, GRID, 40.0, 60.0, 0.8, FRAME)) < len(
        masses
    )


@pytest.mark.parametrize(
    "low, high, overlap", [(40.0, 60.0, 1.0), (40.0, 60.0, 0.0), (60.0, 40.0, 0.9)]
)
def test_the_ladder_refuses_impossible_arguments(low, high, overlap):
    amplitude, phase = lens.newtonian_curve(1.0, GRID)
    with pytest.raises(ValueError):
        lens.mass_ladder(amplitude, phase, GRID, low, high, overlap, FRAME)


def test_one_mass_is_a_ladder_of_one_rung():
    amplitude, phase = lens.newtonian_curve(1.0, GRID)
    masses = lens.mass_ladder(amplitude, phase, GRID, 50.0, 50.0, 0.9, FRAME)
    assert masses.tolist() == [50.0]


def test_the_pixel_evidence_is_the_marginal_likelihood_ratio():
    rho = np.array([0.0, 1.0, 2.5])
    # One SNR: exp(-a^2 / 2) * I0(a * rho).
    single = lens.pixel_evidence(rho, (3.0, 3.0))
    assert single == pytest.approx(-4.5 + np.log(i0(3.0 * rho)))
    # A range of SNRs: grows with rho, also far beyond the range of I0.
    curve = lens.pixel_evidence(np.linspace(0, 40, 81), (4.0, 20.0))
    assert np.all(np.diff(curve) > 0) and np.all(np.isfinite(curve))
    assert curve[0] < -8 and curve[-1] > 100


def test_the_channel_prior_sums_to_one_and_follows_the_log_mass():
    ladders = [np.array([10.0, 20.0, 40.0]), np.array([30.0])]
    prior = np.exp(lens.channel_log_prior(ladders))
    assert len(prior) == 4
    assert prior.sum() == pytest.approx(1.0)
    # Equal steps in log mass weigh the same.
    assert prior[:3] == pytest.approx(prior[0])
