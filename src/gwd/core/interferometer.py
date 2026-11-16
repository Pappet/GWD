"""Interferometer sensitivity: a qualitative learning model and the aLIGO reference.

InterferometerModel is a qualitative learning model (decision D6 = A), not a
calibrated detector model. It shows in which direction the sensitivity moves when the
laser power, the mirrors, the arm length, or a feature changes; its numbers are not
those of a detector. reference_asd and reference_psd give the aLIGO design curve of
PyCBC for comparison (P5.4, CORE-11, APP-13).
"""

import numpy as np

from .constants import K_B, LASER_WAVELENGTH, LIGO_ARM_LENGTH, C, H

# The ad hoc factors of the learning model (CORE-10). None of them is derived from a
# detector; they make the curve look plausible and the sliders visible.
MIRROR_MASS = 40.0  # kg, of the order of the aLIGO test masses
TEMPERATURE = 300.0  # K
SQUEEZING_FACTOR = 3.0  # squeezed light divides the shot noise by this
THERMAL_SCALE = 5e-21  # thermal noise: sqrt(4 kB T / (2 pi f m)) times this
COATING_WEIGHT = 0.5  # thermal noise: times 1 + COATING_WEIGHT (1 - R)
SEISMIC_LEVEL = 1e-17  # seismic noise: SEISMIC_LEVEL / f^2
RADIATION_PRESSURE_SCALE = 1e-3  # radiation pressure noise: times this
BS_LOSS_WEIGHT = 10.0  # loss factor: 1 + BS_LOSS_WEIGHT * beamsplitter_loss
MIRROR_LOSS_WEIGHT = 50.0  # loss factor: + MIRROR_LOSS_WEIGHT (1 - R)
POWER_RECYCLING_FACTOR = 0.6  # power recycling multiplies the loss factor by this
SIGNAL_RECYCLING_DEPTH = 0.5  # signal recycling: a Gaussian dip of this depth ...
SIGNAL_RECYCLING_CENTER = 100.0  # ... at this frequency in Hz ...
SIGNAL_RECYCLING_WIDTH = 50.0  # ... and of this width in Hz

# The reference: PyCBC's aLIGO design PSD (zero detuning, high power).
REFERENCE_PSD_NAME = "aLIGOZeroDetHighPower"

REFERENCE_INSTALL_HINT = (
    "The aLIGO reference curve needs PyCBC, a dependency of gwd except on Windows: "
    "pip install -e . (on Windows, use WSL)"
)


class ReferencePSDError(RuntimeError):
    """The aLIGO reference PSD cannot be computed: PyCBC is missing."""


class InterferometerModel:
    """Amplitude spectral density (ASD) of a GW detector as a qualitative learning model.

    The ASD at self.freq (10 Hz to about 3 kHz) is the quadrature sum of four terms,
    times a loss factor:

    - shot noise: sqrt(h c lambda / (2 pi P L^2)), divided by the square root of the
      cavity gain 2 F / pi (F = finesse) and by SQUEEZING_FACTOR with squeezed light.
      It is flat in frequency; in a real Fabry-Perot interferometer it rises above the
      arm cavity pole.
    - thermal noise: sqrt(4 kB TEMPERATURE / (2 pi f MIRROR_MASS)) times THERMAL_SCALE,
      times 1 + COATING_WEIGHT (1 - R). The formula has no strain unit; THERMAL_SCALE turns
      it into a number of the right order.
    - seismic noise: SEISMIC_LEVEL / f^2.
    - radiation pressure: sqrt(8 hbar P / (MIRROR_MASS lambda c (2 pi f)^2)) times
      RADIATION_PRESSURE_SCALE, without the arm length.
    - loss factor: 1 + BS_LOSS_WEIGHT * beamsplitter_loss + MIRROR_LOSS_WEIGHT (1 - R),
      times POWER_RECYCLING_FACTOR with power recycling, and times a Gaussian dip of
      SIGNAL_RECYCLING_DEPTH and SIGNAL_RECYCLING_WIDTH around SIGNAL_RECYCLING_CENTER
      with signal recycling.

    The arm length enters only the shot noise. The factors are the module constants
    above; tests/core/test_interferometer.py checks only the directions (more power,
    longer arms, and each feature lower the ASD where they should) and the order of
    magnitude against reference_asd, which the default model lies above everywhere.
    """

    def __init__(self):
        # Default parameters
        self.laser_power = 200  # watts
        self.mirror_reflectivity = 0.999
        self.arm_length = LIGO_ARM_LENGTH  # metres
        self.beamsplitter_loss = 1e-4

        # Features
        self.power_recycling = False
        self.signal_recycling = False
        self.squeezed_light = False

        # Frequency range of the curve
        self.freq = np.logspace(1, 3.5, 1000)  # 10 Hz - 3 kHz

    @property
    def finesse(self):
        """Finesse of an arm cavity whose two mirrors both have mirror_reflectivity R.

        F = pi * sqrt(R) / (1 - R), about 3140 for the default R = 0.999. Before P4.5
        finesse was a separate start value of 450, and only the mirror slider of the
        designer set it from R, so the curve jumped on the first touch (APP-12, CORE-11).
        """
        r = self.mirror_reflectivity
        return np.pi * np.sqrt(r) / (1 - r)

    def calculate_shot_noise(self):
        shot_noise = np.sqrt(
            H
            * C
            * LASER_WAVELENGTH
            / (2 * np.pi * self.laser_power * self.arm_length**2)
        )

        cavity_gain = 2 * self.finesse / np.pi
        shot_noise = shot_noise / np.sqrt(cavity_gain)

        if self.squeezed_light:
            shot_noise = shot_noise / SQUEEZING_FACTOR

        return shot_noise * np.ones_like(self.freq)

    def calculate_thermal_noise(self):
        thermal = np.sqrt(4 * K_B * TEMPERATURE / (2 * np.pi * self.freq * MIRROR_MASS))
        coating_factor = (1 - self.mirror_reflectivity) * COATING_WEIGHT
        thermal = thermal * (1 + coating_factor)
        return thermal * THERMAL_SCALE

    def calculate_seismic_noise(self):
        return SEISMIC_LEVEL / self.freq**2

    def calculate_radiation_pressure(self):
        h_bar = H / (2 * np.pi)
        radiation = np.sqrt(
            8
            * h_bar
            * self.laser_power
            / (MIRROR_MASS * LASER_WAVELENGTH * C * (2 * np.pi * self.freq) ** 2)
        )
        return radiation * RADIATION_PRESSURE_SCALE

    def calculate_total_sensitivity(self):
        shot = self.calculate_shot_noise()
        thermal = self.calculate_thermal_noise()
        seismic = self.calculate_seismic_noise()
        radiation = self.calculate_radiation_pressure()

        loss_factor = (
            1.0
            + (self.beamsplitter_loss * BS_LOSS_WEIGHT)
            + ((1 - self.mirror_reflectivity) * MIRROR_LOSS_WEIGHT)
        )

        if self.power_recycling:
            loss_factor *= POWER_RECYCLING_FACTOR

        if self.signal_recycling:
            signal_gain = 1 - SIGNAL_RECYCLING_DEPTH * np.exp(
                -(((self.freq - SIGNAL_RECYCLING_CENTER) / SIGNAL_RECYCLING_WIDTH) ** 2)
            )
            loss_factor *= signal_gain

        total = np.sqrt(shot**2 + thermal**2 + seismic**2 + radiation**2) * loss_factor
        return total


def _reference_psd_function():
    """The LALSimulation function behind pycbc.psd.aLIGOZeroDetHighPower, imported lazily.

    LALSimulation comes with PyCBC (lalsuite). pycbc is imported too, so
    sys.modules["pycbc"] = None blocks the reference as it blocks the PyCBC waveforms.
    """
    try:
        import lalsimulation
        import pycbc  # noqa: F401
    except ImportError as error:
        raise ReferencePSDError(f"{REFERENCE_INSTALL_HINT} ({error})") from error
    return lalsimulation.SimNoisePSDaLIGOZeroDetHighPower


def reference_available():
    """True if the aLIGO reference PSD can be computed here (PyCBC is installed)."""
    try:
        _reference_psd_function()
    except ReferencePSDError:
        return False
    return True


def reference_psd(f, low_freq_cutoff=0.0):
    """One-sided aLIGO design PSD in strain^2 / Hz at the frequencies f (Hz).

    The values of pycbc.psd.aLIGOZeroDetHighPower, which evaluates the same
    LALSimulation function on a grid of k * delta_f; this one evaluates it at any f.
    Like PyCBC, it is 0 below low_freq_cutoff, and it is 0 at f <= 0. A scalar f gives
    a scalar. Raises ReferencePSDError without PyCBC.
    """
    function = _reference_psd_function()
    f = np.asarray(f, dtype=float)
    flat = f.ravel()
    psd = np.zeros_like(flat)
    keep = (flat > 0) & (flat >= low_freq_cutoff)
    psd[keep] = [function(float(value)) for value in flat[keep]]
    return psd.reshape(f.shape)[()]


def reference_asd(f, low_freq_cutoff=0.0):
    """aLIGO design ASD in strain / sqrt(Hz): the square root of reference_psd."""
    return np.sqrt(reference_psd(f, low_freq_cutoff))
