import numpy as np

from .constants import LASER_WAVELENGTH, C


class InterferometerModel:
    """
    Computes the sensitivity (ASD) of a GW detector as a simple learning model (D6).
    (Logic taken from Interferometer Designer V2.)
    """

    def __init__(self):
        # Default parameters
        self.laser_power = 200  # watts
        self.mirror_reflectivity = 0.999
        self.arm_length = 4000  # metres
        self.beamsplitter_loss = 1e-4

        # Features
        self.power_recycling = False
        self.signal_recycling = False
        self.squeezed_light = False

        # Planck constant
        self.h = 6.626e-34

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
            self.h
            * C
            * LASER_WAVELENGTH
            / (2 * np.pi * self.laser_power * self.arm_length**2)
        )

        cavity_gain = 2 * self.finesse / np.pi
        shot_noise = shot_noise / np.sqrt(cavity_gain)

        if self.squeezed_light:
            shot_noise = shot_noise / 3

        return shot_noise * np.ones_like(self.freq)

    def calculate_thermal_noise(self):
        kB = 1.381e-23
        T = 300
        mass = 40
        thermal = np.sqrt(4 * kB * T / (2 * np.pi * self.freq * mass))
        coating_factor = (1 - self.mirror_reflectivity) * 0.5
        thermal = thermal * (1 + coating_factor)
        return thermal * 5e-21

    def calculate_seismic_noise(self):
        return 1e-17 / self.freq**2

    def calculate_radiation_pressure(self):
        h_bar = self.h / (2 * np.pi)
        mass = 40
        radiation = np.sqrt(
            8
            * h_bar
            * self.laser_power
            / (mass * LASER_WAVELENGTH * C * (2 * np.pi * self.freq) ** 2)
        )
        return radiation * 1e-3

    def calculate_total_sensitivity(self):
        shot = self.calculate_shot_noise()
        thermal = self.calculate_thermal_noise()
        seismic = self.calculate_seismic_noise()
        radiation = self.calculate_radiation_pressure()

        loss_factor = (
            1.0 + (self.beamsplitter_loss * 10) + ((1 - self.mirror_reflectivity) * 50)
        )

        if self.power_recycling:
            loss_factor *= 0.6

        if self.signal_recycling:
            signal_gain = 1 - 0.5 * np.exp(-(((self.freq - 100) / 50) ** 2))
            loss_factor *= signal_gain

        total = np.sqrt(shot**2 + thermal**2 + seismic**2 + radiation**2) * loss_factor
        return total
