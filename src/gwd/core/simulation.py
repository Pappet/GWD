"""Time domain simulation of a chirp and the detector response, for the waveform explorer."""

import numpy as np

from . import waveforms
from .constants import LASER_WAVELENGTH
from .noise import generate_colored_noise
from .timebase import time_axis


class TimeDomainSimulator:
    """
    Simulates gravitational wave signals and the detector response in the time domain.
    Uses the shared physics modules of gwd.core.
    """

    def __init__(self, duration=0.5, sample_rate=4096, *, backend, rng):
        # The waveform backend, 'pycbc' or 'newton'. The caller chooses it (P5.2).
        waveforms.check_backend(backend)
        self.backend = backend
        # The noise is drawn from rng, a numpy Generator or RandomState (P5.3).
        self.rng = rng
        self.arm_length = 4000
        self.laser_wavelength = LASER_WAVELENGTH

        self.duration = duration
        self.sample_rate = sample_rate
        self.t = time_axis(sample_rate, duration)

    def generate_chirp(self, mass1, mass2, distance_mpc):
        """
        Delegates the computation to gwd.core.waveforms.
        IMPORTANT: normalize=False, so the result is in physical units (strain).
        """
        strain, freq = waveforms.generate_chirp(
            self.t,
            backend=self.backend,
            t_merger=self.duration * 0.8,  # merger at 80% of the window
            mass1=mass1,
            mass2=mass2,
            distance_mpc=distance_mpc,
            normalize=False,  # physical strain for the simulator
        )
        return strain, freq

    def add_detector_noise(self, signal, noise_level_slider):
        """
        Adds colored detector noise to the signal.
        """
        # The slider (0.1 - 5.0) is scaled to a physical noise amplitude (~1e-22)
        phys_noise_level = noise_level_slider * 1e-22

        # Colored noise from gwd.core.noise
        noise = generate_colored_noise(
            len(signal), self.sample_rate, phys_noise_level, rng=self.rng
        )

        return signal + noise

    def calculate_response(self, strain):
        """
        Computes the interferometer response: (arm length change, intensity).
        """
        delta_L = strain * self.arm_length
        phase_shift = 4 * np.pi * delta_L / self.laser_wavelength
        intensity = np.sin(phase_shift) ** 2
        return delta_L, intensity
