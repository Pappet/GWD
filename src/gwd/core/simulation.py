"""Time domain simulation of a chirp and the detector response, for the waveform explorer."""

import numpy as np

from . import waveforms
from .constants import LASER_WAVELENGTH, LIGO_ARM_LENGTH
from .noise import colored_noise_from_psd
from .timebase import time_axis


class TimeDomainSimulator:
    """
    Simulates gravitational wave signals and the detector response in the time domain.
    Uses the shared physics modules of gwd.core.
    """

    def __init__(self, duration=0.5, sample_rate=4096, *, backend, rng, noise_psd):
        # The waveform backend, 'pycbc' or 'newton'. The caller chooses it (P5.2).
        waveforms.check_backend(backend)
        self.backend = backend
        # The noise is drawn from rng, a numpy Generator or RandomState (P5.3).
        self.rng = rng
        # The one-sided PSD of the detector noise in strain^2 / Hz, a function of the
        # frequency as colored_noise_from_psd takes it, for example
        # gwd.core.interferometer.reference_psd (P5.4).
        self.noise_psd = noise_psd
        self.arm_length = LIGO_ARM_LENGTH
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

    def add_detector_noise(self, signal, noise_scale=1.0):
        """Return signal plus noise_scale times noise drawn from self.noise_psd.

        noise_scale multiplies the amplitude: 1 is the PSD itself, 2 four times its
        power. Before P5.4 a slider value times 1e-22 went into generate_colored_noise,
        whose standard deviation depended on the length and the sampling rate (CORE-9,
        CORE-12).
        """
        noise = colored_noise_from_psd(
            len(signal), self.sample_rate, self.noise_psd, rng=self.rng
        )
        return signal + noise_scale * noise

    def calculate_response(self, strain):
        """
        Computes the interferometer response: (arm length change, intensity).
        """
        delta_L = strain * self.arm_length
        phase_shift = 4 * np.pi * delta_L / self.laser_wavelength
        intensity = np.sin(phase_shift) ** 2
        return delta_L, intensity
