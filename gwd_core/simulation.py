import numpy as np
from .noise import generate_colored_noise
from .waveforms import generate_astrophysical_chirp # Jetzt importieren wir das Super-Hirn

class TimeDomainSimulator:
    """
    Simuliert GW-Signale und Detektorantwort im Zeitbereich.
    Nutzt zentralisierte Physik-Module.
    """
    def __init__(self, duration=0.5, sample_rate=4096):
        self.c = 299792458
        self.arm_length = 4000
        self.laser_wavelength = 1064e-9
        
        self.duration = duration
        self.sample_rate = sample_rate
        self.t = np.linspace(0, duration, int(duration * sample_rate))
        
    def generate_chirp(self, mass1, mass2, distance_mpc):
        """
        Delegiert die Berechnung an waveforms.py.
        WICHTIG: normalize=False, damit wir physikalische Einheiten (Strain) bekommen!
        """
        strain, freq = generate_astrophysical_chirp(
            time_array=self.t,
            t_merger=self.duration * 0.8, # Merger bei 80% der Zeit
            mass1=mass1,
            mass2=mass2,
            distance_mpc=distance_mpc,
            normalize=False # Wir wollen echte Physik für den Simulator
        )
        return strain, freq
    
    def add_detector_noise(self, signal, noise_level_slider):
        """
        Fügt realistisches Rauschen hinzu.
        """
        # Slider (0.1 - 5.0) wird auf physikalische Rauschamplitude (~1e-22) skaliert
        phys_noise_level = noise_level_slider * 1e-22
        
        # Farbiges Rauschen aus noise.py
        noise = generate_colored_noise(len(signal), self.sample_rate, phys_noise_level)
        
        return signal + noise
    
    def calculate_response(self, strain):
        """
        Berechnet Interferometer-Antwort.
        """
        delta_L = strain * self.arm_length
        phase_shift = 4 * np.pi * delta_L / self.laser_wavelength
        intensity = np.sin(phase_shift)**2
        return delta_L, intensity