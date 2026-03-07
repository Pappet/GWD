import numpy as np

class InterferometerModel:
    """
    Berechnet die theoretische Sensitivität eines GW-Detektors.
    (Logik extrahiert aus Interferometer Designer V2)
    """
    def __init__(self):
        # Standard-Parameter
        self.laser_power = 200        # Watt
        self.mirror_reflectivity = 0.999
        self.arm_length = 4000        # Meter
        self.beamsplitter_loss = 1e-4
        self.finesse = 450
        
        # Features
        self.power_recycling = False
        self.signal_recycling = False
        self.squeezed_light = False
        
        # Konstanten
        self.c = 299792458
        self.h = 6.626e-34
        
        # Frequenzbereich für Berechnungen
        self.freq = np.logspace(1, 3.5, 1000)  # 10 Hz - 3 kHz
        
    def calculate_shot_noise(self):
        wavelength = 1064e-9
        shot_noise = np.sqrt(self.h * self.c * wavelength / (2 * np.pi * self.laser_power * self.arm_length**2))
        
        cavity_gain = 2 * self.finesse / np.pi
        shot_noise = shot_noise / np.sqrt(cavity_gain)
        
        if self.squeezed_light:
            shot_noise = shot_noise / 3
            
        return shot_noise * 1.0 * np.ones_like(self.freq)
    
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
        wavelength = 1064e-9
        mass = 40
        radiation = np.sqrt(8 * h_bar * self.laser_power / 
                          (mass * wavelength * self.c * (2 * np.pi * self.freq)**2))
        return radiation * 1e-3
    
    def calculate_total_sensitivity(self):
        shot = self.calculate_shot_noise()
        thermal = self.calculate_thermal_noise()
        seismic = self.calculate_seismic_noise()
        radiation = self.calculate_radiation_pressure()
        
        loss_factor = 1.0 + (self.beamsplitter_loss * 10) + ((1 - self.mirror_reflectivity) * 50)
        
        if self.power_recycling:
            loss_factor *= 0.6
            
        if self.signal_recycling:
            signal_gain = 1 - 0.5 * np.exp(-((self.freq - 100) / 50)**2)
            loss_factor *= signal_gain
            
        total = np.sqrt(shot**2 + thermal**2 + seismic**2 + radiation**2) * loss_factor
        return total