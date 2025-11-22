# Datei: GWD/gwd_core/noise.py
import numpy as np

def generate_gaussian_noise(length, noise_level=1.0):
    """
    Erzeugt einfaches gaußsches (weißes) Rauschen.
    
    Args:
        length: Anzahl der Datenpunkte
        noise_level: Standardabweichung (Amplitude des Rauschens)
    
    Returns:
        Array mit Rauschwerten
    """
    return np.random.normal(loc=0.0, scale=noise_level, size=length)

def generate_colored_noise(length, sample_rate, noise_level=1.0):
    """
    Erzeugt realistischeres, farbiges Rauschen (z.B. seismisch dominiert).
    Simuliert ein 1/f Verhalten bei tiefen Frequenzen basierend auf weißem Rauschen.
    
    Args:
        length: Anzahl der Datenpunkte
        sample_rate: Abtastrate in Hz (wichtig für Frequenzanalyse)
        noise_level: Basis-Amplitude des Rauschens
    """
    # 1. Weißes Rauschen als Basis erzeugen
    white_noise = generate_gaussian_noise(length, noise_level)
    
    # 2. In den Frequenzbereich wechseln (FFT)
    freqs = np.fft.fftfreq(length, 1/sample_rate)
    noise_fft = np.fft.fft(white_noise)
    
    # 3. Spektrale Gewichtung anwenden (Seismisches Rauschen simulieren)
    # Wir verstärken tiefe Frequenzen: Faktor ~ sqrt(1/f)
    mask = np.abs(freqs) > 0 # Division durch Null bei 0 Hz verhindern
    
    # Der Faktor 10 ist empirisch gewählt, um die Charakteristik sichtbar zu machen
    scaling_factor = (10 / np.abs(freqs[mask]))**0.5
    noise_fft[mask] *= scaling_factor
    
    # 4. Zurück in den Zeitbereich (Inverse FFT)
    colored_noise = np.real(np.fft.ifft(noise_fft))
    
    return colored_noise