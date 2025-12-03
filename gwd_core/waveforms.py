"""
Wellenform-Generierung für Gravitationswellen-Simulationen.

Dieses Modul stellt Funktionen bereit, um realistische Gravitationswellen-Signale
von verschmelzenden schwarzen Löchern (Binary Black Hole Mergers) zu generieren.
"""

import numpy as np

try:
    from pycbc.waveform import get_td_waveform
    HAS_PYCBC = True
except ImportError:
    HAS_PYCBC = False
    print("⚠️ PyCBC nicht installiert - verwende Newton-Approximation als Fallback")


def generate_astrophysical_chirp(time_array, t_merger=None, mass1=30, mass2=30, 
                                 distance_mpc=400, f_lower=25.0, normalize=True, 
                                 spin1z=0.0, spin2z=0.0, inclination=0, 
                                 approximant='IMRPhenomD'):
    """
    Generiert eine Gravitationswelle (Chirp) von verschmelzenden schwarzen Löchern.
    
    Diese Funktion nutzt PyCBC für hochpräzise Physik-Simulationen wenn verfügbar,
    fällt aber auf eine Newton-Approximation zurück wenn PyCBC nicht installiert ist.
    
    Args:
        time_array: Zeitvektor der Simulation (z.B. np.linspace(0, 4, 16384))
        t_merger: Zeitpunkt des Mergers in Sekunden. Falls None, wird er kurz vor 
                  Ende gesetzt (90% der Zeit).
        mass1, mass2: Massen der schwarzen Löcher in Sonnenmassen.
        distance_mpc: Entfernung in Megaparsec (wichtig für physikalische Amplitude).
        f_lower: Untere Frequenzgrenze für die Wellenform-Generierung (Hz).
        normalize: Falls True (für ML), wird das Signal auf Amplitude 1.0 skaliert.
                   Falls False (für Physik-Sim), wird der echte Strain ausgegeben.
        spin1z, spin2z: Spin-Parameter entlang der Sichtlinie (-1 bis +1).
        inclination: Neigung des Systems (0 = von oben, π/2 = von der Seite).
        approximant: Wellenform-Approximation (nur für PyCBC).
    
    Returns:
        tuple: (strain, frequency)
            - strain: np.array mit der Amplitude der Gravitationswelle
            - frequency: np.array mit der instantanen Frequenz über die Zeit
    """
    
    # Standard-Merger-Zeit bestimmen, falls nicht gegeben
    if t_merger is None:
        t_merger = time_array[-1] * 0.9  # 90% der Gesamtzeit

    # --- VARIANTE A: PyCBC (Profi-Physik) ---
    if HAS_PYCBC:
        try:
            dt = time_array[1] - time_array[0]
            
            # Wir holen uns Plus- und Cross-Polarisation
            hp, hc = get_td_waveform(
                approximant=approximant,
                mass1=mass1,
                mass2=mass2,
                spin1z=spin1z,
                spin2z=spin2z,
                inclination=inclination,
                delta_t=dt,
                f_lower=f_lower,
                distance=distance_mpc
            )
            
            # Berechnung der instantanen Frequenz aus der Phase des komplexen Signals
            # h(t) = h_plus - i * h_cross
            complex_strain = hp.numpy() - 1j * hc.numpy()
            phase = np.unwrap(np.angle(complex_strain))
            
            # Frequenz ist die Ableitung der Phase nach der Zeit / 2π
            raw_freq = np.abs(np.gradient(phase, dt) / (2 * np.pi))
            
            waveform = hp.numpy()
            
            # Mapping auf unser time_array (Verschieben zum t_merger)
            # 1. Peak finden
            peak_idx = np.argmax(np.abs(waveform))
            
            # 2. Ziel-Index im time_array finden
            target_idx = np.argmin(np.abs(time_array - t_merger))
            
            # 3. Shift berechnen
            shift = target_idx - peak_idx
            
            # Leere Arrays für das Ergebnis
            strain_out = np.zeros_like(time_array)
            freq_out = np.zeros_like(time_array)
            
            # Kopieren mit Index-Check
            start_src = max(0, -shift)
            end_src = min(len(waveform), len(time_array) - shift)
            
            start_dst = max(0, shift)
            end_dst = start_dst + (end_src - start_src)
            
            if end_dst > start_dst:
                strain_out[start_dst:end_dst] = waveform[start_src:end_src]
                freq_out[start_dst:end_dst] = raw_freq[start_src:end_src]
            
            # Tapering am Anfang des Fensters, um Klicks zu vermeiden
            taper_len = min(int(0.05 / dt), start_dst + 50)  # 50ms Taper
            if start_dst < taper_len and start_dst < len(strain_out):
                # Soft fade in
                window = np.linspace(0, 1, taper_len - start_dst)
                strain_out[start_dst:taper_len] *= window

            # Normalisierung für Machine Learning
            if normalize:
                m = np.max(np.abs(strain_out))
                if m > 0:
                    strain_out /= m
                
            return strain_out, freq_out

        except Exception as e:
            print(f"⚠️ PyCBC Fehler: {e} -> Fallback auf Newton.")
    
    # --- VARIANTE B: Newtonsche Physik (Fallback) ---
    # Formeln für die Inspiral-Phase (vor dem Merger)
    
    # Konstanten
    G = 6.674e-11       # Gravitationskonstante
    c = 2.998e8         # Lichtgeschwindigkeit
    Msun = 1.989e30     # Sonnenmasse in kg
    
    # Gesamtmasse und Chirp-Masse
    M = (mass1 + mass2) * Msun
    ch_mass = (mass1 * mass2)**0.6 / (mass1 + mass2)**0.2 * Msun
    dist_m = distance_mpc * 3.086e22  # Megaparsec in Meter
    
    # Zeit bis zum Merger (tau)
    tau = np.clip(t_merger - time_array, 0.0001, None)
    
    # Frequenz-Entwicklung (0. Ordnung Post-Newton)
    # f(tau) ~ tau^(-3/8)
    freq_newton = (5 / (256 * np.pi))**(3/8) * \
                  (G * ch_mass / c**3)**(-5/8) * \
                  tau**(-3/8)
    
    # Amplitude (Strain)
    # h(tau) ~ 1/D * tau^(-1/4)
    amp_prefactor = (4 / dist_m) * \
                    (G * ch_mass / c**2)**(5/4) * \
                    (5 / (c * tau))**(1/4)
    
    # Phase
    # phi(tau) ~ tau^(5/8)
    phase = -2 * (5 * G * ch_mass / c**3)**(-5/8) * tau**(5/8)
    
    strain_out = amp_prefactor * np.cos(phase)
    
    # Alles nach dem Merger abschneiden (Newton kann keinen Ringdown simulieren)
    mask_post_merger = time_array > t_merger
    strain_out[mask_post_merger] = 0
    freq_newton[mask_post_merger] = 0
    
    # Fade-In Taper um abrupten Start zu vermeiden
    strain_out *= (1 - np.exp(-10 * time_array))
    
    if normalize:
        m = np.max(np.abs(strain_out))
        if m > 0:
            strain_out /= m
        
    return strain_out, freq_newton


def generate_realistic_chirp(time_array, mass1=30, mass2=30, distance=400, 
                             spin1z=0.0, spin2z=0.0, inclination=0, 
                             approximant='IMRPhenomD'):
    """
    Erzeugt ein realistisches GW-Signal mit PyCBC (oder Newton-Fallback).
    
    Diese Funktion ist ein vereinfachter Wrapper um generate_astrophysical_chirp
    mit physikalisch korrekten Defaults.
    
    Args:
        time_array: np.array mit Zeitpunkten (z.B. np.linspace(0, 4, 16384))
        mass1, mass2: Massen in Sonnenmassen
        distance: Distanz in Mpc
        spin1z, spin2z: Spin entlang Sichtlinie (-1 bis +1)
        inclination: Neigung (0 = von oben, π/2 = von der Seite)
        approximant: 'IMRPhenomD' (schnell) oder 'SEOBNRv4' (genauer)
    
    Returns:
        tuple: (strain, freq)
            - strain: np.array (nur Amplitude, kein Rauschen)
            - freq: Instantaneous frequency (für Analyse)
    """
    
    if not HAS_PYCBC:
        # Fallback auf Newton-Approximation
        print("⚠️ PyCBC nicht verfügbar - verwende Newton-Approximation")
        return generate_astrophysical_chirp(
            time_array,
            mass1=mass1,
            mass2=mass2,
            distance_mpc=distance,
            spin1z=spin1z,
            spin2z=spin2z,
            inclination=inclination,
            normalize=False
        )
    
    # PyCBC verfügbar - nutze es direkt
    sample_rate = 1.0 / (time_array[1] - time_array[0])
    
    # Erzeuge Waveform
    hp, hc = get_td_waveform(
        approximant=approximant,
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        distance=distance,
        inclination=inclination,
        delta_t=1.0/sample_rate,
        f_lower=20.0
    )
    
    # Kombiniere beide Polarisationen (vereinfacht: optimal orientiert)
    # In Realität: h(t) = F+ * hp + Fx * hc (Antenna Pattern)
    strain_raw = hp.numpy()
    
    # Berechne instantaneous frequency
    complex_strain = hp.numpy() - 1j * hc.numpy()
    phase = np.unwrap(np.angle(complex_strain))
    dt = 1.0 / sample_rate
    freq = np.abs(np.gradient(phase, dt) / (2 * np.pi))
    
    # Zeitliche Ausrichtung: Peak am Ende
    peak_idx = np.argmax(np.abs(strain_raw))
    strain_raw = np.roll(strain_raw, len(strain_raw) - peak_idx - 100)
    freq = np.roll(freq, len(freq) - peak_idx - 100)
    
    # Resize auf gewünschte Länge
    if len(strain_raw) > len(time_array):
        strain_raw = strain_raw[:len(time_array)]
        freq = freq[:len(time_array)]
    elif len(strain_raw) < len(time_array):
        strain_raw = np.pad(strain_raw, (0, len(time_array) - len(strain_raw)))
        freq = np.pad(freq, (0, len(time_array) - len(freq)))
    
    return strain_raw, freq


def generate_random_bbh_parameters():
    """
    Zieht zufällige, aber realistische Parameter aus LIGO O3-Verteilungen.
    
    Diese Funktion generiert Parameter basierend auf den beobachteten
    Verteilungen echter Gravitationswellen-Events:
    - Massen: Log-uniform verteilt (kleinere Massen häufiger)
    - Distanz: Volumen-gewichtet (größere Distanzen häufiger)
    - Spins: Uniform verteilt im beobachteten Bereich
    - Neigung: Isotropisch verteilt
    
    Returns:
        dict: Dictionary mit den Parametern:
            - mass1, mass2: Massen in Sonnenmassen (mass1 >= mass2)
            - distance: Distanz in Mpc
            - spin1z, spin2z: Spin-Parameter
            - inclination: Neigung in Radiant
    """
    # Massen: Log-uniform zwischen 5-100 Sonnenmassen
    m1 = 10 ** np.random.uniform(np.log10(5), np.log10(100))
    m2 = 10 ** np.random.uniform(np.log10(5), np.log10(m1))  # m2 ≤ m1
    
    # Distanz: Uniform in Volumen → D³ Distribution
    # 0-2000 Mpc deckt den Großteil der LIGO-Detektionen ab
    distance = (np.random.uniform(0, 1)**(1/3)) * 2000
    
    # Spins: Uniform zwischen -0.9 und +0.9 (beobachtete Verteilung)
    spin1z = np.random.uniform(-0.9, 0.9)
    spin2z = np.random.uniform(-0.9, 0.9)
    
    # Inclination: Isotropisch → cos(i) uniform
    cos_inclination = np.random.uniform(-1, 1)
    inclination = np.arccos(cos_inclination)
    
    return {
        'mass1': m1,
        'mass2': m2,
        'distance': distance,
        'spin1z': spin1z,
        'spin2z': spin2z,
        'inclination': inclination
    }