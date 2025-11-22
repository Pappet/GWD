import numpy as np
try:
    from pycbc.waveform import get_td_waveform
    HAS_PYCBC = True
except ImportError:
    HAS_PYCBC = False

def generate_astrophysical_chirp(time_array, t_merger=None, mass1=30, mass2=30, distance_mpc=400, f_lower=25.0, normalize=True):
    """
    Generiert eine Gravitationswelle (Chirp).
    
    Args:
        time_array: Zeitvektor der Simulation
        t_merger: Zeitpunkt des Mergers (in Sekunden). Falls None, wird er kurz vor Ende gesetzt.
        mass1, mass2: Massen der schwarzen Löcher in Sonnenmassen.
        distance_mpc: Entfernung in Megaparsec (wichtig für physikalische Amplitude).
        normalize: Falls True (für ML), wird das Signal auf Amplitude 1.0 skaliert.
                   Falls False (für Physik-Sim), wird der echte Strain (z.B. 1e-21) ausgegeben.
    
    Returns:
        strain (array), frequency (array)
    """
    
    # Standard-Merger-Zeit bestimmen, falls nicht gegeben
    if t_merger is None:
        t_merger = time_array[-1] - 0.1

    # --- VARIANTE A: PyCBC (Profi-Physik) ---
    if HAS_PYCBC:
        try:
            dt = time_array[1] - time_array[0]
            
            # Wir holen uns Plus- und Cross-Polarisation
            hp, hc = get_td_waveform(approximant="IMRPhenomD",
                                     mass1=mass1,
                                     mass2=mass2,
                                     delta_t=dt,
                                     f_lower=f_lower,
                                     distance=distance_mpc)
            
            # Berechnung der instantanen Frequenz aus der Phase des komplexen Signals
            # h(t) = h_plus - i * h_cross
            complex_strain = hp.numpy() - 1j * hc.numpy()
            phase = np.unwrap(np.angle(complex_strain))
            # Frequenz ist die Ableitung der Phase nach der Zeit / 2pi
            # Wir nutzen gradient für die numerische Ableitung
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
            # (Wir schieben die Waveform an die richtige Stelle im Zeitstrahl)
            start_src = max(0, -shift)
            end_src = min(len(waveform), len(time_array) - shift)
            
            start_dst = max(0, shift)
            end_dst = start_dst + (end_src - start_src)
            
            if end_dst > start_dst:
                strain_out[start_dst:end_dst] = waveform[start_src:end_src]
                freq_out[start_dst:end_dst] = raw_freq[start_src:end_src]
            
            # Tapering am Anfang des Fensters, um Klicks zu vermeiden
            taper_len = int(0.05 / dt) # 50ms Taper
            if start_dst < taper_len and start_dst < len(strain_out):
                 # Falls der Chirp ganz am Anfang startet -> Soft fade in
                 strain_out[:taper_len] *= np.linspace(0, 1, taper_len)

            # Normalisierung für Machine Learning
            if normalize:
                m = np.max(np.abs(strain_out))
                if m > 0: strain_out /= m
                
            return strain_out, freq_out

        except Exception as e:
            print(f"⚠️ PyCBC Fehler: {e} -> Fallback auf Newton.")
    
    # --- VARIANTE B: Newtonsche Physik (Fallback) ---
    # Formeln für die Inspiral-Phase
    
    # Konstanten
    G = 6.674e-11
    c = 2.998e8
    Msun = 1.989e30
    M = (mass1 + mass2) * Msun
    ch_mass = (mass1 * mass2)**0.6 / (mass1 + mass2)**0.2 * Msun
    dist_m = distance_mpc * 3.086e22
    
    # Zeit bis zum Merger (tau)
    tau = np.clip(t_merger - time_array, 0.0001, None)
    
    # Frequenz-Entwicklung (0. Ordnung Post-Newton)
    # f(tau) ~ tau^(-3/8)
    freq_newton = (5 / (256 * np.pi))**(3/8) * (G * ch_mass / c**3)**(-5/8) * tau**(-3/8)
    
    # Amplitude (Strain)
    # h(tau) ~ 1/D * tau^(-1/4)
    amp_prefactor = (4 / dist_m) * (G * ch_mass / c**2)**(5/4) * (5 / (c * tau))**(1/4)
    
    # Phase
    # phi(tau) ~ tau^(5/8)
    phase = -2 * (5 * G * ch_mass / c**3)**(-5/8) * tau**(5/8)
    
    strain_out = amp_prefactor * np.cos(phase)
    
    # Alles nach dem Merger abschneiden (Newton kann keinen Ringdown)
    mask_post_merger = time_array > t_merger
    strain_out[mask_post_merger] = 0
    freq_newton[mask_post_merger] = 0
    
    # Fade-In Taper
    strain_out *= (1 - np.exp(-10 * time_array))
    
    if normalize:
        m = np.max(np.abs(strain_out))
        if m > 0: strain_out /= m
        
    return strain_out, freq_newton