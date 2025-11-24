#!/usr/bin/env python3
"""
Gravitationswellen-Detektor Simulator - GUI Frontend
Nutzt gwd_core für die Simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sys
import os

# --- IMPORT SETUP ---
sys.path.append(os.path.dirname(__file__))
from gwd_core.simulation import TimeDomainSimulator

def create_interactive_plot():
    # Instanz der Simulation Engine
    sim = TimeDomainSimulator(duration=0.5, sample_rate=4096)
    
    # Startwerte für die GUI
    gui_params = {
        'mass1': 30,
        'mass2': 30,
        'distance': 400,
        'noise_level': 1.0
    }
    
    # Figure Setup
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Gravitationswellen-Detektor Simulation (GWD Core Powered)', fontsize=14, fontweight='bold')
    
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
    ax_strain = fig.add_subplot(gs[0, :])
    ax_freq = fig.add_subplot(gs[1, :])
    ax_detector = fig.add_subplot(gs[2, :])
    ax_noise = fig.add_subplot(gs[3, :])
    
    # Hilfsfunktion für die Berechnung
    def calculate_model():
        # 1. Chirp berechnen (gwd_core)
        strain, freq = sim.generate_chirp(gui_params['mass1'], gui_params['mass2'], gui_params['distance'])
        # 2. Rauschen hinzufügen (gwd_core)
        strain_noisy = sim.add_detector_noise(strain, gui_params['noise_level'])
        # 3. Antwort berechnen (gwd_core)
        delta_L, intensity = sim.calculate_response(strain_noisy)
        return strain, freq, strain_noisy, delta_L

    # Initiale Daten
    strain, freq, strain_noisy, delta_L = calculate_model()
    
    # Plots initialisieren
    line_strain, = ax_strain.plot(sim.t, strain, 'b-', lw=1.5, label='Reines Signal')
    ax_strain.set_title('Gravitationswellen-Strain')
    ax_strain.grid(True, alpha=0.3)
    ax_strain.legend()
    
    line_freq, = ax_freq.plot(sim.t, freq, 'g-', lw=1.5)
    ax_freq.set_title('Instantane Frequenz')
    ax_freq.grid(True, alpha=0.3)
    
    line_detector, = ax_detector.plot(sim.t, delta_L * 1e15, 'r-', lw=1.5)
    ax_detector.set_ylabel('Längenänderung (fm)')
    ax_detector.set_title('Interferometer Antwort')
    ax_detector.grid(True, alpha=0.3)
    
    line_noise, = ax_noise.plot(sim.t, strain_noisy, 'orange', lw=0.8, alpha=0.7, label='Signal + Rauschen')
    line_clean_ref, = ax_noise.plot(sim.t, strain, 'b-', lw=1.5, alpha=0.5)
    ax_noise.set_title('Detektorsignal mit Rauschen')
    ax_noise.legend()
    
    # --- SLIDER ---
    plt.subplots_adjust(bottom=0.25)
    
    ax_mass1 = plt.axes([0.15, 0.15, 0.65, 0.02])
    ax_mass2 = plt.axes([0.15, 0.12, 0.65, 0.02])
    ax_distance = plt.axes([0.15, 0.09, 0.65, 0.02])
    ax_noise = plt.axes([0.15, 0.06, 0.65, 0.02])
    
    s_mass1 = Slider(ax_mass1, 'Masse 1 (M☉)', 5, 100, valinit=gui_params['mass1'], valstep=1)
    s_mass2 = Slider(ax_mass2, 'Masse 2 (M☉)', 5, 100, valinit=gui_params['mass2'], valstep=1)
    s_dist = Slider(ax_distance, 'Entfernung (Mpc)', 10, 1000, valinit=gui_params['distance'], valstep=10)
    s_noise = Slider(ax_noise, 'Rauschpegel', 0.1, 5, valinit=gui_params['noise_level'], valstep=0.1)
    
    def update(val):
        # Werte aus Slidern holen
        gui_params['mass1'] = s_mass1.val
        gui_params['mass2'] = s_mass2.val
        gui_params['distance'] = s_dist.val
        gui_params['noise_level'] = s_noise.val
        
        # Neu berechnen
        strain, freq, strain_noisy, delta_L = calculate_model()
        
        # Linien updaten
        line_strain.set_ydata(strain)
        line_freq.set_ydata(freq)
        line_detector.set_ydata(delta_L * 1e15)
        line_noise.set_ydata(strain_noisy)
        line_clean_ref.set_ydata(strain)
        
        # Achsen skalieren
        for ax in [ax_strain, ax_freq, ax_detector, ax_noise]:
            ax.relim()
            ax.autoscale_view(scalex=False)
            
        fig.canvas.draw_idle()
    
    s_mass1.on_changed(update)
    s_mass2.on_changed(update)
    s_dist.on_changed(update)
    s_noise.on_changed(update)
    
    # Reset Button
    ax_reset = plt.axes([0.85, 0.10, 0.1, 0.04])
    btn_reset = Button(ax_reset, 'Reset', hovercolor='0.975')
    
    def reset(event):
        s_mass1.reset()
        s_mass2.reset()
        s_dist.reset()
        s_noise.reset()
    btn_reset.on_clicked(reset)
    
    plt.show()

if __name__ == "__main__":
    create_interactive_plot()