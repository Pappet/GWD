#!/usr/bin/env python3
"""
Interferometer Designer V2 - GUI Frontend
Nutzt gwd_core für die Physik-Berechnungen.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import sys
import os

# --- IMPORT SETUP ---
# Füge den aktuellen Ordner zum Pfad hinzu, damit Python 'gwd_core' findet
sys.path.append(os.path.dirname(__file__))
from gwd_core.interferometer import InterferometerModel

def main():
    # Instanz der Physik-Engine erstellen
    sim = InterferometerModel()
    
    # Figure erstellen
    fig = plt.figure(figsize=(16, 9))
    fig.canvas.manager.set_window_title('Interferometer Designer')
    
    # Layout
    ax_schema = plt.subplot2grid((3, 3), (0, 0), rowspan=2)
    ax_sensitivity = plt.subplot2grid((3, 3), (0, 1), colspan=2, rowspan=2)
    ax_info = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    ax_info.axis('off')
    
    plt.subplots_adjust(left=0.05, bottom=0.35, right=0.95, top=0.95, hspace=0.3, wspace=0.3)
    
    def draw_schema():
        """Zeichne Interferometer-Schema"""
        ax_schema.clear()
        ax_schema.set_xlim(0, 10)
        ax_schema.set_ylim(0, 10)
        ax_schema.set_aspect('equal')
        ax_schema.axis('off')
        ax_schema.set_title('Interferometer Aufbau', fontweight='bold', fontsize=12)
        
        # Laser
        laser = FancyBboxPatch((1, 4.5), 1, 1, boxstyle="round,pad=0.1",
                               ec='red', fc='lightcoral', lw=2)
        ax_schema.add_patch(laser)
        ax_schema.text(1.5, 5, 'LASER', ha='center', va='center', fontweight='bold', fontsize=9)
        ax_schema.text(1.5, 4.2, f'{sim.laser_power}W', ha='center', fontsize=7)
        
        # Strahl zum BS
        ax_schema.arrow(2, 5, 1.5, 0, head_width=0.2, head_length=0.2, fc='red', ec='red', lw=2)
        
        # Beamsplitter
        bs = Rectangle((3.5, 4.6), 0.6, 0.6, angle=45, ec='blue', fc='lightblue', lw=2)
        ax_schema.add_patch(bs)
        ax_schema.text(3.8, 4.2, 'BS', ha='center', fontweight='bold', fontsize=8)
        
        # X-Arm
        ax_schema.arrow(4.2, 5, 2, 0, head_width=0.15, head_length=0.15, fc='orange', ec='orange', lw=2)
        ax_schema.plot([4.5, 8], [5, 5], 'g--', lw=1, alpha=0.5)
        ax_schema.text(6.25, 5.3, f'X-Arm: {sim.arm_length}m', ha='center', fontsize=8, color='green')
        
        # ETM X
        etm_x = Rectangle((7.8, 4.5), 0.2, 1, ec='purple', fc='lavender', lw=2)
        ax_schema.add_patch(etm_x)
        ax_schema.text(7.9, 4.2, f'R={sim.mirror_reflectivity*100:.3f}%', ha='center', fontsize=6)
        
        # Y-Arm
        ax_schema.arrow(3.8, 5.3, 0, 2, head_width=0.15, head_length=0.15, fc='orange', ec='orange', lw=2)
        ax_schema.plot([3.8, 3.8], [5.5, 9], 'g--', lw=1, alpha=0.5)
        ax_schema.text(4.2, 7.25, f'Y-Arm:\n{sim.arm_length}m', ha='center', fontsize=8, color='green')
        
        # ETM Y
        etm_y = Rectangle((3.3, 8.8), 1, 0.2, ec='purple', fc='lavender', lw=2)
        ax_schema.add_patch(etm_y)
        
        # Photodetector
        pd = Circle((3.2, 5), 0.3, ec='green', fc='lightgreen', lw=2)
        ax_schema.add_patch(pd)
        ax_schema.text(3.2, 5, 'PD', ha='center', va='center', fontweight='bold', fontsize=8)
        
        # Features anzeigen
        y_pos = 1.5
        if sim.power_recycling:
            ax_schema.text(1, y_pos, '⚡ Power Recycling', fontsize=9, bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.7))
            y_pos -= 0.5
        if sim.signal_recycling:
            ax_schema.text(1, y_pos, '📡 Signal Recycling', fontsize=9, bbox=dict(boxstyle='round', fc='lightblue', alpha=0.7))
            y_pos -= 0.5
        if sim.squeezed_light:
            ax_schema.text(1, y_pos, '🌀 Squeezed Light', fontsize=9, bbox=dict(boxstyle='round', fc='plum', alpha=0.7))
    
    def plot_sensitivity():
        """Plotte Sensitivitätskurve mit Daten aus gwd_core"""
        ax_sensitivity.clear()
        
        # HIER: Berechnung durch das Core-Modul
        sensitivity = sim.calculate_total_sensitivity()
        
        ax_sensitivity.loglog(sim.freq, sensitivity, 'b-', lw=3, label='Dein Detektor')
        
        # Referenzlinien (LIGO Design)
        ligo_ref = 1e-23 * (sim.freq / 100)**(-0.5)
        ligo_ref = np.where(sim.freq < 40, ligo_ref * 5, ligo_ref)
        ax_sensitivity.loglog(sim.freq, ligo_ref, 'k--', lw=1.5, alpha=0.5, label='LIGO Design')
        
        ax_sensitivity.set_xlabel('Frequenz (Hz)', fontsize=12, fontweight='bold')
        ax_sensitivity.set_ylabel('Strain Sensitivity [1/√Hz]', fontsize=12, fontweight='bold')
        ax_sensitivity.set_title('Detektor-Empfindlichkeit', fontsize=13, fontweight='bold')
        ax_sensitivity.grid(True, which='both', alpha=0.3, ls=':')
        ax_sensitivity.legend(loc='upper right')
        ax_sensitivity.set_xlim(10, 3000)
        ax_sensitivity.set_ylim(1e-25, 1e-20)
    
    def update_info():
        ax_info.clear()
        ax_info.axis('off')
        info = [
            "═══ AKTUELLE KONFIGURATION ═══",
            f"Laser-Leistung:     {sim.laser_power:.0f} W",
            f"Spiegel-Reflekt.:   {sim.mirror_reflectivity*100:.4f} %",
            f"Arm-Länge:          {sim.arm_length:.0f} m",
            f"BS-Verluste:        {sim.beamsplitter_loss:.1e}",
            f"Finesse:            {sim.finesse:.0f}",
            f"Power Recycling:    {'✓ AN' if sim.power_recycling else '✗ AUS'}",
            f"Signal Recycling:   {'✓ AN' if sim.signal_recycling else '✗ AUS'}",
            f"Squeezed Light:     {'✓ AN' if sim.squeezed_light else '✗ AUS'}",
        ]
        ax_info.text(0.02, 0.95, '\n'.join(info), transform=ax_info.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', fc='wheat', alpha=0.6))

    def update_all(val=None):
        draw_schema()
        plot_sensitivity()
        update_info()
        fig.canvas.draw_idle()
    
    # --- SLIDER SETUP ---
    slider_left = 0.15
    slider_width = 0.3
    
    ax_laser = plt.axes([slider_left, 0.26, slider_width, 0.02])
    slider_laser = Slider(ax_laser, 'Laser (W)', 100, 500, valinit=sim.laser_power, valstep=50)
    
    ax_mirror = plt.axes([slider_left, 0.23, slider_width, 0.02])
    slider_mirror = Slider(ax_mirror, 'Spiegel R', 0.990, 0.99999, valinit=sim.mirror_reflectivity, valfmt='%.5f')
    
    ax_arm = plt.axes([slider_left, 0.20, slider_width, 0.02])
    slider_arm = Slider(ax_arm, 'Arm (m)', 2000, 8000, valinit=sim.arm_length, valstep=500)
    
    ax_bs = plt.axes([slider_left, 0.17, slider_width, 0.02])
    slider_bs = Slider(ax_bs, 'BS Loss', 1e-5, 1e-3, valinit=sim.beamsplitter_loss, valfmt='%.1e')
    
    ax_checks = plt.axes([0.55, 0.14, 0.3, 0.12])
    ax_checks.set_title('Erweiterte Optionen', fontweight='bold')
    checks = CheckButtons(ax_checks, ['Power Recycling', 'Signal Recycling', 'Squeezed Light'], [False, False, False])
    
    # --- CALLBACKS ---
    def update_laser(val):
        sim.laser_power = val
        update_all()
    
    def update_mirror(val):
        sim.mirror_reflectivity = val
        # Automatische Finesse-Anpassung (Physik-Logik im Client für bessere UX)
        if val < 1.0:
            sim.finesse = (np.pi * np.sqrt(val)) / (1 - val)
        update_all()
    
    def update_arm(val):
        sim.arm_length = val
        update_all()
    
    def update_bs(val):
        sim.beamsplitter_loss = val
        update_all()
        
    def update_checks(label):
        if label == 'Power Recycling': sim.power_recycling = not sim.power_recycling
        elif label == 'Signal Recycling': sim.signal_recycling = not sim.signal_recycling
        elif label == 'Squeezed Light': sim.squeezed_light = not sim.squeezed_light
        update_all()
    
    slider_laser.on_changed(update_laser)
    slider_mirror.on_changed(update_mirror)
    slider_arm.on_changed(update_arm)
    slider_bs.on_changed(update_bs)
    checks.on_clicked(update_checks)
    
    update_all()
    plt.show()

if __name__ == "__main__":
    main()