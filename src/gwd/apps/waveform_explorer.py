#!/usr/bin/env python3
"""
Waveform explorer: GUI for a chirp, its frequency, the detector response, and noise.
The physics comes from gwd.core.simulation.TimeDomainSimulator.

build_figure() builds the figure and its widgets without showing it, so tests can move
the sliders under Agg. main() shows it.
"""

from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from gwd.apps._backend import require_interactive_backend
from gwd.config import SAMPLE_RATE
from gwd.core.simulation import TimeDomainSimulator

# Start values of the sliders
START_PARAMS = {"mass1": 30, "mass2": 30, "distance": 400, "noise_level": 1.0}


def build_figure():
    """Return (fig, state): the figure with plots, sliders, and button, not shown.

    state holds the simulator, the current parameters, the four plot axes, their lines,
    the sliders, and the reset button.
    """
    sim = TimeDomainSimulator(duration=0.5, sample_rate=SAMPLE_RATE)
    gui_params = dict(START_PARAMS)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Gravitational Wave Detector Simulation (gwd.core)",
        fontsize=14,
        fontweight="bold",
    )

    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
    ax_strain = fig.add_subplot(gs[0, :])
    ax_freq = fig.add_subplot(gs[1, :])
    ax_detector = fig.add_subplot(gs[2, :])
    ax_noise = fig.add_subplot(gs[3, :])

    # The whole model for the current parameters
    def calculate_model():
        # 1. Compute the chirp (gwd.core)
        strain, freq = sim.generate_chirp(
            gui_params["mass1"], gui_params["mass2"], gui_params["distance"]
        )
        # 2. Add noise (gwd.core)
        strain_noisy = sim.add_detector_noise(strain, gui_params["noise_level"])
        # 3. Compute the response (gwd.core)
        delta_L, _ = sim.calculate_response(strain_noisy)
        return strain, freq, strain_noisy, delta_L

    strain, freq, strain_noisy, delta_L = calculate_model()

    (line_strain,) = ax_strain.plot(sim.t, strain, "b-", lw=1.5, label="Clean signal")
    ax_strain.set_title("Gravitational wave strain")
    ax_strain.grid(True, alpha=0.3)
    ax_strain.legend()

    (line_freq,) = ax_freq.plot(sim.t, freq, "g-", lw=1.5)
    ax_freq.set_title("Instantaneous frequency")
    ax_freq.grid(True, alpha=0.3)

    (line_detector,) = ax_detector.plot(sim.t, delta_L * 1e15, "r-", lw=1.5)
    ax_detector.set_ylabel("Arm length change (fm)")
    ax_detector.set_title("Interferometer response")
    ax_detector.grid(True, alpha=0.3)

    (line_noise,) = ax_noise.plot(
        sim.t, strain_noisy, "orange", lw=0.8, alpha=0.7, label="Signal + noise"
    )
    (line_clean_ref,) = ax_noise.plot(sim.t, strain, "b-", lw=1.5, alpha=0.5)
    ax_noise.set_title("Detector signal with noise")
    ax_noise.legend()

    # --- Sliders ---
    fig.subplots_adjust(bottom=0.25)

    ax_mass1 = fig.add_axes([0.15, 0.15, 0.65, 0.02])
    ax_mass2 = fig.add_axes([0.15, 0.12, 0.65, 0.02])
    ax_distance = fig.add_axes([0.15, 0.09, 0.65, 0.02])
    # Not ax_noise: that name hid the noise plot, which never rescaled (APP-11).
    ax_s_noise = fig.add_axes([0.15, 0.06, 0.65, 0.02])

    s_mass1 = Slider(
        ax_mass1, "Mass 1 (M☉)", 5, 100, valinit=gui_params["mass1"], valstep=1
    )
    s_mass2 = Slider(
        ax_mass2, "Mass 2 (M☉)", 5, 100, valinit=gui_params["mass2"], valstep=1
    )
    s_dist = Slider(
        ax_distance,
        "Distance (Mpc)",
        10,
        1000,
        valinit=gui_params["distance"],
        valstep=10,
    )
    s_noise = Slider(
        ax_s_noise,
        "Noise level",
        0.1,
        5,
        valinit=gui_params["noise_level"],
        valstep=0.1,
    )

    def update(val):
        gui_params["mass1"] = s_mass1.val
        gui_params["mass2"] = s_mass2.val
        gui_params["distance"] = s_dist.val
        gui_params["noise_level"] = s_noise.val

        strain, freq, strain_noisy, delta_L = calculate_model()

        line_strain.set_ydata(strain)
        line_freq.set_ydata(freq)
        line_detector.set_ydata(delta_L * 1e15)
        line_noise.set_ydata(strain_noisy)
        line_clean_ref.set_ydata(strain)

        # Rescale y only; the time axis stays.
        for ax in [ax_strain, ax_freq, ax_detector, ax_noise]:
            ax.relim()
            ax.autoscale_view(scalex=False)

        fig.canvas.draw_idle()

    s_mass1.on_changed(update)
    s_mass2.on_changed(update)
    s_dist.on_changed(update)
    s_noise.on_changed(update)

    ax_reset = fig.add_axes([0.85, 0.10, 0.1, 0.04])
    btn_reset = Button(ax_reset, "Reset", hovercolor="0.975")

    def reset(event):
        s_mass1.reset()
        s_mass2.reset()
        s_dist.reset()
        s_noise.reset()

    btn_reset.on_clicked(reset)

    state = SimpleNamespace(
        sim=sim,
        params=gui_params,
        ax_strain=ax_strain,
        ax_freq=ax_freq,
        ax_detector=ax_detector,
        ax_noise=ax_noise,
        line_strain=line_strain,
        line_freq=line_freq,
        line_detector=line_detector,
        line_noise=line_noise,
        line_clean_ref=line_clean_ref,
        s_mass1=s_mass1,
        s_mass2=s_mass2,
        s_dist=s_dist,
        s_noise=s_noise,
        btn_reset=btn_reset,
    )
    return fig, state


def main():
    require_interactive_backend("Waveform explorer")
    # The widgets only react while fig and state are referenced.
    fig, state = build_figure()
    plt.show()


if __name__ == "__main__":
    main()
