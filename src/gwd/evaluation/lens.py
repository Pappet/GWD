"""What training changed in a chirp telescope: gwd eval lens.

A chirp telescope (gwd.training.telescope) keeps its start values next to what training
added to them, so a trained model shows what it learned:

- The fitting factor of its lens on signals of its training scenario, with the start
  values and with the trained ones. For a signal without noise, it is the brightest
  pixel of the focal plane divided by the optimal SNR of the signal in the band of the
  lens: the share of that SNR the lens reaches, 1 for a perfect matched filter.
- A plot with four panels: the phase and the amplitude that training added to each
  chirp curve, the gain over the band, and the evidence curve. For a model that
  started from the leading-order formula (--lens-start newton), the first two panels
  also show what PyCBC's waveform differs by from that start: the change that would
  make the lens a matched filter.

The phase of a chirp curve is fixed only up to a constant, a shift in time, and a
stretch, which renames the masses of the channels; none of them changes what the lens
detects. gauge_free() removes them, so that phase curves can be compared. The amplitude
is fixed only up to a common factor, which rescaled() takes out.

TensorFlow is not imported here: the model comes from gwd.registry.load_model, and its
layers are read and called as they are.
"""

import os
from contextlib import contextmanager

import numpy as np
from matplotlib.figure import Figure
from tabulate import tabulate

from gwd.config import EVALUATION_MODELS_PLOTS_DIR, WAVEFORM_BACKEND, ensure_dirs
from gwd.core import lens as optics
from gwd.data.injection import make_sample
from gwd.data.scenarios import training_scenario
from gwd.registry import load_model, model_name, read_metadata, resolve

# The first word of the spawn keys of the signals: 'gwdf' in ASCII, a stream of its own.
SIGNAL_STREAM = 0x67776466
SIGNALS = 200
# A node counts as part of a chirp curve where its start amplitude is above this.
AMPLITUDE_FLOOR = 0.02
# Next to a reference the phase panel is linear up to this many radians and logarithmic
# beyond: the reference reaches tens of radians at the merger, a short run adds tenths.
PHASE_LINEAR = 0.1
# Total masses (solar masses) below and from which the report counts a signal as light
# and as heavy.
LIGHT_BELOW = 40.0
HEAVY_FROM = 80.0


class NotATelescopeError(ValueError):
    """The model has no lens: it is not a chirp telescope."""


def telescope_layers(model):
    """(lens, pooling) of a chirp telescope; NotATelescopeError for any other model."""
    names = {layer.name for layer in model.layers}
    if not {"lens", "pooling"} <= names:
        raise NotATelescopeError(
            "The model is not a chirp telescope (it has no layers 'lens' and "
            "'pooling'); train one with gwd train --arch telescope."
        )
    return model.get_layer("lens"), model.get_layer("pooling")


def plot_path(model_path):
    """The plot of a model: lens_<model name>.png."""
    return os.path.join(
        EVALUATION_MODELS_PLOTS_DIR, f"lens_{model_name(model_path)}.png"
    )


def read_nodes(lens, mother):
    """The nodes of mother curve number mother that at least one channel reads."""
    frame, u = lens.optics_frame, lens.optics_grid.u()
    lightest = min(lens.ladders[mother])
    return (u >= -frame.pre_s / lightest) & (u <= frame.post_s / lightest)


def gauge_free(delta, phase, u, weight):
    """delta without what cannot be seen in a phase curve.

    delta is a change of the phase curve phase over the nodes u. Its best fit by a
    constant, a shift in time (the slope of phase), and a stretch (u times that slope),
    with the weights weight, is subtracted. Nodes of weight 0 take no part in the fit.
    """
    slope = np.gradient(phase, u)
    basis = np.stack([np.ones_like(u), slope, u * slope], axis=1)
    root = np.sqrt(weight)[:, None]
    fit, *_ = np.linalg.lstsq(basis * root, delta * root[:, 0], rcond=None)
    return delta - basis @ fit


def rescaled(amplitude, start, weight):
    """amplitude times the factor that brings it closest to start.

    A common factor on the amplitude of a chirp curve changes nothing the lens detects
    (rho is divided by the noise level of the kernel). The factor is the least-squares
    fit with the weights weight.
    """
    factor = (weight * amplitude * start).sum() / (weight * amplitude**2).sum()
    return factor * amplitude


def curve_changes(lens, reference=None):
    """What training added to each chirp curve, and what a reference differs by.

    One dict per mother curve: mass_ratio; u (the nodes that count: read by a channel,
    and with a start amplitude above AMPLITUDE_FLOOR); phase and amplitude (what
    training added there, the phase gauge_free, the amplitude of the trained curve
    rescaled to the start curve); phase_rms (the root mean square of that phase,
    weighted with the energy of the start curve). With reference = (amplitude, phase)
    of other curves on the same nodes also reference_phase and reference_amplitude
    (what those differ by from the start values, gauge_free and rescaled the same way),
    phase_start_rms, and phase_left_rms (the root mean square of what the start phase
    and the trained phase differ by from the reference, weighted like phase_rms).
    """
    u = lens.optics_grid.u()
    start_amplitude = lens.amplitude_start.numpy().astype(float)
    start_phase = lens.phase_start.numpy().astype(float)
    added_amplitude, added_phase = (
        np.asarray(added, dtype=float) for added in lens.added()
    )
    changes = []
    for mother, mass_ratio in enumerate(lens.mass_ratios):
        counts = read_nodes(lens, mother) & (start_amplitude[mother] > AMPLITUDE_FLOOR)
        weight = np.where(counts, start_amplitude[mother] ** 2 * np.gradient(u), 0.0)

        def rms(values):
            return float(np.sqrt((weight * values**2).sum() / weight.sum()))

        def amplitude_change(amplitude):
            time = np.where(counts, np.gradient(u), 0.0)
            start = start_amplitude[mother]
            return (rescaled(amplitude, start, time) - start)[counts]

        phase = gauge_free(added_phase[mother], start_phase[mother], u, weight)
        change = {
            "mass_ratio": mass_ratio,
            "u": u[counts],
            "phase": phase[counts],
            "amplitude": amplitude_change(
                start_amplitude[mother] + added_amplitude[mother]
            ),
            "phase_rms": rms(phase),
        }
        if reference is not None:
            difference = reference[1][mother] - start_phase[mother]
            difference = gauge_free(difference, start_phase[mother], u, weight)
            change["reference_phase"] = difference[counts]
            change["phase_start_rms"] = rms(difference)
            change["phase_left_rms"] = rms(difference - phase)
            change["reference_amplitude"] = amplitude_change(
                np.asarray(reference[0][mother], dtype=float)
            )
        changes.append(change)
    return changes


@contextmanager
def start_values(lens):
    """Inside the block the lens is as it started: nothing added to the curves, gain 1."""
    trained = [lens.amplitude_delta, lens.phase_delta, lens.gain]
    kept = [variable.numpy() for variable in trained]
    try:
        for variable, value in zip(trained, kept):
            variable.assign(np.zeros_like(value))
        yield
    finally:
        for variable, value in zip(trained, kept):
            variable.assign(value)


def draw_signals(scenario, n, seed=0, backend=WAVEFORM_BACKEND):
    """n signals of scenario without noise: (signals (n, samples), total masses).

    Each is the conditioned signal of one sample (gwd.data.injection.make_sample), as
    it lies in the model's input. Signal i draws from RandomState(MT19937(SeedSequence(
    seed, spawn_key=(SIGNAL_STREAM, i)))).
    """
    signals, masses = [], []
    for index in range(n):
        sequence = np.random.SeedSequence(seed, spawn_key=(SIGNAL_STREAM, index))
        rng = np.random.RandomState(np.random.MT19937(sequence))
        sample = make_sample(scenario, rng, has_signal=True, backend=backend)
        signals.append(np.asarray(sample.signal, dtype=np.float32))
        masses.append(sample.params.mass1 + sample.params.mass2)
    return np.array(signals), np.array(masses)


def fitting_factors(lens, signals, batch=8):
    """The fitting factor of the lens for each signal (see the module doc)."""
    low, high = lens.optics_frame.bins
    spectra = np.fft.rfft(signals.astype(float), axis=1)[:, low:high]
    power = lens.noise_power.numpy().astype(float)
    optimal = np.sqrt(2.0 * (np.abs(spectra) ** 2 / power).sum(axis=1))
    brightest = np.concatenate(
        [
            np.asarray(lens(signals[i : i + batch, :, None])).max(axis=(1, 2))
            for i in range(0, len(signals), batch)
        ]
    )
    return brightest / optimal


def summarize(factors, masses):
    """Median, 10 % quantile, and the medians of the light and the heavy signals."""
    light, heavy = masses < LIGHT_BELOW, masses >= HEAVY_FROM
    return [
        float(np.median(factors)),
        float(np.percentile(factors, 10)),
        float(np.median(factors[light])) if light.any() else float("nan"),
        float(np.median(factors[heavy])) if heavy.any() else float("nan"),
    ]


def plot(lens, pooling, changes, start_evidence, path, title):
    """Write the four panels (see the module doc) to path."""
    fig = Figure(figsize=(13, 9))
    (ax_phase, ax_amplitude), (ax_gain, ax_evidence) = fig.subplots(2, 2)
    reference = "reference_phase" in changes[0]

    for number, change in enumerate(changes):
        color = f"C{number}"
        # Seconds before the peak for the mother's 20 solar masses; after it: dropped.
        before = change["u"] < 0
        seconds = -change["u"][before] * optics.MOTHER_TOTAL_MASS
        label = f"q = {change['mass_ratio']:g}"
        ax_phase.plot(seconds, change["phase"][before], color=color, label=label)
        ax_amplitude.plot(
            seconds, change["amplitude"][before], color=color, label=label
        )
        if reference:
            ax_phase.plot(
                seconds, change["reference_phase"][before], color=color, linestyle="--"
            )
            ax_amplitude.plot(
                seconds,
                change["reference_amplitude"][before],
                color=color,
                linestyle="--",
            )
    note = "\ndashed: PyCBC (IMRPhenomD) minus the start" if reference else ""
    phase_unit = "rad"
    if reference:
        ax_phase.set_yscale("symlog", linthresh=PHASE_LINEAR)
        phase_unit = f"rad, logarithmic beyond {PHASE_LINEAR:g}"
    for ax, heading, label in (
        (ax_phase, "Phase added by training, gauge-free", f"Phase ({phase_unit})"),
        (
            ax_amplitude,
            "Amplitude added by training, up to a common factor",
            "Amplitude (peak of the start curve = 1)",
        ),
    ):
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Time before the peak at 20 solar masses total (s)")
        ax.set_ylabel(label)
        ax.set_title(f"{heading}{note}")
        ax.grid(True, alpha=0.3)
    # Next to a reference the curves leave these corners free.
    ax_phase.legend(
        title="Mass ratio m1 / m2", loc="lower right" if reference else "best"
    )
    ax_amplitude.legend(
        title="Mass ratio m1 / m2", loc="upper left" if reference else "best"
    )

    frequencies = lens.optics_frame.frequencies
    position = np.linspace(0.0, lens.gain_nodes - 1.0, len(frequencies))
    gain = np.interp(position, np.arange(lens.gain_nodes), lens.gain.numpy())
    ax_gain.plot(frequencies, 20.0 * gain / np.log(10.0))
    ax_gain.axhline(0, color="gray", linewidth=0.8)
    ax_gain.set_xlabel("Frequency (Hz)")
    ax_gain.set_ylabel("Gain (dB)")
    ax_gain.set_title("Gain over the band added by training")
    ax_gain.grid(True, alpha=0.3)

    rho = pooling.rho_nodes()
    if start_evidence is not None:
        ax_evidence.plot(rho, start_evidence, color="gray", label="start")
    ax_evidence.plot(rho, pooling.evidence.numpy(), color="C0", label="trained")
    ax_evidence.set_xlabel("rho of a pixel (noise alone: rho^2 is chi-squared, 2 dof)")
    ax_evidence.set_ylabel("Evidence (log likelihood ratio)")
    ax_evidence.set_title("Evidence curve of the pooling")
    ax_evidence.legend()
    ax_evidence.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    ensure_dirs(os.path.dirname(path))
    fig.savefig(path, dpi=150)


def run(model="latest", signals=SIGNALS, seed=0):
    """Report what training changed in a chirp telescope; returns the exit code.

    Args:
        model: 'latest', a model name in models_registry/, or a path to a .keras file.
        signals: Signals drawn from the model's training scenario for the fitting
            factor.
        seed: Seed of those signals.
    """
    try:
        model_path = resolve(model)
    except FileNotFoundError as error:
        print(f"❌ {error}")
        return 1
    loaded = load_model(model_path)
    try:
        lens, pooling = telescope_layers(loaded)
    except NotATelescopeError as error:
        print(f"❌ {model_name(model_path)}: {error}")
        return 1
    metadata = read_metadata(model_path) or {}
    setup = metadata.get("telescope", {})
    scenario = training_scenario(metadata)

    print("=" * 70)
    print(f"🔭 LENS REPORT: {os.path.basename(model_path)}")
    ratios = ", ".join(f"{ratio:g}" for ratio in lens.mass_ratios)
    print(
        f"Chirp telescope: {lens.channels} channels, mass ratios {ratios}; start "
        f"'{setup.get('lens_start', 'unknown')}'; trained: "
        f"{setup.get('trained', 'unknown')}"
    )
    print("=" * 70)

    start_evidence = None
    if scenario is None:
        print(
            "The model names no training scenario, so no signals can be drawn for "
            "the fitting factor."
        )
    else:
        start_evidence = optics.pixel_evidence(pooling.rho_nodes(), scenario.snr_range)
        drawn, masses = draw_signals(scenario, signals, seed)
        with start_values(lens):
            before = fitting_factors(lens, drawn)
        after = fitting_factors(lens, drawn)
        print(
            f"\nFitting factor on {signals} signals of {scenario.name} (seed {seed}): "
            "the share of the optimal SNR in the band that the brightest pixel reaches"
        )
        rows = [
            ["start", *summarize(before, masses)],
            ["trained", *summarize(after, masses)],
        ]
        headers = [
            "Lens",
            "Median",
            "10 % quantile",
            f"Median, total mass < {LIGHT_BELOW:g}",
            f"Median, total mass >= {HEAVY_FROM:g}",
        ]
        print(tabulate(rows, headers=headers, floatfmt=".3f"))

    reference = None
    if setup.get("lens_start") == "newton":
        curves = [
            optics.mother_curve(ratio, lens.optics_grid, backend=WAVEFORM_BACKEND)
            for ratio in lens.mass_ratios
        ]
        reference = ([c[0] for c in curves], [c[1] for c in curves])
    changes = curve_changes(lens, reference)

    print("\nWhat training changed:")
    rows = []
    for change in changes:
        row = [
            f"{change['mass_ratio']:g}",
            change["phase_rms"],
            float(np.abs(change["amplitude"]).max()),
        ]
        if reference is not None:
            row += [change["phase_start_rms"], change["phase_left_rms"]]
        rows.append(row)
    headers = ["Mass ratio", "Phase added (rad, rms)", "Largest amplitude added"]
    if reference is not None:
        headers += [
            "Phase from the start to PyCBC (rad, rms)",
            "Phase left to PyCBC (rad, rms)",
        ]
    print(tabulate(rows, headers=headers, floatfmt=".3f"))
    gain = 20.0 * lens.gain.numpy() / np.log(10.0)
    print(f"Band gain: {gain.min():+.2f} to {gain.max():+.2f} dB")
    if start_evidence is not None:
        moved = np.abs(pooling.evidence.numpy() - start_evidence).max()
        print(f"Evidence curve: moved by at most {moved:.2f}")
    prior = optics.channel_log_prior(lens.ladders)
    moved = np.abs(pooling.bias.numpy() - prior).max()
    print(f"Channel biases: moved by at most {moved:.2f}")

    path = plot_path(model_path)
    plot(lens, pooling, changes, start_evidence, path, model_name(model_path))
    print(f"\nPlot saved: {path}")
    return 0
