"""The chirp telescope: its lens focuses a chirp, and the model trains, saves, and loads.

The tests build a small telescope on a narrow mass range (a handful of channels); the
full one of TRAIN_V3 is trained in tests/training/test_train.py (slow).
"""

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytest.importorskip("gwpy")
pytest.importorskip("pycbc")

from gwd import registry  # noqa: E402
from gwd.config import N_SAMPLES, SAMPLE_RATE  # noqa: E402
from gwd.core import lens as optics  # noqa: E402
from gwd.data.injection import make_sample  # noqa: E402
from gwd.data.preprocessing import normalize  # noqa: E402
from gwd.data.scenarios import APP_V3, TRAIN_V1, TRAIN_V3  # noqa: E402
from gwd.training import telescope  # noqa: E402

# 30 to 40 solar masses each: total masses 60 to 80 at mass ratio 1.
SMALL = replace(TRAIN_V3, name="SMALL_V3", masses=(30.0, 40.0))


def build(**options):
    defaults = dict(seed=3, mass_ratios=(1.0,), noise_samples=48)
    return telescope.build_telescope(SMALL, **{**defaults, **options})


@pytest.fixture(scope="module")
def model():
    return build()


@pytest.fixture(scope="module")
def lens(model):
    return model.get_layer("lens")


def rng(index):
    return np.random.RandomState(1000 + index)


def test_shapes_and_the_channels_of_the_small_range(model, lens):
    assert model.input_shape == (None, N_SAMPLES, 1)
    assert model.output_shape == (None, 1)
    (masses,) = lens.ladders
    assert masses[0] == 60.0 and masses[-1] < 80.0 and 3 < len(masses) < 40
    assert lens.channels == len(masses)
    # Merger times 2.0 to 3.6 s of the scenario, at 1024 Hz.
    assert lens.compute_output_shape((None, N_SAMPLES, 1)) == (None, len(masses), 1640)


def test_the_parameters_are_curves_gain_evidence_and_biases(model, lens):
    nodes = lens.optics_grid.nodes
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    pooling = telescope.EVIDENCE_NODES + lens.channels
    # One mother curve: the knots of what training adds to its amplitude and phase.
    assert lens.curve_knots == telescope.CURVE_KNOTS == 64
    assert trainable == 2 * lens.curve_knots + telescope.GAIN_NODES + pooling
    # Fixed: the start curves, and the noise power and whitening per bin of the band.
    bins = optics.Frame(SAMPLE_RATE, N_SAMPLES, 2.0, 0.4, (24.0, 300.0)).bins
    assert model.count_params() == trainable + 2 * nodes + 2 * (bins[1] - bins[0])


def test_what_training_adds_is_interpolated_between_its_knots():
    lens = build(noise_samples=2).get_layer("lens")
    nodes, knots = lens.optics_grid.nodes, lens.curve_knots
    amplitude, phase = (np.asarray(curve) for curve in lens.added())
    assert amplitude.shape == phase.shape == (1, nodes)
    assert not amplitude.any() and not phase.any()
    # One knot alone is a triangle over the nodes; the knots spread evenly over them.
    lens.phase_delta.assign(np.eye(knots, dtype=np.float32)[None, 10])
    lens.amplitude_delta.assign(np.linspace(0.0, 1.0, knots, dtype=np.float32)[None])
    amplitude, phase = (np.asarray(curve)[0] for curve in lens.added())
    spacing = (nodes - 1) / (knots - 1)
    triangle = np.clip(1 - np.abs(np.arange(nodes) / spacing - 10), 0, None)
    assert phase == pytest.approx(triangle, abs=1e-5)
    # A straight line over the knots is the same line over the nodes.
    assert amplitude == pytest.approx(np.linspace(0.0, 1.0, nodes), abs=1e-5)


@pytest.mark.parametrize(
    "trainable, expected",
    [
        ("pooling", {"gain", "evidence", "bias"}),
        ("lens", {"amplitude_delta", "phase_delta"}),
    ],
)
def test_trainable_selects_the_weights(trainable, expected):
    built = build(trainable=trainable, noise_samples=2)
    assert {w.name for w in built.trainable_weights} == expected


def test_the_layers_compute_in_float32_under_every_policy(model):
    assert tf.keras.mixed_precision.global_policy().name == "float32"
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        mixed = build(noise_samples=2)
    finally:
        tf.keras.mixed_precision.set_global_policy("float32")
    for layer in mixed.layers[1:]:
        assert layer.dtype_policy.compute_dtype == "float32"
    assert mixed.predict(np.zeros((1, N_SAMPLES, 1), np.float32), verbose=0).dtype == (
        np.float32
    )


def test_the_kernels_are_those_of_the_numpy_reference(lens):
    amplitude, phase = lens.amplitude_start.numpy(), lens.phase_start.numpy()
    reference = optics.kernel_spectra(
        amplitude[0], phase[0], lens.optics_grid, lens.ladders[0], lens.optics_frame
    )
    kernels = lens.kernels().numpy()
    assert kernels.shape == reference.shape
    assert np.abs(kernels - reference).max() < 1e-3 * np.abs(reference).max()


def test_noise_gives_every_pixel_two_degrees_of_freedom(lens):
    noise = np.array(
        [
            normalize(make_sample(SMALL, rng(i), has_signal=False).data)
            for i in range(12)
        ]
    )
    rho = lens(noise[:, :, None].astype(np.float32)).numpy()
    assert rho.shape == (12, lens.channels, 1640)
    # rho^2 is chi-squared with 2 degrees of freedom: its mean is 2.
    assert (rho**2).mean() == pytest.approx(2.0, rel=0.1)


def test_a_signal_becomes_one_bright_pixel_at_its_merger_time(lens):
    sample = make_sample(SMALL, rng(50), has_signal=True, snr=12.0)
    signal = np.asarray(sample.signal, dtype=np.float32)
    rho = lens(signal[None, :, None]).numpy()[0]
    channel, pixel = np.unravel_index(rho.argmax(), rho.shape)
    first, _ = lens.optics_frame.window(*lens.window)
    assert (first + pixel) / 1024 == pytest.approx(sample.params.t_merger, abs=0.005)
    total = sample.params.mass1 + sample.params.mass2
    assert lens.ladders[0][channel] == pytest.approx(total, rel=0.1)
    # The brightest pixel holds nearly all of the signal's SNR in the band.
    low, high = lens.optics_frame.bins
    spectrum = np.fft.rfft(signal.astype(float))[low:high]
    optimal = np.sqrt(2 * (np.abs(spectrum) ** 2 / lens.noise_power.numpy()).sum())
    assert 0.9 < rho.max() / optimal < 1.03
    # Half a second before the merger the focal plane is dark in every channel.
    assert rho[:, pixel - 512].max() < 0.1 * rho.max()


def test_the_score_separates_a_clear_signal_from_noise(model):
    loud = make_sample(SMALL, rng(60), has_signal=True, snr=12.0)
    quiet = make_sample(SMALL, rng(61), has_signal=False)
    x = np.array([normalize(loud.data), normalize(quiet.data)], dtype=np.float32)
    scores = model.predict(x[:, :, None], verbose=0).ravel()
    assert scores[0] > 0.99 and scores[1] < 0.5


def test_the_newton_start_draws_no_pycbc_waveform(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("mother_curve must not be called for start='newton'")

    monkeypatch.setattr(optics, "mother_curve", refuse)
    built = build(start="newton", noise_samples=2)
    lens = built.get_layer("lens")
    amplitude, phase = optics.newtonian_curve(1.0, lens.optics_grid)
    assert lens.phase_start.numpy()[0] == pytest.approx(phase, rel=1e-5, abs=1e-3)
    assert lens.amplitude_start.numpy()[0] == pytest.approx(amplitude, abs=1e-6)
    with pytest.raises(ValueError, match="start must be one of"):
        build(start="taylor")


def test_training_changes_the_added_weights_and_keeps_the_start_values():
    built = build(learning_rate=1e-2, noise_samples=8)
    before = {w.name: w.numpy().copy() for w in built.weights}
    samples = [make_sample(SMALL, rng(70 + i), has_signal=i % 2 == 0) for i in range(8)]
    x = np.array([normalize(s.data) for s in samples], dtype=np.float32)[:, :, None]
    y = np.array([float(s.has_signal) for s in samples])
    history = built.fit(x, y, batch_size=4, epochs=2, verbose=0)
    assert np.all(np.isfinite(history.history["loss"]))
    after = {w.name: w.numpy() for w in built.weights}
    assert all(np.all(np.isfinite(value)) for value in after.values())
    fixed = ("amplitude_start", "phase_start", "noise_power", "whitening")
    for name in fixed:
        assert np.array_equal(after[name], before[name])
    for name in ("phase_delta", "amplitude_delta", "gain", "evidence", "bias"):
        assert not np.array_equal(after[name], before[name])


def test_a_saved_telescope_loads_through_the_registry(model, tmp_path):
    path = str(tmp_path / "gwd_model_20260101-000000.keras")
    model.save(path)
    x = np.array(
        [normalize(make_sample(SMALL, rng(80 + i)).data) for i in range(3)],
        dtype=np.float32,
    )[:, :, None]
    expected = model.predict(x, verbose=0)
    loaded = registry.load_model(path)
    again = loaded.predict(x, verbose=0)
    assert again.ravel().tolist() == pytest.approx(expected.ravel().tolist(), abs=1e-6)
    assert loaded.get_layer("lens").ladders == model.get_layer("lens").ladders
    # A fresh process that never imported gwd.training.telescope loads it too.
    np.save(tmp_path / "x.npy", x)
    code = (
        "import sys, numpy as np\n"
        "from gwd import registry\n"
        "assert 'gwd.training.telescope' not in sys.modules\n"
        f"model = registry.load_model({path!r})\n"
        f"scores = model.predict(np.load({str(tmp_path / 'x.npy')!r}), verbose=0)\n"
        "print(' '.join(repr(float(s)) for s in scores.ravel()))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    scores = [float(text) for text in result.stdout.split()]
    assert scores == pytest.approx(expected.ravel().tolist(), abs=1e-6)


@pytest.mark.parametrize(
    "scenario", [None, TRAIN_V1, APP_V3], ids=lambda s: s.name if s else "no scenario"
)
def test_scenarios_without_a_known_psd_or_a_mass_range_are_refused(scenario):
    with pytest.raises(ValueError, match="chirp telescope needs"):
        telescope.build_telescope(scenario)


def test_mass_ratios_outside_the_scenario_are_refused():
    assert telescope.total_mass_range(TRAIN_V3, 1.0) == (10.0, 200.0)
    assert telescope.total_mass_range(TRAIN_V3, 4.0) == (25.0, 125.0)
    with pytest.raises(ValueError, match="between 1 and 20"):
        telescope.total_mass_range(TRAIN_V3, 25.0)
    with pytest.raises(ValueError, match="trainable must be one of"):
        telescope.build_telescope(SMALL, trainable="none")
