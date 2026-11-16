"""BatchNorm recalibration before validation, and the epoch timer (gwd.training.callbacks)."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from gwd.training import callbacks  # noqa: E402
from gwd.training.model import build_model  # noqa: E402


def bn_model():
    tf.keras.utils.set_random_seed(0)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((3,)),
            tf.keras.layers.Dense(4),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss="mse")
    return model


@pytest.mark.parametrize(
    "n, batch_size, expected",
    [
        (10, 4, [(0, 4), (4, 8)]),  # the short last batch is left out
        (8, 4, [(0, 4), (4, 8)]),
        (3, 4, [(0, 3)]),  # fewer than one batch: one batch
    ],
)
def test_recalibration_uses_whole_batches(n, batch_size, expected):
    assert callbacks.recalibration_batches(n, batch_size) == expected


def test_recalibration_sets_the_mean_of_the_batch_statistics():
    model = bn_model()
    X = np.random.default_rng(1).normal(3.0, 2.0, size=(64, 3)).astype("float32")
    weights = [w.numpy().copy() for w in model.trainable_weights]
    assert callbacks.recalibrate_batchnorm(model, X, batch_size=16) == 4
    bn = model.layers[1]
    hidden = model.layers[0](X).numpy()
    batches = hidden.reshape(4, 16, 4)
    np.testing.assert_allclose(
        bn.moving_mean.numpy(), batches.mean(axis=1).mean(0), rtol=1e-5
    )
    np.testing.assert_allclose(
        bn.moving_variance.numpy(), batches.var(axis=1).mean(0), rtol=1e-4
    )
    # Nothing else changes: the weights and the momentum stay.
    for before, after in zip(weights, model.trainable_weights):
        np.testing.assert_array_equal(before, after.numpy())
    assert bn.momentum == pytest.approx(0.99)


def test_recalibration_runs_before_every_validation():
    model = bn_model()
    X = np.random.default_rng(0).normal(size=(16, 3)).astype("float32")
    recalibration = callbacks.BatchNormRecalibration(X * 5 + 7, batch_size=8)
    seen = []
    # Runs after the recalibration and records the moving mean the validation uses.
    probe = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: seen.append(
            model.layers[1].moving_mean.numpy().copy()
        )
    )
    model.fit(
        X, X[:, :1], validation_data=(X, X[:, :1]), epochs=2,
        callbacks=[recalibration, probe], verbose=0,
    )  # fmt: skip
    assert recalibration.calls == 2
    hidden = model.layers[0](X * 5 + 7).numpy()
    np.testing.assert_allclose(seen[-1], hidden.mean(axis=0), rtol=1e-4)


def test_the_cnn_keeps_its_architecture_with_recalibration():
    model = build_model((64, 1))
    count = model.count_params()
    X = np.random.default_rng(0).normal(size=(8, 64, 1)).astype("float32")
    callbacks.recalibrate_batchnorm(model, X, batch_size=4)
    assert model.count_params() == count
    assert len(callbacks.batchnorm_layers(model)) == 4


def test_the_epoch_timer_records_every_epoch():
    model = bn_model()
    X = np.random.default_rng(0).normal(size=(16, 3)).astype("float32")
    timer = callbacks.EpochTimer(probe=lambda: {"drawn": 1})
    model.fit(X, X[:, :1], epochs=2, callbacks=[timer], verbose=0)
    assert len(timer.seconds) == 2 and all(s > 0 for s in timer.seconds)
    assert timer.extra == [{"drawn": 1}, {"drawn": 1}]
