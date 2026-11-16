"""Callbacks of gwd train: BatchNorm recalibration before validation, and an epoch timer.

Why the validation accuracy sat at 0.5. Every BatchNormalization layer of the CNN
normalizes with the statistics of the current batch in training, but with moving
averages of them in inference, so also on the validation data: moving = momentum *
moving + (1 - momentum) * batch, with momentum 0.99, so the averages span roughly the
last 100 steps and start at mean 0 and variance 1. While the weights change quickly,
the averages lag behind them. Measured on the laptop with an online TRAIN_V3 run (8
epochs of 64 steps, seed 1): the training accuracy on fresh samples was 0.77 after the
first epoch and 0.86 after the eighth, while the validation accuracy stayed at 0.50 in
epochs 1 to 6 and reached 0.52 in epoch 7. The model of epoch 7 scored 0.52 on 1024
validation samples in inference mode and 0.82 with batch statistics. Its moving
averages, recomputed from 1024 training samples with its own weights, gave 0.83 in
inference mode. So the moving averages, not the weights, made the validation
predictions useless, and early stopping, the checkpoint, and the learning rate schedule
watched a val_loss that said little about the weights. Waiting a fixed number of steps
(EarlyStopping(start_from_epoch=...)) would not cure this: after 5 epochs (320 steps,
when the start values weigh 0.99**320 = 4 %) the validation accuracy was still 0.50.

The fix (precise BatchNorm, Wu and Johnson 2021, "Rethinking 'Batch' in BatchNorm"):
BatchNormRecalibration recomputes the moving averages before every validation, from a
fixed set of BN_RECALIBRATION_SAMPLES training samples (gwd.config) with the current
weights. The batches run through the model in training mode with momentum k / (k + 1)
for batch k, so each moving average becomes the plain mean of the batch statistics;
no weight changes, and the labels are not used. The validation, the checkpoint (which
saves these averages with the weights), early stopping, and the learning rate schedule
then see the model as it would be used. The architecture is unchanged.
"""

import time

from tensorflow.keras import callbacks, layers

from gwd.config import BATCH_SIZE


def batchnorm_layers(model):
    return [
        layer for layer in model.layers if isinstance(layer, layers.BatchNormalization)
    ]


def recalibration_batches(n, batch_size=BATCH_SIZE):
    """The (start, stop) of the recalibration batches of n samples: whole batches only.

    A short last batch would weigh as much as a full one, and one sample has no
    variance; fewer samples than one batch form a single batch.
    """
    if n <= batch_size:
        return [(0, n)]
    return [
        (start, start + batch_size)
        for start in range(0, n - batch_size + 1, batch_size)
    ]


def recalibrate_batchnorm(model, X, batch_size=BATCH_SIZE):
    """Set the moving mean and variance of every BatchNormalization layer from X.

    Each moving average becomes the mean of its batch statistics over the batches of
    X, with the model's current weights (see the module doc). The weights and the
    momentum stay as they were. Returns the number of batches used.
    """
    bns = batchnorm_layers(model)
    if not bns:
        return 0
    momenta = [layer.momentum for layer in bns]
    batches = recalibration_batches(len(X), batch_size)
    try:
        for k, (start, stop) in enumerate(batches):
            for layer in bns:
                layer.momentum = k / (k + 1)
            model(X[start:stop], training=True)
    finally:
        for layer, momentum in zip(bns, momenta):
            layer.momentum = momentum
    return len(batches)


class BatchNormRecalibration(callbacks.Callback):
    """Recompute the BatchNorm moving averages from X before every validation."""

    def __init__(self, X, batch_size=BATCH_SIZE):
        super().__init__()
        self.X = X
        self.batch_size = batch_size
        self.calls = 0

    def on_test_begin(self, logs=None):
        recalibrate_batchnorm(self.model, self.X, self.batch_size)
        self.calls += 1


class EpochTimer(callbacks.Callback):
    """Records the seconds of every epoch (validation included) in seconds.

    probe, if given, is called at the end of every epoch and returns a dict; the dicts
    are kept in extra (online training: samples drawn and time waited for them).
    """

    def __init__(self, probe=None):
        super().__init__()
        self.probe = probe
        self.seconds = []
        self.extra = []
        self._began = None

    def on_epoch_begin(self, epoch, logs=None):
        self._began = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.seconds.append(time.perf_counter() - self._began)
        if self.probe is not None:
            self.extra.append(self.probe())
