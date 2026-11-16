"""Samples drawn on the fly for online training (gwd train --online).

A file dataset is drawn once and then shown to the CNN in every epoch, so a model can
learn its noise realizations by heart: the first V2 model reached 100 % training and
75 % validation accuracy on 10 000 fixed samples. Online training draws fresh noise and
signals for every training batch from the scenario instead, through the one injection
path (gwd.data.injection.make_sample), so no sample is seen twice.

Every sample is a pure function of (seed, part, epoch, index). It draws from its own
generator, np.random.RandomState(MT19937(SeedSequence(seed, spawn_key=(ONLINE_STREAM,
part, epoch, index)))), where part is 0 for train, 1 for val, 2 for test, and 3 for
bn (PARTS), epoch is the training epoch (0 for the fixed sets, which are drawn once),
and index is the position of the sample in its epoch or set. So the samples do not
depend on the number of workers, the chunk size, or the batch size, and the parts never
share a stream: val and test are fixed, seeded sets, disjoint from every training
batch. bn is a fixed set from the training distribution that gwd train recomputes the
BatchNorm statistics from before every validation (gwd.training.callbacks).
Sample index i carries a signal if i is even, so every epoch and set is balanced.

The streams are disjoint from the other seeded streams of the package as well: the
test sets of gwd.evaluation.testsets use spawn keys that start with 0x67776474
('gwdt'), these start with ONLINE_STREAM = 0x6777646f ('gwdo'), and gwd train seeds
TensorFlow and the split with the plain seed.

SampleSource draws in a pool of worker processes (spawned, not forked, because the
training process runs TensorFlow's threads): draw() fills a fixed set, batches()
yields training batches forever and keeps a bounded number of batches in flight. Each
sample is z-scored (gwd.data.preprocessing.normalize, as gwd train does with a file
sample) and stored as float32. Only numpy, scipy, and the package are used here, so the
workers import no TensorFlow.

Scenarios with real noise (TRAIN_V1) are refused: their samples would reuse the same
real noise chunks in training, validation, and test, which is the leak that P5.6
groups the file split against.
"""

import multiprocessing
import os
import time
from collections import deque
from concurrent.futures import CancelledError, Future, ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from gwd.config import N_SAMPLES, ONLINE_CHUNK, WAVEFORM_BACKEND
from gwd.data.injection import make_sample
from gwd.data.preprocessing import normalize

# The first word of every spawn key: 'gwdo' in ASCII.
ONLINE_STREAM = 0x6777646F
PARTS = ("train", "val", "test", "bn")
# How the streams are keyed, for the model's metadata.
STREAM_TEXT = (
    "RandomState(MT19937(SeedSequence(seed, spawn_key=(0x6777646f, part, epoch, "
    "index)))), part 0 train, 1 val, 2 test, 3 bn; signal for even index"
)


def sample_rng(seed, part, epoch, index):
    """The RandomState of one sample; see the module doc."""
    key = (ONLINE_STREAM, PARTS.index(part), epoch, index)
    sequence = np.random.SeedSequence(seed, spawn_key=key)
    return np.random.RandomState(np.random.MT19937(sequence))


def check_scenario(scenario):
    """Raise ValueError for a scenario that cannot be drawn on the fly (real noise)."""
    if scenario.real_noise_fraction > 0:
        raise ValueError(
            f"{scenario.name} uses real noise chunks, which online training would "
            "reuse in training, validation, and test; train on a dataset with --data, "
            "or choose a scenario without real noise (train_v2, train_v3)."
        )


@dataclass(frozen=True)
class Chunk:
    """Samples start..stop-1 of one part and epoch, as a worker returns them."""

    X: np.ndarray  # float32, (stop - start, N_SAMPLES), each sample z-scored
    y: np.ndarray  # float32, 1 signal, 0 noise only
    optimal_snr: np.ndarray  # float64, 0 without signal
    pid: int  # the process that drew them
    seconds: float  # how long drawing took there


def draw_chunk(scenario, seed, part, epoch, start, stop, backend=WAVEFORM_BACKEND):
    """Draw samples start..stop-1 of part and epoch (runs in a worker process)."""
    began = time.perf_counter()
    n = stop - start
    X = np.empty((n, N_SAMPLES), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    optimal = np.zeros(n)
    for row, index in enumerate(range(start, stop)):
        sample = make_sample(
            scenario,
            sample_rng(seed, part, epoch, index),
            has_signal=index % 2 == 0,
            backend=backend,
        )
        X[row] = normalize(sample.data)
        y[row] = float(sample.has_signal)
        optimal[row] = sample.optimal_snr
    return Chunk(X, y, optimal, os.getpid(), time.perf_counter() - began)


def _done(value):
    future = Future()
    future.set_result(value)
    return future


class SampleSource:
    """Draws samples of scenario for seed in worker processes; see the module doc.

    workers: number of worker processes; 0 draws in the calling process (for tests).
    chunk: samples per task. prefetch: training batches kept in flight by batches();
    None keeps about two tasks per worker queued. Use it as a context manager, or call
    close(), so the workers end.
    """

    def __init__(
        self,
        scenario,
        seed,
        workers,
        chunk=ONLINE_CHUNK,
        prefetch=None,
        backend=WAVEFORM_BACKEND,
    ):
        check_scenario(scenario)
        if workers < 0:
            raise ValueError(f"workers must not be negative, got {workers}")
        if chunk < 1:
            raise ValueError(f"chunk must be at least 1, got {chunk}")
        self.scenario = scenario
        self.seed = seed
        self.workers = workers
        self.chunk = chunk
        self.prefetch = prefetch
        self.backend = backend
        self._executor = None
        self._closed = False
        if workers:
            self._executor = ProcessPoolExecutor(
                workers, mp_context=multiprocessing.get_context("spawn")
            )
        # Statistics: samples drawn, seconds the workers spent on them, seconds the
        # consumer of batches() waited for data, and the worker processes seen.
        self.samples = 0
        self.worker_seconds = 0.0
        self.wait_seconds = 0.0
        self.pids = set()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        """End the worker processes; queued tasks are cancelled."""
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def _submit(self, part, epoch, start, stop):
        args = (self.scenario, self.seed, part, epoch, start, stop, self.backend)
        if self._closed:
            raise RuntimeError("the SampleSource is closed")
        if self._executor is None:
            return _done(draw_chunk(*args))
        return self._executor.submit(draw_chunk, *args)

    def _tasks(self, part, epoch, start, stop):
        return [
            self._submit(part, epoch, low, min(low + self.chunk, stop))
            for low in range(start, stop, self.chunk)
        ]

    def _collect(self, futures):
        """(X, y, optimal SNR) of the chunks in order, with the statistics updated."""
        chunks = []
        for future in futures:
            began = time.perf_counter()
            chunk = future.result()
            self.wait_seconds += time.perf_counter() - began
            self.samples += len(chunk.y)
            self.worker_seconds += chunk.seconds
            self.pids.add(chunk.pid)
            chunks.append(chunk)
        X = np.concatenate([c.X for c in chunks])
        y = np.concatenate([c.y for c in chunks])
        optimal = np.concatenate([c.optimal_snr for c in chunks])
        return X.reshape(len(X), N_SAMPLES, 1), y, optimal

    def draw(self, part, n, epoch=0):
        """Samples 0..n-1 of part and epoch: (X (n, N_SAMPLES, 1), y, optimal SNR)."""
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        return self._collect(self._tasks(part, epoch, 0, n))

    def batch_samples(self, batch_size, steps_per_epoch, number):
        """(epoch, start, stop): which training samples training batch number holds."""
        epoch, step = divmod(number, steps_per_epoch)
        return epoch, step * batch_size, (step + 1) * batch_size

    def batches(self, batch_size, steps_per_epoch):
        """Yield training batches (X, y) forever: steps_per_epoch per epoch.

        Batch k is the k-th of the whole run, so the epochs follow from k alone and do
        not depend on when Keras starts or ends an epoch. Batch k holds the training
        samples step * batch_size ... of epoch k // steps_per_epoch, where step is
        k % steps_per_epoch.
        """
        tasks_per_batch = -(-batch_size // self.chunk)
        ahead = self.prefetch or max(2, -(-2 * max(self.workers, 1) // tasks_per_batch))
        queue = deque()
        number = 0
        try:
            while True:
                while len(queue) < ahead:
                    epoch, start, stop = self.batch_samples(
                        batch_size, steps_per_epoch, number
                    )
                    queue.append(self._tasks("train", epoch, start, stop))
                    number += 1
                X, y, _ = self._collect(queue.popleft())
                yield X, y
        except (CancelledError, RuntimeError):
            # After close(), the queued tasks are cancelled and no new ones can be
            # submitted: the consumer has finished. Anything else is an error.
            if not self._closed:
                raise
            return

    def rate(self):
        """Samples per second and process, measured in the workers (None before any)."""
        if not self.worker_seconds:
            return None
        return self.samples / self.worker_seconds
