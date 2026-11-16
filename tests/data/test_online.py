"""Samples drawn on the fly for online training (gwd.data.online).

Every sample is a pure function of (seed, part, epoch, index), so the batches repeat
with the same seed, change from epoch to epoch, do not depend on the number of
workers or the chunk size, and the fixed val and test sets never share a stream with
the training batches.
"""

import itertools
import os

import numpy as np
import pytest

pytest.importorskip("gwpy")
pytest.importorskip("pycbc")

from gwd.config import N_SAMPLES  # noqa: E402
from gwd.data import online  # noqa: E402
from gwd.data.scenarios import EVAL_V1, TRAIN_V1, TRAIN_V2, TRAIN_V3  # noqa: E402
from gwd.evaluation.testsets import part_rng  # noqa: E402


def first_batches(source, count, batch_size=4, steps_per_epoch=2):
    return list(itertools.islice(source.batches(batch_size, steps_per_epoch), count))


def test_the_same_seed_gives_the_same_first_batches():
    with online.SampleSource(TRAIN_V2, 3, workers=0, chunk=2) as source:
        first = first_batches(source, 3)
    with online.SampleSource(TRAIN_V2, 3, workers=0, chunk=2) as source:
        second = first_batches(source, 3)
    with online.SampleSource(TRAIN_V2, 4, workers=0, chunk=2) as source:
        other_seed = first_batches(source, 1)
    for (X1, y1), (X2, y2) in zip(first, second):
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
    assert not np.array_equal(first[0][0], other_seed[0][0])


def test_the_samples_do_not_depend_on_workers_or_chunks():
    with online.SampleSource(TRAIN_V3, 5, workers=0, chunk=4) as source:
        alone = first_batches(source, 2)
    with online.SampleSource(TRAIN_V3, 5, workers=2, chunk=1) as source:
        pooled = first_batches(source, 2)
    for (X1, y1), (X2, y2) in zip(alone, pooled):
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


def test_every_epoch_draws_new_samples():
    """Batch 2 is the first of epoch 1: the same labels, other noise and signals."""
    with online.SampleSource(TRAIN_V2, 3, workers=0) as source:
        batches = first_batches(source, 4, batch_size=4, steps_per_epoch=2)
    (X0, y0), (X1, _), (X2, y2), (X3, _) = batches
    np.testing.assert_array_equal(y0, y2)
    for a, b in itertools.combinations([X0, X1, X2, X3], 2):
        for row_a, row_b in itertools.product(a, b):
            assert not np.array_equal(row_a, row_b)


def test_batches_are_balanced_normalized_float32():
    with online.SampleSource(TRAIN_V2, 1, workers=0) as source:
        ((X, y),) = first_batches(source, 1, batch_size=6)
    assert X.shape == (6, N_SAMPLES, 1)
    assert X.dtype == np.float32
    np.testing.assert_array_equal(y, [1, 0, 1, 0, 1, 0])
    np.testing.assert_allclose(X.mean(axis=1), 0, atol=1e-5)
    np.testing.assert_allclose(X.std(axis=1), 1, atol=1e-4)


def test_batch_samples_maps_a_run_batch_to_its_epoch():
    source = online.SampleSource(TRAIN_V2, 1, workers=0)
    assert source.batch_samples(4, 3, 0) == (0, 0, 4)
    assert source.batch_samples(4, 3, 2) == (0, 8, 12)
    assert source.batch_samples(4, 3, 4) == (1, 4, 8)


def test_val_and_test_are_fixed_and_disjoint_from_the_training_stream():
    with online.SampleSource(TRAIN_V2, 7, workers=0) as source:
        X_val, y_val, _ = source.draw("val", 4)
        X_val_again, _, _ = source.draw("val", 4)
        X_test, y_test, _ = source.draw("test", 4)
        train = first_batches(source, 4, batch_size=4, steps_per_epoch=2)
    np.testing.assert_array_equal(X_val, X_val_again)
    np.testing.assert_array_equal(y_val, [1, 0, 1, 0])
    np.testing.assert_array_equal(y_val, y_test)
    training = np.concatenate([X for X, _ in train])
    rows = [*X_val, *X_test, *training]
    for a, b in itertools.combinations(rows, 2):
        assert not np.array_equal(a, b)


def test_the_streams_differ_by_part_and_from_the_test_sets():
    first = {
        part: online.sample_rng(0, part, 0, 0).randint(2**31) for part in online.PARTS
    }
    assert len(set(first.values())) == len(online.PARTS)
    # The fixed test sets of gwd.evaluation.testsets key their streams with 'gwdt'.
    testset = part_rng("indist_v2", 0, "test").randint(2**31)
    plain = np.random.RandomState(0).randint(2**31)
    assert testset not in first.values() and plain not in first.values()


def test_the_worker_count_is_respected():
    with online.SampleSource(TRAIN_V2, 1, workers=2, chunk=1) as source:
        source.draw("val", 8)
        assert source._executor._max_workers == 2
        assert 1 <= len(source.pids) <= 2
        assert os.getpid() not in source.pids
    with online.SampleSource(TRAIN_V2, 1, workers=0) as source:
        source.draw("val", 2)
        assert source.pids == {os.getpid()}


def test_batches_keep_a_bounded_number_of_tasks_in_flight(monkeypatch):
    source = online.SampleSource(TRAIN_V2, 1, workers=0, chunk=2, prefetch=3)
    submitted = []
    real = source._submit
    monkeypatch.setattr(
        source, "_submit", lambda *args: submitted.append(args) or real(*args)
    )
    batches = source.batches(4, 10)
    next(batches)
    # Three batches of two tasks each were queued, and one was handed out.
    assert len(submitted) == 6
    next(batches)
    assert len(submitted) == 8


def test_closing_ends_the_batches():
    source = online.SampleSource(TRAIN_V2, 1, workers=1, chunk=2)
    batches = source.batches(2, 4)
    next(batches)
    source.close()
    assert len(list(itertools.islice(batches, 50))) < 50


@pytest.mark.parametrize("scenario", [TRAIN_V1])
def test_scenarios_with_real_noise_are_refused(scenario):
    with pytest.raises(ValueError, match="real noise"):
        online.SampleSource(scenario, 1, workers=0)


def test_v1_scenarios_with_white_noise_are_drawn():
    with online.SampleSource(EVAL_V1, 1, workers=0) as source:
        X, y, _ = source.draw("val", 2)
    assert X.shape == (2, N_SAMPLES, 1)
