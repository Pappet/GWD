"""gwd eval compare: two plots, one scoring pass per model, one color per model (P4.2),
and since P5.8 each model's operating point at its registry threshold."""

from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.colors import to_rgba

from gwd import registry
from gwd.evaluation import compare


class StubModel:
    """Scores the test set by its label plus noise; strength sets how well."""

    def __init__(self, name, strength, log):
        self.name, self.strength, self.log = name, strength, log

    def predict(self, X, verbose=0):
        self.log.append(("predict", self.name))
        rng = np.random.RandomState(len(self.name))
        labels = X[:, 0, 0]
        return (labels * self.strength + rng.rand(len(X))).reshape(-1, 1)


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """compare with three stub models and a test set whose first value is the label."""
    log, figures = [], []
    strengths = {"gwd_model_a": 0.2, "gwd_model_b": 2.0, "gwd_model_c": 0.8}

    def load(path):
        log.append(("load", path))
        return StubModel(path, strengths[path], log)

    def test_set(name, n, seed):
        y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
        return SimpleNamespace(
            X=np.repeat(y[:, None, None], 4, axis=1),
            y=y,
            label=lambda: f"{name} (seed {seed}, n {n})",
            short_hash=lambda: "0" * 12,
        )

    def save(fig, filename):
        figures.append((filename, fig))
        return filename

    monkeypatch.setattr(compare, "resolve", lambda spec: spec)
    monkeypatch.setattr(compare, "load_model", load)
    monkeypatch.setattr(compare.testsets, "build", test_set)
    monkeypatch.setattr(compare.testsets, "describe", lambda built: [])
    monkeypatch.setattr(compare, "save_plot", save)
    monkeypatch.setattr(registry, "get_threshold", lambda path: 0.9)
    return list(strengths), log, figures


def test_two_plots_with_different_names(stubbed):
    names, _, figures = stubbed
    assert compare.run(names) == 0
    filenames = [filename for filename, _ in figures]
    assert len(filenames) == 2
    assert len(set(filenames)) == 2
    assert filenames[1] == filenames[0].replace(".png", "_zoom.png")


def test_models_load_one_after_another_and_predict_once(stubbed):
    names, log, _ = stubbed
    assert compare.run(names) == 0
    expected = [step for name in names for step in (("load", name), ("predict", name))]
    assert log == expected


def test_curves_and_bars_share_one_color_per_model(stubbed):
    names, _, figures = stubbed
    assert compare.run(names) == 0
    colors = compare.model_colors(names)
    (_, comparison), (_, zoom) = figures
    roc_axis, bar_axis = comparison.axes

    def curve_colors(axis):
        return {
            line.get_label().split(" (AUC")[0]: to_rgba(line.get_color())
            for line in axis.get_lines()
            if line.get_label().startswith("gwd_model")
        }

    expected = {name: to_rgba(colors[name]) for name in names}
    assert curve_colors(roc_axis) == expected
    assert curve_colors(zoom.axes[0]) == expected

    labels = [tick.get_text() for tick in bar_axis.get_yticklabels()]
    bars = {
        label: patch.get_facecolor() for label, patch in zip(labels, bar_axis.patches)
    }
    assert bars == expected
    # The bars are ranked by AUC, so their order differs from the load order.
    assert labels == ["gwd_model_b", "gwd_model_c", "gwd_model_a"]


def test_model_colors_depend_on_the_names_only():
    assert compare.model_colors(["b", "a"]) == compare.model_colors(["a", "b"])
    assert len(set(compare.model_colors(list("abcdefghijkl")).values())) == 12


def test_short_name():
    assert (
        compare.short_name("gwd_model_20251205-232013") == "gwd_model_20251205-232013"
    )
    assert compare.short_name("x" * 40) == "x" * 30 + "..."


def test_a_model_that_fails_to_load_is_skipped(stubbed, monkeypatch, capsys):
    names, log, figures = stubbed
    real_load = compare.load_model

    def load(path):
        if path == "gwd_model_c":
            raise OSError("broken file")
        return real_load(path)

    monkeypatch.setattr(compare, "load_model", load)
    assert compare.run(names) == 0
    assert "broken file" in capsys.readouterr().out
    assert len(figures) == 2


@pytest.mark.parametrize("loadable, rc", [(0, 1), (1, 0)])
def test_fewer_than_two_loadable_models_make_no_plot(
    stubbed, monkeypatch, loadable, rc
):
    names, _, figures = stubbed
    real_load = compare.load_model

    def load(path):
        if names.index(path) >= loadable:
            raise OSError("broken file")
        return real_load(path)

    monkeypatch.setattr(compare, "load_model", load)
    assert compare.run(names) == rc
    assert figures == []


def test_all_models_share_the_named_test_set(stubbed, monkeypatch):
    names, _, figures = stubbed
    calls = []
    build = compare.testsets.build

    def record(name, n, seed):
        calls.append((name, n, seed))
        return build(name, n, seed)

    monkeypatch.setattr(compare.testsets, "build", record)
    assert compare.run(names, testset="indist_v1", seed=4, n=10) == 0
    assert calls == [("indist_v1", 10, 4)]
    title = figures[0][1].axes[0].get_title()
    assert title == "ROC curves on indist_v1 (seed 4, n 10), SHA-256 000000000000"


def test_compare_uses_the_registry_threshold_of_each_model(
    stubbed, monkeypatch, capsys
):
    """P5.8: the operating point of every model is at exactly registry.get_threshold."""
    names, _, figures = stubbed
    asked, used = [], []
    original = compare.calculate_operating_point

    def get_threshold(path):
        asked.append(path)
        return 0.123

    def spy(y_true, y_scores, threshold):
        used.append(threshold)
        return original(y_true, y_scores, threshold)

    monkeypatch.setattr(registry, "get_threshold", get_threshold)
    monkeypatch.setattr(compare, "calculate_operating_point", spy)
    assert compare.run(names) == 0
    assert asked == names
    assert used == [0.123] * 3
    out = capsys.readouterr().out
    assert out.count("threshold 0.123") == 3
    assert "at FPR" in out
    stars = [
        line for line in figures[0][1].axes[0].get_lines() if line.get_marker() == "*"
    ]
    assert len(stars) == 3


def test_score_models_marks_the_operating_point(monkeypatch):
    class Perfect:
        def predict(self, X, verbose=0):
            return X[:, 0, 0].reshape(-1, 1)

    y = np.r_[np.zeros(4), np.ones(4)]
    monkeypatch.setattr(compare, "load_model", lambda path: Perfect())
    monkeypatch.setattr(registry, "get_threshold", lambda path: 0.5)
    (curve,) = compare.score_models(["gwd_model_x"], y[:, None, None], y)
    assert (curve.threshold, curve.op_tpr, curve.op_fpr) == (0.5, 1.0, 0.0)
    assert curve.roc_auc == 1.0


def test_a_whitening_model_is_scored_on_the_whitened_counterpart(
    stubbed, monkeypatch, capsys
):
    """V3: with --testset indist_v2, a whitening model gets indist_v3 (same labels),
    the others indist_v2; each set is built once and named in the title."""
    names, log, figures = stubbed
    built, seen = [], {}
    build = compare.testsets.build

    def record(name, n, seed):
        built.append(name)
        test_set = build(name, n, seed)
        test_set.X = test_set.X + (100.0 if name == "indist_v3" else 0.0)
        test_set.spec = compare.testsets.get(name)
        return test_set

    def set_for_model(name, path):
        if path == "gwd_model_c":
            return "indist_v3", "whiten_bandpass", "note for c"
        return name, "bandpass", None

    original = compare.score_models

    def spy(paths, X_test, y_true, inputs=None):
        for path in paths:
            seen[path] = float((inputs or {}).get(path, X_test)[0, 0, 0])
        return original(paths, X_test, y_true, inputs)

    monkeypatch.setattr(compare.testsets, "build", record)
    monkeypatch.setattr(compare.testsets, "set_for_model", set_for_model)
    monkeypatch.setattr(compare, "score_models", spy)
    assert compare.run(names, testset="indist_v2", seed=1, n=10) == 0
    assert sorted(built) == ["indist_v2", "indist_v3"]
    assert seen == {"gwd_model_a": 0.0, "gwd_model_b": 0.0, "gwd_model_c": 100.0}
    assert "gwd_model_c: note for c" in capsys.readouterr().out
    title = figures[0][1].axes[0].get_title()
    assert "models with whiten_bandpass on indist_v3" in title
