"""gwd eval compare: two plots, one scoring pass per model, one color per model (P4.2)."""

import numpy as np
import pytest
from matplotlib.colors import to_rgba

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

    def test_set(scenario, n):
        y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
        return np.repeat(y[:, None, None], 4, axis=1), y

    def save(fig, filename):
        figures.append((filename, fig))
        return filename

    monkeypatch.setattr(compare, "resolve", lambda spec: spec)
    monkeypatch.setattr(compare, "load_model", load)
    monkeypatch.setattr(compare, "make_test_set", test_set)
    monkeypatch.setattr(compare, "save_plot", save)
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
