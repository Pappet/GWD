#!/usr/bin/env python3
"""
ROC comparison: several models on one test set, independent of a threshold.

Run it with ``gwd eval compare``; it writes the comparison plot and a zoom on low FPR.
Since P5.7 every model is scored on the part 'test' of one fixed, hashed test set
(gwd.evaluation.testsets; --testset, --seed, --n; default: see below). Since P5.8 each
model's curve also marks its operating point at its own threshold
(gwd.registry.get_threshold), and the ranking, by ROC AUC, lists the TPR there with the
FPR it reached.

Since V3 every model sees data with its own preprocessing
(gwd.evaluation.testsets.set_for_model): with --testset indist_v2, a whitening model is
scored on indist_v3, which holds the same events and noise, whitened; the labels are
the same, so all curves still describe the same physical samples.

Models with preprocessings that no pair of counterpart sets covers cannot share one
set. With --testset, a model that the set (or its counterpart) cannot give the
preprocessing it was trained with (a whitening model on a V1 set) is left out, with the
reason. Without --testset, each model gets its default set
(gwd.evaluation.testsets.default_for_model: indist_v2 for V2, indist_v3 for V3,
stress_v1 for V1 and for models that do not name their scenario), and the models are
compared in groups, one per family of sets with the same draws (indist_v2 and indist_v3
are one); each group gets its own plots, named with its set, and its own ranking. A
group with one model has nothing to compare and is only listed.
"""

import datetime
import os
import sys
from typing import NamedTuple

import matplotlib
import numpy as np

from gwd.config import (
    EVALUATION_MODELS_PLOTS_DIR,
    TESTSET_SEED,
    TESTSET_SIZE,
    ensure_dirs,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

from gwd import registry
from gwd.data.scenarios import DEFAULT_PREPROCESSING, counterpart_set
from gwd.evaluation import testsets
from gwd.evaluation.metrics import calculate_operating_point
from gwd.registry import list_models, load_model, model_name, resolve

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class Curve(NamedTuple):
    """One model's ROC curve and its operating point at its own threshold."""

    name: str
    fpr: object
    tpr: object
    roc_auc: float
    threshold: float
    op_tpr: float
    op_fpr: float


def score_models(paths, X_test, y_true, inputs=None):
    """Load the models one after another and compute each ROC curve once.

    Only one model is in memory at a time. Returns a Curve per model that loads, in the
    order of paths (oldest first when they come from list_models()); its operating point
    is at the model's threshold from gwd.registry.get_threshold. inputs maps a path to
    the samples of that model, if they are not X_test (another preprocessing of the
    same samples, with the same labels y_true).
    """
    inputs = inputs or {}
    curves = []
    for path in paths:
        try:
            model = load_model(path)
        except Exception as e:
            print(f"✗ Error for {path}: {e}")
            continue
        name = model_name(path)
        print(f"✓ Loaded: {name}")
        y_scores = model.predict(inputs.get(path, X_test), verbose=0).flatten()
        del model
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        threshold = registry.get_threshold(path)
        op_tpr, op_fpr = calculate_operating_point(y_true, y_scores, threshold)
        curves.append(
            Curve(
                name, fpr, tpr, auc(fpr, tpr), threshold, float(op_tpr), float(op_fpr)
            )
        )
    return curves


def model_colors(names):
    """One fixed color per model name, the same in every plot of a run.

    The names are sorted, so the same set of models gets the same colors in every run.
    """
    unique = sorted(set(names))
    cmap = plt.cm.tab10 if len(unique) <= 10 else plt.cm.tab20
    return {name: cmap(i % cmap.N) for i, name in enumerate(unique)}


def short_name(name, length=30):
    """The model name for a legend, cut to length characters."""
    return name if len(name) <= length else name[:length] + "..."


def save_plot(fig, filename):
    """Save fig in the evaluation plot folder and return the path."""
    ensure_dirs(EVALUATION_MODELS_PLOTS_DIR)
    path = os.path.join(EVALUATION_MODELS_PLOTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"📊 Plot: {path}")
    return path


def mark_operating_point(ax, curve, color):
    """A star at the model's operating point; not in the legend."""
    ax.plot(
        curve.op_fpr,
        curve.op_tpr,
        marker="*",
        markersize=14,
        color=color,
        markeredgecolor="black",
        linestyle="none",
        label="_operating point",
    )


def plot_name(testset, zoom=False):
    """plot_compare_roc_<timestamp>_<testset>.png, or ..._zoom.png for the zoom."""
    return (
        f"plot_compare_roc_{TIMESTAMP}_{testset}" + ("_zoom" if zoom else "") + ".png"
    )


def compare_models_roc(curves, colors, title="ROC Curve Comparison", testset="roc"):
    """Plot all ROC curves and the AUC ranking. Returns the ranking, best first.

    testset names the plot file (plot_name).
    """

    print("\n🧠 Plotting the ROC curves of all models...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for curve in curves:
        ax1.plot(
            curve.fpr,
            curve.tpr,
            color=colors[curve.name],
            lw=2,
            label=f"{short_name(curve.name)} (AUC={curve.roc_auc:.3f})",
        )
        mark_operating_point(ax1, curve, colors[curve.name])

    # Random classifier line
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Classifier")

    ax1.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax1.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- AUC Bar Chart ---
    auc_scores_sorted = sorted(
        ((curve.name, curve.roc_auc) for curve in curves),
        key=lambda x: x[1],
        reverse=True,
    )
    names = [short_name(name) for name, _ in auc_scores_sorted]
    aucs = [x[1] for x in auc_scores_sorted]

    bars = ax2.barh(
        range(len(names)),
        aucs,
        color=[colors[name] for name, _ in auc_scores_sorted],
    )
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("ROC AUC", fontsize=12, fontweight="bold")
    ax2.set_title("Model Ranking by AUC", fontsize=14, fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax2.grid(True, alpha=0.3, axis="x")

    # Write the values into the bars
    for i, (bar, auc_val) in enumerate(zip(bars, aucs)):
        ax2.text(
            auc_val - 0.05,
            i,
            f"{auc_val:.4f}",
            va="center",
            ha="right",
            fontweight="bold",
            color="white",
        )

    fig.tight_layout()
    save_plot(fig, plot_name(testset))

    return auc_scores_sorted


def zoom_to_relevant_region(curves, colors, testset="roc"):
    """
    Zoomed ROC plot: gravitational wave searches care mostly about very low FPR,
    and a zoom on FPR <= 0.1 shows the differences there better. testset names the
    plot file (plot_name).
    """
    print("\n🔍 Zooming in on the relevant region (FPR <= 0.1)...")

    fig, ax = plt.subplots(figsize=(10, 8))

    for curve in curves:
        # Plot only FPR <= 0.1
        mask = curve.fpr <= 0.1
        ax.plot(
            curve.fpr[mask],
            curve.tpr[mask],
            color=colors[curve.name],
            lw=2,
            marker="o",
            markersize=3,
            label=f"{short_name(curve.name)} (AUC={curve.roc_auc:.3f})",
        )
        if curve.op_fpr <= 0.1:
            mark_operating_point(ax, curve, colors[curve.name])

    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title(
        "ROC Curve - Zoom on the relevant region (FPR <= 0.1)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    save_plot(fig, plot_name(testset, zoom=True))


def family(name):
    """The set that names a family of sets with the same draws: the bandpass one.

    indist_v3 belongs to indist_v2, stress_v3 to stress_v2; every other set is its own.
    """
    spec = testsets.get(name)
    return (counterpart_set(spec, DEFAULT_PREPROCESSING) or spec).name


def group_models(paths, testset=None):
    """{family set name: [(path, set name, preprocessing, note)]}, in the order of paths.

    With testset, one group: each model gets testset or its counterpart with the
    model's preprocessing (testsets.set_for_model); a model for which neither has its
    preprocessing is left out, with the reason. Without, each model gets its default
    set (testsets.default_for_model), and models whose sets share their draws form one
    group. Prints what it chose for each model.
    """
    groups = {}
    for path in paths:
        name = model_name(path)
        asked = testset
        if asked is None:
            asked, reason = testsets.default_for_model(path)
            print(f"{name}: {testsets.describe_default(asked, reason)}")
        chosen, preprocessing, note = testsets.set_for_model(asked, path)
        if testsets.get(chosen).preprocessing != preprocessing:
            print(f"{name}: left out. {note}")
            continue
        if note:
            print(f"{name}: {note}")
        key = testset if testset is not None else family(chosen)
        groups.setdefault(key, []).append((path, chosen, preprocessing, note))
    return groups


def compare_group(testset, entries, seed, n):
    """Compare the models of one group on testset and its counterparts. Exit code.

    entries are (path, set name, preprocessing, note) from group_models. The reference
    set is testset if a model uses it, else the set of the first model; the other
    models get the counterpart they need, which must have the same labels.
    """
    if len(entries) < 2:
        names = ", ".join(model_name(path) for path, *_ in entries) or "none"
        print(f"\n⚠️ Only one model on {testset} ({names}): nothing to compare it with.")
        return 0

    paths = [path for path, *_ in entries]
    sets = [chosen for _, chosen, _, _ in entries]
    reference = testset if testset in sets else sets[0]

    # The test sets, before any model loads (drawing them needs PyCBC).
    try:
        built = testsets.build(reference, n, seed)
        others = {
            name: testsets.build(name, n, seed)
            for name in dict.fromkeys(sets)
            if name != reference
        }
    except testsets.InvalidTestsetError as error:
        print(f"❌ {error}", file=sys.stderr)
        return 1
    print("\n".join(testsets.describe(built)))
    for other in others.values():
        print("\n".join(testsets.describe(other)))
    X_test, y_true = built.X, built.y
    inputs = {}
    for path, name in zip(paths, sets):
        if name != reference:
            if not np.array_equal(others[name].y, y_true):
                raise ValueError(f"{name} does not have the labels of {reference}")
            inputs[path] = others[name].X

    # Load each model in turn and score it once
    curves = score_models(paths, X_test, y_true, inputs)

    if len(curves) == 0:
        print("\n❌ No models to compare.")
        return 1

    if len(curves) == 1:
        print("\n⚠️ Only one model found. Train more models to compare them.")
        return 0

    print(f"\n✅ {len(curves)} models loaded.")
    colors = model_colors([curve.name for curve in curves])

    # Compare
    title = f"ROC curves on {built.label()}, SHA-256 {built.short_hash()}"
    for other in others.values():
        title += (
            f"\nmodels with {other.spec.preprocessing} on {other.spec.name}, "
            f"SHA-256 {other.short_hash()}"
        )
    rankings = compare_models_roc(curves, colors, title, testset)

    # Print the ranking
    print("\n" + "=" * 70)
    print(f"📊 FINAL RANKING on {built.label()}:")
    print("=" * 70)
    by_name = {curve.name: curve for curve in curves}
    n_signal, n_noise = int((y_true == 1).sum()), int((y_true == 0).sum())
    print(
        f"  Ranked by ROC AUC. TPR at each model's threshold (★ in the plots), on n "
        f"{n_signal} signal, at the FPR it reached on n {n_noise} noise samples."
    )
    for i, (name, auc_val) in enumerate(rankings, 1):
        curve = by_name[name]
        print(
            f"  {i}. {name[:40]:<40s} | AUC: {auc_val:.4f} | TPR {curve.op_tpr:6.1%} "
            f"at FPR {curve.op_fpr:6.2%} | threshold {curve.threshold:.4g}"
        )
    print("=" * 70)

    # Bonus: Zoom
    print("\n💡 Creating the zoom plot for low FPR...")
    zoom_to_relevant_region(curves, colors, testset)
    return 0


def run(models=None, testset=None, seed=TESTSET_SEED, n=TESTSET_SIZE):
    """Compare the given models (default: all in models_registry/). Returns the exit code.

    testset, seed, and n name the set (gwd.evaluation.testsets.build). testset None
    gives each model its default set and compares the models in groups of sets with
    the same draws (group_models); a model that a named set cannot give its
    preprocessing is left out.
    """
    print("=" * 70)
    print("🏆 ROC CURVE COMPARISON TOOL")
    print("=" * 70)

    # 1. Find the models
    try:
        paths = [resolve(spec) for spec in models] if models else list_models()
    except FileNotFoundError as error:
        print(f"❌ {error}")
        return 1

    if len(paths) == 0:
        print("\n❌ No models to compare.")
        return 1

    if len(paths) == 1:
        print("\n⚠️ Only one model found. Train more models to compare them.")
        return 0

    # 2. Which set each model is scored on, and the groups that can share one
    groups = group_models(paths, testset)
    if not groups:
        print("\n❌ No models to compare.")
        return 1
    if len(groups) > 1:
        print(
            f"\nThe models were trained on scenarios whose test sets do not share "
            f"their samples, so they are compared in {len(groups)} groups: "
            + ", ".join(f"{name} ({len(group)})" for name, group in groups.items())
        )

    # 3. Each group on its own set
    status = 0
    for name, entries in groups.items():
        if len(groups) > 1:
            print("\n" + "=" * 70)
            print(f"Group {name}: {', '.join(model_name(p) for p, *_ in entries)}")
        status = max(status, compare_group(name, entries, seed, n))
    return status


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit("The options are on the gwd command: gwd eval compare --help")
    sys.exit(run())
