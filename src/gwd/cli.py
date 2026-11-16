"""The gwd command: one entry point for the whole pipeline and the apps (P3.6).

Only dispatch lives here. Every handler imports its module when it runs, so
``gwd --help`` and ``gwd data generate`` never load TensorFlow, and a broken optional
package only breaks the commands that need it. build_parser() is its own function, so
tests and the menu in main.py can check argument lists without running anything.

Exit codes: 0 on success, 1 when a command fails (for example an unknown model, or
a missing PyCBC for a command that draws waveforms), 2 for invalid arguments
(argparse), 130 on Ctrl+C.
"""

import argparse
import os
import sys

# What --model accepts; gwd.registry.resolve() turns it into a file.
MODEL_HELP = (
    "'latest' (newest timestamp in the name), a model name in models_registry/, "
    "or a path to a .keras file"
)


# Read by TensorFlow's C++ runtime when it is imported. Level 3 hides its info,
# warning, and error lines (cudart_stub, oneDNN, cpu_feature_guard, and the cuInit error
# on a machine without a CUDA driver); Python exceptions still surface.
TF_ENVIRONMENT = {
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "TF_ENABLE_ONEDNN_OPTS": "0",  # oneDNN off, as in training before P3.7
}


def quiet_tensorflow():
    """Set TF_ENVIRONMENT before TensorFlow is imported; values the caller set win.

    Every handler that can load TensorFlow calls it first. Once TensorFlow is
    imported, the variables have no effect any more.
    """
    for name, value in TF_ENVIRONMENT.items():
        os.environ.setdefault(name, value)


# --- handlers: each returns an exit code ---


def _fetch_noise(args):
    from gwd.data import fetch_noise

    options = {"detector": args.detector}
    if args.gps is not None:
        options["start_gps"] = args.gps
    if args.duration is not None:
        options["duration"] = args.duration
    return fetch_noise.fetch_and_slice(**options)


def _generate(args):
    from gwd.config import DATA_DIR_TRAIN, SAMPLES_FOR_TRAINING
    from gwd.data.generate import OutputFolderError, generate_dataset
    from gwd.data.scenarios import SCENARIOS

    # generate_dataset seeds its own RNG and records the seed in manifest.json.
    try:
        generate_dataset(
            num_samples=SAMPLES_FOR_TRAINING if args.n is None else args.n,
            output_folder=args.out or DATA_DIR_TRAIN,
            scenario=SCENARIOS[args.scenario.upper()],
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except OutputFolderError as error:
        print(f"❌ {error}", file=sys.stderr)
        return 1
    return 0


def _validate(args):
    from gwd.config import DATA_DIR_TRAIN, VALIDATION_DATASET_PLOTS_DIR
    from gwd.data import validate

    # --plots without a folder is the empty string; no --plots means no plots.
    plots = args.plots
    if plots == "":
        plots = VALIDATION_DATASET_PLOTS_DIR
    return validate.run(data_folder=args.data or DATA_DIR_TRAIN, output_folder=plots)


def _train(args):
    from gwd.config import DATA_DIR_TRAIN, LABELS_FILE

    # Checked before TensorFlow is imported, so a missing dataset fails at once.
    data_folder = args.data or DATA_DIR_TRAIN
    if not os.path.exists(os.path.join(data_folder, LABELS_FILE)):
        print(
            f"❌ No dataset in '{data_folder}': {LABELS_FILE} is missing. "
            "Run 'gwd data generate' first.",
            file=sys.stderr,
        )
        return 1

    quiet_tensorflow()
    from gwd.data.validate import InvalidDatasetError
    from gwd.training.train import TrainConfig, train

    overrides = {
        "data_folder": args.data,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "patience": args.patience,
        "mixed_precision": args.mixed_precision or None,
        "deterministic": args.deterministic or None,
        # train() seeds Python, numpy, TensorFlow, and the split and records the
        # seed; without --seed it draws one.
        "seed": args.seed,
    }
    cfg = TrainConfig(
        **{name: value for name, value in overrides.items() if value is not None}
    )
    try:
        train(cfg)
    except InvalidDatasetError as error:
        # The dataset is checked before training; its findings name the files.
        print(f"❌ {error}", file=sys.stderr)
        return 1
    return 0


def _testset_options(args):
    """The test set arguments given; the functions default to gwd.config (P5.7).

    The set seeds its own random stream, so nothing seeds the global RNG any more.
    """
    options = {"testset": args.testset, "seed": args.seed, "n": args.n}
    return {name: value for name, value in options.items() if value is not None}


def _eval_build_testset(args):
    from gwd.evaluation import testsets

    try:
        options = _testset_options(args)
        if "testset" in options:
            options["name"] = options.pop("testset")
        built = testsets.build(rebuild=args.rebuild, **options)
    except testsets.InvalidTestsetError as error:
        print(f"❌ {error}", file=sys.stderr)
        return 1
    print("\n".join(testsets.describe(built)))
    return 0


def _eval_run(args):
    quiet_tensorflow()
    from gwd.evaluation import evaluate

    return evaluate.run(model=args.model, real=args.real, **_testset_options(args))


def _eval_thresholds(args):
    quiet_tensorflow()
    from gwd.evaluation import thresholds

    return thresholds.run(model=args.model, save=args.save, **_testset_options(args))


def _eval_compare(args):
    quiet_tensorflow()
    from gwd.evaluation import compare

    return compare.run(models=args.model, **_testset_options(args))


def _eval_leaderboard(args):
    from gwd.evaluation import leaderboard

    return leaderboard.show()


APPS = {
    "waveform": "gwd.apps.waveform_explorer",
    "interferometer": "gwd.apps.interferometer_designer",
    "glitch-hunter": "gwd.apps.glitch_hunter.app",
}


def _app(args):
    if args.app == "glitch-hunter":
        # It takes --model; without PyCBC it starts with the simulation off (P4.4).
        quiet_tensorflow()
        from gwd.apps.glitch_hunter.app import main

        return main(model=args.model or "latest")
    if args.model is not None:
        print("❌ --model is an option of glitch-hunter only.", file=sys.stderr)
        return 2

    # The other apps start in their __main__ block; run it as python -m would.
    import runpy

    runpy.run_module(APPS[args.app], run_name="__main__", alter_sys=True)
    return 0


def _env(args):
    from gwd import env

    return env.main()


# --- parser ---


def _add_seed(parser, what, default="unseeded"):
    parser.add_argument(
        "--seed",
        type=int,
        help=f"Seed for {what}. Default: {default}.",
    )


def _non_negative(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must not be negative: {value}")
    return value


def _at_least_two(text):
    value = int(text)
    if value < 2:
        raise argparse.ArgumentTypeError(f"must be at least 2: {value}")
    return value


def _add_testset(parser, *aliases):
    """--testset, --seed, and --n: which fixed test set (gwd.evaluation.testsets).

    The defaults stay None here and come from gwd.config in the functions, so the
    parser does not import gwd.config (which needs a project root).
    """
    from gwd.data.scenarios import TESTSETS

    parser.add_argument(
        "--testset",
        *aliases,
        choices=sorted(TESTSETS),
        help="The fixed test set: indist_v1 (the training distribution TRAIN_V1), "
        "stress_v1 (EVAL_V1, as before P5.7), indist_v2 (TRAIN_V2), stress_v2 "
        "(TRAIN_V2 with an extrapolation bin), or indist_v3 and stress_v3 (the same "
        "events as the V2 sets, whitened). A model trained with whitening is scored on "
        "the V3 counterpart of a V2 set, and the other way round. Default: "
        "DEFAULT_TESTSET in gwd.config (stress_v1).",
    )
    parser.add_argument(
        "--seed",
        type=_non_negative,
        help="Seed of the test set (its own random stream, disjoint from training). "
        "Default: TESTSET_SEED in gwd.config (0).",
    )
    parser.add_argument(
        "--n",
        type=_at_least_two,
        help="Samples in the balanced part 'test'; the noise-only part calib_bg has as "
        "many. Default: TESTSET_SIZE in gwd.config (1000).",
    )


def _add_model(parser):
    parser.add_argument(
        "--model", default="latest", help=f"{MODEL_HELP}. Default: latest."
    )


def build_parser():
    """The full gwd parser. Importing nothing but the scenario names keeps it cheap."""
    from gwd.data.scenarios import SCENARIOS

    parser = argparse.ArgumentParser(
        prog="gwd",
        description="Gravitational wave detection: data, training, evaluation, and apps.",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    # gwd data ...
    data = commands.add_parser("data", help="Fetch noise, generate and validate data.")
    data_commands = data.add_subparsers(
        dest="data_command", metavar="COMMAND", required=True
    )

    fetch = data_commands.add_parser(
        "fetch-noise",
        help="Download real detector noise from GWOSC into data/noise_background/.",
        description="Download real detector noise from GWOSC, condition it, and cut it "
        "into chunks in data/noise_background/. Existing chunks with the same names are "
        "overwritten. Needs network access.",
    )
    fetch.add_argument("--detector", default="H1", help="Detector. Default: H1.")
    fetch.add_argument(
        "--gps",
        type=int,
        help="Start GPS time. Default: START_GPS in gwd.data.fetch_noise (a quiet O3b stretch).",
    )
    fetch.add_argument(
        "--duration",
        type=int,
        help="Seconds to download. Default: NOISE_FETCH_SECONDS in gwd.config.",
    )
    fetch.set_defaults(handler=_fetch_noise)

    generate = data_commands.add_parser(
        "generate",
        help="Generate a training dataset.",
        description="Write samples, labels.csv, and manifest.json into an empty or "
        "new folder. Real noise chunks are used where the scenario and "
        "data/noise_background/ allow it.",
    )
    generate.add_argument(
        "--n",
        type=int,
        help="Number of samples. Default: SAMPLES_FOR_TRAINING in gwd.config.",
    )
    _add_seed(generate, "the drawn samples")
    generate.add_argument("--out", help="Output folder. Default: data/training/.")
    generate.add_argument(
        "--overwrite",
        action="store_true",
        help="Write into a folder that is not empty: first remove its sample_*.npy "
        "and labels.csv (nothing else). Default: stop with exit code 1.",
    )
    generate.add_argument(
        "--scenario",
        type=str.lower,
        choices=sorted(name.lower() for name in SCENARIOS),
        default="train_v1",
        help="Scenario to draw from: train_v1, train_v2 (simulated aLIGO noise, "
        "conditioned with the signal, optimal SNR; needs gwpy), or train_v3 (train_v2 "
        "whitened with the aLIGO PSD before the band-pass; needs gwpy). Default: "
        "train_v1.",
    )
    generate.set_defaults(handler=_generate)

    validate = data_commands.add_parser(
        "validate",
        help="Check a dataset: every sample, orphans, manifest, scenario ranges.",
        description="Check every sample that labels.csv lists, sample files it does "
        "not list, manifest.json, and, if the manifest names its scenario, the signal "
        "parameter ranges and the class balance. Exit code 1 if there is any finding.",
    )
    validate.add_argument("--data", help="Dataset folder. Default: data/training/.")
    validate.add_argument(
        "--plots",
        nargs="?",
        const="",
        metavar="FOLDER",
        help="Also write the report plots, to FOLDER or, without one, to "
        "plots/validation_dataset/. Default: no plots.",
    )
    validate.set_defaults(handler=_validate)

    # gwd train
    train = commands.add_parser(
        "train",
        help="Train the CNN and save it in models_registry/.",
        description="Train the CNN on a dataset and save the model and its .json "
        "in models_registry/.",
    )
    train.add_argument("--data", help="Dataset folder. Default: data/training/.")
    train.add_argument(
        "--epochs",
        type=int,
        help="Maximum number of epochs. Default: EPOCHS in gwd.config.",
    )
    train.add_argument(
        "--batch-size", type=int, help="Batch size. Default: BATCH_SIZE in gwd.config."
    )
    train.add_argument(
        "--lr",
        type=float,
        help="Adam learning rate. Default: LEARNING_RATE in gwd.config.",
    )
    train.add_argument(
        "--patience",
        type=int,
        help="Epochs without a better val_loss before early stopping. "
        "Default: EARLY_STOPPING_PATIENCE in gwd.config.",
    )
    train.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Train with mixed_float16 even without a GPU. Default: mixed_float16 "
        "only if TensorFlow finds a GPU, else float32.",
    )
    train.add_argument(
        "--deterministic",
        action="store_true",
        help="Also make TensorFlow's ops deterministic, so the same --seed gives the "
        "same model on the CPU (slower; not guaranteed on a GPU).",
    )
    _add_seed(
        train,
        "Python, numpy, TensorFlow, and the train/val/test split",
        "a drawn seed, printed and recorded in the model's .json",
    )
    train.set_defaults(handler=_train)

    # gwd eval ...
    evaluation = commands.add_parser("eval", help="Evaluate, tune, and compare models.")
    eval_commands = evaluation.add_subparsers(
        dest="eval_command", metavar="COMMAND", required=True
    )

    build_testset = eval_commands.add_parser(
        "build-testset",
        help="Build a fixed, hashed test set once (later calls load it).",
        description="Draw a test set once into data/eval/<testset>_<seed>_<n>.npz, "
        "with its manifest and SHA-256 in results/testsets/. If both exist, load and "
        "check them instead (a cache hit); exit code 1 if they do not match or are "
        "stale.",
    )
    # --scenario is the name the cleanup plan used for this option (P5.7).
    _add_testset(build_testset, "--scenario")
    build_testset.add_argument(
        "--rebuild",
        action="store_true",
        help="Draw and write the set again, also over a cached one. Default: no.",
    )
    build_testset.set_defaults(handler=_eval_build_testset)

    run = eval_commands.add_parser(
        "run",
        help="ROC evaluation of one model on a fixed test set.",
        description="ROC evaluation on a fixed, hashed test set (built on first use), "
        "with metrics per peak SNR, optimal SNR, total mass, and noise source, and, "
        "with --real, the field test on real events, at the model's threshold from "
        "gwd eval thresholds (0.5 with a warning if it has none). Adds a row to "
        "results/leaderboard_v2.csv.",
    )
    _add_model(run)
    run.add_argument(
        "--real",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the real data field test (needs network access). Default: --no-real.",
    )
    _add_testset(run)
    run.set_defaults(handler=_eval_run)

    thresholds = eval_commands.add_parser(
        "thresholds",
        help="Calibrate one model's detection threshold at FPR 1%%.",
        description="Calibrate one model's detection threshold on the noise-only part "
        "calib_bg of a fixed test set, at a false alarm rate of 1%, and save it in the "
        "model's metadata in models_registry/, where gwd eval run, gwd eval compare, "
        "and the Glitch Hunter read it. Also writes a report to "
        "results/thresholds/<model name>.json, one per model.",
    )
    _add_model(thresholds)
    thresholds.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the threshold in the model's metadata and the report. "
        "Default: --save.",
    )
    _add_testset(thresholds)
    thresholds.set_defaults(handler=_eval_thresholds)

    compare = eval_commands.add_parser(
        "compare",
        help="Compare models in one ROC plot.",
        description="Compare models in one ROC plot on the same fixed test set.",
    )
    compare.add_argument(
        "--model",
        action="append",
        help=f"One model to compare, repeatable: {MODEL_HELP}. Default: all models.",
    )
    _add_testset(compare)
    compare.set_defaults(handler=_eval_compare)

    leaderboard = eval_commands.add_parser(
        "leaderboard",
        help="Show leaderboard v2, one table per test set, best ROC AUC first.",
    )
    leaderboard.set_defaults(handler=_eval_leaderboard)

    # gwd app ...
    app = commands.add_parser(
        "app", help="Start an interactive app (matplotlib window)."
    )
    app.add_argument(
        "app",
        choices=list(APPS),
        help="waveform: waveform explorer; interferometer: interferometer designer; "
        "glitch-hunter: the CNN rates simulated or real data.",
    )
    app.add_argument(
        "--model",
        help=f"glitch-hunter only: the model for 'Ask AI', {MODEL_HELP}. "
        "Default: latest; without any model, 'Ask AI' is off.",
    )
    app.set_defaults(handler=_app)

    # gwd env
    env = commands.add_parser(
        "env", help="Show versions, extras, waveform backend, paths, and GUI backend."
    )
    env.set_defaults(handler=_env)

    return parser


def _is_backend_error(error):
    """True for gwd.core.waveforms.WaveformBackendError, without importing numpy.

    The error can only come from that module, so it is in sys.modules if it was raised.
    """
    waveforms = sys.modules.get("gwd.core.waveforms")
    return waveforms is not None and isinstance(error, waveforms.WaveformBackendError)


def main(argv=None):
    """Run gwd with argv (default: sys.argv[1:]) and return the exit code."""
    args = build_parser().parse_args(argv)
    try:
        returncode = args.handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        # A missing PyCBC is an environment problem, not a bug: no traceback (P5.2).
        if not _is_backend_error(error):
            raise
        print(f"❌ {error}", file=sys.stderr)
        return 1
    return 0 if returncode is None else int(returncode)


if __name__ == "__main__":
    sys.exit(main())
