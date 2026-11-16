"""The gwd command: one entry point for the whole pipeline and the apps (P3.6).

Only dispatch lives here. Every handler imports its module when it runs, so
``gwd --help`` and ``gwd data generate`` never load TensorFlow, and a broken optional
package only breaks the commands that need it. build_parser() is its own function, so
tests and the menu in main.py can check argument lists without running anything.

Exit codes: 0 on success, 1 when a command fails (for example an unknown model),
2 for invalid arguments (argparse), 130 on Ctrl+C.
"""

import argparse
import os
import sys

# What --model accepts; gwd.registry.resolve() turns it into a file.
MODEL_HELP = (
    "'latest' (newest timestamp in the name), a model name in models_registry/, "
    "or a path to a .keras file"
)


def seed_everything(seed):
    """Seed Python's random and the global numpy RNG, for the eval commands.

    A minimal stand-in until P5.3 and P5.6 pass an rng explicitly. seed=None changes
    nothing, so runs stay unseeded as before. gwd data generate and gwd train pass the
    seed to their function instead, which also records it.
    """
    if seed is None:
        return
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


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

    # Read by TensorFlow when it is imported; values the caller has set win.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide info and warnings
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # oneDNN off, as before
    from gwd.data.validate import InvalidDatasetError
    from gwd.training.train import TrainConfig, train

    overrides = {
        "data_folder": args.data,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "patience": args.patience,
        "mixed_precision": args.mixed_precision or None,
        # train() seeds Python, numpy, and TensorFlow and records the seed.
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


def _eval_run(args):
    from gwd.evaluation import evaluate

    seed_everything(args.seed)
    return evaluate.run(model=args.model, real=args.real)


def _eval_thresholds(args):
    from gwd.evaluation import thresholds

    seed_everything(args.seed)
    return thresholds.run(model=args.model, save=args.save)


def _eval_compare(args):
    from gwd.evaluation import compare

    seed_everything(args.seed)
    return compare.run(models=args.model)


def _eval_leaderboard(args):
    from gwd.evaluation import leaderboard

    return leaderboard.show()


APPS = {
    "waveform": "gwd.apps.waveform_explorer",
    "interferometer": "gwd.apps.interferometer_designer",
    "glitch-hunter": "gwd.apps.glitch_hunter.app",
}


def _app(args):
    # The apps start in their __main__ block; run it as python -m would.
    import runpy

    runpy.run_module(APPS[args.app], run_name="__main__", alter_sys=True)
    return 0


def _env(args):
    from gwd import env

    return env.main()


# --- parser ---


def _add_seed(parser, what):
    parser.add_argument(
        "--seed",
        type=int,
        help=f"Seed for {what}. Default: unseeded.",
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
        help="Scenario to draw from. Default: train_v1.",
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
    _add_seed(train, "Python, numpy, and TensorFlow")
    train.set_defaults(handler=_train)

    # gwd eval ...
    evaluation = commands.add_parser("eval", help="Evaluate, tune, and compare models.")
    eval_commands = evaluation.add_subparsers(
        dest="eval_command", metavar="COMMAND", required=True
    )

    run = eval_commands.add_parser(
        "run",
        help="ROC evaluation of one model; adds a leaderboard row.",
        description="ROC evaluation on simulated data (EVAL_V1) and, with --real, the "
        "field test on real events. Adds a row to results/model_leaderboard.csv.",
    )
    _add_model(run)
    run.add_argument(
        "--real",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the real data field test (needs network access). Default: --no-real.",
    )
    _add_seed(run, "the simulated test set")
    run.set_defaults(handler=_eval_run)

    thresholds = eval_commands.add_parser(
        "thresholds",
        help="Find detection thresholds for one model.",
        description="Find detection thresholds for one model from its ROC curve and "
        "save them in results/optimal_thresholds.json.",
    )
    _add_model(thresholds)
    thresholds.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the thresholds. Default: --save.",
    )
    _add_seed(thresholds, "the simulated test set")
    thresholds.set_defaults(handler=_eval_thresholds)

    compare = eval_commands.add_parser(
        "compare",
        help="Compare models in one ROC plot.",
        description="Compare models in one ROC plot on the same simulated test set.",
    )
    compare.add_argument(
        "--model",
        action="append",
        help=f"One model to compare, repeatable: {MODEL_HELP}. Default: all models.",
    )
    _add_seed(compare, "the simulated test set")
    compare.set_defaults(handler=_eval_compare)

    leaderboard = eval_commands.add_parser(
        "leaderboard", help="Show the leaderboard, best ROC AUC first."
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
    app.set_defaults(handler=_app)

    # gwd env
    env = commands.add_parser(
        "env", help="Show versions, extras, waveform backend, paths, and GUI backend."
    )
    env.set_defaults(handler=_env)

    return parser


def main(argv=None):
    """Run gwd with argv (default: sys.argv[1:]) and return the exit code."""
    args = build_parser().parse_args(argv)
    try:
        returncode = args.handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0 if returncode is None else int(returncode)


if __name__ == "__main__":
    sys.exit(main())
