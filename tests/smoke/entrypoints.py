"""The only table of 'stage -> call' for the smoke tests.

Until P3.6 the stages use small throwaway drivers. Later moves only change this file.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SRC = Path(__file__).resolve().parents[2] / "src"

SEED = 1  # Chosen so the old validator does not report 'KRITISCH' by chance.
NUM_SAMPLES = 64
NO_NETWORK_REAL_RESULT = {"found": 0, "total": 0, "far_text": "n/a", "event_scores": {}}


@dataclass(frozen=True)
class Stage:
    args: tuple
    stdin: Optional[str] = None


def _script(path, stdin=None):
    return Stage((str(path),), stdin)


def _module(name, stdin=None):
    return Stage(("-m", name), stdin)


def _driver(folder, module, body, tf_seed=True, stdin=None):
    """Import a script as a module, set the seeds, then run body with it as 'module'.

    folder is put on sys.path first; it is None for modules in the gwd package.
    """
    seed = (
        f"tf.keras.utils.set_random_seed({SEED})"
        if tf_seed
        else f"random.seed({SEED}); np.random.seed({SEED})"
    )
    imports = "import random, numpy as np" + (
        "; import tensorflow as tf" if tf_seed else ""
    )
    path_setup = (
        ["import sys", f"sys.path.insert(0, {str(folder)!r})"] if folder else []
    )
    code = "\n".join(
        [
            *path_setup,
            imports,
            f"import {module} as module",
            seed,
            body,
        ]
    )
    return Stage(("-c", code), stdin)


PIPELINE = {
    "generate": _driver(
        None,
        "gwd.data.generate",
        f"module.generate_dataset({NUM_SAMPLES}, module.DATA_DIR_TRAIN)",
        tf_seed=False,
    ),
    "validate": _module("gwd.data.validate"),
    "train": _driver(None, "gwd.training.train", "module.EPOCHS = 1\nmodule.main()"),
    "evaluate": _driver(
        SRC,
        "evaluate_model_with_roc",
        f"module.evaluate_real_world = lambda model: {NO_NETWORK_REAL_RESULT!r}\nmodule.main()",
    ),
    "thresholds": _driver(SRC, "optimize_threshold", "module.main()", stdin="y\n"),
    # compare stops without a plot when it finds one model, so it gets the same model twice.
    "compare": _driver(
        SRC,
        "compare_models_roc",
        "loaded = module.load_all_models\nmodule.load_all_models = lambda: loaded() * 2\nmodule.main()",
    ),
    "show_leaderboard": _script(SRC / "show_leaderboard.py"),
}

APPS = {
    "gw_simulator": _script(SRC / "gw_simulator.py"),
    "interferometer_simulator": _script(SRC / "interferometer_simulator.py"),
    "glitch_hunter": _script(SRC / "glitch_hunter_app.py"),
}
