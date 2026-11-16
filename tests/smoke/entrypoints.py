"""The only table of 'stage -> call' for the smoke tests.

Until P3.6 the stages use small throwaway drivers. Later moves only change this file.
"""

from dataclasses import dataclass
from typing import Optional

SEED = 1  # Chosen so the old validator does not report 'KRITISCH' by chance.
NUM_SAMPLES = 64
NO_NETWORK_REAL_RESULT = {"found": 0, "total": 0, "far_text": "n/a", "event_scores": {}}


@dataclass(frozen=True)
class Stage:
    args: tuple
    stdin: Optional[str] = None


def _module(name, stdin=None):
    return Stage(("-m", name), stdin)


def _driver(module, body, tf_seed=True, stdin=None):
    """Import a gwd module, set the seeds, then run body with it as 'module'."""
    seed = (
        f"tf.keras.utils.set_random_seed({SEED})"
        if tf_seed
        else f"random.seed({SEED}); np.random.seed({SEED})"
    )
    imports = "import random, numpy as np" + (
        "; import tensorflow as tf" if tf_seed else ""
    )
    code = "\n".join(
        [
            imports,
            f"import {module} as module",
            seed,
            body,
        ]
    )
    return Stage(("-c", code), stdin)


PIPELINE = {
    "generate": _driver(
        "gwd.data.generate",
        f"module.generate_dataset({NUM_SAMPLES}, module.DATA_DIR_TRAIN)",
        tf_seed=False,
    ),
    "validate": _module("gwd.data.validate"),
    "train": _driver("gwd.training.train", "module.EPOCHS = 1\nmodule.main()"),
    "evaluate": _driver(
        "gwd.evaluation.evaluate",
        f"module.evaluate_real_world = lambda model: {NO_NETWORK_REAL_RESULT!r}\nmodule.main()",
    ),
    "thresholds": _driver("gwd.evaluation.thresholds", "module.main()", stdin="y\n"),
    # compare stops without a plot when it finds one model, so it gets the same model twice.
    "compare": _driver(
        "gwd.evaluation.compare",
        "loaded = module.load_all_models\nmodule.load_all_models = lambda: loaded() * 2\nmodule.main()",
    ),
    "show_leaderboard": _module("gwd.evaluation.leaderboard"),
}

APPS = {
    "waveform_explorer": _module("gwd.apps.waveform_explorer"),
    "interferometer_designer": _module("gwd.apps.interferometer_designer"),
    "glitch_hunter": _module("gwd.apps.glitch_hunter.app"),
}
