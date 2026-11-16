"""The only table of 'stage -> gwd command' for the smoke tests.

Since P3.6 every stage is a real gwd command line, run as ``python -m gwd.cli``. Later
moves only change this file.
"""

from dataclasses import dataclass
from typing import Optional

SEED = "1"
NUM_SAMPLES = 64


@dataclass(frozen=True)
class Stage:
    args: tuple
    stdin: Optional[str] = None


PIPELINE = {
    "generate": Stage(("data", "generate", "--n", str(NUM_SAMPLES), "--seed", SEED)),
    "validate": Stage(("data", "validate")),
    "train": Stage(("train", "--epochs", "1", "--seed", SEED)),
    # P5.7: the fixed test set; evaluate, thresholds, and compare then load it.
    "build_testset": Stage(("eval", "build-testset", "--seed", SEED)),
    # Saves by default; the empty stdin shows that it no longer asks (EVAL-17). Since
    # P5.8 it runs before evaluate, which then uses the calibrated threshold.
    "thresholds": Stage(("eval", "thresholds", "--seed", SEED), stdin=""),
    "evaluate": Stage(("eval", "run", "--no-real", "--seed", SEED)),
    # compare stops without a plot when it finds one model, so it gets the same model twice.
    "compare": Stage(
        ("eval", "compare", "--model", "latest", "--model", "latest", "--seed", SEED)
    ),
    "show_leaderboard": Stage(
        (
            "eval",
            "leaderboard",
        )
    ),
}

APPS = {
    "waveform_explorer": Stage(("app", "waveform")),
    "interferometer_designer": Stage(("app", "interferometer")),
    "glitch_hunter": Stage(("app", "glitch-hunter")),
}
