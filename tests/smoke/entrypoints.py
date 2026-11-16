"""The only table of 'app -> gwd command' for the smoke tests.

Since P3.6 every app is started as a real gwd command line, run as ``python -m
gwd.cli``. The pipeline stages moved to tests/test_e2e.py in P7.4.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Stage:
    args: tuple
    stdin: Optional[str] = None


APPS = {
    "waveform_explorer": Stage(("app", "waveform")),
    "interferometer_designer": Stage(("app", "interferometer")),
    "glitch_hunter": Stage(("app", "glitch-hunter")),
}
