"""Play a sample as sound, without blocking the GUI (P4.4, APP-9).

The sound file lives in a temporary folder that Player removes on close(), not in the
working directory. Each platform has its own branch: winsound on Windows, afplay on
macOS, and paplay (PulseAudio and PipeWire, also WSLg) or else aplay (ALSA) on Linux.
The players run with subprocess.Popen, so play() returns at once. A new sound stops
the one before it.
"""

import os
import platform
import shutil
import subprocess
import tempfile

import numpy as np
import scipy.io.wavfile as wavfile

# Played at twice the sample rate, so the chirp sweeps through the audible range faster.
PLAYBACK_SPEED = 2.0
# Peak amplitude of the written sound, below 1 to leave headroom.
PEAK = 0.8
FILE_NAME = "sample.wav"


def to_audio(data):
    """data scaled to the peak PEAK as float32; silence for data without a peak."""
    data = np.asarray(data, dtype=np.float64)
    peak = np.max(np.abs(data)) if data.size else 0.0
    if peak == 0 or not np.isfinite(peak):
        return np.zeros(data.shape, dtype=np.float32)
    return (data / peak * PEAK).astype(np.float32)


def player_command(path, system=None, which=shutil.which):
    """The command that plays path on this platform, or None if there is none.

    Windows plays through winsound and needs no command, so it returns None too.
    """
    system = system or platform.system()
    if system == "Darwin":
        return ["afplay", path]
    if system == "Linux":
        for player in ("paplay", "aplay"):
            if which(player):
                return [player, path]
    return None


class Player:
    """Writes each sound into its own temporary folder and plays it in the background."""

    def __init__(self, system=None, popen=subprocess.Popen, which=shutil.which):
        self.system = system or platform.system()
        self.popen = popen
        self.which = which
        self.folder = tempfile.TemporaryDirectory(
            prefix="gwd-glitch-hunter-", ignore_cleanup_errors=True
        )
        self.process = None

    @property
    def path(self):
        return os.path.join(self.folder.name, FILE_NAME)

    def play(self, data, sample_rate):
        """Write data as a WAV file and start playing it.

        Returns None on success, else a message that says why nothing plays.
        """
        self.stop()
        rate = int(sample_rate * PLAYBACK_SPEED)
        wavfile.write(self.path, rate, to_audio(data))
        if self.system == "Windows":
            import winsound

            winsound.PlaySound(self.path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            return None
        command = player_command(self.path, self.system, self.which)
        if command is None:
            if self.system == "Linux":
                return "No audio player found: install paplay (pulseaudio-utils) or aplay (alsa-utils)."
            return f"No audio player known for {self.system}."
        try:
            self.process = self.popen(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except OSError as error:
            return f"Cannot start {command[0]}: {error}"
        return None

    def stop(self):
        """Stop the sound that is playing, if any."""
        if self.system == "Windows":
            import winsound

            winsound.PlaySound(None, 0)
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        self.process = None

    def close(self):
        """Stop playing and remove the temporary folder."""
        self.stop()
        self.folder.cleanup()
