# File: GWD/main.py
"""Menu for the GWD scripts. Uses only the standard library and works from any folder."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# (choice, label, kind, target). 'module' runs `python -m target`; since P2.5 every entry
# is a module in the gwd package.
MENU = [
    ("1", "[DATA]   Fetch real noise", "module", "gwd.data.fetch_noise"),
    ("2", "[DATA]   Generate training data", "module", "gwd.data.generate"),
    ("3", "[DATA]   Validate dataset", "module", "gwd.data.validate"),
    ("4", "[TRAIN]  Train CNN model", "module", "gwd.training.train"),
    ("5", "[EVAL]   Evaluate model (ROC & plots)", "module", "gwd.evaluation.evaluate"),
    ("6", "[EVAL]   Optimize threshold", "module", "gwd.evaluation.thresholds"),
    ("7", "[EVAL]   Compare models", "module", "gwd.evaluation.compare"),
    ("8", "[EVAL]   Show leaderboard", "module", "gwd.evaluation.leaderboard"),
    (
        "9",
        "[APP]    Glitch Hunter (interactive)",
        "module",
        "gwd.apps.glitch_hunter.app",
    ),
    ("10", "[APP]    Waveform explorer", "module", "gwd.apps.waveform_explorer"),
    (
        "11",
        "[APP]    Interferometer designer",
        "module",
        "gwd.apps.interferometer_designer",
    ),
]


def command(kind, target):
    """The argument list for one menu entry, independent of the working directory."""
    if kind == "module":
        return [sys.executable, "-m", target]
    raise ValueError(f"Unknown menu entry kind: {kind!r}")


def run_script(kind, target):
    """Run one menu entry in the project folder and return its exit code."""
    print(f"\n🚀 Starting: {target}...\n" + "-" * 50)
    returncode = subprocess.run(command(kind, target), cwd=ROOT).returncode
    if returncode == 0:
        print("\n✅ Finished successfully.")
    else:
        print(f"\n❌ Failed with exit code {returncode}.")
    input("\n[Enter] to return to the menu...")
    return returncode


def main_menu():
    entries = {choice: (kind, target) for choice, _, kind, target in MENU}
    while True:
        # Screen clear (optional)
        os.system("cls" if os.name == "nt" else "clear")

        print("==========================================")
        print("🌊 GRAVITATIONAL WAVE DETECTOR - HUB 🌊")
        print("==========================================")
        for choice, label, _, _ in MENU:
            print(f"{choice}. {label}")
        print("0. Exit")
        print("==========================================")

        choice = input("Choose an option: ")

        if choice in entries:
            run_script(*entries[choice])
        elif choice == "0":
            sys.exit()
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main_menu()
