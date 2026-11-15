# File: GWD/main.py
"""Menu for the GWD scripts. Uses only the standard library and works from any folder."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# (choice, label, kind, target). 'module' runs `python -m target`; 'path' runs the file
# target, relative to this file. Only modules in the gwd package are listed as 'module'.
MENU = [
    ("1", "[DATA]   Fetch real noise", "module", "gwd.data.fetch_noise"),
    ("2", "[DATA]   Generate training data", "module", "gwd.data.generate"),
    ("3", "[DATA]   Validate dataset", "module", "gwd.data.validate"),
    ("4", "[TRAIN]  Train CNN model", "module", "gwd.training.train"),
    (
        "5",
        "[EVAL]   Evaluate model (ROC & plots)",
        "path",
        "src/evaluate_model_with_roc.py",
    ),
    ("6", "[EVAL]   Optimize threshold", "path", "src/optimize_threshold.py"),
    ("7", "[EVAL]   Compare models", "path", "src/compare_models_roc.py"),
    ("8", "[APP]    Glitch Hunter (interactive)", "path", "src/glitch_hunter_app.py"),
    ("9", "[APP]    Waveform simulator", "path", "src/gw_simulator.py"),
]


def command(kind, target):
    """The argument list for one menu entry, independent of the working directory."""
    if kind == "module":
        return [sys.executable, "-m", target]
    if kind == "path":
        return [sys.executable, os.path.join(ROOT, target)]
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
