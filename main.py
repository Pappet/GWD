"""Menu for the gwd command. Uses only the standard library and works from any folder.

Each entry is an argument list for gwd and runs as ``python -m gwd.cli ...`` in a
subprocess. tests/test_menu.py parses every list with gwd.cli.build_parser(), so the
menu and the CLI cannot drift apart. Exits with 0 on '0' or end of input.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# (text, gwd arguments). The choice is the position, starting at 1.
MENU = [
    ("[DATA]   Fetch real noise (network)", ["data", "fetch-noise"]),
    ("[DATA]   Generate training data", ["data", "generate"]),
    ("[DATA]   Validate dataset", ["data", "validate"]),
    ("[TRAIN]  Train CNN model", ["train"]),
    ("[EVAL]   Evaluate model (ROC, plots, field test)", ["eval", "run"]),
    ("[EVAL]   Optimize threshold", ["eval", "thresholds"]),
    ("[EVAL]   Compare models", ["eval", "compare"]),
    ("[EVAL]   Show leaderboard", ["eval", "leaderboard"]),
    ("[APP]    Glitch Hunter (interactive)", ["app", "glitch-hunter"]),
    ("[APP]    Waveform explorer", ["app", "waveform"]),
    ("[APP]    Interferometer designer", ["app", "interferometer"]),
    ("[INFO]   Show environment", ["env"]),
]


def command(args):
    """The command line for one menu entry, independent of the working directory."""
    return [sys.executable, "-m", "gwd.cli", *args]


def ask(prompt):
    """input() that returns None at the end of input instead of raising EOFError."""
    try:
        return input(prompt)
    except EOFError:
        print()
        return None


def clear_screen():
    """Clear the terminal with cls or clear, without a shell; only on a terminal."""
    if not sys.stdout.isatty():
        return
    clear = ["cmd", "/c", "cls"] if os.name == "nt" else ["clear"]
    try:
        subprocess.run(clear, check=False)
    except OSError:
        pass


def run_entry(args):
    """Run one menu entry in the project folder and return its exit code."""
    print(f"\nStarting: gwd {' '.join(args)}\n" + "-" * 50, flush=True)
    returncode = subprocess.run(command(args), cwd=ROOT).returncode
    if returncode == 0:
        print("\nFinished successfully.")
    else:
        print(f"\nFailed with exit code {returncode}.")
    return returncode


def main_menu():
    """Show the menu until '0' or the end of input. Returns the exit code.

    That is 0, or after an entry ran, that entry's exit code if the input ends at the
    '[Enter]' prompt that follows it.
    """
    while True:
        clear_screen()
        print("==========================================")
        print("GRAVITATIONAL WAVE DETECTOR - HUB")
        print("==========================================")
        for number, (text, _) in enumerate(MENU, 1):
            print(f"{number:>2}. {text}")
        print(" 0. Exit")
        print("==========================================")

        choice = ask("Choose an option: ")
        if choice is None or choice.strip() == "0":
            return 0
        choice = choice.strip()

        if choice.isdigit() and 1 <= int(choice) <= len(MENU):
            returncode = run_entry(MENU[int(choice) - 1][1])
        else:
            print(f"Invalid choice: {choice!r}")
            returncode = 0
        if ask("\n[Enter] to return to the menu...") is None:
            return returncode


if __name__ == "__main__":
    try:
        sys.exit(main_menu())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
