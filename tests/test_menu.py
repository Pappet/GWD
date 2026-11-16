"""The menu in main.py runs gwd commands, works from any folder, and exits cleanly (P3.6)."""

import importlib.util
import os
import subprocess
import sys

import pytest

from gwd.cli import build_parser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(ROOT, "main.py")


def load_main():
    spec = importlib.util.spec_from_file_location("gwd_menu", MAIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MENU = load_main().MENU


def run_menu(stdin, cwd):
    return subprocess.run(
        [sys.executable, MAIN],
        input=stdin,
        cwd=cwd,
        env={**os.environ, "TERM": "dumb"},
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("entry", MENU, ids=lambda entry: " ".join(entry[1]))
def test_every_menu_entry_parses_as_a_gwd_command(entry):
    _, args = entry
    # parse_args raises SystemExit on an unknown command or option.
    parsed = build_parser().parse_args(args)
    assert callable(parsed.handler)


def test_menu_covers_every_command_once():
    commands = [tuple(args) for _, args in MENU]
    assert len(commands) == len(set(commands)) == 13


def test_only_the_field_test_entry_needs_the_network():
    """gwd eval run is offline by default (D10); one entry adds --real (P4.3)."""
    evaluations = [args for _, args in MENU if args[:2] == ["eval", "run"]]
    assert evaluations == [["eval", "run"], ["eval", "run", "--real"]]
    assert build_parser().parse_args(["eval", "run"]).real is False


def test_commands_do_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    menu = load_main()
    monkeypatch.chdir(tmp_path)
    for _, args in menu.MENU:
        assert menu.command(args) == [sys.executable, "-m", "gwd.cli", *args]


def test_run_entry_returns_the_real_exit_code(capfd):
    menu = load_main()
    # An unknown option: argparse exits with 2 before any module is imported.
    assert menu.run_entry(["eval", "run", "--no-such-option"]) == 2
    assert "exit code 2" in capfd.readouterr().out


def test_menu_exits_with_zero_from_another_folder(tmp_path):
    result = run_menu("0\n", tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr


def test_end_of_input_exits_with_zero(tmp_path):
    result = subprocess.run(
        [sys.executable, MAIN],
        stdin=subprocess.DEVNULL,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    assert "EOFError" not in result.stderr


def test_invalid_choice_then_end_of_input(tmp_path):
    result = run_menu("99\n", tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Invalid choice: '99'" in result.stdout
    assert "Traceback" not in result.stderr


def test_menu_reports_the_exit_code_of_a_failed_entry(tmp_path):
    # The validate entry checks data/training/ of an empty project: no labels.csv, rc 1.
    choice = next(
        i for i, (_, args) in enumerate(MENU, 1) if args == ["data", "validate"]
    )
    env = {**os.environ, "TERM": "dumb", "GWD_HOME": str(tmp_path), "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, MAIN],
        input=f"{choice}\n",
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert "Failed with exit code 1." in result.stdout
    # The input ends at the [Enter] prompt, so the menu exits with that entry's code.
    assert result.returncode == 1
    # The Starting line is flushed before the child writes.
    assert result.stdout.index("Starting: gwd data validate") < result.stdout.index(
        "Dataset folder:"
    )


def test_only_generate_asks_and_its_answer_parses():
    menu = load_main()
    assert list(menu.CONFIRM) == [("data", "generate")]
    for args, (question, extra) in menu.CONFIRM.items():
        assert question.rstrip().endswith("[y/N]")
        parsed = build_parser().parse_args([*args, *extra])
        assert parsed.overwrite is True


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("y", ["data", "generate", "--overwrite"]),
        ("YES ", ["data", "generate", "--overwrite"]),
        ("n", ["data", "generate"]),
        ("", ["data", "generate"]),
        (None, None),
    ],
)
def test_confirm_adds_overwrite_only_on_yes(monkeypatch, answer, expected):
    menu = load_main()
    monkeypatch.setattr(menu, "ask", lambda prompt: answer)
    assert menu.confirm(["data", "generate"]) == expected
    # Entries without a question run as they are, without asking.
    monkeypatch.setattr(menu, "ask", lambda prompt: pytest.fail("asked"))
    assert menu.confirm(["data", "validate"]) == ["data", "validate"]


def test_no_to_the_question_keeps_an_existing_dataset(tmp_path):
    training = tmp_path / "data" / "training"
    training.mkdir(parents=True)
    (training / "labels.csv").write_text("filename,has_signal\n")
    choice = next(
        i for i, (_, args) in enumerate(MENU, 1) if args == ["data", "generate"]
    )
    env = {**os.environ, "TERM": "dumb", "GWD_HOME": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, MAIN],
        input=f"{choice}\nn\n",
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert "Replace an existing dataset" in result.stdout
    assert "Starting: gwd data generate\n" in result.stdout
    assert "Failed with exit code 1." in result.stdout
    assert "--overwrite" in result.stderr
    assert [p.name for p in training.iterdir()] == ["labels.csv"]
