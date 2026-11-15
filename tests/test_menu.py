"""The menu in main.py lists only runnable entries and works from any folder (P2.4)."""

import importlib.util
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(ROOT, "main.py")


def load_main():
    spec = importlib.util.spec_from_file_location("gwd_menu", MAIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MENU = load_main().MENU


def test_menu_choices_are_unique_and_do_not_use_exit():
    choices = [choice for choice, _, _, _ in MENU]
    assert len(choices) == len(set(choices))
    assert "0" not in choices


@pytest.mark.parametrize("entry", MENU, ids=lambda entry: entry[3])
def test_menu_entry_is_runnable(entry):
    _, _, kind, target = entry
    assert kind in ("module", "path")
    if kind == "module":
        # Only gwd modules: the editable install also makes the scripts in src/ importable.
        assert target.startswith("gwd.")
        assert importlib.util.find_spec(target) is not None
    else:
        assert os.path.isfile(os.path.join(ROOT, target))


def test_path_commands_do_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    menu = load_main()
    monkeypatch.chdir(tmp_path)
    for _, _, kind, target in menu.MENU:
        args = menu.command(kind, target)
        assert args[0] == sys.executable
        if kind == "path":
            assert os.path.isabs(args[1]) and os.path.isfile(args[1])


def test_run_script_returns_the_real_exit_code(tmp_path, monkeypatch, capsys):
    menu = load_main()
    script = tmp_path / "fails.py"
    script.write_text("raise SystemExit(3)\n")
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert menu.run_script("path", str(script)) == 3
    assert "exit code 3" in capsys.readouterr().out


def test_menu_exits_with_zero_from_another_folder(tmp_path):
    result = subprocess.run(
        [sys.executable, MAIN],
        input="0\n",
        cwd=tmp_path,
        env={**os.environ, "TERM": "dumb"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
