"""The menu in main.py lists only runnable entries and works from any folder (P2.4, P2.5)."""

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


def test_menu_lists_every_entry_point_as_a_module():
    assert len(MENU) == 11
    assert all(kind == "module" for _, _, kind, _ in MENU)


@pytest.mark.parametrize("entry", MENU, ids=lambda entry: entry[3])
def test_menu_entry_is_runnable(entry):
    _, _, kind, target = entry
    assert kind == "module"
    assert target.startswith("gwd.")
    assert importlib.util.find_spec(target) is not None


def test_commands_do_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    menu = load_main()
    monkeypatch.chdir(tmp_path)
    for _, _, kind, target in menu.MENU:
        assert menu.command(kind, target) == [sys.executable, "-m", target]


def test_unknown_entry_kind_is_rejected():
    with pytest.raises(ValueError):
        load_main().command("path", "src/script.py")


def test_run_script_returns_the_real_exit_code(tmp_path, monkeypatch, capsys):
    menu = load_main()
    (tmp_path / "gwd_menu_fails.py").write_text("raise SystemExit(3)\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert menu.run_script("module", "gwd_menu_fails") == 3
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
