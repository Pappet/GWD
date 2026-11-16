"""The gwd command: help, exit codes, lazy imports, and seeds (P3.6). No TensorFlow needed."""

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gwd import cli

ROOT = Path(__file__).resolve().parent.parent
GWD = Path(sys.executable).parent / "gwd"
UNKNOWN_MODEL = "gwd_model_19990101-000000"


def subcommands(parser, prefix=()):
    """Yield the argument path of every command and subcommand, e.g. ('eval', 'run')."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                yield (*prefix, name)
                yield from subcommands(sub, (*prefix, name))


COMMANDS = list(subcommands(cli.build_parser()))


def run_gwd(args, home, **kwargs):
    """Run the installed gwd script with GWD_HOME in home."""
    env = {**os.environ, "GWD_HOME": str(home), "MPLBACKEND": "Agg"}
    return subprocess.run(
        [str(GWD), *args], cwd=home, env=env, capture_output=True, text=True, **kwargs
    )


def run_python(code, home):
    env = {**os.environ, "GWD_HOME": str(home), "MPLBACKEND": "Agg"}
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=home,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def test_console_script_lists_all_commands(tmp_path):
    result = run_gwd(["--help"], tmp_path)
    assert result.returncode == 0, result.stderr
    for command in ("data", "train", "eval", "app", "env"):
        assert f"    {command} " in result.stdout


def test_every_command_is_known():
    assert set(COMMANDS) == {
        ("data",),
        ("data", "fetch-noise"),
        ("data", "generate"),
        ("data", "validate"),
        ("train",),
        ("eval",),
        ("eval", "run"),
        ("eval", "thresholds"),
        ("eval", "compare"),
        ("eval", "leaderboard"),
        ("app",),
        ("env",),
    }


@pytest.mark.parametrize("command", COMMANDS, ids=" ".join)
def test_help_exits_with_zero(command, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([*command, "--help"])
    assert exit_info.value.code == 0
    assert "usage: gwd" in capsys.readouterr().out


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["bogus"],
        ["data"],
        ["eval", "run", "--no-such-option"],
        ["data", "generate", "--n", "many"],
        ["data", "generate", "--scenario", "train_v9"],
        ["app", "bogus"],
    ],
    ids=lambda args: " ".join(args) or "no arguments",
)
def test_invalid_arguments_exit_with_two(args, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(args)
    assert exit_info.value.code == 2


@pytest.mark.parametrize(
    "args",
    [
        ["eval", "run", "--model", UNKNOWN_MODEL],
        ["eval", "thresholds", "--model", UNKNOWN_MODEL],
        ["eval", "compare", "--model", UNKNOWN_MODEL],
        ["eval", "compare"],
        ["data", "validate"],
    ],
    ids=" ".join,
)
def test_failing_command_exits_with_one(args, tmp_path):
    # An empty project: no models, no data/training/labels.csv.
    result = run_gwd(args, tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr


def test_thresholds_do_not_ask_for_input(tmp_path):
    result = run_gwd(
        ["eval", "thresholds", "--model", UNKNOWN_MODEL, "--no-save"],
        tmp_path,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 1
    assert "EOFError" not in result.stderr


def test_no_input_calls_in_the_package():
    calls = []
    for path in sorted((ROOT / "src" / "gwd").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "input"
            ):
                calls.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert calls == []


def test_generate_with_a_seed_is_reproducible_and_loads_no_tensorflow(tmp_path):
    code = "\n".join(
        [
            "import sys",
            "from gwd.cli import main",
            "for name in ('a', 'b', 'c'):",
            "    seed = '4' if name == 'c' else '3'",
            "    args = ['data', 'generate', '--n', '6', '--seed', seed, '--out', name]",
            "    assert main(args) == 0",
            "print('tensorflow' in sys.modules, 'gwpy' in sys.modules)",
        ]
    )
    result = run_python(code, tmp_path)
    assert result.stdout.splitlines()[-1] == "False False"

    labels = {name: (tmp_path / name / "labels.csv").read_text() for name in "abc"}
    assert labels["a"] == labels["b"]
    assert labels["a"] != labels["c"]
    for i in range(6):
        sample = f"sample_{i:05d}.npy"
        np.testing.assert_array_equal(
            np.load(tmp_path / "a" / sample), np.load(tmp_path / "b" / sample)
        )


def test_generate_takes_the_scenario(tmp_path):
    result = run_gwd(
        ["data", "generate", "--n", "2", "--scenario", "EVAL_V1", "--out", "d"],
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "EVAL_V1" in result.stdout
    assert (tmp_path / "d" / "labels.csv").exists()


def test_help_and_the_parser_load_no_heavy_modules(tmp_path):
    code = "\n".join(
        [
            "import sys",
            "from gwd.cli import build_parser",
            "build_parser()",
            "heavy = ('tensorflow', 'gwpy', 'pycbc', 'matplotlib', 'pandas', 'numpy')",
            "print([name for name in heavy if name in sys.modules])",
        ]
    )
    assert run_python(code, tmp_path).stdout.strip() == "[]"


def test_env_reports_paths_and_loads_no_tensorflow(tmp_path):
    code = "\n".join(
        [
            "import sys",
            "from gwd.cli import main",
            "rc = main(['env'])",
            "print(rc, 'tensorflow' in sys.modules, 'gwpy' in sys.modules)",
        ]
    )
    result = run_python(code, tmp_path)
    assert result.stdout.splitlines()[-1] == "0 False False"
    assert f"GWD_HOME         {tmp_path}" in result.stdout
    assert f"Models           {tmp_path / 'models_registry'}" in result.stdout
    assert "Waveform backend:" in result.stdout
    assert "matplotlib       Agg (not interactive)" in result.stdout


def test_seed_everything_seeds_random_and_numpy():
    import random

    cli.seed_everything(5)
    first = (random.random(), np.random.rand())
    cli.seed_everything(5)
    assert (random.random(), np.random.rand()) == first


def test_seed_none_leaves_the_rngs_alone(monkeypatch):
    monkeypatch.setattr(np.random, "seed", lambda seed: pytest.fail("seeded"))
    cli.seed_everything(None)


class FakeTraining:
    """Stands in for gwd.training.train, so the train handler runs without TensorFlow."""

    def __init__(self):
        self.calls = []

    def TrainConfig(self, **options):
        return options

    def train(self, cfg):
        self.calls.append(cfg)


@pytest.fixture
def fake_training(monkeypatch, tmp_path):
    fake = FakeTraining()
    monkeypatch.setitem(sys.modules, "gwd.training.train", fake)
    # The handler checks for labels.csv in the default dataset folder first.
    (tmp_path / "labels.csv").touch()
    monkeypatch.setattr("gwd.config.DATA_DIR_TRAIN", str(tmp_path))
    for name in ("TF_CPP_MIN_LOG_LEVEL", "TF_ENABLE_ONEDNN_OPTS"):
        # setenv first, so monkeypatch also undoes what the handler sets.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    return fake


def test_train_without_options_uses_the_config_defaults(fake_training):
    assert cli.main(["train"]) == 0
    # Only what is given on the command line is passed; TrainConfig has the rest.
    assert fake_training.calls == [{}]
    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "2"
    assert os.environ["TF_ENABLE_ONEDNN_OPTS"] == "0"


def test_train_passes_the_overrides(fake_training, monkeypatch, tmp_path):
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "0")
    args = f"train --data {tmp_path} --epochs 3 --batch-size 16 --lr 0.001"
    args += " --patience 2 --mixed-precision --seed 5"
    assert cli.main(args.split()) == 0
    assert fake_training.calls == [
        {
            "data_folder": str(tmp_path),
            "epochs": 3,
            "batch_size": 16,
            "learning_rate": 0.001,
            "patience": 2,
            "mixed_precision": True,
            "seed": 5,
        }
    ]
    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "0"


def test_train_without_a_dataset_fails_cleanly_and_loads_no_tensorflow(tmp_path):
    """P3.7 left a traceback here; now it is one message on stderr and rc 1."""
    code = "\n".join(
        [
            "import sys",
            "from gwd.cli import main",
            "rc = main(['train', '--data', 'missing'])",
            "print(rc, 'tensorflow' in sys.modules)",
        ]
    )
    result = run_python(code, tmp_path)
    assert result.stdout.splitlines()[-1] == "1 False"
    assert "No dataset in 'missing': labels.csv is missing" in result.stderr
    assert "gwd data generate" in result.stderr
    assert "Traceback" not in result.stderr


def test_train_with_the_default_dataset_missing_exits_with_one(tmp_path):
    result = run_gwd(["train"], tmp_path)
    assert result.returncode == 1
    assert "labels.csv is missing" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "argv",
    [
        ["eval", "thresholds"],
        ["eval", "compare", "--model", "a", "--model", "b"],
    ],
)
def test_without_pycbc_a_command_that_draws_signals_exits_with_1(argv, tmp_path):
    """P5.2: no Newton fallback, no traceback; the hint names the install command."""
    for name in ("a", "b"):
        (tmp_path / "models_registry").mkdir(exist_ok=True)
        (tmp_path / "models_registry" / f"{name}.keras").touch()
    code = "\n".join(
        [
            "import sys",
            "sys.modules['pycbc'] = None",
            "from gwd.cli import main",
            f"rc = main({argv!r})",
            "print(rc, 'tensorflow' in sys.modules)",
        ]
    )
    result = run_python(code, tmp_path)
    assert result.stdout.splitlines()[-1] == "1 False"
    assert "pip install -e '.[physics]'" in result.stderr
    assert "Traceback" not in result.stderr
