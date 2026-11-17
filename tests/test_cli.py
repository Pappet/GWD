"""The gwd command: help, exit codes, lazy imports, and seeds (P3.6). No TensorFlow needed."""

import argparse
import ast
import os
import subprocess
import sys
import types
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
        ("eval", "build-testset"),
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


@pytest.mark.parametrize(
    "argv, module, expected",
    [
        (["eval", "run"], "evaluate", {"model": "latest", "real": False}),
        (
            ["eval", "run", "--testset", "indist_v1", "--seed", "3", "--n", "10"],
            "evaluate",
            {
                "model": "latest",
                "real": False,
                "testset": "indist_v1",
                "seed": 3,
                "n": 10,
            },
        ),
        (
            ["eval", "thresholds", "--no-save", "--seed", "0"],
            "thresholds",
            {"model": "latest", "save": False, "seed": 0},
        ),
        (
            ["eval", "compare", "--model", "a", "--testset", "stress_v1"],
            "compare",
            {"models": ["a"], "testset": "stress_v1"},
        ),
    ],
)
def test_the_eval_commands_pass_the_test_set(argv, module, expected, monkeypatch):
    """P5.7: --testset, --seed, --n go to the function; nothing seeds the global RNG."""
    calls = []
    target = __import__(f"gwd.evaluation.{module}", fromlist=["run"])
    monkeypatch.setattr(target, "run", lambda **options: calls.append(options) or 0)
    monkeypatch.setattr(np.random, "seed", lambda seed: pytest.fail("seeded"))
    assert cli.main(argv) == 0
    assert calls == [expected]


def test_build_testset_passes_name_seed_n_and_rebuild(monkeypatch, capsys):
    from gwd.evaluation import testsets

    calls = []
    monkeypatch.setattr(
        testsets, "build", lambda **options: calls.append(options) or "built"
    )
    monkeypatch.setattr(testsets, "describe", lambda built: [f"described {built}"])
    argv = ["eval", "build-testset", "--scenario", "indist_v1", "--seed", "7"]
    assert cli.main([*argv, "--n", "64", "--rebuild"]) == 0
    assert calls == [{"name": "indist_v1", "seed": 7, "n": 64, "rebuild": True}]
    assert "described built" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["eval", "run", "--seed", "-1"],
        ["eval", "run", "--n", "1"],
        ["eval", "build-testset", "--testset", "train_v1"],
    ],
)
def test_invalid_test_set_options_exit_with_2(argv):
    with pytest.raises(SystemExit) as exit_:
        cli.main(argv)
    assert exit_.value.code == 2


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
    clear_tf_environment(monkeypatch)
    return fake


def clear_tf_environment(monkeypatch):
    for name in cli.TF_ENVIRONMENT:
        # setenv first, so monkeypatch also undoes what the handler sets.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def test_train_without_options_uses_the_config_defaults(fake_training):
    assert cli.main(["train"]) == 0
    # Only what is given on the command line is passed; TrainConfig has the rest.
    assert fake_training.calls == [{}]
    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "3"
    assert os.environ["TF_ENABLE_ONEDNN_OPTS"] == "0"


# Every handler that can load TensorFlow, with the function it hands over to.
TENSORFLOW_COMMANDS = {
    ("eval", "run"): "gwd.evaluation.evaluate.run",
    ("eval", "thresholds"): "gwd.evaluation.thresholds.run",
    ("eval", "compare"): "gwd.evaluation.compare.run",
    ("app", "glitch-hunter"): "gwd.apps.glitch_hunter.app.main",
}


@pytest.fixture
def tf_handlers(monkeypatch):
    """Stub the functions of TENSORFLOW_COMMANDS; each records the environment it
    sees, which is the one TensorFlow would be imported with."""
    seen = {}
    for command, target in TENSORFLOW_COMMANDS.items():
        module_name, function = target.rsplit(".", 1)

        def record(*args, _command=command, **kwargs):
            seen[_command] = {name: os.environ.get(name) for name in cli.TF_ENVIRONMENT}
            return 0

        if command[0] == "app":
            # A stand-in module, so the test needs no matplotlib GUI or app imports.
            module = types.ModuleType(module_name)
            setattr(module, function, record)
            monkeypatch.setitem(sys.modules, module_name, module)
        else:
            monkeypatch.setattr(target, record)
    clear_tf_environment(monkeypatch)
    return seen


@pytest.mark.parametrize("command", TENSORFLOW_COMMANDS)
def test_every_tensorflow_command_quiets_tensorflow(tf_handlers, command):
    """P3.7 set the variables in gwd train only; eval and the Glitch Hunter too now."""
    assert cli.main(list(command)) == 0
    assert tf_handlers[command] == cli.TF_ENVIRONMENT


@pytest.mark.parametrize("command", TENSORFLOW_COMMANDS)
def test_the_caller_tensorflow_environment_wins(tf_handlers, monkeypatch, command):
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "0")
    monkeypatch.setenv("TF_ENABLE_ONEDNN_OPTS", "1")
    assert cli.main(list(command)) == 0
    assert tf_handlers[command] == {
        "TF_CPP_MIN_LOG_LEVEL": "0",
        "TF_ENABLE_ONEDNN_OPTS": "1",
    }


def test_train_passes_the_overrides(fake_training, monkeypatch, tmp_path):
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "0")
    args = f"train --data {tmp_path} --epochs 3 --batch-size 16 --lr 0.001"
    args += " --patience 2 --mixed-precision --deterministic --seed 5"
    assert cli.main(args.split()) == 0
    assert fake_training.calls == [
        {
            "data_folder": str(tmp_path),
            "epochs": 3,
            "batch_size": 16,
            "learning_rate": 0.001,
            "patience": 2,
            "mixed_precision": True,
            "deterministic": True,
            "seed": 5,
        }
    ]
    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "0"


def test_train_online_passes_the_scenario_and_needs_no_dataset(
    fake_training, monkeypatch, tmp_path
):
    # No labels.csv anywhere: online training reads no dataset.
    monkeypatch.setattr("gwd.config.DATA_DIR_TRAIN", str(tmp_path / "missing"))
    args = "train --online --scenario train_v3 --samples-per-epoch 256"
    args += " --val-samples 64 --test-samples 32 --workers 3 --seed 2"
    assert cli.main(args.split()) == 0
    assert fake_training.calls == [
        {
            "seed": 2,
            "scenario": "TRAIN_V3",
            "samples_per_epoch": 256,
            "val_samples": 64,
            "test_samples": 32,
            "workers": 3,
        }
    ]


@pytest.mark.parametrize(
    "args, message",
    [
        ("--scenario train_v3", "--scenario needs --online"),
        ("--online", "--online needs --scenario"),
        ("--online --scenario train_v1", "TRAIN_V1 uses real noise"),
        ("--online --scenario train_v2 --data x", "reads no --data"),
        ("--workers 2 --val-samples 9", "--val-samples, --workers only work with"),
    ],
)
def test_train_refuses_online_options_that_do_not_fit(
    fake_training, capsys, args, message
):
    assert cli.main(["train", *args.split()]) == 2
    assert message in capsys.readouterr().err
    assert fake_training.calls == []


@pytest.mark.parametrize(
    "args", ["--workers -1", "--samples-per-epoch 0", "--scenario nope"]
)
def test_train_rejects_invalid_online_values(args):
    with pytest.raises(SystemExit) as exit_:
        cli.main(["train", "--online", *args.split()])
    assert exit_.value.code == 2


def test_train_passes_the_telescope_options(fake_training, monkeypatch, tmp_path):
    monkeypatch.setattr("gwd.config.DATA_DIR_TRAIN", str(tmp_path / "missing"))
    args = "train --online --scenario train_v3 --arch telescope --lens-start newton"
    args += " --telescope-train lens --mass-ratios 1,2.5"
    assert cli.main(args.split()) == 0
    assert fake_training.calls == [
        {
            # The telescope's own default; --batch-size overrides it.
            "batch_size": 32,
            "architecture": "telescope",
            "lens_start": "newton",
            "telescope_train": "lens",
            "mass_ratios": (1.0, 2.5),
            "scenario": "TRAIN_V3",
        }
    ]
    assert cli.main([*args.split(), "--batch-size", "8"]) == 0
    assert fake_training.calls[1]["batch_size"] == 8
    # --arch cnn is the default and passes nothing.
    assert cli.main("train --online --scenario train_v3 --arch cnn".split()) == 0
    assert fake_training.calls[2] == {"scenario": "TRAIN_V3"}


@pytest.mark.parametrize(
    "args, message",
    [
        ("--lens-start newton", "--lens-start only work with --arch telescope"),
        ("--arch cnn --mass-ratios 1", "--mass-ratios only work with --arch telescope"),
        ("--arch telescope --mixed-precision", "computes in float32"),
        ("--arch telescope --online --scenario app_v3", "APP_V3 is not one"),
        (
            "--arch telescope --online --scenario train_v3 --mass-ratios 1,30",
            "--mass-ratios must be at most 20; got 30",
        ),
    ],
)
def test_train_refuses_telescope_options_that_do_not_fit(
    fake_training, capsys, args, message
):
    assert cli.main(["train", *args.split()]) == 2
    assert message in capsys.readouterr().err
    assert fake_training.calls == []


@pytest.mark.parametrize("ratios", ["0.5", "1,one", "2,2", ""])
def test_train_rejects_invalid_mass_ratios(ratios):
    with pytest.raises(SystemExit) as exit_:
        cli.main(["train", "--arch", "telescope", "--mass-ratios", ratios])
    assert exit_.value.code == 2


def test_the_telescope_needs_a_dataset_with_a_v2_or_v3_scenario(
    fake_training, capsys, tmp_path
):
    # The fixture's dataset has labels.csv and no manifest.json: no known scenario.
    assert cli.main(["train", "--arch", "telescope"]) == 1
    assert "this dataset names none" in capsys.readouterr().err
    assert fake_training.calls == []


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
        ["eval", "build-testset"],
        ["eval", "thresholds"],
        ["eval", "compare", "--model", "a", "--model", "b"],
    ],
)
def test_without_pycbc_a_command_that_draws_signals_exits_with_1(argv, tmp_path):
    """P5.2: no fallback, no traceback; the hint names the install command."""
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
    assert "pip install -e ." in result.stderr
    assert "Traceback" not in result.stderr


def run_glitch_hunter(argv, home, blocked=()):
    """Run gwd app glitch-hunter in a subprocess under Agg; blocked modules are None."""
    code = "\n".join(
        [
            "import os, sys",
            "os.environ['GWD_HEADLESS_OK'] = '1'",
            *(f"sys.modules[{name!r}] = None" for name in blocked),
            "from gwd.cli import main",
            f"rc = main({['app', 'glitch-hunter', *argv]!r})",
            "print(rc, 'tensorflow' in sys.modules)",
        ]
    )
    return run_python(code, home)


def test_without_pycbc_the_glitch_hunter_starts_with_the_simulation_off(tmp_path):
    """P4.4: real data need no PyCBC, so only the simulation is off (was rc 1 in P5.2)."""
    result = run_glitch_hunter([], tmp_path, blocked=["pycbc"])
    assert result.stdout.splitlines()[-1] == "0 False"
    assert "the simulation is off" in result.stderr
    assert "pip install -e ." in result.stderr
    assert "Traceback" not in result.stderr


def test_without_pycbc_and_gwpy_the_glitch_hunter_exits_with_1(tmp_path):
    result = run_glitch_hunter([], tmp_path, blocked=["pycbc", "gwpy"])
    assert result.stdout.splitlines()[-1] == "1 False"
    assert "pip install -e ." in result.stderr
    assert "pip install -e '.[gwdata]'" in result.stderr
    assert "Traceback" not in result.stderr


def test_an_unknown_glitch_hunter_model_exits_with_1(tmp_path):
    result = run_glitch_hunter(["--model", "nope"], tmp_path)
    assert result.stdout.splitlines()[-1] == "1 False"
    assert "Model 'nope' is neither a file nor a model" in result.stderr


def test_model_is_an_option_of_the_glitch_hunter_only(tmp_path):
    result = run_gwd(["app", "waveform", "--model", "latest"], tmp_path)
    assert result.returncode == 2
    assert "--model is an option of glitch-hunter only" in result.stderr
