"""Every command and relative link in the documentation works (P7.1).

The scanned files are the Markdown files in the project root and in docs/ (until P7.4
the German working folder docs/aufraeumen/, which quoted old commands on purpose, was
left out; P7.4 removed it). The test checks:

- every ``gwd ...`` line in a code block parses with gwd.cli.build_parser(), options
  and values included (nothing runs);
- every inline ``gwd ...`` span parses the same way. A synopsis such as
  ``gwd train [--epochs --seed]`` parses up to the bracket, and each option in the
  brackets must exist on that command; ``a\\|b`` lists alternatives;
- every ``python -m X`` names an importable module (find_spec) and every
  ``python file.py`` an existing file;
- every ``pytest -m`` expression uses only markers registered in pyproject.toml;
- every relative link points to an existing file.
"""

import argparse
import importlib.util
import itertools
import re
import shlex
import tomllib
from pathlib import Path

import pytest

from gwd.cli import build_parser

ROOT = Path(__file__).resolve().parent.parent
DOCS = sorted([*ROOT.glob("*.md"), *(ROOT / "docs").glob("*.md")])

# Inline spans with placeholders that cannot parse; anything else with '<' fails.
TEMPLATES = {"gwd <command> --help"}

FENCE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE = re.compile(r"`([^`\n]+)`")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def scan(path):
    """(code lines, prose lines) of a Markdown file, as (line number, text) pairs.

    Code lines come from fenced blocks, with continuation lines (ending in a backslash)
    joined onto the line they continue.
    """
    code, prose = [], []
    in_block = False
    pending = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if FENCE.match(line):
            in_block = not in_block
            continue
        if not in_block:
            prose.append((number, line))
            continue
        if pending is not None:
            line = pending[1] + " " + line.strip()
            number = pending[0]
            pending = None
        if line.rstrip().endswith("\\"):
            pending = (number, line.rstrip()[:-1])
            continue
        code.append((number, line))
    return code, prose


def words(text):
    """The shell words of one command line, without comments, a prompt, or env vars."""
    text = text.strip()
    if text.startswith("$ "):
        text = text[2:]
    try:
        tokens = shlex.split(text, comments=True)
    except ValueError:
        return []
    while tokens and ENV_ASSIGNMENT.match(tokens[0]):
        tokens.pop(0)
    return tokens


def collect():
    """All checks found in the docs, as (kind, location, payload) triples."""
    found = []
    for path in DOCS:
        name = path.relative_to(ROOT).as_posix()
        code, prose = scan(path)
        for number, line in code:
            tokens = words(line)
            if tokens:
                found.append(("command", f"{name}:{number}", tokens))
        for number, line in prose:
            for span in INLINE_CODE.findall(line):
                if span.startswith(("gwd ", "python ", "pytest ")):
                    found.append(("span", f"{name}:{number}", span))
            # Links in inline code are not links.
            for target in LINK.findall(INLINE_CODE.sub("", line)):
                found.append(("link", f"{name}:{number}", (path, target)))
    return found


FOUND = collect()


def label(kind, payload):
    if kind == "command":
        return " ".join(payload)
    return payload[1] if kind == "link" else payload


def of_kind(*kinds):
    return [
        pytest.param(payload, id=f"{location} {label(kind, payload)}")
        for kind, location, payload in FOUND
        if kind in kinds
    ]


def parse_gwd(args):
    """Parse gwd args; fail with argparse's message if they do not parse."""
    try:
        build_parser().parse_args(args)
    except SystemExit as exit_:
        # --help exits with 0 after printing the help, which is fine.
        assert exit_.code == 0, f"'gwd {' '.join(args)}' does not parse"


def subparser(args):
    """The parser of the (sub)command that args name, such as ['eval', 'run']."""
    parser = build_parser()
    for arg in args:
        commands = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        parser = commands.choices[arg]
    return parser


def registered_markers():
    with open(ROOT / "pyproject.toml", "rb") as f:
        lines = tomllib.load(f)["tool"]["pytest"]["ini_options"]["markers"]
    return {line.split(":")[0].strip() for line in lines}


def check_python(tokens):
    """python -m X must name an importable module; python file.py an existing file."""
    if len(tokens) >= 3 and tokens[1] == "-m":
        module = tokens[2]
        assert importlib.util.find_spec(module) is not None, f"no module {module}"
        if module == "pytest":
            check_pytest(tokens[2:])
    elif len(tokens) >= 2 and tokens[1].endswith(".py"):
        assert (ROOT / tokens[1]).is_file(), f"no file {tokens[1]}"


def check_pytest(tokens):
    """Marker names in pytest -m must be registered; test paths must exist."""
    markers = registered_markers()
    for i, token in enumerate(tokens[1:], 1):
        if tokens[i - 1] == "-m":
            names = set(re.findall(r"[A-Za-z_]\w*", token)) - {"and", "or", "not"}
            assert names <= markers, f"unregistered markers: {names - markers}"
        elif token.startswith("tests"):
            assert (ROOT / token.split("::")[0]).exists(), f"no test path {token}"


def check(tokens):
    if tokens[0] == "gwd":
        parse_gwd(tokens[1:])
    elif tokens[0] in ("python", "python3"):
        check_python(tokens)
    elif tokens[0] == "pytest":
        check_pytest(tokens)


def test_the_docs_have_commands_to_check():
    commands = [p for k, _, p in FOUND if k == "command" and p[0] == "gwd"]
    spans = [p for k, _, p in FOUND if k == "span" and p.startswith("gwd ")]
    links = [p for k, _, p in FOUND if k == "link"]
    assert len(commands) >= 10 and len(spans) >= 10 and len(links) >= 3


@pytest.mark.parametrize("tokens", of_kind("command"))
def test_code_block_command(tokens):
    check(tokens)


@pytest.mark.parametrize("span", of_kind("span"))
def test_inline_command(span):
    if "<" in span:
        assert span in TEMPLATES, f"unknown placeholder in '{span}'"
        return
    base, _, options = span.partition("[")
    # A Markdown table escapes '|' as '\|'; 'a\|b' lists alternatives.
    alternatives = [word.replace("\\|", "|").split("|") for word in words(base)]
    for tokens in itertools.product(*alternatives):
        tokens = list(tokens)
        if tokens[0] != "gwd":
            check(tokens)
            continue
        parse_gwd(tokens[1:])
        if options:
            known = set(subparser(tokens[1:])._option_string_actions)
            for word in options.rstrip("]").split():
                for option in word.split("/"):
                    if option.startswith("-"):
                        assert option in known, (
                            f"'gwd {' '.join(tokens[1:])}' has no {option}"
                        )


@pytest.mark.parametrize("link", of_kind("link"))
def test_relative_link(link):
    path, target = link
    if re.match(r"^[a-z]+:", target) or target.startswith("#"):
        return
    file = target.split("#")[0]
    assert (path.parent / file).exists(), f"broken link {target} in {path.name}"
