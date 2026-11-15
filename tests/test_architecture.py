"""Import rules from docs/architecture.md, checked on the source with ast (P2.6).

Each component has a table of what it may import. TOP means at module level (and
inside functions), LAZY means only inside a function body; a try block at module level
is not lazy. The standard library is always allowed. Anything not in the table is a
violation, so a stray 'import matplotlib' in gwd.core fails. The tables also limit
where tensorflow, gwpy, and pycbc may appear at all, lazy or not. Dynamic imports
(__import__, importlib) are not checked.

Known violations are in ALLOWLIST. It may only shrink: an entry that no longer matches
a real violation fails the test, so fixing the code forces removing the entry. It must
be empty by P7.4.
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "gwd"
STDLIB = sys.stdlib_module_names

TOP = "top"
LAZY = "lazy"
ANY = "*"  # every non-stdlib import

RULES = {
    "config": {},
    "core": {"gwd.core": TOP, "numpy": TOP, "scipy": TOP, "pycbc": LAZY},
    "data": {
        **dict.fromkeys(["gwd.data", "gwd.core", "gwd.config"], TOP),
        **dict.fromkeys(["numpy", "scipy", "pandas"], TOP),
    },
    "registry": {"gwd.config": TOP},
    "training": {
        **dict.fromkeys(["gwd.training", "gwd.data", "gwd.core", "gwd.config"], TOP),
        "gwd.registry": TOP,
        **dict.fromkeys(["tensorflow", "sklearn", "numpy", "pandas"], TOP),
        "matplotlib": LAZY,
    },
    "evaluation": {
        **dict.fromkeys(["gwd.evaluation", "gwd.data", "gwd.core", "gwd.config"], TOP),
        "gwd.registry": TOP,
        **dict.fromkeys(["sklearn", "pandas", "matplotlib", "tabulate", "numpy"], TOP),
    },
    "apps": {
        **dict.fromkeys(["gwd.apps", "gwd.core", "gwd.data", "gwd.config"], TOP),
        "gwd.registry": TOP,
        **dict.fromkeys(["matplotlib", "numpy", "scipy.io.wavfile"], TOP),
        **dict.fromkeys(["tensorflow", "gwpy"], LAZY),
    },
    "cli": {ANY: LAZY},
    "main": {},
}

# Narrower exceptions that apply to a single file on top of its component's rules.
FILE_RULES = {
    "src/gwd/data/gwosc.py": {"gwpy": LAZY},
    "src/gwd/data/preprocessing.py": {"gwpy": LAZY},
    "src/gwd/data/validate.py": {"matplotlib": LAZY},
    "src/gwd/registry.py": {"tensorflow": LAZY},
}

# (file, import or 'print'). Each entry names the finding and the step that removes it.
ALLOWLIST = {
    # CORE-17: pycbc is imported at module level in a try block (P5.2).
    ("src/gwd/core/waveforms.py", "pycbc"),
    # CORE-17: print in library code (P4.1).
    ("src/gwd/core/waveforms.py", "print"),
    # DATA-13: module-level side effects in training, matplotlib on top (P3.7).
    ("src/gwd/training/train.py", "matplotlib"),
    # HYG-16: the dependency check imports pycbc and gwd.core on top (P3.6, gwd env).
    ("src/gwd/env.py", "pycbc"),
    ("src/gwd/env.py", "gwd.core"),
}


def component(path):
    rel = path.relative_to(ROOT).as_posix()
    if rel == "main.py":
        return "main"
    parts = path.relative_to(PACKAGE).with_suffix("").parts
    if parts[0] in ("cli", "env"):
        return "cli"
    if parts[0] in ("config", "__init__"):
        return "config"
    return parts[0]


def python_files():
    return sorted([*PACKAGE.rglob("*.py"), ROOT / "main.py"])


def imports(path):
    """Yield (module, lazy) for every import statement in the file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    # src/gwd/core/noise.py -> ["gwd", "core"]; main.py -> []
    package = list(path.relative_to(ROOT).with_suffix("").parts[1:-1])

    def visit(node, lazy):
        for child in ast.iter_child_nodes(node):
            inner = lazy or isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
            )
            if isinstance(child, ast.Import):
                for alias in child.names:
                    yield alias.name, lazy
            elif isinstance(child, ast.ImportFrom):
                base = child.module or ""
                if child.level:
                    parent = package[: len(package) + 1 - child.level]
                    base = ".".join([*parent, base] if base else parent)
                for alias in child.names:
                    # 'from gwd import config' -> gwd.config and 'from scipy.io import
                    # wavfile' -> scipy.io.wavfile; rules match on the dotted prefix.
                    yield f"{base}.{alias.name}", lazy
            yield from visit(child, inner)

    yield from visit(tree, False)


def key(module):
    """The name an import is reported and allowlisted under."""
    parts = module.split(".")
    return ".".join(parts[:2]) if parts[0] == "gwd" else parts[0]


def allowed(module, lazy, rules):
    for name, where in rules.items():
        matches = name == ANY or module == name or module.startswith(name + ".")
        if matches and (lazy or where == TOP):
            return True
    return False


def violations():
    found = set()
    for path in python_files():
        rel = path.relative_to(ROOT).as_posix()
        # Unknown component: strictest rules; test_rules_cover_every_component names it.
        rules = {**RULES.get(component(path), {}), **FILE_RULES.get(rel, {})}
        for module, lazy in imports(path):
            if module.split(".")[0] in STDLIB or allowed(module, lazy, rules):
                continue
            found.add((rel, key(module)))
        if component(path) == "core":
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if any(
                isinstance(n, ast.Call) and getattr(n.func, "id", None) == "print"
                for n in ast.walk(tree)
            ):
                found.add((rel, "print"))
    return found


def test_imports_follow_the_architecture_rules():
    new = violations() - ALLOWLIST
    assert not new, "Import rules from docs/architecture.md broken:\n" + "\n".join(
        f"  {path}: {name}" for path, name in sorted(new)
    )


def test_allowlist_only_shrinks():
    stale = ALLOWLIST - violations()
    assert not stale, "Fixed; remove from ALLOWLIST:\n" + "\n".join(
        f"  {path}: {name}" for path, name in sorted(stale)
    )


def test_rules_cover_every_component():
    assert {component(path) for path in python_files()} <= RULES.keys()


def test_no_search_path_edits():
    # Built from parts, so a plain text search for the attribute finds nothing here.
    pattern = re.compile(r"\bsys\s*\.\s*path\b|\bfrom\s+sys\s+import\s+path\b")
    files = [*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("tests").rglob("*.py")]
    hits = [
        f"  {path.relative_to(ROOT)}:{number}"
        for path in [*files, ROOT / "main.py"]
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert not hits, "Search path edits found:\n" + "\n".join(hits)


def test_no_byte_identical_python_files():
    # Tracked and untracked, not ignored files, so a fresh copy is caught (c4e3dd8).
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    seen = {}
    duplicates = []
    for rel in listed:
        path = ROOT / rel
        if not path.is_file():
            continue  # deleted in the working tree
        content = path.read_bytes()
        if path.name == "__init__.py" and not content.strip():
            continue
        if content in seen:
            duplicates.append(f"  {seen[content]} == {rel}")
        seen.setdefault(content, rel)
    assert not duplicates, "Byte-identical Python files:\n" + "\n".join(duplicates)
