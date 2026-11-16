# AGENTS.md

Rules for AI assistants working in this repository.

## Files and data

- Move files only with `git mv`, never by copying (see c4e3dd8).
- Never run `git clean -x` or `rm -rf` on `analysis/`, `data/`,
  `models_registry/`, `plots/`, `results/`, or the legacy paths
  `gw_noise_background/`, `gw_training_data/`, and `./model_leaderboard.csv`.
  Ignored files in these paths cannot be restored from git.

## Commits and pull requests

- One topic per commit, with a prefix: `chore`, `fix`, `refactor`, `test`,
  `docs`, or `golden` (see c54acd5).
- Name the step ID from the step checklist in `docs/architecture.md` in the pull
  request.
- Each step PR ticks its own step in the checklist in `docs/architecture.md`.
- Before every commit, run `pytest -m 'not slow and not network'`.
  Before opening a pull request, run all tests.
- CI must be green before a pull request is merged.
- Change golden values only in `golden:` commits, and state the reason.
- Do not commit tool state.

## Environment

- Use the project's Python 3.12 virtual environment; never run tests on the
  system Python.

## Language

- Everything is in English (decision D11 = B): identifiers, data columns, file
  names, comments, docstrings, console output, CLI help texts, the menu,
  documentation, commit messages, and PR text.
- New files and new lines are written in English. Existing German text is
  translated when a step touches its file; a module that a step rewrites is
  translated completely, in its own `docs:` commit in the same pull request.
- A German name that other code or stored data depends on, such as the leaderboard
  column `Physik_Gap`, is renamed only in the step that changes that schema.

## Documentation

- Do not make unverified claims in the documentation.
