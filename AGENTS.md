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
- Name the step ID from `docs/aufraeumen/PLAN.md` in the pull request.
- Before every commit, run `pytest -m 'not slow and not network'`.
  Before opening a pull request, run all tests.
- Change golden values only in `golden:` commits, and state the reason.
- Do not commit tool state.

## Environment

- Use the project's Python 3.12 virtual environment; never run tests on the
  system Python.

## Language

- Write all new code, comments, console output, documentation, commit messages,
  and PR text in English.

## Documentation

- Do not make unverified claims in the documentation.
