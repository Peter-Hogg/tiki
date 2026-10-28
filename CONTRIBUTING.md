# Contributing to TIKI

Thank you for helping make scientific image analysis easier to use. Bug
reports, workflow feedback, documentation fixes and code contributions are all
welcome.

## Before opening an issue

- Search the existing issues first.
- Include your operating system, Python version and TIKI version or commit.
- For file-reading problems, describe the file format, dimensions and
  acquisition software. Share only non-sensitive example data.
- Include the full terminal traceback when one is available.

## Development setup

```bash
git clone https://github.com/Peter-Hogg/tiki.git
cd tiki
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,docs]'
```

Run the application and tests:

```bash
tiki
QT_QPA_PLATFORM=offscreen python -m pytest -q  # headless on a desktop with a GPU
sphinx-build -W --keep-going -b html docs docs/_build/html
```

## Pull requests

1. Keep changes focused and explain the biological workflow they improve.
2. Add or update tests for behavior changes.
3. Update the documentation when adding a command or changing a shortcut.
4. Avoid loading complete large movies into memory; preserve memory mapping
   and chunked operations where practical.
5. Confirm the test and documentation builds pass before opening the PR.

Without `QT_QPA_PLATFORM=offscreen` the tests open real windows on your
display. Offscreen mode borrows OpenGL from your GPU driver; machines without
one (like CI runners) need `xvfb-run` plus Qt's xcb libraries instead, as in
`.github/workflows/tests.yml`. Tests stub out napari's layer-control panels
(see the `tiki` fixture in `tests/test_app.py`).

By contributing, you agree that your contribution is licensed under the
project's BSD 3-Clause License.

## Design principles

- Favor familiar ImageJ behavior when it does not conflict with data safety.
- Keep processing functions independent of Qt and napari where possible.
- Never silently guess an ambiguous axis when asking the user is safer.
- Prefer non-destructive operations that create new layers.
- Errors should explain what the user can do next and must not crash Qt.
