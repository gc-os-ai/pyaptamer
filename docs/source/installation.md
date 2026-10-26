# Installation

`pyaptamer` requires Python 3.10 or newer.

## From PyPI

```bash
pip install --pre pyaptamer
```

The current release is a prerelease, so `pip` needs the `--pre` flag.
Check the install:

```bash
python -c "import pyaptamer; print(pyaptamer.__version__)"
```

## Development install

Clone the repository and install in editable mode:

```bash
git clone https://github.com/gc-os-ai/pyaptamer.git
cd pyaptamer
pip install -e ".[dev]"
```

The `dev` extra adds `ruff`, `pytest`, `pytest-cov`, and `pre-commit`.

## Building the documentation

```bash
pip install -e ".[docs]"
cd docs
make html
```

Open `docs/build/html/index.html` in a browser.
