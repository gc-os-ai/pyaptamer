# Installation

`pyaptamer` requires Python 3.10 or newer.

## From PyPI

```bash
pip install pyaptamer
```

The current release is a prerelease. To pin it explicitly:

```bash
pip install pyaptamer==0.1.0a1
```

## Development install

Clone the repository and install in editable mode:

```bash
git clone https://github.com/gc-os-ai/pyaptamer.git
cd pyaptamer
pip install -e ".[dev]"
```

The `dev` extra adds `ruff`, `pytest`, and `pre-commit`.

## Building the documentation

```bash
pip install -e ".[docs]"
cd docs
make html
```

The rendered site is written to `docs/build/html`.
