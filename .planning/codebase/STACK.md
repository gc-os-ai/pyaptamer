# Technology Stack

**Analysis Date:** 2026-04-13

## Languages

**Primary:**
- Python >= 3.10 - All application code and examples.

**Secondary:**
- CSV/PDB - Data formats for protein and aptamer structures.

## Runtime

**Environment:**
- Python 3.10+
- PyTorch (for deep learning models)
- Lightning (for model training pipelines)

**Package Manager:**
- setuptools (pip)
- Build backend: `setuptools.build_meta`

## Frameworks

**Core:**
- PyTorch >= 2.5.1 - Deep learning backend.
- Lightning >= 2.5.3 - Training and pipeline abstraction.
- Scikit-learn >= 1.3.0 - Standardized API patterns and utility functions.
- Biopython >= 1.83 - Bioinformatics processing (PDB, sequences).

**Testing:**
- Pytest >= 8.0.0 - Unit and integration testing.

**Build/Dev:**
- Ruff >= 0.12.0 - Linting and formatting.
- Pre-commit - Hook management for code quality.

## Key Dependencies

**Critical:**
- `numpy` >= 2.0.2 - Numerical computations.
- `pandas` >= 2.0.0 - Data manipulation.
- `datasets` >= 4.0.0 - Hugging Face datasets integration.
- `skorch` - Scikit-learn compatible neural network library.
- `imblearn` - Imbalanced learning utilities.

**Infrastructure:**
- `requests` - External API interactions (UniProt, Hugging Face).

## Configuration

**Environment:**
- Configured via `pyproject.toml` for tool settings (ruff, setuptools).
- No major environment variables detected for core functionality.

**Build:**
- `pyproject.toml` - Main project configuration.
- `.pre-commit-config.yaml` - Pre-commit hooks configuration.

## Platform Requirements

**Development:**
- Cross-platform (Linux, macOS, Windows).
- Requires Python 3.10 or higher.

**Production:**
- Distributed as a Python package via PyPI.
- Runs on any environment with compatible PyTorch/Lightning versions.

---

*Stack analysis: 2026-04-13*
