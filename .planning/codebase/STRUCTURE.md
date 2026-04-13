# Directory Structure

**Analysis Date:** 2026-04-13

## Project Layout

```text
pyaptamer/
├── pyproject.toml             # Project config and dependencies
├── README.md                  # Project overview
├── CODE_OF_CONDUCT.md         # Community guidelines
├── LICENSE                    # BSD 3-Clause License
├── pyaptamer/                 # Main package core
│   ├── aptanet/               # AptaNet model implementation
│   ├── aptatrans/             # AptaTrans (Transformer) model
│   │   ├── layers/            # Neural network modules (Conv, Encoder, etc.)
│   │   ├── tests/             # Model-specific tests
│   │   └── _model.py          # Main AptaTrans class
│   ├── benchmarking/          # Model evaluation tools
│   ├── data/                  # Basic data loaders
│   ├── datasets/              # Advanced dataset loading (PDB, CSV, HF)
│   │   ├── _loaders/          # Specific dataset loaders (Aptacom, PFOA, etc.)
│   │   ├── data/              # Raw data files (CSV, PDB)
│   │   └── dataclasses/       # Structured data representations
│   ├── experiments/           # Experiment runners/scripts
│   ├── mcts/                  # Monte Carlo Tree Search implementation
│   ├── pseaac/                # Pseudo Amino Acid Composition extraction
│   ├── trafos/                # Data transformers (fit/transform API)
│   ├── tests/                 # Global/integration tests
│   └── utils/                 # General utility functions (Bioinformatics, augmentation)
├── build_tools/               # Tools for building/CI
├── examples/                  # Usage tutorials and examples
└── .planning/                 # Project planning documents (GSD)
```

## Key Locations

- **Models:** `pyaptamer/aptatrans/_model.py`, `pyaptamer/aptanet/_aptanet_nn.py`
- **Data Loading:** `pyaptamer/datasets/_loaders/`
- **Feature Extraction:** `pyaptamer/pseaac/`, `pyaptamer/trafos/`
- **Utilities:** `pyaptamer/utils/`
- **Tests:** `pyaptamer/**/tests/`

## Naming Conventions
- **Internal files:** Often prefixed with `_` (e.g., `_model.py`, `_base.py`) suggesting they are implementation details rather than public entry points.
- **Directories:** All lowercase, reflecting Python package standards.

---

*Structure analysis: 2026-04-13*
