# Coding Conventions

**Analysis Date:** 2026-04-13

## Python Standards
- **Style:** PEP8 compatible, formatted with `ruff` (88 char line length).
- **Naming:**
  - Classes: `PascalCase` (e.g., `AptaTrans`, `ConvBlock`).
  - Functions/Variables: `snake_case` (e.g., `forward_encoder`, `in_dim`).
  - Internal components: Prefixed with `_` (e.g., `_make_layer`).

## Documentation
- **Docstrings:** NumPy style is strictly followed for all public and most internal classes/methods.
  - Sections: `Parameters`, `Attributes`, `Returns`, `Raises`, `Examples`, `References`.
- **References:** Academic references are included in docstrings using reStructuredText format.

## Type Hinting
- Comprehensive use of type hints for method signatures (`arg: type -> ret_type`).
- Uses `tuple`, `list`, `Callable`, `Tensor`, etc. from `collections.abc`, `torch`, and `typing`.

## Patterns
- **Scikit-learn API:** Usage of `fit`, `predict`, `transform` where applicable.
- **PyTorch/Lightning:** `nn.Module` for models, `nn.Sequential` for layer grouping.
- **Factory methods:** Internal helper methods like `_make_layer` or `_make_encoder` are used to build complex objects.

## Error Handling
- Explicit raising of clear error types:
  - `AssertionError` for invalid configurations.
  - `ValueError` for invalid string inputs or types.
  - `ImportError` (likely) for missing optional dependencies.

---

*Conventions analysis: 2026-04-13*
