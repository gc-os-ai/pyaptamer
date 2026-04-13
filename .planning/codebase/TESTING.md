# Testing Practices

**Analysis Date:** 2026-04-13

## Framework
- **Primary:** `pytest`
- **Utility:** `unittest.mock` (via `@patch`) for mocking external API calls.

## Test Structure
Tests are co-located with the code they test in `tests` sub-directories within each package:
- `pyaptamer/aptatrans/tests/`
- `pyaptamer/utils/tests/`
- `pyaptamer/datasets/tests/`

## Test Types
1. **Unit Tests:** Focus on individual layers and utility functions.
2. **Integration Tests:** (Likely) testing full model forward passes and dataset loaders.
3. **Doctests:** `test_doctest.py` suggests that examples provided in docstrings are also tested.

## Mocking & Data
- **External APIs:** `requests.get` is mocked in UniProt tests to avoid network dependency.
- **Sample Data:** `pyaptamer/datasets/data/` contains small/dummy versions of PDB and CSV files for testing.
- **Fixtures:** Usage of standard `pytest` fixture patterns.

## Execution
Run tests from the project root:
```bash
pytest
```

---

*Testing audit: 2026-04-13*
