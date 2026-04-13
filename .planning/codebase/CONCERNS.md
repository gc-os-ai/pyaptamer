# Technical Concerns

**Analysis Date:** 2026-04-13

## Technical Debt & Fragility

### 1. Partial Implementations
- `pyaptamer/datasets/dataclasses/_masked.py` contains a TODO noting that the masking method is currently limited to the AptaTrans implementation and may need generalization.
- `pyaptamer/aptatrans/_pipeline.py` has an incomplete interaction map plotting feature.

### 2. Duplication & Code Quality
- `pyaptamer/utils/_aptatrans_utils.py` contains logic that should be merged with more generic RNA utilities (`rna2vec`).
- Large CSV files (`AptaCom03.csv`) are tracked in the repository and contain placeholder characters like `XXX`, which might cause issues for certain parsers if not handled.

### 3. Performance
- `pyaptamer/utils/_aptatrans_utils.py` notes a TODO regarding performance optimizations.

## Potential Risks

### 1. Data Integrity
- The reliance on `requests.get` for UniProt sequence mapping without robust caching or retry logic could lead to flakiness in environments with unstable internet.
- Pretrained weights are downloaded automatically; changes to remote repository structures or availability on Hugging Face could break model initialization.

### 2. Complexity
- The intersection of PyTorch, Lightning, Biopython, and Scikit-learn API patterns creates a high barrier to entry for new contributors.
- Nested directory structures and heavy use of internal (`_`) files can make navigation difficult.

## Areas for Improvement
- Unify bioinformatics utilities across models to reduce duplication.
- Finish plotting and visualization utilities for models.
- Implement more robust local caching for external biological data.

---

*Concerns audit: 2026-04-13*
