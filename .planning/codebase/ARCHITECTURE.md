# Architecture

**Analysis Date:** 2026-04-13

## System Pattern
The project follows a **Scikit-learn compatible API** pattern. It uses base classes from `scikit-base` (or equivalent patterns) to provide a standardized `fit`/`predict`/`transform` interface. This allows for composability and interoperability with the wider Python scientific stack.

## Core Components

### 1. Neural Network Models (`aptanet`, `aptatrans`)
- **AptaNet:** Likely a CNN or MLP based model for aptamer classification.
- **AptaTrans:** A Transformer-based architecture for predicting aptamer-protein interactions. It uses pretrained encoders for both aptamers and proteins and processes their interaction via an Interaction Map followed by a 2D CNN head.

### 2. Dataset Management (`datasets`, `data`)
- Centralized data loading for standard datasets (1BRQ, 5NU7, etc.).
- Support for both local files (CSV, PDB) and remote ones (Hugging Face).
- Use of dataclasses for structured data representation.

### 3. Feature Extraction (`pseaac`, `trafos`)
- **PseAAC:** Pseudo Amino Acid Composition for protein feature extraction.
- **Trafos:** Domain-specific transformers/encoders for sequence data.

### 4. Search & Optimization (`mcts`)
- Monte Carlo Tree Search implemented for aptamer discovery or optimization.

## Data Flow
1. **Loading:** `datasets` loaders fetch raw data (PDB/CSV).
2. **Preprocessing:** `utils` and `trafos` convert raw structures to numerical features (sequences, interaction maps).
3. **Training/Inference:** `aptanet`/`aptatrans` models take these features to predict binding or optimize sequences.
4. **Validation:** `benchmarking` and `tests` evaluate model performance.

## Design Principles
- **Modularity:** Heavy use of sub-packages for specific domains (modeling, data, utils).
- **Standards:** Consistent use of docstrings, type hints, and standardized project structure.
- **Extensibility:** Plugins or easily swappable modules for new features/models.

---

*Architecture analysis: 2026-04-13*
