# PyAptamer

Python library for aptamer design.

`pyaptamer` provides loaders for sequence and structure files, feature
encoders, aptamer-protein interaction models and aptamer generators. All of
them follow the `scikit-learn` API: `fit`, `predict` and `transform`.

Source code, issue tracker and tutorial notebooks are on
[GitHub](https://github.com/gc-os-ai/pyaptamer).

## Install

```bash
pip install --pre pyaptamer
```

The `--pre` flag is required while `pyaptamer` is a prerelease. See
{doc}`installation` for development installs and optional extras.

## Quick start

```python
import numpy as np
from pyaptamer.aptanet import AptaNetPipeline
from pyaptamer.data import MoleculeLoader

aptamer = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
protein = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"

X = MoleculeLoader(data={"aptamer": [aptamer] * 40, "protein": [protein] * 40})
y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

pipe = AptaNetPipeline().fit(X, y)

X_new = MoleculeLoader(data={"aptamer": [aptamer] * 5, "protein": [protein] * 5})
pipe.predict(X_new)
```

The {doc}`user guide <user_guide/index>` has one page per tool. The
notebooks in
[examples](https://github.com/gc-os-ai/pyaptamer/tree/main/examples)
walk through AptaNet, AptaMCTS, AptaTrans and benchmarking end to end.

## Algorithms

:::{list-table}
:header-rows: 1
:widths: 20 80

* - Algorithm
  - Use
* - {doc}`AptaMCTS <user_guide/aptamcts>`
  - Predict whether an aptamer and a protein interact, and generate candidate
    aptamers for a target protein.
* - {doc}`AptaNet <user_guide/aptanet>`
  - Predict whether an aptamer and a protein interact, from sequence pairs.
* - {doc}`AptaTrans <user_guide/aptatrans>`
  - Score aptamer-protein interaction and recommend candidate aptamers for a
    target.
* - {doc}`DeepAptamer <user_guide/deepaptamer>`
  - Predict whether a DNA aptamer binds its target, from the sequence and its
    predicted DNA shape.
* - {doc}`AptaDiff <user_guide/aptadiff>`
  - Generate aptamer sequences with a diffusion model conditioned on latent
    vectors.
* - {doc}`MCTS <user_guide/mcts>`
  - Search for a high-scoring sequence with Monte Carlo tree search, given a
    scoring experiment.
:::

## Data and features

:::{list-table}
:header-rows: 1
:widths: 20 80

* - Tool
  - Use
* - {doc}`MoleculeLoader <user_guide/molecule_loader>`
  - Table of sequences and structure files, read lazily. Tiling, indexing
    and metadata columns.
* - {doc}`Feature encodings <user_guide/encodings>`
  - `PSeAAC` and `ConjointTriad` for proteins, `KMerFrequencies` for
    nucleotides, column-wise encoding with `ApplyToCols`.
* - {class}`DeepDNAShape <pyaptamer.deepdnashape.DeepDNAShape>`
  - Predicted DNA shape features of DNA sequences: minor groove width,
    propeller twist, helical twist and roll.
* - {doc}`Datasets <user_guide/datasets>`
  - Bundled structures, the Li 2014 interaction dataset and AptaCom from
    Hugging Face, each as a `MoleculeLoader`.
* - {doc}`Benchmarking <user_guide/benchmarking>`
  - Cross-validated comparison of estimators.
:::

The package is in early development. The public API is unstable and may change
between releases.

```{toctree}
:hidden:
:maxdepth: 1

installation
user_guide/index
api_reference
```
