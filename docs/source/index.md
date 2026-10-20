# PyAptamer

Python library for aptamer design.

`pyaptamer` provides aptamer-protein interaction models and aptamer candidate
generation behind a `scikit-learn`-style API. Four algorithms are available today:

:::{list-table}
:header-rows: 1
:widths: 20 80

* - Algorithm
  - Use
* - {doc}`AptaDiff <user_guide/aptadiff>`
  - Design new aptamer sequences from the SELEX reads of a selection round,
    with a diffusion model.
* - {doc}`AptaMCTS <user_guide/aptamcts>`
  - Predict whether an aptamer and a protein interact, and generate candidate
    aptamers for a target protein.
* - {doc}`AptaNet <user_guide/aptanet>`
  - Predict whether an aptamer and a protein interact, from sequence pairs.
* - {doc}`AptaTrans <user_guide/aptatrans>`
  - Score aptamer-protein interaction and recommend candidate aptamers for a target.
:::

The package is in early development. The public API is unstable and may change
between releases.

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

aptamer = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
protein = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"

X = [(aptamer, protein) for _ in range(40)]
y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

pipe = AptaNetPipeline().fit(X, y)
pipe.predict(X[:5])
```

```{toctree}
:hidden:
:maxdepth: 1

installation
user_guide/index
api_reference
```
