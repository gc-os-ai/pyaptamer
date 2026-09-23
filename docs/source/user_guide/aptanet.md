# AptaNet

AptaNet predicts whether an aptamer and a protein interact. It takes
`(aptamer, protein)` sequence pairs, derives numeric features from each sequence,
selects features with a random forest, and classifies with a multi-layer
perceptron. The estimator follows the `scikit-learn` `fit` / `predict` API.

The entry point is {class}`~pyaptamer.aptanet.AptaNetPipeline`.

## Predicting interactions

```python
import numpy as np
from pyaptamer.aptanet import AptaNetPipeline
from pyaptamer.data import MoleculeLoader

aptamers = [
    "GGGAGGACGAAGACGACUCGAGACAGGCUAGGGAGGGA",
    "AAGCGUCGGAUCUACACGUGCGAUAGCUCAGUACGCGGU",
    "CGGUAUCGAGUACAGGAGUCCGACGGAUAGUCCGGAGC",
]
protein = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"

X = MoleculeLoader(data={"aptamer": aptamers * 10, "protein": [protein] * 30})
y = np.array([0, 1, 0] * 10, dtype=np.float32)

pipe = AptaNetPipeline()
pipe.fit(X, y)

X_new = MoleculeLoader(data={"aptamer": aptamers, "protein": [protein] * 3})
labels = pipe.predict(X_new)
probabilities = pipe.predict_proba(X_new)
```

`X` is a {class}`~pyaptamer.data.MoleculeLoader` or a `pandas.DataFrame` with
an aptamer column and a protein column. `y` is a float array of binary labels.
`predict` returns class labels and `predict_proba` returns class probabilities.

The column names default to `aptamer` and `protein` and can be changed:

```python
pipe = AptaNetPipeline(aptamer_col="apt", protein_col="target")
```

## Feature extraction

The pipeline encodes the aptamer column with
{class}`~pyaptamer.trafos.encode.KMerFrequencies` and the protein column with
{class}`~pyaptamer.trafos.encode.PSeAAC`, joined by a `ColumnTransformer`.
The `k` argument sets the longest aptamer k-mer:

```python
pipe = AptaNetPipeline(k=5)
```

See {doc}`encodings` to run the encoders on their own.

## Loading protein sequences

Protein sequences can come from a PDB structure through the dataset loaders:

```python
from pyaptamer.datasets import load_1gnh

protein = load_1gnh(tiling="first").to_dataframe()["sequence"].iloc[0]
```

See {doc}`molecule_loader` for the loader's options.

## Swapping the estimator

`AptaNetPipeline` runs {class}`~pyaptamer.aptanet.AptaNetClassifier` by default.
Pass any `scikit-learn` compatible classifier to replace it:

```python
from sklearn.ensemble import GradientBoostingClassifier

pipe = AptaNetPipeline(estimator=GradientBoostingClassifier())
```

For regression targets, use {class}`~pyaptamer.aptanet.AptaNetRegressor` directly.

## Reference

- Emami, N., Ferdousi, R. AptaNet as a deep learning approach for
  aptamer-protein interaction prediction. *Scientific Reports* 11, 6074 (2021).
  <https://doi.org/10.1038/s41598-021-85629-0>
