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

aptamers = [
    "GGGAGGACGAAGACGACUCGAGACAGGCUAGGGAGGGA",
    "AAGCGUCGGAUCUACACGUGCGAUAGCUCAGUACGCGGU",
    "CGGUAUCGAGUACAGGAGUCCGACGGAUAGUCCGGAGC",
]
protein = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"

X = [(a, protein) for a in aptamers] * 10
y = np.array([0, 1, 0] * 10, dtype=np.float32)

pipe = AptaNetPipeline()
pipe.fit(X, y)

labels = pipe.predict(X[:3])
probabilities = pipe.predict_proba(X[:3])
```

`X` is a list of `(aptamer, protein)` string pairs. `y` is a float array of binary
labels. `predict` returns class labels and `predict_proba` returns class
probabilities.

The `k` argument sets the aptamer k-mer size used for feature extraction:

```python
pipe = AptaNetPipeline(k=5)
```

## Loading protein sequences

Protein sequences can come from a PDB structure through the dataset loaders:

```python
from pyaptamer.datasets import load_1gnh

protein = load_1gnh().to_df_seq()["sequence"].tolist()[0]
```

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
