# DeepAptamer

DeepAptamer predicts whether a DNA aptamer binds its target. It takes aptamer
sequences alone. Each sequence is one-hot encoded and joined with its DNA
shape, predicted by {class}`~pyaptamer.deepdnashape.DeepDNAShape`, and a
convolutional network classifies the result. The target is not an input. The
model learns one target from the labels it is trained on.

The entry point is {class}`~pyaptamer.deepaptamer.DeepAptamerPipeline`.

## Predicting binding

```python
import numpy as np
from pyaptamer.data import MoleculeLoader
from pyaptamer.deepaptamer import DeepAptamerClassifier, DeepAptamerPipeline

aptamers = [
    "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCC",
    "TGCATGCTAGCTAGCTAGCTAGCTAGCTAGCGCTA",
]
X = MoleculeLoader(data={"aptamer": aptamers * 20})
y = np.array([0, 1] * 20)

pipe = DeepAptamerPipeline(estimator=DeepAptamerClassifier(max_epochs=5))
pipe.fit(X, y)

X_new = MoleculeLoader(data={"aptamer": aptamers})
labels = pipe.predict(X_new)
probabilities = pipe.predict_proba(X_new)
```

`X` is a {class}`~pyaptamer.data.MoleculeLoader` or a `pandas.DataFrame`
with an aptamer column. `y` holds one binary label per row. The sequences
must be DNA in upper case, with the letters A, C, G, T and N. RNA raises an
error.

The DNA shape model is downloaded from Hugging Face the first time `fit` or
`predict` runs, and reused from the cache after that.

The default classifier trains for 100 epochs. `max_epochs=5` keeps this
example short.

The column name defaults to `aptamer` and can be changed:

```python
pipe = DeepAptamerPipeline(aptamer_col="sequence")
```

## Sequence length

Sequences are padded to `seq_len`, 35 by default, the length of the aptamers
in the paper. A longer sequence raises an error, so set `seq_len` to the
longest sequence in the data. `full_dna_shape=True` keeps the shape values
at the two edge positions that the paper drops.

```python
pipe = DeepAptamerPipeline(seq_len=40)
```

## Feature extraction

{class}`~pyaptamer.deepaptamer.DeepAptamerFeatures` builds the input of the
network: `4 * seq_len` one-hot columns, then the shape values. With the
defaults a sequence becomes 140 + 126 = 266 numbers.

```python
from pyaptamer.deepaptamer import DeepAptamerFeatures

features = DeepAptamerFeatures().fit_transform(X)
features.shape
```

```text
(40, 266)
```

## Swapping the estimator

`DeepAptamerPipeline` runs
{class}`~pyaptamer.deepaptamer.DeepAptamerClassifier` by default. Its
arguments `max_epochs`, `lr`, `batch_size`, `dropout` and `random_state` set
the training. Any `scikit-learn` classifier can take its place and is fitted
on the same features:

```python
from sklearn.ensemble import RandomForestClassifier

pipe = DeepAptamerPipeline(estimator=RandomForestClassifier())
```

## Reference

- Yang, X., et al. DeepAptamer: Advancing high-affinity aptamer discovery
  with a hybrid deep learning model. *Molecular Therapy Nucleic Acids* 36,
  102436 (2024). <https://doi.org/10.1016/j.omtn.2024.102436>
- Code: <https://github.com/YangX-BIDD/DeepAptamer>
