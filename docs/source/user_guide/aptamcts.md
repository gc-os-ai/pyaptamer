# AptaMCTS

AptaMCTS predicts whether an aptamer and a protein interact, and generates
candidate aptamers for a target protein. It takes `(aptamer, protein)`
sequence pairs, encodes each sequence as k-mer frequencies, and classifies
the pair with a random forest. A fitted classifier then scores the
candidates proposed by a Monte Carlo tree search.

The entry point is {class}`~pyaptamer.aptamcts.AptaMCTSPipeline`.

## Predicting interactions

The li2014 data of the paper is bundled as a
{class}`~pyaptamer.data.MoleculeLoader` with an `aptamer` column, a
`protein` column and a label column:

```python
from pyaptamer.aptamcts import AptaMCTSPipeline
from pyaptamer.datasets import load_li2014

X, y = load_li2014(split="train", return_X_y=True)
pipe = AptaMCTSPipeline().fit(X, y["label"])

X_test, y_test = load_li2014(split="test", return_X_y=True)
labels = pipe.predict(X_test)
probabilities = pipe.predict_proba(X_test)
```

`X` is a {class}`~pyaptamer.data.MoleculeLoader` or a `pandas.DataFrame`
with an aptamer column and a protein column. Other columns are ignored. `y`
holds one binary label per row; li2014 uses the strings `positive` and
`negative`. `predict` returns labels and `predict_proba` returns one row of
class probabilities per pair, in the order of `pipe.pipeline_.classes_`.

AptaMCTS is designed for RNA aptamers. A DNA sequence is converted to RNA
before encoding, with `T` replaced by `U`, so both spellings of one aptamer
give the same features.

The column names default to `aptamer` and `protein` and can be changed:

```python
pipe = AptaMCTSPipeline(aptamer_col="apt", protein_col="target")
```

## Sequences from files

A `MoleculeLoader` cell may be a path to a sequence file. With
`tiling="samples"` a FASTQ of aptamer reads becomes one row per read, and a
protein given as a string is repeated on every row:

```python
from pyaptamer.data import MoleculeLoader
from pyaptamer.datasets import load_1gnh

target = load_1gnh(tiling="first").to_dataframe()["sequence"].iloc[0]

X_new = MoleculeLoader(
    data={"aptamer": ["reads.fastq"], "protein": [target]},
    tiling="samples",
)
probabilities = pipe.predict_proba(X_new)
```

See {doc}`molecule_loader` for the loader's options.

## Feature extraction

The pipeline runs three transformers, each on one column with
{class}`~pyaptamer.trafos.compose.ApplyToCols`:

- {class}`~pyaptamer.trafos.transform.DNAToRNA` on the aptamer column,
- {class}`~pyaptamer.trafos.encode.KMerFrequencies` with `alphabet="ACGU"`
  and `normalize="length"` on the aptamer column, k-mers of length 1 to
  `rna_k`, 340 values by default,
- {class}`~pyaptamer.trafos.encode.ConjointTriad` on the protein column,
  k-mers of length 1 to `prot_k` over the 7 conjoint triad groups, 399
  values by default.

Dividing the counts by the sequence length is what the Apta-MCTS code calls
the improved conjoint triad feature, iCTF. The same features on their own:

```python
from sklearn.pipeline import Pipeline

from pyaptamer.trafos.compose import ApplyToCols
from pyaptamer.trafos.encode import ConjointTriad, KMerFrequencies
from pyaptamer.trafos.transform import DNAToRNA

features = Pipeline(
    [
        ("to_rna", ApplyToCols(DNAToRNA(), cols="aptamer")),
        (
            "aptamer",
            ApplyToCols(
                KMerFrequencies(k=4, alphabet="ACGU", normalize="length"),
                cols="aptamer",
            ),
        ),
        ("protein", ApplyToCols(ConjointTriad(k=3), cols="protein")),
    ]
)

Xt = features.fit_transform(X)
Xt.shape, Xt.columns[:3].tolist(), Xt.columns[340:343].tolist()
```

```text
((2320, 739), ['aptamer__0', 'aptamer__1', 'aptamer__2'], ['protein__0', 'protein__1', 'protein__2'])
```

The `rna_k` and `prot_k` arguments set the longest k-mer of each encoder:

```python
pipe = AptaMCTSPipeline(rna_k=3, prot_k=2)
```

See {doc}`encodings` for the encoders on their own.

## Generating candidates

A fitted pipeline scores candidates for one target protein through
{class}`~pyaptamer.experiments.AptamerEvalAptaMCTS`. The score of a
candidate is its predicted probability of the positive class when paired
with the target. {class}`~pyaptamer.mcts.MCTS` searches for a sequence with
a high score. `depth` is the length of the generated sequence:

```python
from pyaptamer.experiments import AptamerEvalAptaMCTS
from pyaptamer.mcts import MCTS

experiment = AptamerEvalAptaMCTS(target=target, pipeline=pipe)
mcts = MCTS(experiment=experiment, depth=20, n_iterations=1000)
result = mcts.run(verbose=False)
result["candidate"], result["score"]
```

## Swapping the estimator

`AptaMCTSPipeline` runs
`sklearn.ensemble.RandomForestClassifier(class_weight="balanced")` by
default. The paper trained forests of 35 to 200 trees and kept the best.
Pass any `scikit-learn` classifier to replace it:

```python
from sklearn.ensemble import RandomForestClassifier

pipe = AptaMCTSPipeline(
    estimator=RandomForestClassifier(n_estimators=200, class_weight="balanced")
)
```

## Reference

- Lee, G., Jang, G.H., Kang, H.Y., Song, G. Predicting aptamer sequences
  that interact with target proteins using an aptamer-protein interaction
  classifier and a Monte Carlo tree search approach. *PLoS ONE* 16,
  e0253760 (2021). <https://doi.org/10.1371/journal.pone.0253760>
- Code: <https://github.com/leekh7411/Apta-MCTS>
