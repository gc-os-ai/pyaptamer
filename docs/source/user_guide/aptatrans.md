# AptaTrans

AptaTrans predicts whether an aptamer and a protein interact, and generates
candidate aptamers for a target protein. It encodes each sequence as token
ids and embeds them with two transformer encoders, one for aptamers and one
for proteins. A convolutional network classifies the pair from the map of
interactions between the two embeddings. A fitted pipeline then scores the
candidates proposed by a Monte Carlo tree search.

The entry point is {class}`~pyaptamer.aptatrans.AptaTransPipeline`.

## Predicting interactions

The li2014 data of the paper is bundled as a
{class}`~pyaptamer.data.MoleculeLoader` with an `aptamer` column, a
`protein` column and a label column:

```python
from pyaptamer.aptatrans import AptaTransPipeline
from pyaptamer.datasets import load_li2014

X, y = load_li2014(split="train", return_X_y=True)
pipe = AptaTransPipeline().fit(X, y["label"])

X_test, y_test = load_li2014(split="test", return_X_y=True)
labels = pipe.predict(X_test)
probabilities = pipe.predict_proba(X_test)
```

`X` is a {class}`~pyaptamer.data.MoleculeLoader` or a `pandas.DataFrame`
with an aptamer column and a protein column. Other columns are ignored. `y`
holds one binary label per row; li2014 uses the strings `positive` and
`negative`. `predict` returns labels and `predict_proba` returns one row of
class probabilities per pair, in the order of `pipe.pipeline_.classes_`.

A DNA aptamer is converted to RNA before encoding, with `T` replaced by `U`,
so both spellings of one aptamer give the same prediction.

The column names default to `aptamer` and `protein` and can be changed:

```python
pipe = AptaTransPipeline(aptamer_col="apt", protein_col="target")
```

## The network

The network and its training settings are held by
{class}`~pyaptamer.aptatrans.AptaTransClassifier`. The default network has
the sizes of the paper. Each pair gives an interaction map of 275 aptamer
positions by 867 protein positions, so training it needs a GPU. A smaller
network on shorter sequences trains on a CPU:

```python
from pyaptamer.aptatrans import AptaTransClassifier

clf = AptaTransClassifier(
    apta_max_len=64,
    in_dim=32,
    n_heads=4,
    n_encoder_layers=1,
    conv_layers=[1, 1, 1],
    max_epochs=2,
)
pipe = AptaTransPipeline(apta_max_len=64, prot_max_len=128, estimator=clf)
```

Training runs with Lightning. In each `fit`, the fraction
`validation_fraction` of the pairs, 5% by default, is held out. Training
keeps the weights of the epoch with the lowest validation loss, and stops
after `patience` epochs without improvement, 10 by default.

## Feature extraction

The pipeline encodes each pair as one row of integer token ids:

- {class}`~pyaptamer.transformations.transform.DNAToRNA` converts the aptamer
  to RNA.
- {class}`~pyaptamer.transformations.encode.KMerIndexEncoder` with `k=3` and
  `alphabet="ACGUN"` encodes the aptamer as the ids of its overlapping
  triplets, `apta_max_len` ids, 275 by default.
- {class}`~pyaptamer.transformations.encode.GreedyEncoder` encodes the protein
  by greedy longest match against the protein words, `prot_max_len` ids, 867
  by default.

Both blocks are zero-padded or truncated to their length, and id 0 is
padding. The protein words come from the bundled `protein_word_freq`
dataset of the paper. Words with a frequency above the mean are kept, 713
words. Pass your own word frequencies to replace them:

```python
pipe = AptaTransPipeline(prot_words={"DHR": 0.5, "AIQ": 0.5, "AAG": 0.2})
```

The token rows of a fitted pipeline:

```python
tokens = pipe.pipeline_[:-1].transform(X_test.to_dataframe())
tokens.shape
```

```text
(580, 1142)
```

The first 275 columns are the aptamer ids and the other 867 the protein ids.

## Interaction map

The interaction map is the position-by-position map between the aptamer and
protein embeddings that the convolutional network reads:

```python
imap = pipe.interaction_map(X_test)
imap.shape
```

```text
(580, 1, 275, 867)
```

## Generating candidates

A fitted pipeline generates candidate aptamers for one target protein. Each
candidate comes from one run of {class}`~pyaptamer.mcts.MCTS`, with the
predicted probability of the second class in `classes_` as the score:

```python
target = X_test.to_dataframe()["protein"].iloc[0]
candidates = pipe.generate(target, n_candidates=3, depth=20, n_iterations=1000)
```

`generate` returns one `(candidate, sequence, score)` tuple per candidate,
highest score first. `depth` is the length of each candidate and must be at
least 3, the length of one triplet. `sequence` is the raw move string of the
search.

To drive the search yourself, pass the pipeline to
{class}`~pyaptamer.experiments.AptamerEvalAptaTrans`. See {doc}`mcts`.

## Pretraining the encoders

The paper pretrains the two encoders before training the classifier. The
aptamer encoder learns the secondary structure of RNA sequences from bpRNA,
and the protein encoder the secondary structure of proteins from the Protein
Data Bank. {class}`~pyaptamer.aptatrans.AptaTransEncoderLightning` runs this
step. The tutorial notebook `examples/aptatrans_tutorial.ipynb` walks
through it.

## References

- Shin, I., et al. AptaTrans: a deep neural network for predicting
  aptamer-protein interaction using pretrained encoders. *BMC Bioinformatics*
  24, 447 (2023).
- Lee, G., et al. Predicting aptamer sequences that interact with target proteins
  using an aptamer-protein interaction classifier and a Monte Carlo tree search
  approach. *PLoS ONE* 16, e0253760 (2021).
- Code: <https://github.com/PNUMLB/AptaTrans>
