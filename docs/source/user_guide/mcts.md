# Monte Carlo tree search

{class}`~pyaptamer.mcts.MCTS` searches for a sequence with a high score. It
builds the sequence one nucleotide at a time, at either end, and spends more
rounds on the branches that scored well earlier. The score comes from an
experiment object whose `evaluate` method maps a sequence to a number, higher
is better. This is the search behind the candidate generation of AptaMCTS and
AptaTrans.

## Running a search

A fitted interaction classifier gives the score through one of
{class}`~pyaptamer.experiments.AptamerEvalAptaMCTS`,
{class}`~pyaptamer.experiments.AptamerEvalAptaNet` and
{class}`~pyaptamer.experiments.AptamerEvalAptaTrans`. The score is the
predicted probability that the candidate binds the target protein.

```python
from pyaptamer.aptamcts import AptaMCTSPipeline
from pyaptamer.datasets import load_li2014
from pyaptamer.experiments import AptamerEvalAptaMCTS
from pyaptamer.mcts import MCTS

X, y = load_li2014(split="test", return_X_y=True)
pipe = AptaMCTSPipeline().fit(X, y["label"])
target = X.to_dataframe()["protein"].iloc[0]

experiment = AptamerEvalAptaMCTS(target=target, pipeline=pipe)
mcts = MCTS(experiment=experiment, depth=20, n_iterations=200)
result = mcts.run(verbose=False)
result["candidate"], result["score"]
```

`depth` is the length of the generated sequence. `n_iterations` is the number
of search rounds per position. More rounds score more candidates and take
longer.

`run` returns a dict with `candidate`, the generated sequence, `score`, its
score, and `sequence`, the raw move string of the search.

## Alphabet

`states` lists the moves. A letter before `_` is added at the start of the
sequence, a letter after `_` at the end. The default is RNA:
`["A_", "C_", "G_", "U_", "_A", "_C", "_G", "_U"]`. For DNA:

```python
mcts = MCTS(
    experiment=experiment,
    states=["A_", "C_", "G_", "T_", "_A", "_C", "_G", "_T"],
    depth=20,
)
```

## Your own score

Any object with an `evaluate(candidate)` method works as the experiment. This
one rewards GC content:

```python
class GCContent:
    def evaluate(self, candidate):
        return sum(letter in "GC" for letter in candidate) / len(candidate)


mcts = MCTS(experiment=GCContent(), depth=10, n_iterations=50)
mcts.run(verbose=False)
```

The search is random, so the candidate differs between runs. Its score is
1.0 whenever it contains only G and C.

## Reference

- Lee, G., et al. Predicting aptamer sequences that interact with target
  proteins using an aptamer-protein interaction classifier and a Monte Carlo
  tree search approach. *PLoS ONE* 16, e0253760 (2021).
  <https://doi.org/10.1371/journal.pone.0253760>
