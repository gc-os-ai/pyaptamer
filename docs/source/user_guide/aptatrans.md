# AptaTrans

AptaTrans is a transformer model for aptamer-protein interaction. It does two
things: it scores how strongly a candidate aptamer interacts with a target
protein, and it recommends new candidate aptamers for a target using a Monte
Carlo Tree Search guided by that score.

The entry point is {class}`~pyaptamer.aptatrans.AptaTransPipeline`. The search
itself is exposed separately as {class}`~pyaptamer.mcts.MCTS`.

## Building a pipeline

`AptaTransPipeline` wraps an {class}`~pyaptamer.aptatrans.AptaTrans` model. The
model is configured with one {class}`~pyaptamer.aptatrans.EncoderPredictorConfig`
per encoder (aptamer and protein). `prot_words` maps protein subsequences to
integer ids and should come from the dataset used to pretrain the protein
encoder.

```python
import torch
from pyaptamer.aptatrans import (
    AptaTrans,
    AptaTransPipeline,
    EncoderPredictorConfig,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

apta_embedding = EncoderPredictorConfig(128, 16, max_len=100)
prot_embedding = EncoderPredictorConfig(128, 16, max_len=100)
model = AptaTrans(apta_embedding, prot_embedding)

prot_words = {"DHR": 0.5, "AIQ": 0.5, "AAG": 0.2}
pipeline = AptaTransPipeline(
    device, model, prot_words, depth=5, n_iterations=5
)
```

`depth` sets the length of generated candidates and the search tree depth.
`n_iterations` sets the number of MCTS iterations per candidate.

## Scoring an interaction

```python
target = "DHRNENIAIQ"
aptamer = "ACGUA"

score = pipeline.predict(aptamer, target)
interaction_map = pipeline.get_interaction_map(aptamer, target)
```

`predict` returns an interaction score tensor. `get_interaction_map` returns the
position-by-position interaction tensor between the two sequences.

## Recommending candidates

```python
candidates = pipeline.recommend(target, n_candidates=3, verbose=False)
```

`recommend` returns a set of `(candidate, sequence, score)` tuples. Search runs
until `n_candidates` unique candidates are found.

## Running the search directly

To use the search without the pipeline, drive {class}`~pyaptamer.mcts.MCTS`
yourself with an experiment that scores sequences:

```python
import torch
from pyaptamer.aptatrans import AptaTrans, EncoderPredictorConfig
from pyaptamer.experiments import AptamerEvalAptaTrans
from pyaptamer.mcts import MCTS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AptaTrans(
    EncoderPredictorConfig(128, 16, max_len=128),
    EncoderPredictorConfig(128, 16, max_len=128),
).to(device)

experiment = AptamerEvalAptaTrans(
    target="DHRNE",
    model=model,
    device=device,
    prot_words={"DHR": 1, "RNE": 2, "NE": 3},
)
mcts = MCTS(depth=5, n_iterations=2, experiment=experiment)

result = mcts.run(verbose=False)
candidate = result["candidate"]
```

## References

- Shin, I., et al. AptaTrans: a deep neural network for predicting
  aptamer-protein interaction using pretrained encoders. *BMC Bioinformatics*
  24, 447 (2023).
- Lee, G., et al. Predicting aptamer sequences that interact with target proteins
  using an aptamer-protein interaction classifier and a Monte Carlo tree search
  approach. *PLoS ONE* 16, e0253760 (2021).
