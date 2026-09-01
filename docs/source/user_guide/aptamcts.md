# AptaMCTS

AptaMCTS predicts whether an aptamer and a protein interact. It takes
`(aptamer, protein)` sequence pairs, derives numeric features using the Improved Conjoint Triad Feature (iCTF) encoding, and classifies them using a Random Forest model. The estimator follows the `scikit-learn` `fit` / `predict` API and is designed to act as the scoring function for generating novel aptamer sequences via a Monte Carlo Tree Search (MCTS).

The entry point is {class}`~pyaptamer.aptamcts.AptaMCTSPipeline`.

## Predicting interactions

Because AptaMCTS derives features from two different molecule types, it requires inputs to be passed as a {class}`~pyaptamer.data.MoleculeLoader`.

```python
import numpy as np
from pyaptamer.aptamcts import AptaMCTSPipeline
from pyaptamer.data import MoleculeLoader

aptamers = [
    "GGGAGGACGAAGACGACUCGAGACAGGCUAGGGAGGGA",
    "AAGCGUCGGAUCUACACGUGCGAUAGCUCAGUACGCGGU",
    "CGGUAUCGAGUACAGGAGUCCGACGGAUAGUCCGGAGC",
]
protein = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"

X = MoleculeLoader(data={"aptamer": aptamers * 10, "protein": [protein] * 30})
y = np.array([0, 1, 0] * 10, dtype=np.float32)

pipe = AptaMCTSPipeline()
pipe.fit(X, y)

labels = pipe.predict(X[:3])
probabilities = pipe.predict_proba(X[:3])
```

`predict` returns class labels and `predict_proba` returns class probabilities.

The `rna_k` and `prot_k` arguments set the k-mer sizes used for the iCTF feature extraction, while `aptamer_col` and `protein_col` map to the MoleculeLoader keys:

```python
pipe = AptaMCTSPipeline(
    rna_k=4, 
    prot_k=3, 
    aptamer_col="aptamer", 
    protein_col="protein"
)
```

## Swapping the estimator
AptaMCTSPipeline runs {class}`~pyaptamer.aptamcts.AptaMCTSClassifier` (a Random Forest implementation) by default. Pass any scikit-learn compatible classifier to replace it:

```python
from sklearn.svm import SVC

pipe = AptaMCTSPipeline(estimator=SVC(probability=True))
```

## Running the search
To generate new candidate aptamers for a specific target protein, you can use the fitted pipeline as the score function for a Monte Carlo Tree Search. Drive {class}`~pyaptamer.mcts.MCTS` by wrapping your pipeline in the AptaMCTS experiment class:

```python
from pyaptamer.experiments import AptamerEvalAptaMCTS
from pyaptamer.mcts import MCTS

pipeline = AptaMCTSPipeline().fit(X, y)

target_protein = "ACDEFGHIKLMN"
experiment = AptamerEvalAptaMCTS(target=target_protein, pipeline=pipeline)

mcts = MCTS(depth=10, n_iterations=5000, experiment=experiment)
result = mcts.run(verbose=False)
candidate = result["candidate"]
```

## Reference
- Lee, G., Jang, G.H., Kang, H.Y., Song, G. Predicting aptamer sequences that interact with target proteins using an aptamer-protein interaction classifier and a Monte Carlo tree search approach. *PLoS ONE* 16, e0253760 (2021). <https://doi.org/10.1371/journal.pone.0253760>
