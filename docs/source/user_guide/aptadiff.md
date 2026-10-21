# AptaDiff

AptaDiff generates aptamer sequences with a diffusion model. Training pairs
each sequence with a latent vector, for example its embedding from a RaptGen
variational autoencoder. Generation starts from random noise and denoises it,
step by step, into a sequence that matches a given latent vector. Moving
through the latent space then gives new sequences close to known ones.

The entry point is {class}`~pyaptamer.aptadiff.AptaDiffGenerator`.

## Training and generating

`fit` takes the latent vectors as `X` and the one-hot encoded sequences as
`y`. {class}`~pyaptamer.transformations.encode.SequenceOneHotEncoder`
produces the layout the model expects and decodes the generated output
again.

```python
import numpy as np
import pandas as pd
from pyaptamer.aptadiff import AptaDiffGenerator
from pyaptamer.data import MoleculeLoader
from pyaptamer.transformations.encode import SequenceOneHotEncoder

sequences = [
    "GGGAGGACGAAGACGACTCG",
    "AAGCGTCGGATCTACACGTG",
    "CGGTATCGAGTACAGGAGTC",
    "TTGACCGTAGCATCGATGCA",
]
encoder = SequenceOneHotEncoder()
y = encoder.fit_transform(MoleculeLoader(data={"aptamer": sequences * 10}))

rng = np.random.default_rng(0)
X = rng.normal(size=(40, 4)).astype(np.float32)

generator = AptaDiffGenerator(
    num_timesteps=20, dim=32, depth=1, heads=4, max_epochs=3, random_state=0
)
generator.fit(X, y)

generated = generator.predict(X[:3])
encoder.inverse_transform(pd.DataFrame(generated, columns=y.columns))
```

All sequences in `y` must have the same length. For sequences of length `L`
over four nucleotides, `y` has `4 * L` columns, and `predict` returns the
same layout with one row per row of `X`.

Here the latent vectors are random numbers, so the example only shows the
mechanics. In practice they come from an encoder fitted on the same
sequences, so that nearby latent vectors give similar sequences.

## Model size and training

The defaults match the paper: `dim=512`, `depth=12`, `heads=16`,
`num_timesteps=1000` and `max_epochs=1000`. The small values above make the
example run in seconds. Training uses a GPU when one is available.

A share of the data, `validation_fraction`, is held out, and the weights of
the epoch with the lowest validation loss are kept. `patience` stops training
after that many epochs without improvement. `random_state` makes the split,
the training and the generation reproducible.

```python
generator = AptaDiffGenerator(max_epochs=200, patience=20, random_state=0)
```

## Reference

- Wang, Z., et al. AptaDiff: de novo design and optimization of aptamers
  based on diffusion models. *Briefings in Bioinformatics* 25, bbae517
  (2024). <https://doi.org/10.1093/bib/bbae517>
- Code: <https://github.com/wz-create/AptaDiff>
