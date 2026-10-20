# AptaDiff

AptaDiff designs new aptamers from SELEX reads. It takes the raw reads of a
single selection round, strips their primer regions, reads the latent
coordinates of each read out of a pretrained variational autoencoder, and
trains a multinomial diffusion model to generate sequences conditioned on
those coordinates. A Gaussian mixture fitted to the same coordinates supplies
the latent points that new sequences are generated from.

The entry point is {class}`~pyaptamer.aptadiff.AptaDiff`. The diffusion model
is exposed separately as {class}`~pyaptamer.aptadiff.AptaDiffGenerator`.

AptaDiff is unsupervised and generative, so it has no `predict`. New sequences
come from `sample`, and `y` is ignored by `fit`.

## Building the estimator

`AptaDiff` composes three parts that are passed in: the latent model `vae`,
the `trimmer` that cleans the reads, and the `generator` that trains the
diffusion model.

```python
import torch
from pyaptamer.aptadiff import AptaDiff, AptaDiffGenerator
from pyaptamer.datasets import load_sample_fastq
from pyaptamer.raptgen import CNNPHMMVAE
from pyaptamer.transformations.transform import PrimerTrimmer

X = load_sample_fastq()

vae = CNNPHMMVAE(motif_len=40, embed_size=2)
# vae.load_state_dict(torch.load("raptgen_weights.pt"))

trimmer = PrimerTrimmer(
    start_primer="TAATACGACTCACTATAGGGAGAACTTCGACCAGAAG",
    end_primer="TATGTGCGCATACATGGATCCTC",
    variable_length=40,
)

generator = AptaDiffGenerator(
    num_timesteps=5, dim=32, depth=1, max_epochs=1, random_state=0
)

aptadiff = AptaDiff(
    vae=vae,
    trimmer=trimmer,
    generator=generator,
    n_clusters=3,
    gmm_restarts=5,
    random_state=0,
)
```

`X` is a {class}`~pyaptamer.data.MoleculeLoader` or a `pandas.DataFrame` with
one read per row in a single column. A `MoleculeLoader` cell may be a path to
a FASTQ file, which `tiling="samples"` turns into one row per read, see
{doc}`molecule_loader` for the loader's options.

`vae` maps integer-encoded sequences to latent coordinates. It is used for
inference only, and is never trained, inspected or migrated by `AptaDiff`.
Pass either an object with a `transform` method or a module with `encoder` and
`h2mu` attributes, such as {class}`~pyaptamer.raptgen.CNNPHMMVAE`. The weights
must be trained, as an untrained model gives random, unusable coordinates and
therefore meaningless sequences.

`trimmer` removes the constant primer binding sites of the SELEX design with
{class}`~pyaptamer.transformations.transform.PrimerTrimmer`, keeping the random
region. The diffusion model generates one fixed length, so without a trimmer
`X` must already hold sequences that are all the same length.

Reads are encoded with
{class}`~pyaptamer.transformations.encode.SequenceOneHotEncoder`. The default
treats `A`, `T`, `G` and `C` as four classes, with `U` sharing the index of
`T`, and drops any read holding another character. Pass a configured encoder
to supply a custom vocabulary or to make unknown characters raise:

```python
from pyaptamer.transformations.encode import SequenceOneHotEncoder

aptadiff = AptaDiff(vae=vae, encoder=SequenceOneHotEncoder(handle_unknown="raise"))
```

## Generating sequences

`fit` trains the diffusion model and the mixture. `sample` then draws latent
points from the mixture and generates one sequence per point:

```python
aptadiff.fit(X)

sequences = aptadiff.sample(n_samples=10)
```

`sample` returns a `pandas.DataFrame` with the generated sequences in a column
named `sequence`, all of length `seq_len_`. The reads that survived cleaning
are kept as `sequences_`, and their latent coordinates as `latents_`.

`reconstruct` regenerates the input sequences from their own latents, and
`score` reports the average log-likelihood of sequences in bits per character,
negated so that higher is better:

```python
reconstructed = aptadiff.reconstruct(X)
bits_per_character = aptadiff.score(X)
```

## Designing from cluster centers

Generation is conditioned only on latent coordinates, so steering it towards
better binders means choosing better latent points. The component means of the
fitted mixture are available as `gmm_centers_`, one point per learned cluster.
Passing them to `inverse_transform` generates one sequence per cluster, which
is the de novo design procedure of the paper:

```python
candidates = aptadiff.inverse_transform(aptadiff.gmm_centers_)
```

`inverse_transform` accepts any latent points with `embed_size_` columns, not
only the cluster centers. Points proposed by an outer optimisation loop over
wet-lab affinity measurements, such as Bayesian optimisation, are passed the
same way. `transform` maps sequences in the other direction, to their latent
coordinates.

`n_clusters` sets the number of mixture components, and `gmm_restarts` the
number of fits attempted, of which the one with the lowest AIC is kept:

```python
aptadiff = AptaDiff(vae=vae, n_clusters=8, gmm_restarts=100)
```

## Configuring the diffusion model

`AptaDiff` delegates diffusion training to
{class}`~pyaptamer.aptadiff.AptaDiffGenerator`, whose defaults match the
original implementation and are sized for full SELEX datasets. Pass a
configured generator to change them:

```python
generator = AptaDiffGenerator(
    num_timesteps=1000,
    dim=512,
    depth=12,
    max_epochs=1000,
    patience=10,
    batch_size=32,
    lr=1e-4,
    verbose=1,
)

aptadiff = AptaDiff(vae=vae, generator=generator)
```

`num_timesteps` is the number of diffusion steps, and `predict` runs one
denoiser pass per step. `patience` stops training early when the validation
loss stops improving, the default trains for all `max_epochs`, as the original
training script does.

`transformer_type` selects the attention backend. The default `"native"` uses
PyTorch's own transformer layer, giving exact attention and hardware
acceleration. `"linear"` is the linear attention approximation of the paper,
for strict reproducibility.

`num_classes` must match the number of classes the encoder produces, four by
default. A mismatch raises `ValueError`.

Training runs on the fastest available device, and the generator can also be used independently with latent vectors and one-hot encoded sequences.

## References

- Wang, Z., Liu, Z., Zhang, W., Li, Y., Feng, Y., Chen, S., Zhao, Y.
  AptaDiff: de novo design and optimization of aptamers based on diffusion
  models. *Briefings in Bioinformatics* 25(6), bbae517 (2024).
  <https://doi.org/10.1093/bib/bbae517>
- Code: <https://github.com/wz-create/AptaDiff>
- Iwano, N., Adachi, T., Aoki, K., Nakamura, Y., Hamada, M. Generative
  aptamer discovery using RaptGen. *Nature Computational Science* 2, 378-386
  (2022). <https://doi.org/10.1038/s43588-022-00249-6>
