# AptaDiff

AptaDiff generates aptamer sequences with a diffusion model. Each training
sequence is paired with its latent coordinates from a pretrained encoder,
for example a RaptGen variational autoencoder. The diffusion model learns to
produce a sequence from its coordinates. New sequences come from new
coordinates: points drawn from a Gaussian mixture fitted to the training
coordinates, the mixture centers, or points chosen by an optimization loop.

The entry point is {class}`~pyaptamer.aptadiff.AptaDiff`.

## Fitting on SELEX reads

`fit` takes raw SELEX reads as a {class}`~pyaptamer.data.MoleculeLoader` or
a `pandas.DataFrame` with one read per row. The reads pass through four
components:

- `trimmer` strips the constant primer regions, typically a
  {class}`~pyaptamer.transformations.transform.PrimerTrimmer`. Without a
  trimmer, all reads must have the same length.
- `encoder` one-hot encodes the trimmed reads. The default drops reads with
  characters other than A, T, G, C and U.
- `vae` is the pretrained latent model. It is used for inference only.
- `generator` is the diffusion model, an
  {class}`~pyaptamer.aptadiff.AptaDiffGenerator`.

```python
from pyaptamer.aptadiff import AptaDiff, AptaDiffGenerator
from pyaptamer.datasets import load_sample_fastq
from pyaptamer.raptgen import CNNPHMMVAE
from pyaptamer.transformations.transform import PrimerTrimmer

vae = CNNPHMMVAE(motif_len=40, embed_size=2)

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
aptadiff.fit(load_sample_fastq())
```

The VAE here is untrained and the diffusion model is tiny, so the example
runs in seconds and shows the mechanics only. In practice `vae` is a RaptGen
model trained on the same reads, and `generator` keeps its defaults, which
match the paper.

After fitting, `latents_` holds the coordinates of the training reads,
`sequences_` the reads they belong to, and `gmm_centers_` the means of the
mixture.

## Generating sequences

`sample` draws coordinates from the mixture and decodes them:

```python
new = aptadiff.sample(10)
```

`inverse_transform` decodes any coordinates. The original implementation
generates one sequence per cluster center:

```python
centers = aptadiff.inverse_transform(aptadiff.gmm_centers_)
```

Both return a `pandas.DataFrame` with a `sequence` column.

Optimization is external. `transform` gives the coordinates of measured
sequences, an optimization loop proposes new coordinates from their measured
affinities, and `inverse_transform` turns the proposals into sequences to
test next.

`score` returns the mean log-likelihood of reads in bits per character, and
`score_samples` one value per read. Higher is better.

## The diffusion model on its own

{class}`~pyaptamer.aptadiff.AptaDiffGenerator` can be used without the
pipeline. `fit` takes the latent vectors as `X` and the one-hot encoded
sequences as `y`, in the layout of
{class}`~pyaptamer.transformations.encode.SequenceOneHotEncoder`. `predict`
returns one one-hot sequence per row of `X`.

The defaults match the paper: `dim=512`, `depth=12`, `heads=16`,
`num_timesteps=1000` and `max_epochs=1000`. Training uses a GPU when one is
available. A share of the data, `validation_fraction`, is held out, and the
weights of the epoch with the lowest validation loss are kept. `patience`
stops training after that many epochs without improvement.

```python
generator = AptaDiffGenerator(max_epochs=200, patience=20, random_state=0)
```

## Reference

- Wang, Z., et al. AptaDiff: de novo design and optimization of aptamers
  based on diffusion models. *Briefings in Bioinformatics* 25, bbae517
  (2024). <https://doi.org/10.1093/bib/bbae517>
- Code: <https://github.com/wz-create/AptaDiff>
