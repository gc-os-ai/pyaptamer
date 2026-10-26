# User guide

```{toctree}
:hidden:
:maxdepth: 1

molecule_loader
encodings
datasets
aptamcts
aptanet
aptatrans
deepaptamer
aptadiff
mcts
benchmarking
```

## Data

:::{list-table}
:widths: 30 70

* - {doc}`MoleculeLoader <molecule_loader>`
  - Sequences and structure files as a lazy table. Tiling, indexing,
    metadata columns, bundled datasets.
* - {doc}`Datasets <datasets>`
  - Interaction pairs with labels, protein structures and SELEX reads,
    bundled or downloaded.
:::

## Transformers

:::{list-table}
:widths: 30 70

* - {doc}`Feature encodings <encodings>`
  - {class}`~pyaptamer.transformations.encode.PSeAAC` and
    {class}`~pyaptamer.transformations.encode.ConjointTriad` for proteins,
    {class}`~pyaptamer.transformations.encode.KMerFrequencies` for nucleotides,
    column-wise encoding with
    {class}`~pyaptamer.transformations.compose.ApplyToCols` or `ColumnTransformer`.
:::

## Algorithms

:::{list-table}
:widths: 30 70

* - {doc}`AptaMCTS <aptamcts>`
  - Aptamer-protein interaction prediction and candidate aptamer generation.
* - {doc}`AptaNet <aptanet>`
  - Aptamer-protein interaction prediction.
* - {doc}`AptaTrans <aptatrans>`
  - Interaction scoring and candidate aptamer generation.
* - {doc}`DeepAptamer <deepaptamer>`
  - Binding prediction for DNA aptamers from sequence and DNA shape.
* - {doc}`AptaDiff <aptadiff>`
  - Sequence generation with a diffusion model.
* - {doc}`Monte Carlo tree search <mcts>`
  - Sequence search with a score of your choice.
:::

## Evaluation

:::{list-table}
:widths: 30 70

* - {doc}`Benchmarking <benchmarking>`
  - Cross-validated comparison of estimators on the same folds.
:::
