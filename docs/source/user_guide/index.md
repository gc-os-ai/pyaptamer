# User guide

```{toctree}
:hidden:
:maxdepth: 1

molecule_loader
encodings
aptanet
aptatrans
```

## Data

:::{list-table}
:widths: 30 70

* - {doc}`MoleculeLoader <molecule_loader>`
  - Sequences and structure files as a lazy table. Tiling, indexing,
    metadata columns, bundled datasets.
:::

## Transformers

:::{list-table}
:widths: 30 70

* - {doc}`Feature encodings <encodings>`
  - {class}`~pyaptamer.trafos.encode.PSeAAC` for proteins,
    {class}`~pyaptamer.trafos.encode.KMerFrequencies` for nucleotides,
    column-wise encoding with `ColumnTransformer`.
:::

## Algorithms

:::{list-table}
:widths: 30 70

* - {doc}`AptaNet <aptanet>`
  - Aptamer-protein interaction prediction.
* - {doc}`AptaTrans <aptatrans>`
  - Interaction scoring and candidate aptamer generation.
:::
