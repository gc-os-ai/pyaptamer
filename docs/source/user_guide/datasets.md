# Datasets

The loaders in {mod}`pyaptamer.datasets` return a
{class}`~pyaptamer.data.MoleculeLoader`, so their output goes straight into
an estimator or transformer.

| Loader | Content |
| --- | --- |
| {func}`~pyaptamer.datasets.load_li2014` | 2900 aptamer-protein pairs with `positive` and `negative` labels, split into 2320 train and 580 test rows. The benchmark data of AptaTrans and AptaMCTS. |
| {func}`~pyaptamer.datasets.load_aptacom` | AptaCom, aptamer-target pairs with binding affinities, downloaded from Hugging Face. |
| {func}`~pyaptamer.datasets.load_1gnh`, {func}`~pyaptamer.datasets.load_1brq`, {func}`~pyaptamer.datasets.load_5nu7` | Bundled protein structures. |
| {func}`~pyaptamer.datasets.load_from_rcsb` | Any structure by PDB id, downloaded from RCSB and cached. |
| {func}`~pyaptamer.datasets.load_sample_fastq` | Ten SELEX reads, one row per read. |
| {func}`~pyaptamer.datasets.load_csv_dataset` | A bundled CSV file by name, as a `pandas.DataFrame`. |
| {func}`~pyaptamer.datasets.load_hf_to_dataset` | Any Hugging Face dataset, as a `datasets.Dataset`. |

## Interaction pairs

```python
from pyaptamer.datasets import load_li2014

X, y = load_li2014(split="train", return_X_y=True)
X.to_dataframe().head(2)
```

```text
                                          aptamer                                            protein
0              UCGGAGGUGGUUCAGCUCUGCAUCGACAGUUGGC  MDSTNYISKLFEYAQRQGQISDIKFEEVGTDGPDHLKTFTLRVVIK...
1  CGGGGTGTTGTCCTGTGCTCTCCGGAGAGCACAGGACAACACCCCG  MDELFPLIFPAEPAQASGPYVEIIEQPKQRGMRFRYKCEGRSAGSI...
```

`y` is a `DataFrame` with the column `label`. Without `split` both parts are
concatenated, train first. Without `return_X_y` one loader holds all three
columns.

```python
y["label"].value_counts()
```

```text
label
negative    1740
positive     580
Name: count, dtype: int64
```

## Protein structures

The structure loaders take the `tiling` argument of `MoleculeLoader`. The
default keeps all chains of a structure in one cell. `"first"` gives the
first chain as a string and `"samples"` one row per chain.

```python
from pyaptamer.datasets import load_1gnh, load_from_rcsb

protein = load_1gnh(tiling="first").to_dataframe()["sequence"].iloc[0]
chains = load_1gnh(tiling="samples").to_dataframe()

thrombin = load_from_rcsb("1HAO", tiling="first")
```

`load_from_rcsb` stores the file under `~/.cache/pyaptamer/pdb` and reuses
it on later calls.

## SELEX reads

```python
from pyaptamer.datasets import load_sample_fastq

reads = load_sample_fastq().to_dataframe()
reads.shape
```

```text
(10, 1)
```

See {doc}`molecule_loader` to load your own FASTQ files.
