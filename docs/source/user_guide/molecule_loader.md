# Loading molecules with MoleculeLoader

{class}`~pyaptamer.data.MoleculeLoader` is a lazy 2D table of molecules. Each
cell holds either a file path (PDB, FASTA, FASTQ, GenBank, EMBL, ...) or an
in-memory value such as a sequence string, a number, or a label. Nothing is
parsed until you call `to_dataframe()`, which reads every file cell and returns
a `pandas.DataFrame`.

Every estimator and transformer in `pyaptamer` that takes sequences accepts a
`MoleculeLoader` directly, so the loader is the normal way to get files into a
pipeline.

## Building a loader

`data` is anything `pandas.DataFrame` accepts: a dict of columns, a list of
rows, or a DataFrame. Column names are yours to choose.

```python
from pathlib import Path
from pyaptamer.data import MoleculeLoader

loader = MoleculeLoader(
    data={
        "aptamer": [Path("round5.fastq")],
        "protein": [Path("1gnh.pdb")],
        "label": [1],
    }
)
df = loader.to_dataframe()
```

A cell is read as a file when it is a `pathlib.Path` (always) or a `str` with a
suffix such as `.pdb` or `.fastq`. A `str` without a suffix stays a literal,
so sequence strings are never mistaken for filenames. The parser is chosen from
the suffix: `.pdb` files are read through Biopython's `pdb-seqres` parser,
everything else through `Bio.SeqIO.parse`. FASTQ quality scores are discarded.

## Where multiple sequences go: `tiling`

A PDB file may hold several chains, and a FASTQ file holds many reads.
`tiling` decides where these sequences go in the output table.

| `tiling` | Result for a cell holding a file with several sequences |
| --- | --- |
| `"bag"` (default) | One cell holding a `list` of strings. A single-sequence file stays a plain `str`. |
| `"first"` | One cell holding the first sequence. |
| `"concat"` | One cell holding all sequences joined into one string, ordered by chain ID. |
| `"features"` | One column per sequence: `<col>_0`, `<col>_1`, ... Shorter rows are padded with `None`. |
| `"samples"` | One row per sequence. Literal columns are repeated on every new row. |
| `"samples_product"` | Like `"samples"`, but with several file columns in one row it takes their cartesian product. |

For a multi-chain protein structure, `"bag"`, `"first"` or `"concat"` keep the
molecule in one cell. For SELEX reads, `"samples"` gives the one-read-per-row
table that transformers such as
{class}`~pyaptamer.trafos.transform.PrimerTrimmer` expect.

```python
loader = MoleculeLoader(data={"read": [Path("round5.fastq")]}, tiling="samples")
reads = loader.to_dataframe()  # one row per read
```

`"samples"` refuses a row with two or more file columns, because zipping the
chains of one file against the reads of another has no meaning. Use
`"samples_product"` for every pairing, or load the protein once as a string and
put that string next to the FASTQ file:

```python
protein = MoleculeLoader(data={"protein": [Path("1gnh.pdb")]}, tiling="first")
protein = protein.to_dataframe().loc[0, "protein"]

loader = MoleculeLoader(
    data={"aptamer": [Path("round5.fastq")], "protein": [protein]},
    tiling="samples",
)
```

## Attaching metadata to files

Any literal column next to a file column is repeated onto every sequence from
that file. This is how you label SELEX rounds: one row per FASTQ file and a
`round` column beside it.

```python
loader = MoleculeLoader(
    data={
        "read": [Path("round3.fastq"), Path("round4.fastq"), Path("round5.fastq")],
        "round": [3, 4, 5],
    },
    tiling="samples",
    indexing="new",
)
selex = loader.to_dataframe()
selex["round"].value_counts()
```

## What happens to chain and read IDs: `indexing`

PDB chain IDs and FASTQ read IDs are kept when a file expands into rows.
`indexing` chooses where they go. It only has an effect with `"samples"` and
`"samples_product"`.

| `indexing` | Result |
| --- | --- |
| `"preserve"` (default) | Row index `<original row label>__<id>`, for example `0__A`. |
| `"new"` | A fresh `RangeIndex`. IDs are dropped. |
| `"keep_as_column"` | A fresh `RangeIndex` plus a `<col>_chain_id` column holding the ID. |

```python
loader = MoleculeLoader(
    data={"read": [Path("round5.fastq")]},
    tiling="samples",
    indexing="keep_as_column",
)
loader.to_dataframe()  # columns: read, read_chain_id
```

## Flat or two-level index: `multiindex`

With `indexing="preserve"`, `multiindex` sets the shape of the index.

| `multiindex` | Result |
| --- | --- |
| `"flatten"` (default) | One string label per row, `<row>__<id>`. |
| `"multiindex"` | A `pandas.MultiIndex` with levels `row` and `sequence`. |
| `"auto"` | `"multiindex"` when some row expanded into several, `"flatten"` otherwise. |

## Dropping repeated sequences: `ignore_duplicates`

`ignore_duplicates=True` removes identical sequences within each file before
tiling. The first occurrence wins. It drops the copy number, so use it only
when the count of each sequence does not matter.

```python
loader = MoleculeLoader(
    data={"target": [Path("1gnh.pdb")]},  # 10 identical chains
    tiling="samples",
    ignore_duplicates=True,
)
loader.to_dataframe()  # one row
```

## Bundled datasets

The dataset loaders in {mod}`pyaptamer.datasets`, such as
{func}`~pyaptamer.datasets.load_1gnh` and
{func}`~pyaptamer.datasets.load_sample_fastq`, return a `MoleculeLoader` set
to `tiling="samples"`. Call `to_dataframe()` on the result, or inspect the
`tiling`, `indexing` and `multiindex` attributes first.

```python
from pyaptamer.datasets import load_1gnh

protein = load_1gnh().to_dataframe()["sequence"].iloc[0]
```

## Passing a loader to a transformer

Transformers call `to_dataframe()` themselves. The loader's settings decide
the table they see, so pick `tiling="samples"` for anything that works on one
read per row.

```python
from pyaptamer.trafos.transform import PrimerTrimmer

trimmer = PrimerTrimmer(
    start_primer="TAATACGACTCACTATAGGGAGAACTTCGACCAGAAG",
    end_primer="TATGTGCGCATACATGGATCCTC",
    variable_length=40,
)

loader = MoleculeLoader(data={"read": [Path("round5.fastq")]}, tiling="samples")
trimmed = trimmer.fit_transform(loader)
```

## Quick reference

- Choose `tiling` by where multi-sequence files should go: a cell (`bag`,
  `first`, `concat`), columns (`features`), or rows (`samples`,
  `samples_product`).
- For SELEX reads use `tiling="samples"` and put round labels in a literal
  column next to the file.
- Use `indexing="keep_as_column"` to keep read or chain IDs as data,
  `"preserve"` to keep them in the index, `"new"` to drop them.
- `ignore_duplicates=True` deduplicates within each file and drops the counts.
