# MoleculeLoader

{class}`~pyaptamer.data.MoleculeLoader` is a lazy 2D table of molecules. Each
cell contains either a file path (PDB, FASTA, FASTQ, GenBank, EMBL, ...) or an
in-memory value such as a sequence string, a number, or a label. No file is
parsed until you call `to_dataframe()`, which reads every file cell and returns
a `pandas.DataFrame`.

Every estimator and transformer in `pyaptamer` that takes sequences accepts a
`MoleculeLoader` directly.

## Build a loader

`data` is anything `pandas.DataFrame` accepts: a dict of columns, a list of
rows, or a DataFrame.

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

A cell is read as a file when it is a `pathlib.Path`, or a `str` with a
suffix such as `.pdb` or `.fastq`. A `str` without a suffix remains a literal.

The file suffix selects the parser:

- `.pdb` files are read with Biopython's `pdb-seqres` parser. Each SEQRES
  chain is one sequence.
- `.fastq` files are read with Biopython's `FastqGeneralIterator`. Only the
  read ID and the sequence are kept. Quality strings are not used.
- Any other suffix is passed to `Bio.SeqIO.parse` as the format name.

Gzipped files are read directly. `reads.fastq.gz` is decompressed and parsed
as FASTQ.

## Handle multiple sequences per file: `tiling`

A PDB file can contain several chains, and a FASTQ file contains many reads.
`tiling` sets the layout of these sequences in the output table.

| `tiling` | Result for a cell containing a file with several sequences |
| --- | --- |
| `"bag"` (default) | One cell containing a `list` of strings. A single-sequence file remains a plain `str`. |
| `"first"` | One cell containing the first sequence. |
| `"concat"` | One cell containing all sequences joined into one string, ordered by chain ID. |
| `"features"` | One column per sequence: `<col>_0`, `<col>_1`, ... Shorter rows are padded with `None`. |
| `"samples"` | One row per sequence. Literal columns are repeated on every new row. |
| `"samples_product"` | Like `"samples"`, but with several file columns in one row it takes their cartesian product. |

For a multi-chain protein structure, use `"bag"`, `"first"` or `"concat"` to
keep the molecule in one cell. For SELEX reads, use `"samples"` to get the
one-read-per-row table that transformers such as
{class}`~pyaptamer.trafos.transform.PrimerTrimmer` expect.

```python
loader = MoleculeLoader(data={"read": [Path("round5.fastq")]}, tiling="samples")
reads = loader.to_dataframe()  # one row per read
```

With `"samples"`, a row with two or more file columns raises an error, since
there is no defined way to pair the chains of one file with the reads of
another. To get every pairing, use `"samples_product"`. To pair one protein
with every read, load the protein as a string first and place that string next
to the FASTQ file:

```python
protein = MoleculeLoader(data={"protein": [Path("1gnh.pdb")]}, tiling="first")
protein = protein.to_dataframe().loc[0, "protein"]

loader = MoleculeLoader(
    data={"aptamer": [Path("round5.fastq")], "protein": [protein]},
    tiling="samples",
)
```

## Attach metadata to files

A literal column next to a file column is repeated onto every sequence from
that file. To label SELEX rounds, use one row per FASTQ file and a `round`
column beside it:

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

## Keep or drop chain and read IDs: `indexing`

When a file expands into rows, `indexing` sets where the PDB chain IDs or
FASTQ read IDs are placed. It only applies to `"samples"` and
`"samples_product"`.

| `indexing` | Result |
| --- | --- |
| `"preserve"` (default) | Row index `<original row label>__<id>`, for example `0__A`. |
| `"new"` | A new `RangeIndex`. IDs are dropped. |
| `"keep_as_column"` | A new `RangeIndex` plus a `<col>_chain_id` column that contains the ID. |

```python
loader = MoleculeLoader(
    data={"read": [Path("round5.fastq")]},
    tiling="samples",
    indexing="keep_as_column",
)
loader.to_dataframe()  # columns: read, read_chain_id
```

## Choose a flat or two-level index: `multiindex`

With `indexing="preserve"`, `multiindex` sets the shape of the row index.

| `multiindex` | Result |
| --- | --- |
| `"flatten"` (default) | One string label per row, `<row>__<id>`. |
| `"multiindex"` | A `pandas.MultiIndex` with levels `row` and `sequence`. |
| `"auto"` | `"multiindex"` when at least one row expanded into several, `"flatten"` otherwise. |

## Remove repeated sequences: `ignore_duplicates`

`ignore_duplicates=True` removes identical sequences within each file before
tiling. The first occurrence is kept. The copy number is lost, so use this
option only when the count of each sequence does not matter.

```python
loader = MoleculeLoader(
    data={"target": [Path("1gnh.pdb")]},  # 10 identical chains
    tiling="samples",
    ignore_duplicates=True,
)
loader.to_dataframe()  # one row
```

## Use the bundled datasets

The dataset loaders in {mod}`pyaptamer.datasets`, such as
{func}`~pyaptamer.datasets.load_1gnh` and
{func}`~pyaptamer.datasets.load_sample_fastq`, return a `MoleculeLoader`. The
PDB loaders take a `tiling` argument with the default `"bag"`, so a
multi-chain structure stays in one cell. Pass `tiling="samples"` for one row
per chain, or `tiling="first"` for the first chain as a string.

```python
from pyaptamer.datasets import load_1gnh

protein = load_1gnh(tiling="first").to_dataframe()["sequence"].iloc[0]
```

## Pass a loader to a transformer

Transformers call `to_dataframe()` on the loader. The loader settings define
the table the transformer receives, so use `tiling="samples"` for any
transformer that works on one read per row.

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

- Set `tiling` by where you want the sequences of a multi-sequence file: in
  one cell (`bag`, `first`, `concat`), in columns (`features`), or in rows
  (`samples`, `samples_product`).
- For SELEX reads, use `tiling="samples"` and put round labels in a literal
  column next to the file column.
- Set `indexing="keep_as_column"` to keep read or chain IDs as data,
  `"preserve"` to keep them in the index, or `"new"` to drop them.
- Set `ignore_duplicates=True` to deduplicate within each file. Counts are
  lost.
