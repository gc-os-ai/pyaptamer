# Encode sequences as features

The transformers in {mod}`pyaptamer.trafos.encode` turn a column of sequence
strings into a numeric table with one row per sequence. They take a one-column
`pandas.DataFrame` or a {class}`~pyaptamer.data.MoleculeLoader` and return a
`pandas.DataFrame` with the same index. Columns are numbered by position.

| Transformer | Input | Output |
| --- | --- | --- |
| {class}`~pyaptamer.trafos.encode.PSeAAC` | protein sequences | pseudo amino acid composition, 350 values by default |
| {class}`~pyaptamer.trafos.encode.KMerFrequencies` | nucleotide sequences | frequency of every k-mer up to length `k`, 340 values by default |
| {class}`~pyaptamer.trafos.encode.GreedyEncoder` | any sequences | one token index per position |

## Pseudo amino acid composition of proteins

{class}`~pyaptamer.trafos.encode.PSeAAC` computes the pseudo amino acid
composition of Chou. For every group of physicochemical properties the vector
holds the 20 amino acid frequencies, in the order `ACDEFGHIKLMNPQRSTVWY`,
followed by `lambda_val` sequence-order correlation factors. Sequences must
be longer than `lambda_val`, which is 30 by default.

From amino acid strings:

```python
import pandas as pd
from pyaptamer.trafos.encode import PSeAAC

X = pd.DataFrame(
    {
        "protein": [
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL",
        ]
    }
)

features = PSeAAC().fit_transform(X)
features
```

```text
     0    1      2      3      4    ...    345    346    347   348   349
0  0.021  0.0  0.018  0.017  0.007  ...  0.019  0.021  0.021  0.02  0.02

[1 rows x 350 columns]
```

Columns 0 to 19 are the normalized frequencies of A, C, D, E, F and so on.
Columns 20 to 49 are the 30 correlation factors of the first property group.
The same 50 columns repeat for each of the 7 groups.

```python
features.iloc[:, 20:24]
```

```text
      20     21     22     23
0  0.026  0.025  0.026  0.025
```

From a PDB structure, through a loader. The dataset loaders return a
`MoleculeLoader` with one row per chain, and the index carries the chain
label:

```python
from pyaptamer.datasets import load_1gnh
from pyaptamer.trafos.encode import PSeAAC

features = PSeAAC().fit_transform(load_1gnh())
features.iloc[:3, :6]
```

```text
          0      1      2      3      4      5
0__A  0.011  0.002  0.011  0.017  0.017  0.021
0__B  0.011  0.002  0.011  0.017  0.017  0.021
0__C  0.011  0.002  0.011  0.017  0.017  0.021
```

The three chains of 1GNH have the same sequence, so their rows are equal. Any
`MoleculeLoader` works the same way, so a column of PDB or FASTA paths becomes
a feature table in one call. See {doc}`molecule_loader` for the loader
options.

Three arguments choose the properties and how they are grouped:

- `prop_indices` selects properties by 0-based index out of the 21 available.
  `None` uses all of them.
- `group_props` groups the selected properties into consecutive chunks of that
  size. `None` means chunks of 3.
- `custom_groups` gives the groups explicitly as lists of indices into the
  selected properties and overrides `group_props`.

```python
PSeAAC(prop_indices=[0, 1, 2, 3, 4, 5], group_props=2)
PSeAAC(custom_groups=[[0, 1], [2, 3], [4, 5], [6, 7]])
PSeAAC(lambda_val=20, weight=0.1)
```

AptaNet uses the defaults: all 21 properties in 7 groups of 3, `lambda_val=30`
and `weight=0.05`.

Lowercase letters are uppercased. A letter that is not one of the 20 standard
amino acids is replaced with `N` and a warning is issued.

## K-mer frequencies of nucleotides

{class}`~pyaptamer.trafos.encode.KMerFrequencies` counts every k-mer of length
1 to `k` over an alphabet and divides by the total count, so each row sums to
one. K-mers are ordered by length, then alphabetically, so with the default
alphabet the columns are `A, C, G, T, AA, AC, AG, AT, CA, ...`.

```python
import pandas as pd
from pyaptamer.trafos.encode import KMerFrequencies

X = pd.DataFrame({"aptamer": ["AGCTTAGCGTACAGCTTAAAAGGGTTTCCCC", "GGGTTTCCCCTGCCCGCGTAC"]})

KMerFrequencies(k=1).fit_transform(X)
```

```text
          0         1         2         3
0  0.258065  0.258065  0.225806  0.258065
1  0.047619  0.428571  0.285714  0.238095
```

With `k=1` the four columns are the base frequencies of A, C, G and T. The
default `k=4` adds the 16 dimers, 64 trimers and 256 tetramers:

```python
KMerFrequencies(k=4).fit_transform(X).iloc[:, :6]
```

```text
          0         1         2         3         4         5
0  0.064516  0.088710  0.056452  0.064516  0.024194  0.008065
1  0.011905  0.142857  0.071429  0.059524  0.000000  0.035714
```

Substrings with a letter outside the alphabet are not counted. The default
alphabet is DNA, so an RNA sequence loses every k-mer containing `U`:

```python
X = pd.DataFrame({"aptamer": ["GGGAGGACGAAGACGACUCGAGACAGGCUAGGGAGGGA"]})

KMerFrequencies(k=1).fit_transform(X)
```

```text
          0         1    2    3
0  0.333333  0.166667  0.5  0.0
```

Pass `alphabet="ACGU"` for RNA. The last column is now `U`:

```python
KMerFrequencies(k=1, alphabet="ACGU").fit_transform(X)
```

```text
          0         1         2         3
0  0.315789  0.157895  0.473684  0.052632
```

## Encode several columns at once

Each encoder works on one column. To encode a table with an aptamer column and
a protein column, route each column to its encoder with a `ColumnTransformer`.
This is how {class}`~pyaptamer.aptanet.AptaNetPipeline` builds its features:

```python
from sklearn.compose import ColumnTransformer

from pyaptamer.data import MoleculeLoader
from pyaptamer.trafos.encode import KMerFrequencies, PSeAAC

X = MoleculeLoader(
    data={
        "aptamer": ["AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"],
        "protein": ["ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"],
    }
)

features = ColumnTransformer(
    [
        ("aptamer", KMerFrequencies(k=4), ["aptamer"]),
        ("protein", PSeAAC(), ["protein"]),
    ]
)

Xt = features.fit_transform(X.to_dataframe())
Xt.shape, Xt[0, :4], Xt[0, 340:344]
```

```text
((1, 690), array([0.054, 0.095, 0.06 , 0.06 ]), array([0.012, 0.012, 0.012, 0.012]))
```

`ColumnTransformer` takes a `DataFrame`, so call `to_dataframe()` on a loader
first. Its output is a `numpy` array: the 340 k-mer values, then the 350
PSeAAC values.
