"""Conversion of DNA sequences to RNA."""

__author__ = ["siddharth7113"]
__all__ = ["DNAToRNA"]

import pandas as pd

from pyaptamer.transformations.base import BaseTransform


class DNAToRNA(BaseTransform):
    """Replace T with U in nucleotide sequences.

    Each sequence is upper-cased and every ``T`` becomes ``U``. Other letters
    are unchanged, so an RNA sequence passes through as it is. Use it ahead
    of an encoder with ``alphabet="ACGU"`` when a table mixes DNA and RNA
    spellings.

    Input is a one-column DataFrame or ``MoleculeLoader`` of strings. The
    output has the same shape, column name and index. A missing cell is
    returned as it is.

    Examples
    --------
    >>> import pandas as pd
    >>> from pyaptamer.transformations.transform import DNAToRNA
    >>> X = pd.DataFrame({"aptamer": ["ACGT", "acgu"]})
    >>> DNAToRNA().fit_transform(X)["aptamer"].tolist()
    ['ACGU', 'ACGU']

    Reads from a FASTQ file come in through a ``MoleculeLoader``. The bundled
    sample file is loaded with ``tiling="samples"``, one read per row:

    >>> from pyaptamer.datasets import load_sample_fastq
    >>> Xt = DNAToRNA().fit_transform(load_sample_fastq())
    >>> "T" in "".join(Xt["sequence"])
    False

    In a table with several columns, name the column with ``ApplyToCols``:

    >>> from pyaptamer.data import MoleculeLoader
    >>> from pyaptamer.transformations.compose import ApplyToCols
    >>> X = MoleculeLoader(data={"aptamer": ["ACGT"], "protein": ["MKT"]})
    >>> Xt = ApplyToCols(DNAToRNA(), cols="aptamer").fit_transform(X)
    >>> Xt["aptamer"].tolist()
    ['ACGU']
    """

    _tags = {
        "authors": ["siddharth7113"],
        "maintainers": ["siddharth7113"],
        "output_type": "str",
        "property:fit_is_empty": True,
        "property:elementwise": True,
        "capability:multivariate": False,
    }

    def _transform_element(self, X):
        """Upper-case one sequence and replace T with U."""
        if pd.isna(X):
            return X
        return X.upper().replace("T", "U")
