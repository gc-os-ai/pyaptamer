__author__ = "satvshr"
__all__ = ["AnyToAASeq"]

import os

import pandas as pd
from Bio import SeqIO

from pyaptamer.trafos.base import BaseTransform


class AnyToAASeq(BaseTransform):
    """Transformer that converts any supported sequence file format
    into amino acid sequences represented as strings.

    This transformer relies on Biopython's :mod:`Bio.SeqIO` module to parse
    biological sequence files of arbitrary formats and extract only the
    sequence information. It supports all formats recognized by ``SeqIO.parse``.

    Supported Formats
    -----------------
    The file formats supported by :func:`Bio.SeqIO.parse` are mentioned below:

    - https://biopython.org/docs/latest/api/Bio.SeqIO.html#file-formats
    - https://biopython.org/wiki/SeqIO

    Attributes
    ----------
    format : str
        The file format to be parsed. Must be a valid format supported by Biopython's
        :func:`Bio.SeqIO.parse`.
    """

    _tags = {
        "output_type": "string",
        "property:fit_is_empty": True,
    }

    def __init__(self, format: str):
        """
        Parameters
        ----------
        format : str
            File format string understood by :func:`Bio.SeqIO.parse`
            (e.g., ``"genbank"``, ``"embl"``, ``"fasta"``).
        """
        super().__init__()
        self.format = format

    def _check_X(self, X):  # noqa: N802
        """Check X is a valid file path."""
        if not os.path.exists(X):
            raise FileNotFoundError(f"Input file not found: {X}")
        return X

    def _transform(self, X):
        """Convert an input sequence file into amino acid sequences.

        Parameters
        ----------
        X : str
            Path to a valid sequence file readable by ``SeqIO.parse``.

        Returns
        -------
        pandas.DataFrame
            DataFrame with a single column ``"sequence"``, where each row
            corresponds to one parsed sequence as a string.

        Raises
        ------
        ValueError
            If the specified format is not supported or parsing fails.
        """
        file_path = self._check_X(X)
        records = [
            str(record.seq) for record in SeqIO.parse(file_path, format=self.format)
        ]
        return pd.DataFrame(records, columns=["sequence"])
