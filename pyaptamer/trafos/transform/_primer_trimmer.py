"""Removal of the constant primer regions from SELEX reads."""

__author__ = ["aditi-dsi", "siddharth7113"]
__all__ = ["PrimerTrimmer"]

import warnings

import pandas as pd

from pyaptamer import logger
from pyaptamer.trafos.base import BaseTransform


class PrimerTrimmer(BaseTransform):
    """Remove the constant primer regions from SELEX reads.

    A SELEX library is built to a fixed design. Every read should be a
    constant 5' region, then a random region of a known length, then a
    constant 3' region::

        TAATACG...CAGAAG  NNNNNNNNNN...NNN  TATGTG...GATCCTC
        |-- start_primer --|-- variable_length --|-- end_primer --|

    Only the random region carries the candidate aptamer. The constant regions
    are the primer binding sites used to amplify the library and are identical
    in every read. This transformer strips them and returns the random region.

    A read is dropped if it does not begin with ``start_primer``, does not end
    with ``end_primer``, or its random region is not exactly
    ``variable_length`` long. A random region of the wrong length means the
    read carries an insertion, a deletion, or a truncation, so the aptamer
    candidate it holds is not one the library was built to produce.

    Parameters
    ----------
    start_primer : str
        The constant 5' region, matched as a literal prefix of each read.
    end_primer : str
        The constant 3' region, matched as a literal suffix of each read,
        given in the orientation in which it appears in the read.
    variable_length : int
        The designed length of the random region, for example 40 for an N40
        library.

    Notes
    -----
    Estimating the constant regions or the library design from the reads is
    not supported. See the AptaDiff project for prior art on inferring them.

    Examples
    --------
    >>> from pyaptamer.datasets import load_sample_fastq
    >>> from pyaptamer.trafos.transform import PrimerTrimmer
    >>>
    >>> transform = PrimerTrimmer(
    ...     start_primer="TAATACGACTCACTATAGGGAGAACTTCGACCAGAAG",
    ...     end_primer="TATGTGCGCATACATGGATCCTC",
    ...     variable_length=40,
    ... )
    >>> Xt = transform.fit_transform(load_sample_fastq())
    >>> len(Xt)
    6
    >>> Xt["sequence"].str.len().unique().tolist()
    [40]
    """

    _tags = {
        "authors": ["aditi-dsi", "siddharth7113"],
        "maintainers": ["aditi-dsi"],
        "property:fit_is_empty": True,
        "capability:multivariate": False,
    }

    def __init__(
        self,
        start_primer: str,
        end_primer: str,
        variable_length: int,
    ):
        self.start_primer = start_primer
        self.end_primer = end_primer
        self.variable_length = variable_length

        super().__init__()

    def _check_reads(self, X):
        """Check that the first column of X holds one str read per row.

        Parameters
        ----------
        X : pd.DataFrame
            Input data. Only the first column is read.

        Raises
        ------
        TypeError
            If the first column of X does not hold str reads.
        """
        kind = pd.api.types.infer_dtype(X.iloc[:, 0], skipna=False)

        if kind != "string":
            raise TypeError(
                f"{type(self).__name__} expects one str read per row, but the "
                f"first column of X is {kind!r}. If it holds several reads per "
                "cell, the MoleculeLoader was built with the default "
                'tiling="bag" - use tiling="samples" so each read is one row.'
            )

    def _validate_params(self):
        """Check the constructor arguments, raising ValueError if unusable."""
        if not self.start_primer or not self.end_primer:
            raise ValueError(
                "start_primer and end_primer are both required and must be "
                f"non-empty, got {self.start_primer!r} and {self.end_primer!r}."
            )

        if self.variable_length <= 0:
            raise ValueError(
                f"variable_length must be positive, got {self.variable_length}."
            )

    def _transform(self, X):
        """Strip the constant regions, dropping reads that miss the design.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform. Only the first column is used.

        Returns
        -------
        Xt : pd.DataFrame
            One row per usable read, holding its random region, with the column
            name of the input and the index entries of the reads kept.
        """
        self._validate_params()
        self._check_reads(X)

        column = X.columns[0]
        reads = X[column]

        start_len = len(self.start_primer)
        read_length = start_len + self.variable_length + len(self.end_primer)

        keep = (
            (reads.str.len() == read_length)
            & reads.str.startswith(self.start_primer)
            & reads.str.endswith(self.end_primer)
        )

        dropped = len(X) - int(keep.sum())
        if dropped and dropped == len(X):
            warnings.warn(
                f"{type(self).__name__} dropped all {len(X)} reads: none of them "
                f"start with {self.start_primer!r}, end with {self.end_primer!r}, "
                f"and hold a {self.variable_length} nt random region in between. "
                "Check the primers against the library design, including the "
                "orientation of end_primer.",
                UserWarning,
                stacklevel=2,
            )
        elif dropped:
            logger.info(
                f"{type(self).__name__} dropped {dropped} of {len(X)} reads "
                "that did not fit the library design."
            )

        Xt = X.loc[keep, [column]].copy()
        Xt[column] = Xt[column].str.slice(start_len, start_len + self.variable_length)

        return Xt
