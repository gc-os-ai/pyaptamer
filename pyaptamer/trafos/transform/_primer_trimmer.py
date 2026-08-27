"""Removal of the constant primer regions from SELEX reads."""

__author__ = ["aditi-dsi", "siddharth7113"]
__all__ = ["PrimerTrimmer"]

import warnings
from typing import Literal

import pandas as pd

from pyaptamer import logger
from pyaptamer.trafos.base import BaseTransform


class PrimerTrimmer(BaseTransform):
    """Remove the constant primer regions from SELEX reads.

    A SELEX library is built with a fixed design. Every read should have a
    constant 5' region, then a random region of a known length, then a
    constant 3' region::

        TAATACG...CAGAAG     NNNNNNNNNN...NNN     TATGTG...GATCCTC
        |-- start_primer --|-- variable_length --|-- end_primer --|

    Only the random region is considered a candidate aptamer. The constant
    regions are the primer binding sites used to amplify the SELEX library and
    are identical in every read. This transformer strips them and returns the
    random region.

    A read is considered corrupted if it fails any part of that design: it does
    not begin with ``start_primer``, does not end with ``end_primer``, or its
    random region is not exactly ``variable_length`` long. A random region of
    the wrong length means the read carries an insertion, a deletion, or a
    truncation. ``on_unmatched`` decides what happens to a corrupted read.

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
    on_unmatched : {"drop", "na", "raise"}, default="drop"
        What to do with a corrupted read.

        - "drop" : leave it out, so ``Xt`` has one row per surviving read.
        - "na" : keep its row but mark the sequence as missing, so ``Xt`` has
          one row per input read. Find those rows with ``isna``: the exact
          missing value differs between pandas versions, so comparing against
          ``None`` is not reliable.
        - "raise" : raise a ``ValueError`` saying how many reads are corrupted.

    Notes
    -----
    Both primers are matched exactly, and the random region length must be
    given. Approximate matching, which would tolerate sequencing errors inside
    a primer, and inferring the random region length, which would allow
    libraries of mixed design, are not supported.

    Examples
    --------
    Corrupted reads are dropped, leaving 6 of the 10 sample reads:

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

    With ``on_unmatched="na"`` every row is kept, and the corrupted ones
    are marked as missing:

    >>> transform = PrimerTrimmer(
    ...     start_primer="TAATACGACTCACTATAGGGAGAACTTCGACCAGAAG",
    ...     end_primer="TATGTGCGCATACATGGATCCTC",
    ...     variable_length=40,
    ...     on_unmatched="na",
    ... )
    >>> Xt = transform.fit_transform(load_sample_fastq())
    >>> len(Xt)
    10
    >>> int(Xt["sequence"].isna().sum())
    4
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
        on_unmatched: Literal["drop", "na", "raise"] = "drop",
    ):
        self.start_primer = start_primer
        self.end_primer = end_primer
        self.variable_length = variable_length
        self.on_unmatched = on_unmatched

        super().__init__()

    def _check_reads(self, X):
        """Check that the first column of X holds one str read per row.

        Missing cells are ignored here. They cannot match the library design,
        so ``_transform`` handles them as unmatched reads.

        Parameters
        ----------
        X : pd.DataFrame
            Input data. Only the first column is read.

        Raises
        ------
        TypeError
            If the first column of X does not hold str reads.
        """
        kind = pd.api.types.infer_dtype(X.iloc[:, 0], skipna=True)

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

        valid_on_unmatched = {"drop", "na", "raise"}
        if self.on_unmatched not in valid_on_unmatched:
            raise ValueError(
                f"on_unmatched must be one of {sorted(valid_on_unmatched)}, "
                f"got {self.on_unmatched!r}."
            )

    def _transform(self, X):
        """Strip the constant regions from every read that fits the design.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform. Only the first column is used.

        Returns
        -------
        Xt : pd.DataFrame
            The random region of each read, same index and column name as
            input are returned. ``on_unmatched`` decides the shape of output
            dataframe.

        Raises
        ------
        ValueError
            If ``on_unmatched="raise"`` and any read is corrupted.
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

        corrupted = len(X) - int(keep.sum())
        # a corrupted read fails at least one part of the design, not all of it
        design = (
            f"start_primer + {self.variable_length} nt + end_primer, "
            f"{read_length} nt in total"
        )

        if self.on_unmatched == "raise" and corrupted:
            raise ValueError(
                f"{corrupted} of {len(X)} reads do not match the library design "
                f"({design}). Check both primers, including the orientation of "
                'end_primer, or use on_unmatched="drop" or "na".'
            )

        if corrupted:
            verb = "marked" if self.on_unmatched == "na" else "dropped"
            tail = " as missing" if self.on_unmatched == "na" else ""

            if corrupted == len(X):
                warnings.warn(
                    f"{type(self).__name__} {verb} all {len(X)} reads{tail}: none "
                    f"match the library design ({design}). Check both primers, "
                    "including the orientation of end_primer.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                logger.info(
                    f"{type(self).__name__} {verb} {corrupted} of {len(X)} "
                    f"reads{tail} that do not match the library design."
                )

        if self.on_unmatched == "na":
            Xt = X[[column]].copy()
            Xt.loc[keep, column] = Xt.loc[keep, column].str.slice(
                start_len, start_len + self.variable_length
            )
            Xt.loc[~keep, column] = None
        else:
            Xt = X.loc[keep, [column]].copy()
            Xt[column] = Xt[column].str.slice(
                start_len, start_len + self.variable_length
            )

        return Xt
