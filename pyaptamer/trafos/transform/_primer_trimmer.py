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

    A read fails to match the design if it does not begin with
    ``start_primer``, does not end with ``end_primer``, or its random region
    is not exactly ``variable_length`` long. A random region of the wrong
    length means the read carries an insertion, a deletion, or a truncation,
    so the aptamer candidate it holds is not one the library was built to
    produce. What happens to a read that fails to match is controlled by
    ``on_unmatched``.

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
        What to do with a read that fails to match the design.

        - "drop" : omit the read from the output. ``Xt`` holds only the
          reads that matched, indexed by the reads kept.
        - "na" : keep the read in the output, with its sequence value set to
          ``None``. ``Xt`` holds one row per input read.
        - "raise" : raise a ``ValueError`` if any read fails to match.

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
        on_unmatched: str = "drop",
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
        """Strip the constant regions, handling reads that miss the design.

        What happens to a read that fails to match the design is controlled
        by ``on_unmatched``: it is omitted (``"drop"``), kept with its value
        set to ``None`` (``"na"``), or turned into a raised exception
        (``"raise"``).

        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform. Only the first column is used.

        Returns
        -------
        Xt : pd.DataFrame
            Holds the random region of each read that matched the design.
            Under ``on_unmatched="drop"``, one row per matched read, indexed
            by the reads kept. Under ``on_unmatched="raise"``, this is only
            reached if every read matched, so it is one row per input read.
            Under ``on_unmatched="na"``, one row per input read, with
            unmatched reads holding ``None``.

        Raises
        ------
        ValueError
            If ``on_unmatched="raise"`` and at least one read fails to match
            the design.
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

        unmatched_count = len(X) - int(keep.sum())

        if self.on_unmatched == "raise" and unmatched_count:
            raise ValueError(
                f"{type(self).__name__} found {unmatched_count} of {len(X)} "
                "reads that do not fit the library design: they do not start "
                f"with {self.start_primer!r}, end with {self.end_primer!r}, and "
                f"hold a {self.variable_length} nt random region in between. "
                "Check the primers against the library design, including the "
                'orientation of end_primer, or set on_unmatched="drop" or '
                '"na" to tolerate unmatched reads.'
            )

        if unmatched_count and unmatched_count == len(X):
            if self.on_unmatched == "na":
                warnings.warn(
                    f"{type(self).__name__} set all {len(X)} reads to None: "
                    f"none of them start with {self.start_primer!r}, end with "
                    f"{self.end_primer!r}, and hold a {self.variable_length} nt "
                    "random region in between. Check the primers against the "
                    "library design, including the orientation of end_primer.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"{type(self).__name__} dropped all {len(X)} reads: none of "
                    f"them start with {self.start_primer!r}, end with "
                    f"{self.end_primer!r}, and hold a {self.variable_length} nt "
                    "random region in between. Check the primers against the "
                    "library design, including the orientation of end_primer.",
                    UserWarning,
                    stacklevel=2,
                )
        elif unmatched_count:
            if self.on_unmatched == "na":
                logger.info(
                    f"{type(self).__name__} set {unmatched_count} of {len(X)} "
                    "reads to None because they did not fit the library design."
                )
            else:
                logger.info(
                    f"{type(self).__name__} dropped {unmatched_count} of "
                    f"{len(X)} reads that did not fit the library design."
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
