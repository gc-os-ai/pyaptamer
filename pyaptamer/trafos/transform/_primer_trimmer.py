"""Transformation class for SELEX data"""

from collections import Counter, defaultdict
from typing import Literal

import numpy as np
import pandas as pd

from pyaptamer import logger
from pyaptamer.data import MoleculeLoader
from pyaptamer.trafos.base import BaseTransform


def _infer_adapter(
    read_counts: Counter,
    total_reads: int,
    target_length: int,
    is_forward: bool = True,
    verbose: bool = False,
) -> str:
    """Estimate one primer from the reads, using the AptaDiff heuristic.

    Extends the primer one base at a time, taking the most common prefix
    or suffix at each length. Stops when that candidate covers less than
    half the reads, or much less than the previous length did.

    Parameters
    ----------
    read_counts : Counter
        Count of each unique read, indexed by sequence.
    total_reads : int
        Number of reads, used for the 50% cutoff.
    target_length : int
        Longest primer to try.
    is_forward : bool, default=True
        If True, use read prefixes. If False, use suffixes.
    verbose : bool, default=False
        If True, log the estimated primer.

    Returns
    -------
    est_adapter : str
        The estimated primer, or an empty string if none was found.

    References
    ----------
    [1] https://github.com/wz-create/AptaDiff

    """
    max_count = None
    est_adapter = ""

    for i in range(1, target_length):
        freq = defaultdict(int)

        for seq, count in read_counts.most_common():
            if len(seq) < i:
                continue

            sub_seq = seq[:i] if is_forward else seq[-i:]
            if len(freq) > 100 and sub_seq not in freq:
                continue

            freq[sub_seq] += count

        if not freq:
            break

        top_seq, top_count = max(freq.items(), key=lambda x: x[1])

        if max_count is not None and top_count < max_count * 0.5:
            if verbose:
                direction = "forward" if is_forward else "reverse"
                logger.info(
                    f"Estimated {direction} adapter len is {i - 1} : {est_adapter}"
                )
            break

        max_count = top_count

        if max_count < total_reads * 0.5:
            if verbose:
                logger.info("No match found covering >50% of reads.")
            break

        est_adapter = top_seq

    return est_adapter


class SELEXTransform(BaseTransform):
    """Remove the primer regions from SELEX reads.

    Keeps only the variable region of each read. Primers are either given by
    the user or estimated from the reads. Reads that do not start and end
    with the primers, or whose length differs from the target length
    by more than ``tolerance``, are set to ``pd.NA``.

    Parameters
    ----------
    method : {"explicit", "heuristic"}, default="explicit"
        With ``"explicit"``, both primers must be given. With ``"heuristic"``,
        any primer not given is estimated from the reads.
    max_len : int, optional
        Maximum length of the variable region. Longer regions are cropped from
        the centre.
    start_primer : str, optional
        The primer at the start of each read or forward_adapter.
    end_primer : str, optional
        The primer at the end of each read or reverse_adapter.
    tolerance : int, default=0
        How far a read length may differ from the target length and still be
        kept. The default keeps only reads of exactly the target length.
    verbose : bool, default=False
        If True, log the estimated primers.

    Attributes
    ----------
    start_primer_ : str
        The start primer used by ``transform``.
    end_primer_ : str
        The end primer used by ``transform``.
    target_length_ : int
        The most common read length seen during ``fit``.
    """

    def __init__(
        self,
        method: Literal["explicit", "heuristic"] = "explicit",
        max_len: int = None,
        start_primer: str = None,
        end_primer: str = None,
        tolerance: int = 0,
        verbose: bool = False,
    ):
        self.method = method
        self.max_len = max_len
        self.start_primer = start_primer
        self.end_primer = end_primer
        self.tolerance = tolerance
        self.verbose = verbose

    def _check_X(self, X):  # noqa: N802
        """Require a MoleculeLoader, then defer to the base coercion/checks."""
        if not isinstance(X, MoleculeLoader):
            raise TypeError(
                f"{type(self).__name__} accepts only a MoleculeLoader as input, "
                f"got {type(X).__name__}."
            )
        return super()._check_X(X)

    def fit(self, X, y=None):
        """Set the primers to remove.

        The target length is taken from the data by both methods. The
        heuristic method also estimates any primer that was not given.

        Parameters
        ----------
        X : MoleculeLoader
            Input data to fit the transformer. Only the first column is read.
        y : array-like, shape (n_samples,), optional
            Target values. Only used if the transformer has
            the tag ``capability:y`` set to True.

        Returns
        -------
        self : object
            Returns self.
        """
        if self.method not in ("explicit", "heuristic"):
            raise ValueError(
                f"method must be 'explicit' or 'heuristic', got {self.method}."
            )

        X = self._check_X(X)

        X = X[X.columns[0]]

        read_counts = Counter(X)
        total_reads = len(X)

        length_counts = defaultdict(int)

        for seq, count in read_counts.items():
            length_counts[len(seq)] += count

        self.target_length_ = sorted(length_counts.items(), key=lambda x: -x[1])[0][0]

        if self.method == "explicit":
            if not self.start_primer or not self.end_primer:
                raise ValueError("Both primers required for explicit method.")

            self.start_primer_ = self.start_primer
            self.end_primer_ = self.end_primer

        else:
            if self.start_primer:
                self.start_primer_ = self.start_primer
            else:
                self.start_primer_ = _infer_adapter(
                    read_counts, total_reads, self.target_length_, verbose=self.verbose
                )

            if self.end_primer:
                self.end_primer_ = self.end_primer
            else:
                self.end_primer_ = _infer_adapter(
                    read_counts,
                    total_reads,
                    self.target_length_,
                    is_forward=False,
                    verbose=self.verbose,
                )

        return self

    def _transform(self, X):
        """
        Removes the primers from each read.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform. Only the first column is used.

        Returns
        -------
        Xt : pd.DataFrame
            Transformed data. One row per read, with the same index and column
            name as the input. Each value is the variable region of the read,
            or ``pd.NA`` if the primers did not match or the read length was
            outside the tolerance.
        """
        X = X[X.columns[0]]

        fwd_len = len(self.start_primer_)
        rev_len = len(self.end_primer_)

        stop_idx = -rev_len if rev_len > 0 else None

        length_ok = (X.str.len() - self.target_length_).abs() <= self.tolerance

        valid_mask = (
            X.str.startswith(self.start_primer_)
            & X.str.endswith(self.end_primer_)
            & length_ok
        )

        X_transformed = pd.Series(
            np.where(valid_mask, X.str.slice(start=fwd_len, stop=stop_idx), pd.NA),
            index=X.index,
            name=X.name,
            dtype=object,
        )

        # Center crop if `max_len` is given
        if self.max_len is not None:
            X_transformed.loc[valid_mask] = X_transformed.loc[valid_mask].apply(
                lambda seq: seq[
                    len(seq) // 2 - self.max_len // 2 : len(seq) // 2
                    - self.max_len // 2
                    + self.max_len
                ]
                if len(seq) > self.max_len
                else seq
            )

        return X_transformed.to_frame()
