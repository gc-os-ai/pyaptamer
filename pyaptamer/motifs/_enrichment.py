"""Track sequence enrichment across SELEX rounds."""

__author__ = ["Alleny244"]
__all__ = ["EnrichmentTracker"]

from collections import Counter

import pandas as pd
from skbase.base import BaseObject

from pyaptamer import logger


class EnrichmentTracker(BaseObject):
    """Track sequence enrichment across multiple SELEX rounds.

    Computes fold-enrichment and frequency changes for sequences
    observed across sequential selection rounds, enabling identification
    of sequences that are being positively selected.

    Parameters
    ----------
    min_count : int, optional, default=2
        Minimum count threshold for a sequence to be considered
        in enrichment calculations. Sequences below this threshold
        in all rounds are filtered out.
    pseudocount : float, optional, default=1.0
        Pseudocount added to frequencies to avoid division by zero
        when computing fold-enrichment.

    Attributes
    ----------
    round_counts_ : list[Counter]
        Raw sequence counts per round after fitting.
    enrichment_ : pd.DataFrame
        Enrichment statistics computed after calling ``compute``.

    Examples
    --------
    >>> from pyaptamer.motifs import EnrichmentTracker
    >>> rounds = [
    ...     ["ACGT", "ACGT", "TGCA", "AAAA"],
    ...     ["ACGT", "ACGT", "ACGT", "TGCA"],
    ... ]
    >>> tracker = EnrichmentTracker(min_count=1)
    >>> tracker.add_round(rounds[0])
    >>> tracker.add_round(rounds[1])
    >>> result = tracker.compute()
    """

    def __init__(self, min_count: int = 2, pseudocount: float = 1.0):
        self.min_count = min_count
        self.pseudocount = pseudocount
        self.round_counts_: list[Counter] = []
        self.enrichment_: pd.DataFrame | None = None
        super().__init__()

    def add_round(self, sequences: list[str]) -> "EnrichmentTracker":
        """Add a SELEX round of sequences.

        Parameters
        ----------
        sequences : list[str]
            List of nucleotide sequences from one SELEX round.

        Returns
        -------
        self : EnrichmentTracker
        """
        self.round_counts_.append(Counter(sequences))
        self.enrichment_ = None
        return self

    def compute(self) -> pd.DataFrame:
        """Compute enrichment statistics across all added rounds.

        Returns
        -------
        enrichment : pd.DataFrame
            DataFrame with columns for each round's frequency and
            fold-enrichment between consecutive rounds. Index is
            the sequence string.

        Raises
        ------
        ValueError
            If fewer than 2 rounds have been added.
        """
        if len(self.round_counts_) < 2:
            raise ValueError(
                "At least 2 rounds are required to compute enrichment. "
                f"Got {len(self.round_counts_)}."
            )

        all_seqs = set()
        for counts in self.round_counts_:
            all_seqs.update(counts.keys())

        all_seqs = {
            seq
            for seq in all_seqs
            if any(c[seq] >= self.min_count for c in self.round_counts_)
        }

        data = {}
        for i, counts in enumerate(self.round_counts_):
            total = sum(counts.values())
            freqs = {seq: counts[seq] / total for seq in all_seqs}
            data[f"freq_round_{i}"] = freqs

        df = pd.DataFrame(data)

        for i in range(1, len(self.round_counts_)):
            prev_col = f"freq_round_{i - 1}"
            curr_col = f"freq_round_{i}"
            df[f"fold_enrichment_{i}"] = (
                df[curr_col] + self.pseudocount / sum(self.round_counts_[i].values())
            ) / (
                df[prev_col]
                + self.pseudocount / sum(self.round_counts_[i - 1].values())
            )

        df = df.sort_values(df.columns[-1], ascending=False)

        self.enrichment_ = df
        logger.info(
            "Computed enrichment for %d sequences across %d rounds.",
            len(df),
            len(self.round_counts_),
        )
        return df

    def top_enriched(self, n: int = 10, round_idx: int = -1) -> pd.DataFrame:
        """Return the top-n enriched sequences.

        Parameters
        ----------
        n : int, optional, default=10
            Number of top sequences to return.
        round_idx : int, optional, default=-1
            Which fold-enrichment column to rank by. -1 uses the last.

        Returns
        -------
        top : pd.DataFrame
            Top-n rows from the enrichment DataFrame.
        """
        if self.enrichment_ is None:
            self.compute()

        fold_cols = [c for c in self.enrichment_.columns if c.startswith("fold_")]
        col = fold_cols[round_idx]
        return self.enrichment_.nlargest(n, col)
