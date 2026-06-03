"""Motif discovery from aptamer sequence pools."""

__author__ = ["Alleny244"]
__all__ = ["MotifFinder"]

from collections import Counter

import numpy as np
import pandas as pd
from skbase.base import BaseEstimator

from pyaptamer import logger


class MotifFinder(BaseEstimator):
    """Discover conserved sequence motifs in aptamer pools.

    Identifies over-represented k-mers in a pool of sequences using
    frequency analysis and information-theoretic scoring. Builds a
    position weight matrix (PWM) for each discovered motif.

    Parameters
    ----------
    k : int, optional, default=6
        Length of k-mers to scan for.
    top_n : int, optional, default=10
        Number of top over-represented k-mers to report.
    background : dict[str, float] or None, optional, default=None
        Background nucleotide frequencies. If None, assumes uniform
        distribution (0.25 each for A, C, G, T/U).
    alphabet : str, optional, default="DNA"
        Sequence alphabet. One of "DNA" or "RNA".

    Attributes
    ----------
    kmer_counts_ : Counter
        Raw k-mer counts from fitted sequences.
    kmer_scores_ : pd.DataFrame
        K-mer enrichment scores after fitting.
    motifs_ : list[dict]
        Discovered motifs with PWM and consensus information.
    n_sequences_ : int
        Number of sequences used for fitting.

    References
    ----------
    .. [1] Bailey, Timothy L., and Charles Elkan. "Fitting a mixture model by
    expectation maximization to discover motifs in biopolymers." Proceedings
    of the Second International Conference on Intelligent Systems for
    Molecular Biology. Vol. 2. AAAI Press, 1994.

    Examples
    --------
    >>> from pyaptamer.motifs import MotifFinder
    >>> sequences = ["ACGTACGTAA", "TACGTACGTT", "GACGTACGTC"]
    >>> finder = MotifFinder(k=4, top_n=3)
    >>> finder.fit(sequences)
    MotifFinder(k=4, top_n=3)
    >>> finder.motifs_[0]["consensus"]
    'ACGT'
    """

    _tags = {
        "object_type": "motif_finder",
    }

    ALPHABETS = {
        "DNA": "ACGT",
        "RNA": "ACGU",
    }

    def __init__(
        self,
        k: int = 6,
        top_n: int = 10,
        background: dict[str, float] | None = None,
        alphabet: str = "DNA",
    ):
        self.k = k
        self.top_n = top_n
        self.background = background
        self.alphabet = alphabet
        super().__init__()

    def fit(self, X, y=None):
        """Fit the motif finder to a collection of sequences.

        Parameters
        ----------
        X : list[str] or pd.DataFrame
            Input sequences. If a DataFrame, uses the first column.
        y : ignored

        Returns
        -------
        self : MotifFinder
        """
        sequences = self._to_sequence_list(X)
        self.n_sequences_ = len(sequences)

        if self.alphabet not in self.ALPHABETS:
            raise ValueError(
                f"Unknown alphabet '{self.alphabet}'. "
                f"Must be one of {list(self.ALPHABETS.keys())}."
            )

        chars = self.ALPHABETS[self.alphabet]
        bg = self.background or {c: 1.0 / len(chars) for c in chars}

        self.kmer_counts_ = self._count_kmers(sequences, self.k)
        self.kmer_scores_ = self._score_kmers(self.kmer_counts_, bg, sequences)
        self.motifs_ = self._build_motifs(sequences, chars)

        logger.info(
            "Discovered %d motifs from %d sequences (k=%d).",
            len(self.motifs_),
            self.n_sequences_,
            self.k,
        )
        return self

    def transform(self, X):
        """Encode sequences by motif occurrence counts.

        Parameters
        ----------
        X : list[str] or pd.DataFrame
            Input sequences.

        Returns
        -------
        features : pd.DataFrame
            DataFrame with one column per motif, values are occurrence
            counts of that motif in each sequence.
        """
        sequences = self._to_sequence_list(X)
        motif_seqs = [m["consensus"] for m in self.motifs_]

        data = {}
        for motif in motif_seqs:
            data[f"motif_{motif}"] = [
                self._count_occurrences(seq, motif) for seq in sequences
            ]
        return pd.DataFrame(data)

    def fit_transform(self, X, y=None):
        """Fit and transform in one step.

        Parameters
        ----------
        X : list[str] or pd.DataFrame
            Input sequences.
        y : ignored

        Returns
        -------
        features : pd.DataFrame
        """
        return self.fit(X, y).transform(X)

    def get_pwm(self, motif_idx: int = 0) -> pd.DataFrame:
        """Get the position weight matrix for a discovered motif.

        Parameters
        ----------
        motif_idx : int, optional, default=0
            Index of the motif in ``motifs_``.

        Returns
        -------
        pwm : pd.DataFrame
            Position weight matrix with positions as rows and
            nucleotides as columns.
        """
        return self.motifs_[motif_idx]["pwm"]

    def _to_sequence_list(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].tolist()
        return list(X)

    @staticmethod
    def _count_kmers(sequences, k):
        counts = Counter()
        for seq in sequences:
            seq_upper = seq.upper()
            for i in range(len(seq_upper) - k + 1):
                counts[seq_upper[i : i + k]] += 1
        return counts

    @staticmethod
    def _count_occurrences(sequence, motif):
        count = 0
        seq_upper = sequence.upper()
        motif_upper = motif.upper()
        start = 0
        while True:
            idx = seq_upper.find(motif_upper, start)
            if idx == -1:
                break
            count += 1
            start = idx + 1
        return count

    def _score_kmers(self, kmer_counts, background, sequences):
        total_kmers = sum(kmer_counts.values())

        records = []
        for kmer, count in kmer_counts.items():
            observed_freq = count / total_kmers
            expected_freq = 1.0
            for c in kmer:
                expected_freq *= background.get(c.upper(), 0.25)

            if expected_freq > 0:
                score = np.log2(observed_freq / expected_freq)
            else:
                score = 0.0

            records.append(
                {
                    "kmer": kmer,
                    "count": count,
                    "frequency": observed_freq,
                    "expected_frequency": expected_freq,
                    "enrichment_score": score,
                }
            )

        df = pd.DataFrame(records).set_index("kmer")
        return df.sort_values("enrichment_score", ascending=False)

    def _build_motifs(self, sequences, chars):
        top_kmers = self.kmer_scores_.head(self.top_n).index.tolist()
        motifs = []

        for kmer in top_kmers:
            windows = []
            for seq in sequences:
                seq_upper = seq.upper()
                start = 0
                while True:
                    idx = seq_upper.find(kmer, start)
                    if idx == -1:
                        break
                    windows.append(seq_upper[idx : idx + self.k])
                    start = idx + 1

            if not windows:
                continue

            pwm_data = {c: [] for c in chars}
            for pos in range(self.k):
                col_counts = Counter(w[pos] for w in windows)
                total = len(windows)
                for c in chars:
                    pwm_data[c].append(col_counts.get(c, 0) / total)

            pwm = pd.DataFrame(pwm_data, index=[f"pos_{i}" for i in range(self.k)])

            consensus = ""
            for pos in range(self.k):
                best = max(chars, key=lambda c: pwm_data[c][pos])
                consensus += best

            info_content = 0.0
            for pos in range(self.k):
                for c in chars:
                    p = pwm_data[c][pos]
                    if p > 0:
                        info_content += p * np.log2(p / (1.0 / len(chars)))

            motifs.append(
                {
                    "consensus": consensus,
                    "pwm": pwm,
                    "info_content": info_content,
                    "n_occurrences": len(windows),
                    "score": self.kmer_scores_.loc[kmer, "enrichment_score"],
                }
            )

        return motifs
