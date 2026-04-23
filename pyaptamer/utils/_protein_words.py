"""Protein word extraction utilities.

Core behavior mirrors the AptaTrans vocabulary pre-processing pattern:
1) collect overlapping substrings for k in [min_k, max_k],
2) optionally keep only above-average-frequency words,
3) map surviving words to compact integer ids via ``filter_words``.
"""

from collections import Counter

from pyaptamer.utils._base import filter_words


def compute_protein_words(
    protein_sequences,
    min_k: int = 1,
    max_k: int = 3,
    apply_frequency_filter: bool = True,
):
    """Build a protein-word vocabulary from raw sequences.

    Parameters
    ----------
    protein_sequences : iterable[str]
        Protein sequences.
    min_k : int, optional, default=1
        Minimum word length.
    max_k : int, optional, default=3
        Maximum word length.
    apply_frequency_filter : bool, optional, default=True
        If True, keeps only words with frequency above the mean and renumbers
        the surviving words into consecutive integer ids.

    Returns
    -------
    dict
        If ``apply_frequency_filter`` is True, returns only the words whose
        frequency is above the average frequency, mapped to integer ids.
        The original frequencies are discarded for the filtered output.
        Otherwise, returns raw counts as ``dict[str, int]`` (word -> frequency).

    Notes
    -----
    The returned id mapping is stable with respect to insertion order of
    counted words (Python 3.7+ dict ordering), after applying the mean filter.
    """
    if min_k < 1 or max_k < min_k:
        raise ValueError("Expected 1 <= min_k <= max_k.")

    word_counts = Counter()

    for seq in protein_sequences:
        # Be permissive with upstream inputs: ignore missing entries.
        if seq is None:
            continue

        # Normalize simple formatting noise and case before counting words.
        sequence = str(seq).strip().upper()
        if not sequence:
            continue

        seq_len = len(sequence)
        for k in range(min_k, max_k + 1):
            # No k-mer can be extracted when sequence is shorter than k.
            if seq_len < k:
                continue

            # Overlapping substrings, e.g. MKLAVT -> MKL, KLA, LAV, AVT for k=3.
            for start in range(seq_len - k + 1):
                word_counts[sequence[start : start + k]] += 1

    if not apply_frequency_filter:
        return dict(word_counts)

    # filter_words expects frequency values and returns word->id mapping.
    return filter_words({word: float(freq) for word, freq in word_counts.items()})
