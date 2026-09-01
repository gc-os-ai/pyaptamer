"""Sequence statistics utilities for aptamer design and screening."""

__author__ = ["Vaishnav88sk"]
__all__ = [
    "gc_content",
    "nucleotide_composition",
    "sequence_summary",
]

from collections import Counter

import numpy as np
import pandas as pd


def gc_content(sequence: str) -> float:
    """Compute the GC content of a DNA or RNA sequence.

    GC content is the proportion of bases in the sequence that are either
    guanine (G) or cytosine (C). This is a key metric in aptamer design as
    it affects secondary structure stability, melting temperature, and
    binding affinity.

    Parameters
    ----------
    sequence : str
        A DNA (containing A, C, G, T) or RNA (containing A, C, G, U) sequence.
        Case-insensitive.

    Returns
    -------
    float
        GC content as a fraction between 0.0 and 1.0.
        Returns 0.0 for empty sequences.

    Raises
    ------
    TypeError
        If ``sequence`` is not a string.

    Examples
    --------
    >>> from pyaptamer.utils import gc_content
    >>> gc_content("ACGT")
    0.5
    >>> gc_content("GGCC")
    1.0
    >>> gc_content("AAAA")
    0.0
    >>> gc_content("ACGU")
    0.5
    """
    if not isinstance(sequence, str):
        raise TypeError(
            f"`sequence` must be a string, got {type(sequence).__name__}."
        )

    if len(sequence) == 0:
        return 0.0

    sequence = sequence.upper()
    gc_count = sequence.count("G") + sequence.count("C")
    return gc_count / len(sequence)


def nucleotide_composition(sequence: str) -> dict[str, dict[str, float]]:
    """Compute nucleotide composition of a DNA or RNA sequence.

    Returns the count and frequency (proportion) of each nucleotide
    found in the sequence. Unknown characters are grouped under an
    ``"other"`` key.

    Parameters
    ----------
    sequence : str
        A DNA (A, C, G, T) or RNA (A, C, G, U) sequence. Case-insensitive.

    Returns
    -------
    dict[str, dict[str, float]]
        A dictionary mapping each nucleotide (and ``"other"`` for unknowns)
        to a sub-dictionary with keys ``"count"`` (int) and ``"frequency"``
        (float between 0 and 1).

    Raises
    ------
    TypeError
        If ``sequence`` is not a string.

    Examples
    --------
    >>> from pyaptamer.utils import nucleotide_composition
    >>> comp = nucleotide_composition("AACG")
    >>> comp["A"]["count"]
    2
    >>> comp["A"]["frequency"]
    0.5
    """
    if not isinstance(sequence, str):
        raise TypeError(
            f"`sequence` must be a string, got {type(sequence).__name__}."
        )

    sequence = sequence.upper()
    valid_bases = set("ACGTU")
    total = len(sequence) if len(sequence) > 0 else 1

    counts = Counter(sequence)
    composition = {}

    for base in sorted(valid_bases):
        c = counts.get(base, 0)
        if c > 0:
            composition[base] = {
                "count": c,
                "frequency": c / total,
            }

    # group unknowns
    other_count = sum(v for k, v in counts.items() if k not in valid_bases)
    if other_count > 0:
        composition["other"] = {
            "count": other_count,
            "frequency": other_count / total,
        }

    return composition


def sequence_summary(sequences: list[str]) -> pd.DataFrame:
    """Compute summary statistics for a batch of DNA/RNA sequences.

    Generates a DataFrame with per-sequence statistics including length,
    GC content, and nucleotide counts. This is useful for quickly profiling
    a set of candidate aptamers.

    Parameters
    ----------
    sequences : list[str]
        A list of DNA or RNA sequence strings.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per sequence and the following columns:

        - ``"sequence"``: the original sequence string
        - ``"length"``: length of the sequence (int)
        - ``"gc_content"``: GC content as a float between 0 and 1
        - ``"A"``, ``"C"``, ``"G"``, ``"T"``, ``"U"``: nucleotide counts (int)

    Raises
    ------
    TypeError
        If ``sequences`` is not a list.
    ValueError
        If ``sequences`` is empty.

    Examples
    --------
    >>> from pyaptamer.utils import sequence_summary
    >>> df = sequence_summary(["ACGT", "GGCC", "AAUU"])
    >>> df["gc_content"].tolist()
    [0.5, 1.0, 0.0]
    >>> df["length"].tolist()
    [4, 4, 4]
    """
    if not isinstance(sequences, list):
        raise TypeError(
            f"`sequences` must be a list, got {type(sequences).__name__}."
        )

    if len(sequences) == 0:
        raise ValueError("`sequences` must not be empty.")

    records = []
    for seq in sequences:
        upper_seq = seq.upper()
        records.append(
            {
                "sequence": seq,
                "length": len(seq),
                "gc_content": gc_content(seq),
                "A": upper_seq.count("A"),
                "C": upper_seq.count("C"),
                "G": upper_seq.count("G"),
                "T": upper_seq.count("T"),
                "U": upper_seq.count("U"),
            }
        )

    return pd.DataFrame(records)
