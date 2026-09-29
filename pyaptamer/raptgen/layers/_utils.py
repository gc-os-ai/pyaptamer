"""Utility functions and classes for RaptGen layers"""

__author__ = ["NoorMajdoub"]
__all__ = ["NucleotideIndex", "State", "Transition", "seq_to_indices"]

from enum import IntEnum


def seq_to_indices(seq):
    """Convert a nucleotide sequence string into a list of integer indices."""
    return [int(NucleotideIndex[char]) for char in seq]


class NucleotideIndex(IntEnum):
    """Nucleotide-to-integer index mapping.

    U shares T's index so RNA and DNA sequences encode identically.
    """

    A = 0
    T = 1
    G = 2
    C = 3
    PAD = 4
    SOS = 5
    EOS = 6
    U = 1


class State(IntEnum):
    """HMM states: Match, Insert, Delete."""

    M = 0
    I = 1  # noqa: E741
    D = 2


class Transition(IntEnum):
    """HMM state-transition types."""

    M2M = 0
    M2I = 1
    M2D = 2
    I2M = 3
    I2I = 4
    D2M = 5
    D2D = 6
