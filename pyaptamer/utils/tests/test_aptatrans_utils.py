"""Tests for the generic sequence-to-vector utility.

Targets the missing branches in ``pyaptamer/utils/_aptatrans_utils.py``
(issue #725):
  * lines 88-91  - splitting a sequence into chunks at ``seq_max_len``
  * line 98      - skipping a primary character with no vocabulary match
  * line 119     - returning empty (zero) arrays when nothing tokenizes
"""

__author__ = ["gandzekas"]

import numpy as np
import pytest

from pyaptamer.utils._aptatrans_utils import seq2vec


def test_seq2vec_splits_at_seq_max_len():
    """A sequence longer than ``seq_max_len`` is split into chunks (lines 88-91)."""
    # vocabulary: single "A" and digram "AA" so "AAAAA" tokenizes greedily
    words = {"A": 1, "AA": 2}
    sequences = (["AAAAA"], ["HHHHH"])
    primary, secondary = seq2vec(sequences, words, seq_max_len=2)

    # first chunk is exactly seq_max_len and holds the split boundary
    assert primary.shape == (2, 2)
    assert secondary.shape == (2, 2)
    # first row is the full first chunk (digrams "AA","AA")
    assert list(primary[0]) == [2, 2]
    # second row is the remainder, padded
    assert list(primary[1]) == [1, 0]


def test_seq2vec_skips_unmatched_primary_char():
    """A primary character absent from the vocabulary is skipped (line 98)."""
    words = {"A": 1}
    # "Z" is not in the vocabulary -> no match -> skipped, no crash
    primary, secondary = seq2vec((["AZA"], ["HHH"]), words, seq_max_len=4)
    assert primary.shape[1] == 4
    # the two "A" tokens survive; "Z" is dropped
    assert list(primary[0]) == [1, 1, 0, 0]


def test_seq2vec_empty_when_nothing_tokenizes():
    """No matches yields zero-shaped arrays (line 119)."""
    words = {"A": 1}
    primary, secondary = seq2vec((["Z"], ["H"]), words, seq_max_len=4)
    assert primary.shape == (0, 4)
    assert secondary.shape == (0, 4)
    assert primary.dtype == np.float64
    assert secondary.dtype == np.float64
