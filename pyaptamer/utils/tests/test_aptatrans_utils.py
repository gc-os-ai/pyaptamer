"""Tests for seq2vec."""

import numpy as np

from pyaptamer.utils._aptatrans_utils import seq2vec


def test_seq2vec_empty_input_returns_zero_shaped_arrays():
    """Check seq2vec returns zero-length, correctly shaped arrays for empty input."""
    seq_out, ss_out = seq2vec(([], []), words={}, seq_max_len=5)
    assert seq_out.shape == (0, 5)
    assert ss_out.shape == (0, 5)


def test_seq2vec_splits_sequences_longer_than_seq_max_len():
    """Check seq2vec splits and encodes a sequence longer than seq_max_len correctly."""
    words = {"A": 1, "C": 2}
    sequences = (["ACACA"], ["HBEGI"])

    seq_out, ss_out = seq2vec(sequences, words, seq_max_len=2)

    np.testing.assert_array_equal(seq_out, [[1, 2], [1, 2], [1, 0]])
    np.testing.assert_array_equal(ss_out, [[1, 2], [3, 4], [5, 0]])


def test_seq2vec_skips_unmatched_characters():
    """Check a character with no vocabulary match is skipped, not zero-padded."""
    words = {"A": 1, "C": 2}
    sequences = (["ACZCA"], ["HBEGI"])

    seq_out, ss_out = seq2vec(sequences, words, seq_max_len=3)

    np.testing.assert_array_equal(seq_out, [[1, 2, 2], [1, 0, 0]])
    np.testing.assert_array_equal(ss_out, [[1, 2, 4], [5, 0, 0]])
