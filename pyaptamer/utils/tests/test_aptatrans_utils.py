"""Tests for seq2vec."""

import numpy as np

from pyaptamer.utils._aptatrans_utils import seq2vec


def test_seq2vec_matches_docstring_example():
    words = {"AA": 1, "AC": 2, "A": 3}
    sequences = (["AAAC"], ["HHHC"])

    seq_out, ss_out = seq2vec(sequences, words, seq_max_len=4)

    expected_seq = np.array([[1.0, 2.0, 0.0, 0.0]])
    expected_ss = np.array([[9.0, 0.0, 0.0, 0.0]])

    np.testing.assert_array_equal(seq_out, expected_seq)
    np.testing.assert_array_equal(ss_out, expected_ss)


def test_seq2vec_empty_input_returns_zero_shaped_arrays():
    seq_out, ss_out = seq2vec(([], []), words={}, seq_max_len=5)

    assert seq_out.shape == (0, 5)
    assert ss_out.shape == (0, 5)


def test_seq2vec_splits_sequences_longer_than_seq_max_len():
    words = {"A": 1}
    sequences = (["AAAAA"], ["HHHHH"])

    seq_out, ss_out = seq2vec(sequences, words, seq_max_len=2)

    assert seq_out.shape[1] == 2
    assert seq_out.shape[0] == 3


def test_seq2vec_skips_unmatched_characters():
    # "Z" is not in the vocabulary at word lengths 1-3, forcing the
    # "skip character if no match found" branch (i += 1) to execute.
    words = {"A": 1}
    sequences = (["AZA"], ["HHH"])

    seq_out, _ = seq2vec(sequences, words, seq_max_len=5)

    # only the two "A" tokens match; "Z" is skipped entirely
    assert seq_out[0].tolist()[:2] == [1.0, 1.0]
