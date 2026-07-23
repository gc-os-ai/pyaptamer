"""Tests for seq2vec."""

from pyaptamer.utils._aptatrans_utils import seq2vec


def test_seq2vec_empty_input_returns_zero_shaped_arrays():
    """Check seq2vec returns zero-length, correctly shaped arrays for empty input."""
    seq_out, ss_out = seq2vec(([], []), words={}, seq_max_len=5)
    assert seq_out.shape == (0, 5)
    assert ss_out.shape == (0, 5)


def test_seq2vec_splits_sequences_longer_than_seq_max_len():
    """Check a sequence longer than seq_max_len is split across multiple rows."""
    words = {"A": 1}
    sequences = (["AAAAA"], ["HHHHH"])
    seq_out, ss_out = seq2vec(sequences, words, seq_max_len=2)
    assert seq_out.shape[1] == 2
    assert seq_out.shape[0] == 3


def test_seq2vec_skips_unmatched_characters():
    """Check a character with no vocabulary match is skipped rather than erroring."""
    # "Z" is not in the vocabulary at word lengths 1-3, forcing the
    # "skip character if no match found" branch (i += 1) to execute.
    words = {"A": 1}
    sequences = (["AZA"], ["HHH"])
    seq_out, _ = seq2vec(sequences, words, seq_max_len=5)
    # only the two "A" tokens match; "Z" is skipped entirely
    assert seq_out[0].tolist()[:2] == [1.0, 1.0]
