"""Test suite for the base generic utilities."""

__author__ = ["nennomp"]

import pytest

from pyaptamer.utils._base import compute_protein_word_frequencies, filter_words


def test_filter_words_basic_filtering():
    """Test filter_words with basic filtering logic."""
    # mean = (5 + 2 + 8 + 1) / 4 = 4.0
    # words above mean: apple (5.0), cherry (8.0)
    words = {"apple": 5.0, "banana": 2.0, "cherry": 8.0, "date": 1.0}

    result = filter_words(words)

    expected = {"apple": 1, "cherry": 2}
    assert result == expected


def test_filter_words_all_below_mean():
    """Test filter_words when all words are below the mean."""
    # mean = 1.0, no words above mean
    words = {"word1": 1.0, "word2": 1.0, "word3": 1.0}

    result = filter_words(words)

    assert result == {}


def test_filter_words_preserves_order():
    """Test filter_words preserves the order of words."""
    # mean = (10 + 8 + 6 + 2) / 4 = 6.5
    # words above mean: zebra (10.0), alpha (8.0)
    words = {"zebra": 10.0, "alpha": 8.0, "beta": 6.0, "gamma": 2.0}

    result = filter_words(words)

    # indices should reflect order of iteration over dict
    expected = {"zebra": 1, "alpha": 2}
    assert result == expected


class TestComputeProteinWordFrequencies:
    """Tests for compute_protein_word_frequencies function."""

    def test_basic_3mer(self):
        """Test computing 3-mer frequencies from simple sequences."""
        sequences = ["MKTVR", "MKTVE"]
        result = compute_protein_word_frequencies(sequences, n=3)
        expected = {"MKT": 2, "KTV": 2, "TVR": 1, "TVE": 1}
        assert result == expected

    def test_2mer(self):
        """Test computing 2-mer frequencies."""
        sequences = ["ABC", "ABD"]
        result = compute_protein_word_frequencies(sequences, n=2)
        expected = {"AB": 2, "BC": 1, "BD": 1}
        assert result == expected

    def test_1mer(self):
        """Test computing 1-mer frequencies (single amino acids)."""
        sequences = ["AAA", "AAB"]
        result = compute_protein_word_frequencies(sequences, n=1)
        expected = {"A": 5, "B": 1}
        assert result == expected

    def test_empty_sequences(self):
        """Test with empty list of sequences."""
        result = compute_protein_word_frequencies([], n=3)
        assert result == {}

    def test_sequences_shorter_than_n(self):
        """Test sequences shorter than n are skipped."""
        sequences = ["AB", "A"]
        result = compute_protein_word_frequencies(sequences, n=3)
        assert result == {}

    def test_mixed_case(self):
        """Test that sequences are converted to uppercase."""
        sequences = ["abc", "AbC"]
        result = compute_protein_word_frequencies(sequences, n=2)
        expected = {"AB": 2, "BC": 2}
        assert result == expected

    def test_non_alphabet_characters(self):
        """Test non-amino acid characters are included as-is (no validation)."""
        sequences = ["AX*Y", "AXY"]
        result = compute_protein_word_frequencies(sequences, n=2)
        expected = {"AX": 2, "X*": 1, "*Y": 1, "XY": 1}
        assert result == expected

    def test_parallel_words(self):
        """Test that overlapping n-grams are counted correctly."""
        sequences = ["AAAA"]
        result = compute_protein_word_frequencies(sequences, n=2)
        expected = {"AA": 3}
        assert result == expected

    def test_invalid_n(self):
        """Test that n < 1 raises ValueError."""
        sequences = ["ABC"]
        with pytest.raises(ValueError, match="n must be at least 1"):
            compute_protein_word_frequencies(sequences, n=0)
