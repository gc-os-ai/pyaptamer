"""Test suite for the AptaNet utility functions."""

import numpy as np
import pytest

from pyaptamer.utils._aptanet_utils import generate_kmer_vecs


class TestGenerateKmerVecs:
    """Tests for generate_kmer_vecs function."""

    def test_dna_sequence_returns_correct_shape(self):
        """Check that DNA sequences produce correct output shape."""
        seq = "ACGTACGT"
        result = generate_kmer_vecs(seq, k=2)
        # k=2: 4 unigrams + 16 bigrams = 20
        assert result.shape == (20,)

    def test_dna_sequence_k4_shape(self):
        """Check shape for default k=4 with DNA bases."""
        seq = "ACGTACGT"
        result = generate_kmer_vecs(seq, k=4)
        # 4 + 16 + 64 + 256 = 340
        assert result.shape == (340,)

    def test_frequencies_are_normalized(self):
        """Check that the output frequencies sum to 1."""
        seq = "ACGTACGT"
        result = generate_kmer_vecs(seq, k=2)
        assert np.isclose(result.sum(), 1.0)

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_various_k_values(self, k):
        """Check correct output shape for various k values."""
        seq = "ACGTACGT"
        result = generate_kmer_vecs(seq, k=k)
        expected_len = sum(4**i for i in range(1, k + 1))
        assert result.shape == (expected_len,)

    def test_single_base_sequence(self):
        """Check that a single base sequence works."""
        result = generate_kmer_vecs("A", k=1)
        assert result.shape == (4,)
        assert result.sum() > 0

    def test_empty_sequence(self):
        """Check that an empty sequence returns all zeros."""
        result = generate_kmer_vecs("", k=2)
        assert np.all(result == 0)

    def test_rna_sequence_raises_error(self):
        """Check that RNA sequences with 'U' raise a ValueError."""
        with pytest.raises(ValueError, match="non-DNA characters"):
            generate_kmer_vecs("ACGUACGU", k=2)

    def test_invalid_characters_raise_error(self):
        """Check that sequences with invalid characters raise a ValueError."""
        with pytest.raises(ValueError, match="non-DNA characters"):
            generate_kmer_vecs("ACGXACGT", k=2)

    def test_lowercase_dna_raises_error(self):
        """Check that lowercase DNA sequences raise a ValueError."""
        with pytest.raises(ValueError, match="non-DNA characters"):
            generate_kmer_vecs("acgtacgt", k=2)
