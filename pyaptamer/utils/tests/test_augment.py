"""Test suite for the data augmentation utilities."""

__author__ = ["nennomp"]

import numpy as np
import pytest

from pyaptamer.utils._augment import augment_complement, augment_reverse


def test_augment_reverse_single_array():
    sequences = np.array(["AAC", "BBB", "ATCG"])
    result = augment_reverse(sequences)

    expected = (np.array(["AAC", "BBB", "ATCG", "CAA", "BBB", "GCTA"]),)
    assert len(result) == 1
    assert len(result[0]) == 6
    np.testing.assert_array_equal(result[0], expected[0])


def test_augment_reverse_multiple_arrays():
    seq1 = np.array(["ABC", "DEF"])
    seq2 = np.array(["XYZ"])
    seq3 = np.array(["123", "456", "789"])

    result = augment_reverse(seq1, seq2, seq3)

    expected = (
        np.array(["ABC", "DEF", "CBA", "FED"]),
        np.array(["XYZ", "ZYX"]),
        np.array(["123", "456", "789", "321", "654", "987"]),
    )
    assert len(result) == 3
    assert len(result[0]) == 4
    assert len(result[1]) == 2
    assert len(result[2]) == 6

    np.testing.assert_array_equal(result[0], expected[0])
    np.testing.assert_array_equal(result[1], expected[1])
    np.testing.assert_array_equal(result[2], expected[2])


# ---------- Tests for augment_complement ----------


class TestAugmentComplement:
    """Tests for the augment_complement function."""

    def test_rna_complement_basic(self):
        """Test basic RNA reverse complement: A<->U, C<->G."""
        seqs = np.array(["AUGC"])
        (result,) = augment_complement(seqs, molecule_type="rna")

        assert len(result) == 2
        # reverse complement of AUGC -> complement UACG -> reverse GCAU
        assert result[0] == "AUGC"
        assert result[1] == "GCAU"

    def test_dna_complement_basic(self):
        """Test basic DNA reverse complement: A<->T, C<->G."""
        seqs = np.array(["ATGC"])
        (result,) = augment_complement(seqs, molecule_type="dna")

        assert len(result) == 2
        # reverse complement of ATGC -> complement TACG -> reverse GCAT
        assert result[0] == "ATGC"
        assert result[1] == "GCAT"

    def test_rna_complement_preserves_case(self):
        """Test that case is preserved in complement."""
        seqs = np.array(["AuGc"])
        (result,) = augment_complement(seqs, molecule_type="rna")

        assert result[1] == "gCaU"

    def test_complement_doubles_array_size(self):
        """Test that output is exactly double the input."""
        seqs = np.array(["AAAA", "CCCC", "GGGG"])
        (result,) = augment_complement(seqs, molecule_type="rna")
        assert len(result) == 6

    def test_complement_multiple_arrays(self):
        """Test augmentation with multiple arrays."""
        seq1 = np.array(["AUGC"])
        seq2 = np.array(["GCAU", "AAAA"])
        r1, r2 = augment_complement(seq1, seq2, molecule_type="rna")

        assert len(r1) == 2
        assert len(r2) == 4

    def test_complement_invalid_molecule_type(self):
        """Test that invalid molecule_type raises ValueError."""
        seqs = np.array(["AUGC"])
        with pytest.raises(ValueError, match="molecule_type must be"):
            augment_complement(seqs, molecule_type="protein")

    def test_complement_roundtrip(self):
        """Test that double complement returns original sequences."""
        seqs = np.array(["AUGCUAGC"])
        (first,) = augment_complement(seqs, molecule_type="rna")
        # first[1] is the reverse complement; applying again should return original
        rc = first[1]
        (second,) = augment_complement(np.array([rc]), molecule_type="rna")
        assert second[1] == seqs[0]

    def test_dna_complement_poly_sequences(self):
        """Test complement of poly-nucleotide sequences."""
        seqs = np.array(["AAAA", "TTTT", "CCCC", "GGGG"])
        (result,) = augment_complement(seqs, molecule_type="dna")

        assert result[4] == "TTTT"  # rc(AAAA) = TTTT
        assert result[5] == "AAAA"  # rc(TTTT) = AAAA
        assert result[6] == "GGGG"  # rc(CCCC) = GGGG
        assert result[7] == "CCCC"  # rc(GGGG) = CCCC

