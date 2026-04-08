"""Test suite for the data augmentation utilities."""

__author__ = ["nennomp"]

import numpy as np

from pyaptamer.utils._augment import augment_reverse


def test_augment_reverse_dna_sequences():
    """Check reverse complement with standard DNA nucleotides."""
    sequences = np.array(["AACG", "TTTT", "ATCG"])
    result = augment_reverse(sequences)

    # AACG -> complement TTGC -> reverse CGTT
    # TTTT -> complement AAAA -> reverse AAAA
    # ATCG -> complement TAGC -> reverse CGAT
    expected = (np.array(["AACG", "TTTT", "ATCG", "CGTT", "AAAA", "CGAT"]),)
    assert len(result) == 1
    assert len(result[0]) == 6
    np.testing.assert_array_equal(result[0], expected[0])


def test_augment_reverse_rna_sequences():
    """Check reverse complement with RNA nucleotides (U instead of T).

    The complement table uses DNA convention (A→T) so U in the input is
    complemented to A, while A is complemented to T. This matches biological
    base-pairing rules and the downstream dna2rna() handles T→U conversion.
    """
    sequences = np.array(["ACGU", "AACU"])
    result = augment_reverse(sequences)

    # ACGU -> complement TGCA -> reverse ACGT
    # AACU -> complement TTGA -> reverse AGTT
    expected = (np.array(["ACGU", "AACU", "ACGT", "AGTT"]),)
    assert len(result) == 1
    assert len(result[0]) == 4
    np.testing.assert_array_equal(result[0], expected[0])


def test_augment_reverse_multiple_arrays():
    """Check augmentation works correctly across multiple arrays."""
    seq1 = np.array(["ACG", "TGA"])
    seq2 = np.array(["ACGT"])

    result = augment_reverse(seq1, seq2)

    # ACG -> complement TGC -> reverse CGT
    # TGA -> complement ACT -> reverse TCA
    # ACGT -> complement TGCA -> reverse ACGT (palindrome)
    expected = (
        np.array(["ACG", "TGA", "CGT", "TCA"]),
        np.array(["ACGT", "ACGT"]),
    )
    assert len(result) == 2
    assert len(result[0]) == 4
    assert len(result[1]) == 2

    np.testing.assert_array_equal(result[0], expected[0])
    np.testing.assert_array_equal(result[1], expected[1])


def test_augment_reverse_non_nucleotide_characters():
    """Check that non-nucleotide characters pass through unchanged."""
    sequences = np.array(["ANXG"])
    result = augment_reverse(sequences)

    # A->T, N->N (unchanged), X->X (unchanged), G->C
    # complement: TNXC -> reverse: CXNT
    expected = (np.array(["ANXG", "CXNT"]),)
    np.testing.assert_array_equal(result[0], expected[0])


def test_augment_reverse_empty_array():
    """Check that empty arrays are handled correctly."""
    sequences = np.array([], dtype=str)
    result = augment_reverse(sequences)

    assert len(result) == 1
    assert len(result[0]) == 0
